# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import multiprocessing as mp
import os
import re
import shutil
from argparse import ArgumentParser

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_MIN = torch.finfo(torch.float8_e4m3fn).min


# All linear weight prefixes that should be MXFP8 quantized in GLM-5.2
FP8_PATTERNS = [
    r"model\.layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa$",
    r"model\.layers\.(\d+)\.self_attn\.kv_b_proj$",
    r"model\.layers\.(\d+)\.self_attn\.q_a_proj$",
    r"model\.layers\.(\d+)\.self_attn\.q_b_proj$",
    r"model\.layers\.(\d+)\.self_attn\.o_proj$",
    r"model\.layers\.(\d+)\.self_attn\.indexer\.wk$",
    r"model\.layers\.(\d+)\.self_attn\.indexer\.wq_b$",
    r"model\.layers\.(\d+)\.mlp\.(?:gate_proj|up_proj|down_proj)$",
    r"model\.layers\.(\d+)\.mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj)$",
    r"model\.layers\.(\d+)\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$",
]
FP8_PATTERNS_COMPILED = [re.compile(p) for p in FP8_PATTERNS]


def match_fp8_pattern(prefix):
    for pat in FP8_PATTERNS_COMPILED:
        if pat.match(prefix):
            return True
    return False


def mxfp8_quantize(weight, block_size=32):
    """Quantize a 2D weight (out_dim, in_dim) to MXFP8 with ue8m0 scale"""
    out_dim, in_dim = weight.shape

    pad_cols = (block_size - in_dim % block_size) % block_size
    if pad_cols > 0:
        weight = torch.nn.functional.pad(weight, (0, pad_cols))

    padded_in_dim = weight.shape[1]
    num_blocks = padded_in_dim // block_size

    weight_blocks = weight.float().reshape(out_dim, num_blocks, block_size)

    # Compute max abs per block -> (out_dim, num_blocks)
    max_abs = weight_blocks.abs().amax(dim=-1)

    # ue8m0 scale: power-of-2 scaling
    scale = max_abs / FP8_MAX
    scale = scale.clamp(min=2**-127)
    log2_scale = torch.ceil(torch.log2(scale))
    exponent = (log2_scale + 127).clamp(0, 254).to(torch.uint8)
    actual_scale = torch.pow(2.0, exponent.float() - 127.0)

    # Quantize
    scale_inv = (1.0 / actual_scale).unsqueeze(-1)  # (out_dim, num_blocks, 1)
    quantized = torch.clamp(weight_blocks * scale_inv, min=FP8_MIN, max=FP8_MAX).to(
        torch.float8_e4m3fn
    )

    # Reshape back and trim padding
    quantized = quantized.reshape(out_dim, padded_in_dim)[:, :in_dim].contiguous()
    scale_ue8m0 = exponent.contiguous()  # (out_dim, num_blocks)

    return quantized, scale_ue8m0


def process_shard(
    rank, file_names, input_path, output_path, return_dict, ignore_dict, block_size=32
):
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    device = f"cuda:{rank % world_size}" if torch.cuda.is_available() else "cpu"

    local_ignore = []

    for file_name in tqdm(file_names, desc=f"Worker {rank}"):
        state_dict = {}
        index = {}

        with safe_open(os.path.join(input_path, file_name), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for weight_name in keys:
                tensor = f.get_tensor(weight_name)

                if weight_name.endswith(".weight"):
                    prefix = weight_name[: -len(".weight")]

                    if match_fp8_pattern(prefix):
                        w = tensor.to(device)
                        quant_w, scale_ue8m0 = mxfp8_quantize(w, block_size=block_size)
                        state_dict[weight_name] = quant_w.cpu()
                        state_dict[f"{prefix}.weight_scale"] = scale_ue8m0.cpu()
                        index[weight_name] = file_name
                        index[f"{prefix}.weight_scale"] = file_name
                        del w, quant_w, scale_ue8m0
                        continue
                    else:
                        if tensor.ndim == 2:
                            local_ignore.append(prefix)

                state_dict[weight_name] = tensor
                index[weight_name] = file_name

        save_file(state_dict, os.path.join(output_path, file_name))
        del state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return_dict[file_name] = index

    ignore_dict[rank] = local_ignore


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--block_size",
        type=int,
        default=32,
        help="Block size for MXFP8 quantization (default: 32)",
    )
    parser.add_argument(
        "--quant_config_type",
        type=str,
        default="modelopt_mxfp8",
        choices=["mxfp8", "modelopt_mxfp8"],
        help="Config format: 'mxfp8' or 'modelopt_mxfp8' (default: modelopt_mxfp8)",
    )
    args = parser.parse_args()
    print(args)

    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    config_path = os.path.join(input_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    if "quantization_config" in config:
        raise AssertionError("Input model already has quantization_config, not supported.")

    num_hidden_layers = config["num_hidden_layers"]
    print(f"num_hidden_layers: {num_hidden_layers}")

    index_path = os.path.join(input_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        model_index = json.load(f)
    safetensor_files = sorted(set(model_index["weight_map"].values()))
    print(f"Found {len(safetensor_files)} safetensor files")

    num_workers = min(args.num_workers, len(safetensor_files))
    file_subsets = [safetensor_files[i::num_workers] for i in range(num_workers)]

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    ignore_dict = manager.dict()

    processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=process_shard,
            args=(
                i,
                file_subsets[i],
                input_path,
                output_path,
                return_dict,
                ignore_dict,
                args.block_size,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    ignored_layers = []
    for worker_list in ignore_dict.values():
        ignored_layers.extend(worker_list)
    ignored_layers = sorted(set(ignored_layers))
    print(f"ignored_layers: {len(ignored_layers)} entries")
    for layer in ignored_layers:
        print(f"  {layer}")

    new_weight_map = {}
    for result in return_dict.values():
        new_weight_map.update(result)

    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {"metadata": {}, "weight_map": dict(sorted(new_weight_map.items()))}, f, indent=2
        )
    print("Saved model.safetensors.index.json")

    out_config = dict(config)
    if args.quant_config_type == "mxfp8":
        out_config["quantization_config"] = {
            "quant_method": "mxfp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [1, args.block_size],
            "ignored_layers": ignored_layers,
        }
    elif args.quant_config_type == "modelopt_mxfp8":
        out_config["quantization_config"] = {
            "quant_method": "modelopt",
            "quantization": {
                "quant_algo": "MXFP8",
                "kv_cache_quant_algo": None,
                "exclude_modules": ignored_layers,
            },
        }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(out_config, f, indent=4)
    print("Saved config.json")

    generated_files = {"model.safetensors.index.json", "config.json"}
    for item in os.listdir(input_path):
        if item.endswith(".safetensors"):
            continue
        if item in generated_files:
            continue
        src = os.path.join(input_path, item)
        dst = os.path.join(output_path, item)
        if os.path.exists(dst):
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"  Copied {item}")

    print(f"\nDone! Output: {output_path}")


if __name__ == "__main__":
    main()
