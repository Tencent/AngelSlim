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

# All linear weight prefixes that should be FP8 block-wise quantized in GLM-5.2
FP8_PATTERNS = [
    # Attention projections
    r"model\.layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa$",
    r"model\.layers\.(\d+)\.self_attn\.kv_b_proj$",
    r"model\.layers\.(\d+)\.self_attn\.q_a_proj$",
    r"model\.layers\.(\d+)\.self_attn\.q_b_proj$",
    r"model\.layers\.(\d+)\.self_attn\.o_proj$",
    # Indexer
    r"model\.layers\.(\d+)\.self_attn\.indexer\.wk$",
    r"model\.layers\.(\d+)\.self_attn\.indexer\.wq_b$",
    # Dense MLP (first_k_dense_replace layers)
    r"model\.layers\.(\d+)\.mlp\.(?:gate_proj|up_proj|down_proj)$",
    # Shared experts
    r"model\.layers\.(\d+)\.mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj)$",
    # Routed experts
    r"model\.layers\.(\d+)\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$",
]
FP8_PATTERNS_COMPILED = [re.compile(p) for p in FP8_PATTERNS]


def match_fp8_pattern(prefix):
    for pat in FP8_PATTERNS_COMPILED:
        if pat.match(prefix):
            return True
    return False


def fp8_blockwise_quantize(weight, block_size=(128, 128)):
    block_size_m, block_size_n = block_size
    rows, cols = weight.shape

    if block_size_m == -1 or block_size_m > rows:
        block_size_m = rows
    if block_size_n == -1 or block_size_n > cols:
        block_size_n = cols

    pad_rows = (block_size_m - rows % block_size_m) % block_size_m
    pad_cols = (block_size_n - cols % block_size_n) % block_size_n
    if pad_rows > 0 or pad_cols > 0:
        weight = torch.nn.functional.pad(weight, (0, pad_cols, 0, pad_rows))

    padded_rows, padded_cols = weight.shape
    num_blocks_m = padded_rows // block_size_m
    num_blocks_n = padded_cols // block_size_n

    weight_blocks = weight.float().reshape(num_blocks_m, block_size_m, num_blocks_n, block_size_n)
    weight_blocks = weight_blocks.permute(0, 2, 1, 3)

    max_abs = weight_blocks.abs().amax(dim=(-1, -2))
    scale_inv = FP8_MAX / max_abs.clamp(min=1e-12)
    scale_inv_expanded = scale_inv.unsqueeze(-1).unsqueeze(-1)

    quantized = torch.clamp(weight_blocks * scale_inv_expanded, min=FP8_MIN, max=FP8_MAX).to(
        torch.float8_e4m3fn
    )

    quantized = quantized.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)
    quantized = quantized[:rows, :cols].contiguous()

    # scale_inv stores 1/scale (i.e. multiply by this to dequantize)
    weight_scale = max_abs / FP8_MAX
    weight_scale = weight_scale.to(torch.bfloat16).contiguous()

    return quantized, weight_scale


def process_shard(rank, file_names, input_path, output_path, block_size, return_dict, ignore_dict):
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
                        quant_w, scale = fp8_blockwise_quantize(w, block_size)
                        state_dict[weight_name] = quant_w.cpu()
                        state_dict[f"{weight_name}_scale_inv"] = scale.cpu()
                        index[weight_name] = file_name
                        index[f"{weight_name}_scale_inv"] = file_name
                        del w, quant_w, scale
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
    parser.add_argument("--block_size", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    print(args)

    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    # Load config
    config_path = os.path.join(input_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    if "quantization_config" in config:
        raise AssertionError("Input model already has quantization_config, not supported.")

    num_hidden_layers = config["num_hidden_layers"]
    print(f"num_hidden_layers: {num_hidden_layers}")

    # Load weight map
    index_path = os.path.join(input_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        model_index = json.load(f)
    safetensor_files = sorted(set(model_index["weight_map"].values()))
    print(f"Found {len(safetensor_files)} safetensor files")

    # Process shards with multiprocessing
    block_size = tuple(args.block_size)
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
                block_size,
                return_dict,
                ignore_dict,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # Collect ignored layers
    ignored_layers = []
    for worker_list in ignore_dict.values():
        ignored_layers.extend(worker_list)
    ignored_layers = sorted(set(ignored_layers))
    print(f"modules_to_not_convert: {len(ignored_layers)} entries")
    for layer in ignored_layers:
        print(f"  {layer}")

    # Build new weight map
    new_weight_map = {}
    for result in return_dict.values():
        new_weight_map.update(result)

    # Write model.safetensors.index.json
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {"metadata": {}, "weight_map": dict(sorted(new_weight_map.items()))}, f, indent=2
        )
    print("Saved model.safetensors.index.json")

    # Write config.json with quantization_config
    out_config = dict(config)
    out_config["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": list(block_size),
        "modules_to_not_convert": ignored_layers,
    }
    # KV cache config: latent kv nope uses FP8 (1x128), rope part stays bf16

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(out_config, f, indent=4)
    print("Saved config.json")

    # Copy other files
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
