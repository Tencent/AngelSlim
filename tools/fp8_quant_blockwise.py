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
import math
import multiprocessing as mp
import os
import shutil
from argparse import ArgumentParser

import accelerate
import torch
from safetensors.torch import safe_open, save_file
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from angelslim.utils import find_layers

SUFFIX_TO_QUANT = [
    ".gate_and_up_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
    ".q_a_proj.weight",
    ".q_b_proj.weight",
    ".kv_a_proj_with_mqa.weight",
    ".kv_b_proj.weight",
    ".qkv_proj.weight",
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
]


def create_quantized_param(param, weight_block_size=(128, 128)):
    """
    Quantizes weights to FP8 format using Block-wise quantization
    """
    # Get FP8 min/max values
    fp8_min = torch.finfo(torch.float8_e4m3fn).min
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    block_size_m, block_size_n = weight_block_size
    rows, cols = param.shape[-2:]

    # Tensor-wise
    if block_size_m == -1 or block_size_m > rows:
        block_size_m = rows
    if block_size_n == -1 or block_size_n > cols:
        block_size_n = cols

    if rows % block_size_m != 0:
        pad = torch.zeros(
            [*param.shape[:-2], block_size_m - rows % block_size_m, cols],
            dtype=param.dtype,
            device=param.device,
        )
        param = torch.concat([param, pad], dim=-2)
    if cols % block_size_n != 0:
        pad = torch.zeros(
            [*param.shape[:-2], rows, block_size_n - cols % block_size_n],
            dtype=param.dtype,
            device=param.device,
        )
        param = torch.concat([param, pad], dim=-1)
    param_value_shape = param.shape

    param_value = (
        param.float()
        .reshape(
            -1,
            math.ceil(rows / block_size_m),
            block_size_m,
            math.ceil(cols // block_size_n),
            block_size_n,
        )
        .permute(0, 1, 3, 2, 4)
    )

    # Calculate scaling factor for each block
    max_abs = torch.amax(torch.abs(param_value), dim=(-1, -2))
    scale_inv = fp8_max / max_abs
    scale_orig_shape = scale_inv.shape
    scale_inv = scale_inv.unsqueeze(-1).unsqueeze(-1)

    # Quantize the weights
    quantized_param = torch.clamp(param_value * scale_inv, min=fp8_min, max=fp8_max).to(
        torch.float8_e4m3fn
    )
    quantized_param = quantized_param.permute(0, 1, 3, 2, 4)
    quantized_param = quantized_param.reshape(param_value_shape)[..., :rows, :cols]

    scale_inv = scale_inv.reshape(scale_orig_shape).squeeze().reciprocal()

    return quantized_param.contiguous(), scale_inv.contiguous()


def process_safetensor(rank, file_name, input_path, output_path, block_size=(128, 128)):
    state_dict = {}
    index = {}
    count = 0
    with safe_open(
        os.path.join(input_path, file_name), framework="pt", device=f"cuda:{rank}"
    ) as f:
        print(f"Processing {file_name} with {len(f.keys())} weights")
        for weight_name in f.keys():
            weight = f.get_tensor(weight_name)
            if any(weight_name.endswith(suffix) for suffix in SUFFIX_TO_QUANT):
                quant_weight, scale = create_quantized_param(weight, block_size)
                state_dict[weight_name] = quant_weight
                index[weight_name] = file_name

                # Reference: https://github.com/vllm-project/vllm/blob/v0.10.1/vllm/model_executor/layers/quantization/fp8.py#L295  # noqa: E501
                if block_size[0] == -1 and block_size[1] == -1:
                    # Tensor-wise
                    state_dict[f"{weight_name}_scale"] = scale
                    index[f"{weight_name}_scale"] = file_name
                else:
                    # Block-wise
                    state_dict[f"{weight_name}_scale_inv"] = scale
                    index[f"{weight_name}_scale_inv"] = file_name
            else:
                state_dict[weight_name] = weight
                index[weight_name] = file_name
            count += 1

    new_safetensor_file = os.path.join(output_path, file_name)
    save_file(state_dict, new_safetensor_file)
    return index


def worker(i, file_names, input_path, output_path, block_size, return_dict):
    world_size = torch.cuda.device_count()
    for file_name in tqdm(file_names, desc=f"Worker {i}"):
        index = process_safetensor(
            i % world_size, file_name, input_path, output_path, block_size
        )
        return_dict[file_name] = index


def main(input_path, output_path, block_size):
    os.makedirs(output_path, exist_ok=True)
    model_index_file = os.path.join(input_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    safetensor_files = set(weight_map.values())
    safetensor_files = list(sorted(safetensor_files))
    print(f"Found {len(safetensor_files)} safetensor files")

    # Analyze model structure to find ignored layers
    config = AutoConfig.from_pretrained(input_path)
    with accelerate.init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    layers = find_layers(model, [nn.Linear])
    print(f"Found {len(layers)} linear layers")
    ignored_layers = []
    for name, _ in layers.items():
        weight_name = f"{name}.weight"
        if not any(weight_name.endswith(suffix) for suffix in SUFFIX_TO_QUANT):
            ignored_layers.append(name)
    print(f"Ignored layers: {ignored_layers}")
    del model

    args.num_workers = min(args.num_workers, len(safetensor_files))
    file_subsets = [
        safetensor_files[i :: args.num_workers] for i in range(args.num_workers)
    ]
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    for i in range(args.num_workers):
        p = mp.Process(
            target=worker,
            args=(i, file_subsets[i], input_path, output_path, block_size, return_dict),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    index = {}
    for result in return_dict.values():
        index.update(result)
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": index}, f, indent=2)

    # Copy config file
    for file in os.listdir(input_path):
        if (
            file.endswith(".py")
            or file.endswith(".json")
            or file.endswith(".md")
            or file.endswith(".txt")
        ):
            src_path = os.path.join(input_path, file)
            dst_path = os.path.join(output_path, file)
            if os.path.exists(dst_path):
                continue
            print(f"cp {src_path} {dst_path}")
            shutil.copy2(src_path, dst_path)

    # Quantization config
    with open(os.path.join(output_path, "config.json"), "r") as f:
        config = json.load(f)
    config["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "modules_to_not_convert": ignored_layers,
    }
    if block_size[0] != -1 and block_size[1] != -1:
        config["quantization_config"]["weight_block_size"] = block_size
    print(f"quant config: {config['quantization_config']}")
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--block_size", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()
    print(args)
    with open(os.path.join(args.input_path, "config.json"), "r", encoding="utf8") as fp:
        json_data = json.load(fp)
        print(json_data)
    if "quantization_config" in json_data.keys():
        raise AssertionError("NOT SUPPORT FP8 DS")

    main(args.input_path, args.output_path, args.block_size)
