# This file is based on DeepSeek code (MIT License).
#
# Original code:
#   Copyright (c) 2023 DeepSeek
#   https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py
#   https://huggingface.co/meituan/DeepSeek-R1-Channel-INT8/blob/main/inference/bf16_cast_channel_int8.py (Meituan fork) # noqa: E501
#
# Additional contributions:
#   Copyright (c) 2026 Kunlunxin (Beijing) Technology Co., Ltd. (Kunlunxin)
#
# Modifications:
# - Merged implementations
# - Added multi-GPU parallel processing
#
# SPDX-License-Identifier: Apache-2.0 AND MIT

import json
import os
import shutil
from argparse import ArgumentParser
from glob import glob

import torch
import torch.multiprocessing as mp
from safetensors.torch import safe_open, save_file

from angelslim.compressor.quant.core.quant_func import weight_dequant

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
    ".indexer.wq_b.weight",
    ".indexer.wk.weight",
]


def process_worker(
    worker_id,
    safetensor_files,
    input_path,
    int8_path,
    weight_map,
    return_dict,
    input_type="bf16",
):
    """
    Process worker.

    Each worker process is responsible for a subset of safetensor files:
    - FP8 → BF16 dequantization
    - BF16 → INT8 quantization
    - Generation of the updated weight_map
    """
    num_gpus = torch.cuda.device_count()
    rank = worker_id % num_gpus
    torch.cuda.set_device(rank)
    quant_count = 0
    new_weight_map = {}
    for safetensor_file in safetensor_files:
        file_name = os.path.basename(safetensor_file)
        print(f"[Worker {worker_id}][GPU {rank}] processing {file_name}")
        with safe_open(safetensor_file, framework="pt", device=f"cuda:{rank}") as f:
            new_state_dict = {}
            keys = set(f.keys())
            for weight_name in keys:
                weight = f.get_tensor(weight_name)
                if any(weight_name.endswith(suffix) for suffix in SUFFIX_TO_QUANT):
                    quant_count += 1
                    if input_type == "fp8":
                        scale_inv_name = f"{weight_name}_scale_inv"
                        scale_inv = get_tensor_from_file(
                            rank, scale_inv_name, weight_map, input_path
                        )
                        weight_bf16 = weight_dequant(weight, scale_inv)
                    else:
                        weight_bf16 = weight
                    int8_weight, scale_inv = weight_quant(weight_bf16)
                    new_state_dict[weight_name] = int8_weight
                    new_scale_name = f"{weight_name}_scale"
                    new_state_dict[new_scale_name] = scale_inv
                    new_weight_map[weight_name] = file_name
                    new_weight_map[new_scale_name] = file_name
                else:
                    if weight_name.endswith("_scale_inv"):
                        continue
                    new_state_dict[weight_name] = weight
                    new_weight_map[weight_name] = file_name

        new_safetensor_file = os.path.join(int8_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
    return_dict[worker_id] = (quant_count, new_weight_map)


# Helper function to get tensor from the correct file
def get_tensor_from_file(rank, tensor_name, weight_map, input_path):
    """
    Retrieves a tensor from mmap safe_tensors

    Args:
        tensor_name (str): The name of the tensor to retrieve.

    Returns:
        torch.Tensor: The retrieved tensor.

    Raises:
        KeyError: If the tensor does not exist in the safetensor file.
    """
    torch.cuda.set_device(rank)
    file_name = weight_map[tensor_name]
    file_path = os.path.join(input_path, file_name)

    with safe_open(file_path, framework="pt", device=f"cuda:{rank}") as f:
        return f.get_tensor(tensor_name)


def weight_quant(tensor: torch.Tensor):
    """
    Quantize a 2D tensor row-wise from BF16/FP32 to INT8.
    Args:
        tensor (torch.Tensor): Input 2D tensor.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Quantized INT8 tensor.
            - Scale tensor (float32) used for quantization.
    """
    assert tensor.dim() == 2
    qmax = 127.0
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]  # [rows, 1]
    scale = abs_max / qmax  # [rows, 1]
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)


def main(input_path, int8_path, num_workers):
    """
    Run the FP8-to-INT8 per-channel quantization pipeline.

    This function:
        1. Copy the config file
        2. Loads FP8 safetensors.
        3. Dequantizes FP8 → BF16, then quantizes BF16 → INT8.
        4. Saves quantized safetensors and updates model index.

    Args:
        input_path (str): Path to directory containing FP8 safetensors.
        int8_path (str): Output directory to save INT8 safetensors.
        num_workers (int): Number of processing workers
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(int8_path, exist_ok=True)
    model_index_file = os.path.join(int8_path, "model.safetensors.index.json")
    config_file = os.path.join(int8_path, "config.json")

    for fname in os.listdir(input_path):
        if fname.endswith(".safetensors"):
            continue
        src = os.path.join(input_path, fname)
        dst = os.path.join(int8_path, fname)
        if os.path.isdir(src):
            print(f"cp -r {src} {dst}")
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif os.path.isfile(src):
            print(f"cp {src} {dst}")
            shutil.copy2(src, dst)

    # modify config.json and save it
    config = json.load(open(config_file))
    # delete quantization_config
    quant_config = config.pop("quantization_config", None)
    input_type = "bf16"
    if quant_config is not None:
        input_type = quant_config.get("quant_method", input_type)
    print("input_type", input_type)
    config["quantization_config"] = {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "memoryless",
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int",
                },
                "output_activations": None,
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int",
                },
                "targets": ["Linear"],
            }
        },
        "format": "int-quantized",
        "ignore": ["lm_head"],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"config.json modified and saved to {config_file}")

    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    safetensor_files = list(glob(os.path.join(input_path, "*.safetensors")))
    safetensor_files.sort()
    quant_count = 0
    new_weight_map = {}

    file_subsets = [safetensor_files[i::num_workers] for i in range(num_workers)]

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=process_worker,
            args=(
                i,
                file_subsets[i],
                input_path,
                int8_path,
                weight_map,
                return_dict,
                input_type,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    for i in range(num_workers):
        qc, wm = return_dict[i]
        quant_count += qc
        new_weight_map.update(wm)
    print(f"{quant_count} weights are quantized.")

    # modify model.safetensors.index.json
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    model_index["weight_map"] = new_weight_map
    with open(model_index_file, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"model.safetensors.index.json modified and saved to {model_index_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-int8-path", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=32)

    args = parser.parse_args()
    main(args.input_path, args.output_int8_path, args.num_workers)
    print("done")
