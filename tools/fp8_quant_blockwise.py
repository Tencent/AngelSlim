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
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

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
    ".experts.gate_up_proj",
    ".experts.down_proj",
]

# Qwen3.5-specific extra suffixes not covered by the generic SUFFIX_TO_QUANT.
# linear_attn projections: in_proj_qkv, in_proj_z, out_proj are quantized
# (consistent with Qwen/Qwen3.5-35B-A3B-FP8 on HuggingFace).
# in_proj_a, in_proj_b, conv1d, A_log, dt_bias, norm are NOT quantized.
# mtp.fc.weight is also NOT quantized (matches the official checkpoint).
# Router gates (mlp.gate / shared_expert_gate) are NOT quantized.
QWEN35_EXTRA_SUFFIX_TO_QUANT = [
    ".linear_attn.in_proj_qkv.weight",
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.out_proj.weight",
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
    original_device = param.device

    # Tensor-wise
    if block_size_m == -1 or block_size_m > rows:
        block_size_m = rows
    if block_size_n == -1 or block_size_n > cols:
        block_size_n = cols

    # Move to CPU for padding to save GPU memory
    param = param.cpu()

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

    # Convert to float on CPU first
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

    # Move back to GPU for quantization
    param_value = param_value.to(original_device)
    del param  # Free CPU memory
    torch.cuda.empty_cache()

    # Calculate scaling factor for each block
    max_abs = torch.amax(torch.abs(param_value), dim=(-1, -2))
    scale_inv = fp8_max / max_abs
    scale_orig_shape = scale_inv.shape
    scale_inv = scale_inv.unsqueeze(-1).unsqueeze(-1)

    # Quantize the weights
    quantized_param = torch.clamp(param_value * scale_inv, min=fp8_min, max=fp8_max).to(
        torch.float8_e4m3fn
    )
    del param_value  # Free GPU memory
    torch.cuda.empty_cache()

    quantized_param = quantized_param.permute(0, 1, 3, 2, 4)
    quantized_param = quantized_param.reshape(param_value_shape)[..., :rows, :cols]

    scale_inv = scale_inv.reshape(scale_orig_shape).squeeze().reciprocal().to(torch.bfloat16)

    return quantized_param.contiguous(), scale_inv.contiguous()


def process_safetensor(rank, file_name, input_path, output_path, block_size=(128, 128)):
    state_dict = {}
    index = {}
    count = 0

    # Load tensors on CPU first to avoid GPU memory issues
    with safe_open(os.path.join(input_path, file_name), framework="pt", device="cpu") as f:
        print(f"Processing {file_name} with {len(f.keys())} weights")
        for weight_name in f.keys():
            weight = f.get_tensor(weight_name)
            if any(weight_name.endswith(suffix) for suffix in SUFFIX_TO_QUANT):
                # Move to GPU only for quantization
                weight = weight.to(f"cuda:{rank}")
                quant_weight, scale = create_quantized_param(weight, block_size)

                # Move back to CPU for saving
                state_dict[weight_name] = quant_weight.cpu()
                index[weight_name] = file_name

                # Reference: https://github.com/vllm-project/vllm/blob/v0.10.1/vllm/model_executor/layers/quantization/fp8.py#L295  # noqa: E501
                if block_size[0] == -1 and block_size[1] == -1:
                    # Tensor-wise
                    state_dict[f"{weight_name}_scale"] = scale.cpu()
                    index[f"{weight_name}_scale"] = file_name
                else:
                    # Block-wise
                    state_dict[f"{weight_name}_scale_inv"] = scale.cpu()
                    index[f"{weight_name}_scale_inv"] = file_name

                # Clean up GPU memory after each weight
                del weight, quant_weight, scale
                torch.cuda.empty_cache()
            else:
                state_dict[weight_name] = weight
                index[weight_name] = file_name
            count += 1

    new_safetensor_file = os.path.join(output_path, file_name)
    save_file(state_dict, new_safetensor_file)

    # Final cleanup
    del state_dict
    torch.cuda.empty_cache()

    return index


def _quant_and_record(weight_name, weight, rank, block_size, state_dict, index, file_name):
    """Quantize a single weight tensor and store result + scale_inv into state_dict/index."""
    weight = weight.to(f"cuda:{rank}")
    quant_weight, scale = create_quantized_param(weight, block_size)
    state_dict[weight_name] = quant_weight.cpu()
    index[weight_name] = file_name
    if block_size[0] == -1 and block_size[1] == -1:
        state_dict[f"{weight_name}_scale"] = scale.cpu()
        index[f"{weight_name}_scale"] = file_name
    else:
        state_dict[f"{weight_name}_scale_inv"] = scale.cpu()
        index[f"{weight_name}_scale_inv"] = file_name
    del weight, quant_weight, scale
    torch.cuda.empty_cache()


def process_safetensor_qwen35(rank, file_name, input_path, output_path, block_size=(128, 128)):
    """
    Variant of process_safetensor for Qwen3.5 (model_type=qwen3_5_moe).

    Differences from the generic version:
    1. Main-model experts are stored as batched tensors
       `experts.gate_up_proj` [N, 2*I, H] and `experts.down_proj` [N, H, I].
       These are split into per-expert keys before quantization:
         experts.{i}.gate_proj.weight, experts.{i}.up_proj.weight
             (split along dim-0 of gate_up_proj)
         experts.{i}.down_proj.weight
             (slice along dim-0 of down_proj)
    2. mtp.* experts are already per-expert in the checkpoint, handled by normal SUFFIX_TO_QUANT.
    3. mtp.fc.weight is a plain linear that is not covered by SUFFIX_TO_QUANT; it is
       added via QWEN35_EXTRA_SUFFIX_TO_QUANT.
    4. Router gates (mlp.gate, shared_expert_gate) are NOT quantized.
    """
    suffix_to_quant = SUFFIX_TO_QUANT + QWEN35_EXTRA_SUFFIX_TO_QUANT
    state_dict = {}
    index = {}

    with safe_open(os.path.join(input_path, file_name), framework="pt", device="cpu") as f:
        print(f"Processing (qwen35) {file_name} with {len(f.keys())} weights")
        for weight_name in f.keys():
            weight = f.get_tensor(weight_name)

            # ------------------------------------------------------------------
            # Batched expert tensors: split and quantize per expert
            # ------------------------------------------------------------------
            if weight_name.endswith(".experts.gate_up_proj"):
                # weight shape: [num_experts, 2*intermediate, hidden]
                num_experts = weight.shape[0]
                # gate_up_proj is stored as [num_experts, 2*intermediate, hidden]
                # split into gate (first half) and up (second half) along dim=1
                gate_w, up_w = weight.chunk(2, dim=1)
                prefix = weight_name[: -len(".experts.gate_up_proj")]
                for i in range(num_experts):
                    gate_name = f"{prefix}.experts.{i}.gate_proj.weight"
                    up_name = f"{prefix}.experts.{i}.up_proj.weight"
                    _quant_and_record(
                        gate_name, gate_w[i], rank, block_size, state_dict, index, file_name
                    )
                    _quant_and_record(
                        up_name, up_w[i], rank, block_size, state_dict, index, file_name
                    )
                del weight, gate_w, up_w
                torch.cuda.empty_cache()
                continue

            if weight_name.endswith(".experts.down_proj"):
                # weight shape: [num_experts, hidden, intermediate]
                num_experts = weight.shape[0]
                prefix = weight_name[: -len(".experts.down_proj")]
                for i in range(num_experts):
                    down_name = f"{prefix}.experts.{i}.down_proj.weight"
                    _quant_and_record(
                        down_name, weight[i], rank, block_size, state_dict, index, file_name
                    )
                del weight
                torch.cuda.empty_cache()
                continue

            # ------------------------------------------------------------------
            # Regular tensors: quantize if suffix matches, else copy verbatim
            # ------------------------------------------------------------------
            if any(weight_name.endswith(suffix) for suffix in suffix_to_quant):
                _quant_and_record(
                    weight_name, weight, rank, block_size, state_dict, index, file_name
                )
            else:
                state_dict[weight_name] = weight
                index[weight_name] = file_name

    new_safetensor_file = os.path.join(output_path, file_name)
    save_file(state_dict, new_safetensor_file)
    del state_dict
    torch.cuda.empty_cache()
    return index


def worker(i, file_names, input_path, output_path, block_size, return_dict):
    world_size = torch.cuda.device_count()
    for file_name in tqdm(file_names, desc=f"Worker {i}"):
        index = process_safetensor(i % world_size, file_name, input_path, output_path, block_size)
        return_dict[file_name] = index


def worker_qwen35(i, file_names, input_path, output_path, block_size, return_dict):
    world_size = torch.cuda.device_count()
    for file_name in tqdm(file_names, desc=f"Worker(qwen35) {i}"):
        index = process_safetensor_qwen35(
            i % world_size, file_name, input_path, output_path, block_size
        )
        return_dict[file_name] = index


def main(input_path, output_path, block_size, num_workers=8):
    os.makedirs(output_path, exist_ok=True)

    # Check if model.safetensors.index.json exists, otherwise use model.safetensors
    model_index_file = os.path.join(input_path, "model.safetensors.index.json")
    has_index = os.path.exists(model_index_file)
    if has_index:
        with open(model_index_file, "r") as f:
            model_index = json.load(f)
        weight_map = model_index["weight_map"]
        safetensor_files = set(weight_map.values())
        safetensor_files = list(sorted(safetensor_files))
    else:
        # If no index file, directly use model.safetensors
        safetensor_files = ["model.safetensors"]
    print(f"Found {len(safetensor_files)} safetensor files")

    # Analyze model structure to find ignored layers
    config = AutoConfig.from_pretrained(input_path)
    model_type = config.model_type
    with accelerate.init_empty_weights():
        if model_type == "qwen3_5_moe":
            model = Qwen3_5MoeForConditionalGeneration._from_config(config)
        elif model_type == "qwen3_5":
            model = Qwen3_5ForConditionalGeneration._from_config(config)
        elif model_type == "qwen3_vl_moe":
            model = Qwen3VLMoeForConditionalGeneration._from_config(config)
        elif model_type == "qwen3_vl":
            model = Qwen3VLForConditionalGeneration._from_config(config)
        else:
            model = AutoModelForCausalLM.from_config(config)
    if model_type == "qwen3_vl_moe":
        layers = find_layers(model, [nn.Linear, Qwen3VLMoeTextExperts])
    else:
        layers = find_layers(model, [nn.Linear])
    print(f"Found {len(layers)} linear layers")

    is_qwen35 = (model_type == "qwen3_5_moe") or (model_type == "qwen3_5")
    active_suffix = SUFFIX_TO_QUANT + (QWEN35_EXTRA_SUFFIX_TO_QUANT if is_qwen35 else [])

    if is_qwen35:
        # For Qwen3.5, scan ALL named modules with a weight parameter (not just
        # nn.Linear) so that Conv1d, Embedding, Router and visual modules are
        # also captured in modules_to_not_convert, matching the official checkpoint.
        # Only 2-D weights are counted (1-D LayerNorm/RMSNorm weights are excluded),
        # which exactly matches the official Qwen/Qwen3.5-35B-A3B-FP8 list.
        # Batched expert tensors (.experts.gate_up_proj / .experts.down_proj) are
        # handled inside process_safetensor_qwen35 and must NOT appear here.
        BATCHED_EXPERT_SUFFIXES = (".experts.gate_up_proj", ".experts.down_proj")
        ignored_layers = []
        for name, module in model.named_modules():
            if not hasattr(module, "weight") or module.weight is None:
                continue
            if module.weight.ndim < 2:  # skip 1-D norm weights
                continue
            weight_name = f"{name}.weight"
            is_quantized = any(weight_name.endswith(s) for s in active_suffix) or any(
                weight_name.endswith(s) for s in BATCHED_EXPERT_SUFFIXES
            )
            if not is_quantized:
                ignored_layers.append(name)
        # mtp layers are not in the transformers model graph; add them manually.
        num_mtp_layers = getattr(config.text_config, "mtp_num_hidden_layers", 0)
        for i in range(num_mtp_layers):
            ignored_layers.append("mtp.fc")
            ignored_layers.append(f"mtp.layers.{i}.mlp.gate")
            ignored_layers.append(f"mtp.layers.{i}.mlp.shared_expert_gate")
    else:
        ignored_layers = []
        for name, _ in layers.items():
            if not name.endswith("mlp.experts"):
                weight_name = f"{name}.weight"
                if not any(weight_name.endswith(suffix) for suffix in active_suffix):
                    ignored_layers.append(name)

    print(f"Ignored layers: {ignored_layers}")
    del model

    target_worker = worker_qwen35 if is_qwen35 else worker

    num_workers = min(num_workers, len(safetensor_files))
    file_subsets = [safetensor_files[i::num_workers] for i in range(num_workers)]
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=target_worker,
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
            or file.endswith(".jinja")
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

    if is_qwen35:
        # Align with the official Qwen/Qwen3.5-35B-A3B-FP8 quantization_config format
        quant_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_per_tensor": False,
            "act_per_tensor": False,
            "modules_to_not_convert": ignored_layers,
        }
        if block_size[0] != -1 and block_size[1] != -1:
            quant_config["weight_block_size"] = list(block_size)
    else:
        quant_config = {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "modules_to_not_convert": ignored_layers,
        }
        if block_size[0] != -1 and block_size[1] != -1:
            quant_config["weight_block_size"] = list(block_size)

    config["quantization_config"] = quant_config
    print(f"quant config: {config['quantization_config']}")
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--block_size", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()
    print(args)
    with open(os.path.join(args.input_path, "config.json"), "r", encoding="utf8") as fp:
        json_data = json.load(fp)
        print(json_data)
    if "quantization_config" in json_data.keys():
        raise AssertionError("NOT SUPPORT FP8 DS")

    main(args.input_path, args.output_path, args.block_size, args.num_workers)
