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

"""Weight quantization utilities (INT4 + FP8).

Provides functions to quantize model weights using:
- FP8 blockwise quantization (for all layers or specified layers)
- INT4 symmetric per-group quantization + packing (W4A8 mixed precision)

Operates directly on safetensors files without loading the full HF model
into GPU memory.
"""

import json
import math
import multiprocessing as mp
import os
import re
import shutil

import accelerate
import torch
from safetensors.torch import safe_open, save_file
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from angelslim.compressor.quant.core.packing_utils import pack_weight_to_int8_gpu
from angelslim.utils import find_layers, print_info

# Weight name suffixes that should be quantized (FP8 or W4)
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


def create_fp8_quantized_param(param, weight_block_size=(128, 128)):
    """Quantize weights to FP8 format using blockwise quantization.

    Args:
        param: Weight tensor to quantize (bf16/fp16/fp32).
        weight_block_size: Tuple of (block_rows, block_cols) for blockwise quantization.
            Use (-1, -1) for tensor-wise quantization.

    Returns:
        Tuple of (quantized_param, scale_inv) where quantized_param is the FP8
        quantized weight and scale_inv is the inverse scale factor.
    """
    fp8_min = torch.finfo(torch.float8_e4m3fn).min
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    block_size_m, block_size_n = weight_block_size
    rows, cols = param.shape[-2:]
    original_device = param.device  # noqa: F841

    # Tensor-wise
    if block_size_m == -1 or block_size_m > rows:
        block_size_m = rows
    if block_size_n == -1 or block_size_n > cols:
        block_size_n = cols

    # Padding (done on the same device as input to avoid unnecessary transfers)
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

    # Convert to float and reshape for blockwise quantization
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

    del param
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
    del param_value
    torch.cuda.empty_cache()

    quantized_param = quantized_param.permute(0, 1, 3, 2, 4)
    quantized_param = quantized_param.reshape(param_value_shape)[..., :rows, :cols]

    scale_inv = scale_inv.reshape(scale_orig_shape).squeeze().reciprocal()

    return quantized_param.contiguous(), scale_inv.contiguous()


def bf16_to_int4_pack(weight, group_size=128):
    """Quantize bf16/fp16 weight to INT4 symmetric and pack to int8.

    Performs symmetric INT4 per-group quantization on bf16 weights,
    following the same format as save.py's _packed_weight.
    All computation (quantization + packing) is done on GPU for speed,
    with results moved to CPU only at the final return.

    Steps:
      1. Compute FP8 tensor-wise scale: fp8_scale = absmax(weight) / 448.0
      2. Compute INT4 per-group scale: int4_scale = per_group_absmax / 8
      3. Quantize: q = clamp(round(bf16_weight / int4_scale), -8, 7)
      4. Pack two int4 values into one int8 using pack_weight_to_int8_gpu (pure PyTorch, GPU)

    Output tensor naming convention (handled by caller):
      - xxx.qweight: packed int4 weight (int8 dtype)
      - xxx.weight_scale: FP8 tensor-wise scale (scalar, bfloat16)
      - xxx.weight_scale.int4: INT4 per-group scale (bfloat16)

    Args:
        weight: Weight tensor in bf16/fp16/fp32, shape (out_features, in_features).
            Should be on GPU for best performance.
        group_size: Group size for per-group quantization.

    Returns:
        Tuple of (packed_weight, fp8_tensor_scale, int4_group_scale), all on CPU:
          - packed_weight: INT8 tensor with two INT4 values packed per byte.
          - fp8_tensor_scale: Scalar bfloat16 tensor, = absmax(weight) / 448.0.
          - int4_group_scale: BFloat16 scale tensor,
                shape (out_features, in_features // group_size), = per_group_absmax / 8.
    """
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0

    assert weight.dim() == 2, "Weight must be 2D tensor"
    out_features, in_features = weight.shape
    assert (
        in_features % group_size == 0
    ), f"in_features ({in_features}) must be divisible by group_size ({group_size})"

    # All computation on the same device as input weight (GPU if available)
    w = weight.float()

    # FP8 tensor-wise scale (scalar): absmax / 448.0
    fp8_tensor_scale = (w.abs().max() / FP8_MAX).to(torch.bfloat16)

    # INT4 per-group scale: per_group_absmax / 8
    w_grouped = w.reshape(-1, group_size)
    group_absmax = w_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    int4_group_scale = group_absmax / 8.0  # INT4 signed range: [-8, 7], scale = absmax / 8

    # Quantize to int4 range using per-group scale
    quant_weight = torch.clamp(torch.round(w_grouped / int4_group_scale), -8, 7)

    # Reshape back to original shape
    quant_weight = quant_weight.reshape(out_features, in_features)
    int4_group_scale = int4_group_scale.reshape(out_features, in_features // group_size).to(
        torch.bfloat16
    )

    # Pack two int4 into one int8 (pure PyTorch, runs on GPU)
    packed_weight = pack_weight_to_int8_gpu(quant_weight)

    # Move everything to CPU at the end
    packed_weight = packed_weight.cpu()
    fp8_tensor_scale = fp8_tensor_scale.cpu()
    int4_group_scale = int4_group_scale.cpu()

    # Free GPU memory
    del w, w_grouped, group_absmax, quant_weight
    torch.cuda.empty_cache()

    return packed_weight, fp8_tensor_scale, int4_group_scale


def _process_safetensor_mixed(
    rank,
    file_name,
    input_path,
    output_path,
    fp8_block_size=(128, 128),
    w4_group_size=128,
    fp8_only_layers=None,
    no_quant_layers=None,
):
    """Process a single safetensor file: apply FP8 or W4 quantization per layer.

    For each weight in the safetensor file:
      - If it matches fp8_only_layers: FP8 blockwise quantize
      - If it matches no_quant_layers: copy as-is (no quantization)
      - If it matches SUFFIX_TO_QUANT but not above: W4 int4 symmetric quantize + pack
      - Otherwise: copy as-is

    Args:
        rank: GPU device rank for quantization.
        file_name: Name of the safetensor file.
        input_path: Directory containing the input safetensor file.
        output_path: Directory to save the quantized safetensor file.
        fp8_block_size: Block size for FP8 blockwise quantization.
        w4_group_size: Group size for W4 per-group quantization.
        fp8_only_layers: List of layer name patterns for FP8-only quantization.
        no_quant_layers: List of layer name patterns to skip all quantization.

    Returns:
        Dictionary mapping weight names to file names (weight_map index).
    """
    import time

    state_dict = {}
    index = {}
    fp8_only_layers = fp8_only_layers or []
    no_quant_layers = no_quant_layers or []
    device = f"cuda:{rank}"

    print_info(f"  [{file_name}] Opening safetensor file (device={device})...")
    t_open = time.time()
    # Load tensors directly to GPU to skip CPU→GPU copy for weights that need quantization.
    # Non-quantized tensors (norms, biases, embeddings) are moved back to CPU immediately.
    with safe_open(os.path.join(input_path, file_name), framework="pt", device=device) as f:
        print_info(
            f"  [{file_name}] Opened in {time.time() - t_open:.1f}s, {len(f.keys())} weights"
        )
        t_quant_start = time.time()
        weight_count = 0
        for weight_name in f.keys():
            t_w = time.time()  # noqa: F841
            weight = f.get_tensor(weight_name)
            weight_count += 1

            # Check if this weight should skip all quantization
            is_no_quant = any(pattern in weight_name for pattern in no_quant_layers)
            if is_no_quant:
                state_dict[weight_name] = weight.cpu()
                index[weight_name] = file_name
                del weight
                continue

            # Check if this weight should be FP8 quantized
            is_fp8_only = any(pattern in weight_name for pattern in fp8_only_layers)

            # Check if this weight is quantizable (matches SUFFIX_TO_QUANT)
            is_quantizable = any(weight_name.endswith(suffix) for suffix in SUFFIX_TO_QUANT)

            if is_fp8_only and is_quantizable:
                # FP8 blockwise quantization (weight already on GPU)
                quant_weight, scale = create_fp8_quantized_param(weight, fp8_block_size)

                state_dict[weight_name] = quant_weight.cpu()
                index[weight_name] = file_name

                if fp8_block_size[0] == -1 and fp8_block_size[1] == -1:
                    state_dict[f"{weight_name}_scale"] = scale.cpu()
                    index[f"{weight_name}_scale"] = file_name
                else:
                    state_dict[f"{weight_name}_scale_inv"] = scale.cpu()
                    index[f"{weight_name}_scale_inv"] = file_name

                del weight, quant_weight, scale
                torch.cuda.empty_cache()
                if weight_count % 10 == 0:
                    print_info(
                        f"    [{file_name}] FP8 quant {weight_count} "
                        f"weights done ({time.time() - t_quant_start:.1f}s)"
                    )

            elif is_quantizable and not is_fp8_only:
                # W4 int4 symmetric quantization + pack (weight already on GPU)
                packed_weight, fp8_scale, int4_scale = bf16_to_int4_pack(
                    weight, group_size=w4_group_size
                )

                # Follow save.py naming convention:
                #   xxx.qweight         - packed int4 weight (int8)
                #   xxx.weight_scale    - FP8 tensor-wise scale (scalar, bfloat16)
                #   xxx.weight_scale.int4 - INT4 per-group scale (bfloat16)
                if ".weight" in weight_name:
                    qweight_name = weight_name.replace(".weight", ".qweight")
                    fp8_scale_name = weight_name.replace(".weight", ".weight_scale")
                    int4_scale_name = weight_name.replace(".weight", ".weight_scale.int4")
                else:
                    # Fallback for weight names without .weight suffix (e.g. packed MoE)
                    qweight_name = f"{weight_name}.qweight"
                    fp8_scale_name = f"{weight_name}.weight_scale"
                    int4_scale_name = f"{weight_name}.weight_scale.int4"

                state_dict[qweight_name] = packed_weight
                index[qweight_name] = file_name

                state_dict[fp8_scale_name] = fp8_scale
                index[fp8_scale_name] = file_name

                state_dict[int4_scale_name] = int4_scale
                index[int4_scale_name] = file_name

                del weight, packed_weight, fp8_scale, int4_scale
                torch.cuda.empty_cache()
                if weight_count % 10 == 0:
                    print_info(
                        f"    [{file_name}] W4 quant {weight_count} "
                        f"weights done ({time.time() - t_quant_start:.1f}s)"
                    )

            else:
                # Copy as-is (non-quantizable weights like biases, norms, embeddings)
                # Move to CPU to free GPU memory
                state_dict[weight_name] = weight.cpu()
                index[weight_name] = file_name
                del weight

        print_info(
            f"  [{file_name}] All {weight_count} "
            f"weights quantized in {time.time() - t_quant_start:.1f}s"
        )

    print_info(f"  [{file_name}] Saving ({len(state_dict)} keys)...")
    t_save = time.time()
    new_safetensor_file = os.path.join(output_path, file_name)
    save_file(state_dict, new_safetensor_file)
    print_info(f"  [{file_name}] Saved in {time.time() - t_save:.1f}s")

    del state_dict
    torch.cuda.empty_cache()

    return index


def _worker_mixed_quant(
    i,
    file_names,
    input_path,
    output_path,
    fp8_block_size,
    w4_group_size,
    return_dict,
    fp8_only_layers=None,
    no_quant_layers=None,
):
    """Worker function for multiprocessing mixed weight quantization."""
    import time

    world_size = torch.cuda.device_count()
    gpu_rank = i % world_size
    print_info(
        f"Worker {i} started (GPU {gpu_rank}), processing {len(file_names)} files: {file_names}"
    )
    for file_name in tqdm(file_names, desc=f"Worker {i}"):
        t0 = time.time()
        index = _process_safetensor_mixed(
            gpu_rank,
            file_name,
            input_path,
            output_path,
            fp8_block_size,
            w4_group_size,
            fp8_only_layers,
            no_quant_layers,
        )
        return_dict[file_name] = index
        print_info(f"Worker {i}: {file_name} done in {time.time() - t0:.1f}s")


def _get_ignored_layers(input_path):
    """Analyze model structure to find layers that should not be quantized.

    Args:
        input_path: Path to the model directory.

    Returns:
        List of layer names to ignore during quantization.
    """
    config = AutoConfig.from_pretrained(input_path)
    model_type = config.model_type

    # Try importing VL model classes (optional dependencies)
    extra_layer_types = []  # noqa: F841
    try:
        from transformers import (
            Qwen3VLForConditionalGeneration,
            Qwen3VLMoeForConditionalGeneration,
        )
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeTextExperts,
        )
    except ImportError:
        Qwen3VLForConditionalGeneration = None
        Qwen3VLMoeForConditionalGeneration = None
        Qwen3VLMoeTextExperts = None

    with accelerate.init_empty_weights():
        if model_type == "qwen3_vl_moe" and Qwen3VLMoeForConditionalGeneration is not None:
            model = Qwen3VLMoeForConditionalGeneration._from_config(config)
        elif model_type == "qwen3_vl" and Qwen3VLForConditionalGeneration is not None:
            model = Qwen3VLForConditionalGeneration._from_config(config)
        else:
            model = AutoModelForCausalLM.from_config(config)

    layer_types = [nn.Linear]
    if model_type == "qwen3_vl_moe" and Qwen3VLMoeTextExperts is not None:
        layer_types.append(Qwen3VLMoeTextExperts)

    layers = find_layers(model, layer_types)
    print_info(f"Found {len(layers)} linear layers")

    ignored_layers = []
    for name, _ in layers.items():
        if not name.endswith("mlp.experts"):
            weight_name = f"{name}.weight"
            if not any(weight_name.endswith(suffix) for suffix in SUFFIX_TO_QUANT):
                ignored_layers.append(name)
    print_info(f"Ignored layers: {ignored_layers}")

    del model
    return ignored_layers


def _convert_ignored_layers_for_deepseek_v3(ignored_layers, model_config):
    """Convert YAML ignore_layers patterns to vLLM glob patterns for DeepSeek V3.

    DeepSeek V3 uses specific naming conventions that vLLM expects in glob format.
    This function maps the user-friendly YAML patterns to the vLLM-compatible
    ignored_layers format.

    Mapping rules:
      - self_attn related patterns (q_a_proj, q_b_proj, kv_a_proj_with_mqa,
        kv_b_proj, o_proj) → "*self_attn*"
      - mlp.gate_proj, mlp.up_proj → "*gate_up_proj" (vLLM uses fused gate_up_proj)
      - mlp.down_proj → "*down_proj"
      - shared_expert → covered by self_attn and proj patterns
      - model.layers.N. → "*layers.N*"

    Args:
        ignored_layers: Original ignored_layers list from YAML config.
        model_config: Model config dict (from config.json).

    Returns:
        Converted ignored_layers list with vLLM glob patterns.
    """
    if not ignored_layers:
        return ignored_layers

    # Define pattern groups for DeepSeek V3
    SELF_ATTN_PATTERNS = {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"}
    GATE_UP_PATTERNS = {"mlp.gate_proj", "mlp.up_proj"}
    DOWN_PROJ_PATTERNS = {"mlp.down_proj"}
    SHARED_EXPERT_PATTERNS = {"shared_expert"}

    converted = []
    has_self_attn = False
    has_gate_up = False
    has_down_proj = False

    for pattern in ignored_layers:
        if pattern in SELF_ATTN_PATTERNS:
            if not has_self_attn:
                converted.append("*self_attn*")
                has_self_attn = True
        elif pattern in GATE_UP_PATTERNS:
            if not has_gate_up:
                converted.append("*gate_up_proj")
                has_gate_up = True
        elif pattern in DOWN_PROJ_PATTERNS:
            if not has_down_proj:
                converted.append("*down_proj")
                has_down_proj = True
        elif pattern in SHARED_EXPERT_PATTERNS:
            # shared_expert is already covered by self_attn and proj patterns
            # but add explicitly if needed
            pass
        elif pattern.startswith("model.layers."):
            # Convert "model.layers.N." to "*layers.N*"
            # Strip trailing dot if present
            p = pattern.rstrip(".")
            # Extract layer index: "model.layers.61" -> "61"
            parts = p.split(".")
            if len(parts) >= 3:
                layer_idx = parts[2]
                converted.append(f"*layers.{layer_idx}*")
            else:
                converted.append(f"*{p}*")
        else:
            # Keep unknown patterns as-is with wildcard prefix
            converted.append(f"*{pattern}*")

    return converted


def mixed_weight_quantize(
    input_path,
    output_path,
    fp8_block_size=(128, 128),
    w4_group_size=128,
    num_workers=32,
    fp8_only_layers=None,
    no_quant_layers=None,
    modules_to_not_convert=None,
):
    """Quantize model with mixed precision (FP8 + INT4) in a single pass.

    Processes all safetensors files, applying per-layer quantization:
      - fp8_only_layers → FP8 blockwise quantization
      - no_quant_layers → copy as-is (e.g. lm_head)
      - other quantizable layers → W4 INT4 symmetric + pack

    Args:
        input_path: Path to the original model directory (with safetensors).
        output_path: Directory to save the quantized model.
        fp8_block_size: Tuple of (block_rows, block_cols) for FP8 blockwise quantization.
        w4_group_size: Group size for W4 per-group quantization.
        num_workers: Number of parallel workers for processing safetensors files.
        fp8_only_layers: List of layer name patterns for FP8-only quantization.
            These layers will be FP8 quantized instead of W4.
        no_quant_layers: List of layer name patterns to skip all quantization
            (e.g. lm_head). These layers are copied as-is.
        modules_to_not_convert: Optional list of layer name patterns to write
            into config.json's "modules_to_not_convert" field. If None, will
            be auto-detected from model structure.
    """
    # Validate input
    config_path = os.path.join(input_path, "config.json")
    with open(config_path, "r", encoding="utf8") as fp:
        json_data = json.load(fp)
    if "quantization_config" in json_data:
        raise AssertionError(
            "Model already has quantization_config. "
            "Re-quantizing a quantized model is not supported."
        )

    os.makedirs(output_path, exist_ok=True)

    # Discover safetensor files
    model_index_file = os.path.join(input_path, "model.safetensors.index.json")
    has_index = os.path.exists(model_index_file)
    if has_index:
        with open(model_index_file, "r") as f:
            model_index = json.load(f)
        weight_map = model_index["weight_map"]
        safetensor_files = sorted(set(weight_map.values()))
    else:
        safetensor_files = ["model.safetensors"]
    print_info(f"Found {len(safetensor_files)} safetensor files")

    # Determine modules_to_not_convert for config.json
    if modules_to_not_convert:
        ignored_layers = list(modules_to_not_convert)
        print_info(f"Using user-specified modules_to_not_convert: {ignored_layers}")
    else:
        ignored_layers = _get_ignored_layers(input_path)

    # Distribute work across multiple processes
    import time

    num_workers = min(num_workers, len(safetensor_files))
    file_subsets = [safetensor_files[i::num_workers] for i in range(num_workers)]

    print_info(f"Spawning {num_workers} worker processes...")
    t_spawn_start = time.time()
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=_worker_mixed_quant,
            args=(
                i,
                file_subsets[i],
                input_path,
                output_path,
                fp8_block_size,
                w4_group_size,
                return_dict,
                fp8_only_layers,
                no_quant_layers,
            ),
        )
        p.start()
        processes.append(p)
    print_info(f"All {num_workers} workers spawned in {time.time() - t_spawn_start:.1f}s")
    for p in processes:
        p.join()

    # Merge weight index
    index = {}
    for result in return_dict.values():
        index.update(result)
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": index}, f, indent=2)

    # Copy config and other non-safetensor files
    for file in os.listdir(input_path):
        if file.endswith((".py", ".json", ".md", ".txt", ".jinja")):
            src_path = os.path.join(input_path, file)
            dst_path = os.path.join(output_path, file)
            if os.path.exists(dst_path):
                continue
            print_info(f"cp {src_path} {dst_path}")
            shutil.copy2(src_path, dst_path)

    # Write quantization config into config.json
    # Use w4a8_awq format with ignored_layers and ignored_quantization_config
    with open(os.path.join(output_path, "config.json"), "r") as f:
        config = json.load(f)

    # Post-process ignored_layers for DeepSeek V3: convert YAML patterns to vLLM glob patterns
    model_type = config.get("model_type", "")
    if model_type == "deepseek_v3":
        ignored_layers = _convert_ignored_layers_for_deepseek_v3(ignored_layers, config)
        print_info(
            f"DeepSeek V3 detected, converted ignored_layers to vLLM format: {ignored_layers}"
        )

    # Build ignored_quantization_config for FP8-only layers
    ignored_quant_config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "kv_cache_quant_method": "fp8",
    }
    if fp8_block_size and fp8_block_size[0] != -1 and fp8_block_size[1] != -1:
        ignored_quant_config["weight_block_size"] = list(fp8_block_size)

    config["quantization_config"] = {
        "quant_method": "w4a8_awq",
        "weight_group_size": w4_group_size,
        "activation_scheme": "static",
        "kv_cache_quant_method": "fp8",
        "ignored_layers": ignored_layers,
        "ignored_quantization_config": ignored_quant_config,
    }
    print_info(f"quant config: {json.dumps(config['quantization_config'], indent=2)}")
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


# Also keep the original FP8-only quantize function for backward compatibility
# (reuse the same create_fp8_quantized_param + SUFFIX_TO_QUANT)
def _process_safetensor_fp8(
    rank, file_name, input_path, output_path, block_size=(128, 128), fp8_only_layers=None
):
    """Process a single safetensor file: quantize matching weights to FP8.

    Args:
        rank: GPU device rank for quantization.
        file_name: Name of the safetensor file.
        input_path: Directory containing the input safetensor file.
        output_path: Directory to save the quantized safetensor file.
        block_size: Block size for blockwise quantization.
        fp8_only_layers: Optional list of layer name patterns. If provided,
            only weights whose names contain one of these patterns will be
            FP8 quantized. If None, falls back to SUFFIX_TO_QUANT matching.

    Returns:
        Dictionary mapping weight names to file names (weight_map index).
    """
    state_dict = {}
    index = {}
    device = f"cuda:{rank}"

    with safe_open(os.path.join(input_path, file_name), framework="pt", device=device) as f:
        print_info(f"Processing {file_name} with {len(f.keys())} weights")
        for weight_name in f.keys():
            weight = f.get_tensor(weight_name)
            should_quantize = (
                any(pattern in weight_name for pattern in fp8_only_layers)
                if fp8_only_layers
                else any(weight_name.endswith(suffix) for suffix in SUFFIX_TO_QUANT)
            )
            if should_quantize:
                # Weight already on GPU, quantize directly
                quant_weight, scale = create_fp8_quantized_param(weight, block_size)

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
            else:
                # Non-quantizable weight, move to CPU to free GPU memory
                state_dict[weight_name] = weight.cpu()
                index[weight_name] = file_name
                del weight

    new_safetensor_file = os.path.join(output_path, file_name)
    save_file(state_dict, new_safetensor_file)

    del state_dict
    torch.cuda.empty_cache()

    return index


def _worker_fp8(
    i, file_names, input_path, output_path, block_size, return_dict, fp8_only_layers=None
):
    """Worker function for multiprocessing FP8-only quantization."""
    world_size = torch.cuda.device_count()
    for file_name in tqdm(file_names, desc=f"Worker {i}"):
        index = _process_safetensor_fp8(
            i % world_size, file_name, input_path, output_path, block_size, fp8_only_layers
        )
        return_dict[file_name] = index


def fp8_blockwise_quantize(
    input_path,
    output_path,
    block_size=(128, 128),
    num_workers=32,
    fp8_only_layers=None,
    modules_to_not_convert=None,
):
    """Quantize model weights to FP8 format using blockwise quantization.

    Directly operates on safetensors files without loading the full HF model,
    producing quantized .safetensors as the final output.

    Args:
        input_path: Path to the original model directory (with safetensors).
        output_path: Directory to save the quantized model.
        block_size: Tuple of (block_rows, block_cols) for blockwise quantization.
        num_workers: Number of parallel workers for processing safetensors files.
        fp8_only_layers: Optional list of layer name patterns specifying which
            layers should be FP8 quantized. Only weights whose names contain
            one of these patterns will be quantized. If None, falls back to
            SUFFIX_TO_QUANT suffix matching for auto-detection.
        modules_to_not_convert: Optional list of layer name patterns to write
            into config.json's "modules_to_not_convert" field. If None, will
            be auto-detected from model structure.
    """
    # Validate input
    config_path = os.path.join(input_path, "config.json")
    with open(config_path, "r", encoding="utf8") as fp:
        json_data = json.load(fp)
    if "quantization_config" in json_data:
        raise AssertionError(
            "Model already has quantization_config. "
            "Re-quantizing a quantized model is not supported."
        )

    os.makedirs(output_path, exist_ok=True)

    # Discover safetensor files
    model_index_file = os.path.join(input_path, "model.safetensors.index.json")
    has_index = os.path.exists(model_index_file)
    if has_index:
        with open(model_index_file, "r") as f:
            model_index = json.load(f)
        weight_map = model_index["weight_map"]
        safetensor_files = sorted(set(weight_map.values()))
    else:
        safetensor_files = ["model.safetensors"]
    print_info(f"Found {len(safetensor_files)} safetensor files")

    # Determine modules_to_not_convert for config.json
    if modules_to_not_convert:
        ignored_layers = list(modules_to_not_convert)
        print_info(f"Using user-specified modules_to_not_convert: {ignored_layers}")
    elif fp8_only_layers:
        ignored_layers = list(fp8_only_layers)
        print_info(f"Using fp8_only_layers as modules_to_not_convert: {ignored_layers}")
    else:
        ignored_layers = _get_ignored_layers(input_path)

    # Distribute work across multiple processes
    num_workers = min(num_workers, len(safetensor_files))
    file_subsets = [safetensor_files[i::num_workers] for i in range(num_workers)]
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=_worker_fp8,
            args=(
                i,
                file_subsets[i],
                input_path,
                output_path,
                block_size,
                return_dict,
                fp8_only_layers,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # Merge weight index
    index = {}
    for result in return_dict.values():
        index.update(result)
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": index}, f, indent=2)

    # Copy config and other non-safetensor files
    for file in os.listdir(input_path):
        if file.endswith((".py", ".json", ".md", ".txt", ".jinja")):
            src_path = os.path.join(input_path, file)
            dst_path = os.path.join(output_path, file)
            if os.path.exists(dst_path):
                continue
            print_info(f"cp {src_path} {dst_path}")
            shutil.copy2(src_path, dst_path)

    # Write quantization config into config.json
    with open(os.path.join(output_path, "config.json"), "r") as f:
        config = json.load(f)
    config["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "modules_to_not_convert": ignored_layers,
    }
    if block_size[0] != -1 and block_size[1] != -1:
        config["quantization_config"]["weight_block_size"] = list(block_size)
    print_info(f"quant config: {config['quantization_config']}")
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def merge_moe_input_scales(
    moe_expert_stats_path: str,
    output_dir: str,
) -> None:
    """Convert MoE expert stats JSON to input_scale tensors in safetensors.

    Reads the moe_expert_stats.json (with per-expert min/max from vLLM calibration),
    computes input_scale = max(abs(min), abs(max)) / FP8_MAX for each expert layer,
    and writes input_scale tensors into the quantized safetensors files.
    Also updates model.safetensors.index.json and config.json accordingly.

    Handles the gate_up_proj split: vLLM collects stats for the fused
    "gate_up_proj" layer, but safetensors store separate "gate_proj" and
    "up_proj" weights. The same input_scale is written for both.

    JSON key format (per-expert):
        model.layers.X.mlp.experts.N.{gate_up_proj|down_proj}
    Safetensor key format:
        model.layers.X.mlp.experts.N.{gate_proj|up_proj|down_proj}.input_scale

    Args:
        moe_expert_stats_path: Path to moe_expert_stats.json file.
        output_dir: Path to the quantized model directory (with safetensors).
    """
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0

    # Mapping from JSON proj name -> safetensor proj name(s)
    # vLLM collects fused gate_up_proj stats, but safetensors have separate gate_proj/up_proj
    _PROJ_NAME_MAPPING = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "gate_and_up_proj": ["gate_proj", "up_proj"],
        "down_proj": ["down_proj"],
    }

    # Load MoE expert stats
    with open(moe_expert_stats_path, "r") as f:
        moe_expert_stats = json.load(f)
    print_info(f"  Loaded {len(moe_expert_stats)} MoE expert activation stats")

    # Load model index to find which safetensor file each weight belongs to
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print_info(
            "  Warning: model.safetensors.index.json not found, " "skipping input_scale generation"
        )
        return

    with open(index_path, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Helper: find which safetensor file a given layer belongs to
    def _find_shard_file(layer_name):
        # Try exact weight key first
        weight_key = f"{layer_name}.weight"
        if weight_key in weight_map:
            return weight_map[weight_key]
        # Try qweight (for W4 quantized layers)
        qweight_key = f"{layer_name}.qweight"
        if qweight_key in weight_map:
            return weight_map[qweight_key]
        # Fallback: find any key with this layer name prefix
        matching = [k for k in weight_map if k.startswith(layer_name + ".")]
        if matching:
            return weight_map[matching[0]]
        return None

    # Build mapping: safetensor_file -> list of (input_scale_key, scale_value)
    file_to_scales = {}
    skipped = 0

    # Pattern for per-expert keys:
    #   model.layers.X.mlp.experts.N.{gate_up_proj|gate_and_up_proj|down_proj}
    expert_pattern = re.compile(
        r"^(.+\.mlp\.experts)\.(\d+)\.(gate_up_proj|gate_and_up_proj|down_proj)$"
    )

    for layer_name, stats in moe_expert_stats.items():
        abs_max = max(abs(stats["min"]), abs(stats["max"]))
        input_scale = abs_max / FP8_MAX

        m = expert_pattern.match(layer_name)
        if m:
            experts_prefix = m.group(1)  # model.layers.X.mlp.experts
            expert_id = m.group(2)  # N
            proj_name = m.group(3)  # gate_up_proj or down_proj

            sf_proj_names = _PROJ_NAME_MAPPING.get(proj_name, [proj_name])
            for sf_proj in sf_proj_names:
                expanded_name = f"{experts_prefix}.{expert_id}.{sf_proj}"
                input_scale_key = f"{expanded_name}.input_scale"
                shard_file = _find_shard_file(expanded_name)
                if shard_file is None:
                    print_info(
                        f"  Warning: Cannot find safetensor file for {expanded_name} "
                        f"(from MoE expert key {layer_name}), skipping"
                    )
                    skipped += 1
                    continue
                if shard_file not in file_to_scales:
                    file_to_scales[shard_file] = []
                file_to_scales[shard_file].append((input_scale_key, input_scale))
                weight_map[input_scale_key] = shard_file
        else:
            # Non-expert key format — apply the same fused proj split logic
            proj_suffix = layer_name.rsplit(".", 1)[-1] if "." in layer_name else layer_name
            if proj_suffix in _PROJ_NAME_MAPPING:
                layer_prefix = layer_name.rsplit(".", 1)[0]
                for sf_proj in _PROJ_NAME_MAPPING[proj_suffix]:
                    expanded_name = f"{layer_prefix}.{sf_proj}"
                    input_scale_key = f"{expanded_name}.input_scale"
                    shard_file = _find_shard_file(expanded_name)
                    if shard_file is None:
                        print_info(
                            f"  Warning: Cannot find safetensor file for {expanded_name} "
                            f"(expanded from {layer_name}), skipping"
                        )
                        skipped += 1
                        continue
                    if shard_file not in file_to_scales:
                        file_to_scales[shard_file] = []
                    file_to_scales[shard_file].append((input_scale_key, input_scale))
                    weight_map[input_scale_key] = shard_file
            else:
                # Direct mapping (no split needed)
                input_scale_key = f"{layer_name}.input_scale"
                shard_file = _find_shard_file(layer_name)
                if shard_file is None:
                    print_info(
                        f"  Warning: Cannot find safetensor file for {layer_name}, skipping"
                    )
                    skipped += 1
                    continue
                if shard_file not in file_to_scales:
                    file_to_scales[shard_file] = []
                file_to_scales[shard_file].append((input_scale_key, input_scale))
                weight_map[input_scale_key] = shard_file

    total_scales = sum(len(v) for v in file_to_scales.values())
    print_info(f"  Total input_scale keys to write: {total_scales} (skipped: {skipped})")

    # Write input_scale tensors into each safetensor file
    for shard_file, scales in file_to_scales.items():
        shard_path = os.path.join(output_dir, shard_file)
        if not os.path.exists(shard_path):
            print_info(f"  Warning: {shard_path} not found, skipping")
            continue

        # Load existing tensors
        existing_tensors = {}
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                existing_tensors[key] = f.get_tensor(key)

        # Add input_scale tensors
        for scale_key, scale_value in scales:
            existing_tensors[scale_key] = torch.tensor(scale_value, dtype=torch.float32)

        # Save back
        save_file(existing_tensors, shard_path)
        print_info(f"  Added {len(scales)} input_scale tensors to {shard_file}")

    # Update model.safetensors.index.json
    with open(index_path, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

    # Update config.json to mark activation_scheme as static
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        if "quantization_config" in config:
            config["quantization_config"]["activation_scheme"] = "static"
            # Also update ignored_quantization_config if present (w4a8_awq format)
            if "ignored_quantization_config" in config["quantization_config"]:
                # Keep ignored layers' activation_scheme as dynamic (FP8 layers use dynamic)
                pass
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    print_info(f"  Total input_scale tensors written: {total_scales}")
