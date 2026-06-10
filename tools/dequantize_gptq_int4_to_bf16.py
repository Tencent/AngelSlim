#!/usr/bin/env python3
"""
Dequantize GPTQ int4 weights to BF16 safetensors.

This script handles standard GPTQ format with:
- qweight: int32 packed weights (8 int4 values per int32)
- qzeros: int32 packed zeros (8 int4 values per int32)  
- scales: float16/float32 scaling factors
- g_idx: optional group indices for actorder

Output: BF16 weights that can be used for accuracy comparison.
"""

import argparse
import json
import os
import shutil
from typing import Dict, Optional, Set, Tuple

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

DEFAULT_INPUT_PATH = "/apdcephfs_zwfy2/share_301053287/nemlittan/AngelSlim_git/AngelSlim/output/hunyuanv3_a20b_int4_gptq"
DEFAULT_OUTPUT_PATH = "/apdcephfs_zwfy2/share_301053287/nemlittan/AngelSlim_git/AngelSlim/output/hunyuanv3_a20b_int4_gptq_dequant_bf16"

# Target expert layers for MoE models (matches the quantization config)
DEFAULT_TARGET_PATTERN = r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)"


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def load_model_index(input_path: str) -> Tuple[Dict[str, str], Dict]:
    """Load model index from safetensors index file or single file."""
    index_path = os.path.join(input_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        index = load_json(index_path)
        return index["weight_map"], index.get("metadata", {})

    model_file = os.path.join(input_path, "model.safetensors")
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Cannot find model.safetensors.index.json or model.safetensors in {input_path}"
        )
    with safe_open(model_file, framework="pt", device="cpu") as f:
        return {name: "model.safetensors" for name in f.keys()}, {}


def load_quant_config(input_path: str) -> Tuple[int, int, bool]:
    """Load quantization config from config.json."""
    config_path = os.path.join(input_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json: {config_path}")

    config = load_json(config_path)
    quant_config = config.get("quantization_config") or {}
    bits = int(quant_config.get("bits", 4))
    group_size = int(quant_config.get("group_size", 128))
    desc_act = quant_config.get("desc_act", False)
    
    if bits != 4:
        raise ValueError(f"This script expects 4-bit GPTQ weights, got bits={bits}")
    return bits, group_size, desc_act


def is_safetensor_file(file_name: str) -> bool:
    return file_name.endswith(".safetensors")


def get_safetensor_files(weight_map: Dict[str, str]) -> list[str]:
    return sorted(set(weight_map.values()))


def unpack_int32_gptq_qweight(qweight: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """
    Unpack GPTQ qweight exactly as GPTQQuantLinear.forward does.

    Stored qweight shape is (in_features / pack_factor, out_features).
    Returned tensor shape is (in_features, out_features), with unsigned int4 values.
    """
    pack_factor = 32 // bits
    mask = (1 << bits) - 1
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=qweight.device)

    unpacked = torch.bitwise_right_shift(
        qweight.to(torch.int32).unsqueeze(1), shifts.view(1, -1, 1)
    )
    unpacked = torch.bitwise_and(unpacked, mask)
    return unpacked.reshape(qweight.shape[0] * pack_factor, qweight.shape[1]).contiguous()


def unpack_int32_gptq_qzeros(qzeros: torch.Tensor, scales_shape: Tuple[int, int], bits: int = 4) -> torch.Tensor:
    """
    Unpack GPTQ qzeros exactly as GPTQQuantLinear.forward does.

    Stored qzeros shape is (num_groups, out_features / pack_factor).
    Returned tensor shape matches scales, with the +1 offset restored.
    """
    pack_factor = 32 // bits
    mask = (1 << bits) - 1
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=qzeros.device)

    unpacked = torch.bitwise_right_shift(
        qzeros.to(torch.int32).unsqueeze(2), shifts.view(1, 1, -1)
    )
    unpacked = torch.bitwise_and(unpacked, mask)
    return (unpacked + 1).reshape(scales_shape).contiguous()


def dequantize_gptq_weight(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: Optional[torch.Tensor],
    bits: int,
    group_size: int,
    desc_act: bool,
) -> torch.Tensor:
    """
    Dequantize GPTQ weight to BF16 using the same layout/formula as GPTQQuantLinear.forward.

    qweight: (in_features / pack_factor, out_features), packed unsigned int4
    qzeros: (num_groups, out_features / pack_factor), packed unsigned int4 with stored zero-1
    scales: (num_groups, out_features)
    output: (out_features, in_features)
    """
    raw_weight = unpack_int32_gptq_qweight(qweight, bits).to(torch.float32)
    in_features, out_features = raw_weight.shape

    if scales.ndim != 2:
        raise ValueError(f"Expected 2D scales, got shape={tuple(scales.shape)}")
    if scales.shape[1] != out_features and scales.shape[0] == out_features:
        scales = scales.t().contiguous()
    if scales.shape[1] != out_features:
        raise ValueError(
            f"Can not align scales shape={tuple(scales.shape)} with qweight out_features={out_features}"
        )

    zeros = unpack_int32_gptq_qzeros(qzeros, tuple(scales.shape), bits).to(torch.float32)
    scales = scales.to(torch.float32)

    if g_idx is None or g_idx.numel() == 0:
        effective_group_size = group_size if group_size != -1 else in_features
        g_idx = torch.arange(in_features, dtype=torch.long, device=qweight.device) // effective_group_size
    else:
        g_idx = g_idx.to(torch.long)

    if g_idx.numel() != in_features:
        num_itr = g_idx.numel() // in_features
        if num_itr > 1:
            g_idx = g_idx[:in_features]
        else:
            raise ValueError(
                f"g_idx length {g_idx.numel()} does not match in_features {in_features}"
            )

    if int(g_idx.max().item()) >= scales.shape[0]:
        raise ValueError(
            f"g_idx max {int(g_idx.max().item())} is out of range for scales shape={tuple(scales.shape)}"
        )

    dequant_weight = scales.index_select(0, g_idx) * (
        raw_weight - zeros.index_select(0, g_idx)
    )
    return dequant_weight.t().contiguous().to(torch.bfloat16)


def prepare_output_dir(input_path: str, output_path: str, overwrite: bool) -> None:
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    if input_path == output_path:
        raise ValueError("input_path and output_path must be different")
    if os.path.exists(output_path):
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_path}\n"
                "Use another new directory, or pass --overwrite if you really want to replace it."
            )
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=False)


def copy_side_files(input_path: str, output_path: str) -> None:
    """Copy non-weight files (config, tokenizer, etc.) to output directory."""
    for file_name in os.listdir(input_path):
        if is_safetensor_file(file_name) or file_name == "model.safetensors.index.json":
            continue
        src_path = os.path.join(input_path, file_name)
        dst_path = os.path.join(output_path, file_name)
        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def update_configs(output_path: str) -> None:
    """Remove quantization config from output model."""
    config_path = os.path.join(output_path, "config.json")
    if os.path.exists(config_path):
        config = load_json(config_path)
        config.pop("quantization_config", None)
        config["torch_dtype"] = "bfloat16"
        if "dtype" in config:
            config["dtype"] = "bfloat16"
        save_json(config, config_path)

    angelslim_config_path = os.path.join(output_path, "angelslim_config.json")
    if os.path.exists(angelslim_config_path):
        angelslim_config = load_json(angelslim_config_path)
        quantization_config = angelslim_config.get("compression_config", {}).get("quantization", {})
        quant_method = quantization_config.setdefault("quant_method", {})
        quant_method["dequant_to_bf16"] = True
        quantization_config["dequantized_from"] = "int4_gptq"
        save_json(angelslim_config, angelslim_config_path)


def find_quant_tensors(weight_map: Dict[str, str]) -> Tuple[Set[str], Set[str]]:
    """
    Find all quantized weight bases and their auxiliary tensors.
    
    Returns:
        quant_bases: Set of base names (without suffix) that have qweight
        aux_names: Set of all auxiliary tensor names (qweight, qzeros, scales, g_idx)
    """
    quant_bases = set()
    aux_names = set()
    
    for name in weight_map:
        if name.endswith(".qweight"):
            base = name[: -len(".qweight")]
            quant_bases.add(base)
            aux_names.add(name)
            # Add expected auxiliary tensors
            for suffix in [".qzeros", ".scales", ".g_idx"]:
                aux_name = base + suffix
                if aux_name in weight_map:
                    aux_names.add(aux_name)
    
    return quant_bases, aux_names


def process_shard(
    shard_name: str,
    input_path: str,
    output_path: str,
    weight_map: Dict[str, str],
    quant_bases: Set[str],
    aux_names: Set[str],
    bits: int,
    group_size: int,
    desc_act: bool,
    chunk_size: int,
) -> Tuple[Dict[str, str], int, int, int, int]:
    """
    Process a single safetensors shard: dequantize quantized weights, copy others.
    """
    src_path = os.path.join(input_path, shard_name)
    dst_path = os.path.join(output_path, shard_name)
    shard_state = {}
    shard_index = {}
    converted = 0
    skipped_aux = 0
    copied = 0
    total_size = 0

    with safe_open(src_path, framework="pt", device="cpu") as f:
        shard_keys = set(f.keys())
        
        # First pass: collect all quantized weight data
        quant_data = {}
        for name in shard_keys:
            if name in aux_names:
                base = name[: -len(".qweight")]
                for suffix in [".qweight", ".qzeros", ".scales", ".g_idx"]:
                    if base + suffix in shard_keys:
                        quant_data[base + suffix] = f.get_tensor(base + suffix)
        
        # Second pass: process keys
        for name in f.keys():
            # Check if this is a qweight that needs dequantization
            if name.endswith(".qweight") and name[: -len(".qweight")] in quant_bases:
                base = name[: -len(".qweight")]
                
                qweight = quant_data.get(name)
                qzeros_name = base + ".qzeros"
                scales_name = base + ".scales"
                g_idx_name = base + ".g_idx"
                
                if qzeros_name not in quant_data:
                    raise KeyError(f"Missing {qzeros_name} for {name}")
                if scales_name not in quant_data:
                    raise KeyError(f"Missing {scales_name} for {name}")
                
                qzeros = quant_data[qzeros_name]
                scales = quant_data[scales_name]
                g_idx = quant_data.get(g_idx_name)
                
                # Dequantize
                weight = dequantize_gptq_weight(
                    qweight=qweight,
                    qzeros=qzeros,
                    scales=scales,
                    g_idx=g_idx,
                    bits=bits,
                    group_size=group_size,
                    desc_act=desc_act,
                )
                
                out_name = base + ".weight"
                shard_state[out_name] = weight
                shard_index[out_name] = shard_name
                total_size += tensor_nbytes(weight)
                converted += 1
                continue
            
            # Skip auxiliary quant tensors
            if name in aux_names:
                skipped_aux += 1
                continue
            
            # Copy non-quantized tensors
            tensor = f.get_tensor(name)
            shard_state[name] = tensor
            shard_index[name] = shard_name
            total_size += tensor_nbytes(tensor)
            copied += 1
    
    if shard_state:
        save_file(shard_state, dst_path)
    
    return shard_index, converted, skipped_aux, copied, total_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dequantize GPTQ int4 weights to BF16 safetensors."
    )
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path)
    
    # Load config
    bits, group_size, desc_act = load_quant_config(input_path)
    
    # Load model index
    weight_map, metadata = load_model_index(input_path)
    safetensor_files = get_safetensor_files(weight_map)
    
    # Find quantized tensors
    quant_bases, aux_names = find_quant_tensors(weight_map)

    print(f"Input path:  {input_path}", flush=True)
    print(f"Output path: {output_path}", flush=True)
    print(f"Safetensor shards: {len(safetensor_files)}", flush=True)
    print(f"Quantized weight bases: {len(quant_bases)}", flush=True)
    print(f"bits={bits}, group_size={group_size}, desc_act={desc_act}", flush=True)

    # Prepare output directory
    prepare_output_dir(input_path, output_path, args.overwrite)

    # Process all shards
    new_index = {}
    total_converted = 0
    total_skipped_aux = 0
    total_copied = 0
    total_size = 0
    
    for shard_name in tqdm(safetensor_files, desc="Dequantizing shards"):
        shard_index, converted, skipped_aux, copied, shard_size = process_shard(
            shard_name=shard_name,
            input_path=input_path,
            output_path=output_path,
            weight_map=weight_map,
            quant_bases=quant_bases,
            aux_names=aux_names,
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            chunk_size=args.chunk_size,
        )
        new_index.update(shard_index)
        total_converted += converted
        total_skipped_aux += skipped_aux
        total_copied += copied
        total_size += shard_size
        print(
            f"Processed {shard_name}: converted={converted}, "
            f"skipped_quant_aux={skipped_aux}, copied={copied}",
            flush=True,
        )

    # Save index
    metadata = dict(metadata)
    metadata["total_size"] = total_size
    save_json(
        {"metadata": metadata, "weight_map": new_index},
        os.path.join(output_path, "model.safetensors.index.json")
    )
    
    # Copy side files and update configs
    copy_side_files(input_path, output_path)
    update_configs(output_path)

    print("Done.", flush=True)
    print(f"Converted quantized weights: {total_converted}", flush=True)
    print(f"Skipped quant auxiliary tensors: {total_skipped_aux}", flush=True)
    print(f"Copied tensors: {total_copied}", flush=True)
    print(f"Output path: {output_path}", flush=True)
    
    # Verify no qweight remains
    remaining_qweights = [name for name in new_index if name.endswith(".qweight")]
    if remaining_qweights:
        print(f"Warning: output still contains {len(remaining_qweights)} qweight tensors.", flush=True)


if __name__ == "__main__":
    main()
