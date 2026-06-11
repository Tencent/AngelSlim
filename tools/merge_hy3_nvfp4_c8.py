#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Merge NVFP4 expert weights + FP8 KV cache scales + activation input_scales
# into a single HF model directory for vLLM inference.
#
# Inputs:
#   --statistics_path: dir containing activation_stats.json & moe_expert_stats.json
#   --nvfp4_w_path: NVFP4 weight-only model dir (has .weight, .weight_scale, .weight_scale_2)
#   --output_path: where to write the merged model
#   --bf16_model_path: (optional) original bf16 model for config.json/tokenizer;
#                      defaults to nvfp4_w_path (which already has them)
#
# Output model contains:
#   - Non-expert weights (attention, shared_mlp, layernorm, embed, lm_head) in BF16
#   - Expert weights in NVFP4 (.weight, .weight_scale, .weight_scale_2)
#   - Expert input_scale (fp32 scalar) computed from moe_expert_stats
#   - KV cache scales (k_proj.k_scale, v_proj.v_scale) computed from activation_stats
#   - config.json with quantization_config (quant_method=modelopt, NVFP4, kv_cache_scheme)

import argparse
import copy
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# FP8 E4M3 max value
FP8_MAX = 448.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge NVFP4 weights + FP8 KV scales + activation input_scales"
    )
    parser.add_argument(
        "--statistics_path",
        type=str,
        required=True,
        help="Path to calibration statistics (activation_stats.json, moe_expert_stats.json)",
    )
    parser.add_argument(
        "--nvfp4_w_path", type=str, required=True, help="Path to NVFP4 weight-only quantized model"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output path for merged model"
    )
    parser.add_argument(
        "--bf16_model_path",
        type=str,
        default=None,
        help="Path to original bf16 model (for config/tokenizer). Defaults to nvfp4_w_path.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers for processing shards",
    )
    parser.add_argument(
        "--kv_statistics_path",
        type=str,
        default="",
        help="Path to a separate activation_stats JSON file for KV cache scales. "
        "If not set, KV cache stats are loaded from statistics_path/activation_stats.json.",
    )
    return parser.parse_args()


def load_activation_stats(statistics_path):
    """Load activation_stats.json and moe_expert_stats.json."""
    act_path = os.path.join(statistics_path, "activation_stats.json")
    moe_path = os.path.join(statistics_path, "moe_expert_stats.json")

    with open(act_path, "r") as f:
        act_stats = json.load(f)

    moe_stats = {}
    if os.path.exists(moe_path):
        with open(moe_path, "r") as f:
            moe_stats = json.load(f)

    # Merge moe stats into act_stats (same pattern as fp8_quant_with_vllm_activation.py)
    act_stats.update(moe_stats)
    return act_stats


def compute_kv_scales(act_stats, num_layers):
    """
    Compute per-tensor FP8 KV cache scales from activation_stats.

    Keys in act_stats: "model.layers.{L}.self_attn.attn.k_cache" -> {"min": float, "max": float}
    Output keys: "model.layers.{L}.self_attn.k_proj.k_scale" -> float tensor
                 "model.layers.{L}.self_attn.v_proj.v_scale" -> float tensor
    """
    kv_scales = {}
    for layer_idx in range(num_layers):
        for cache_type, scale_name in [
            ("k_cache", "k_proj.k_scale"),
            ("v_cache", "v_proj.v_scale"),
        ]:
            key = f"model.layers.{layer_idx}.self_attn.attn.{cache_type}"
            if key not in act_stats:
                print(f"  [WARNING] Missing KV stats for {key}, skipping")
                continue

            stats = act_stats[key]
            min_val = stats["min"]
            max_val = stats["max"]

            # Per-tensor: min/max are scalars
            if isinstance(min_val, list):
                # Per-head: take max across heads for per-tensor scale
                absmax = max(max(abs(v) for v in min_val), max(abs(v) for v in max_val))
            else:
                absmax = max(abs(min_val), abs(max_val))

            scale = absmax / FP8_MAX
            out_key = f"model.layers.{layer_idx}.self_attn.{scale_name}"
            kv_scales[out_key] = torch.tensor(scale, dtype=torch.float32)

    return kv_scales


def compute_expert_input_scales(act_stats, num_layers, num_experts):
    """
    Compute input_scale for each expert projection from moe_expert_stats.

    Stats keys: "model.layers.{L}.mlp.experts.{E}.gate_up_proj" -> {"min", "max"}
                "model.layers.{L}.mlp.experts.{E}.down_proj" -> {"min", "max"}

    Output keys: "model.layers.{L}.mlp.experts.{E}.gate_proj.input_scale" -> float tensor
                 "model.layers.{L}.mlp.experts.{E}.up_proj.input_scale" -> float tensor
                 "model.layers.{L}.mlp.experts.{E}.down_proj.input_scale" -> float tensor

    gate_proj and up_proj share the same input (gate_up_proj activation).
    """
    input_scales = {}
    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            # gate_up_proj -> input_scale for both gate_proj and up_proj
            gate_up_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_up_proj"
            if gate_up_key in act_stats:
                stats = act_stats[gate_up_key]
                absmax = max(abs(stats["min"]), abs(stats["max"]))
                scale = absmax / FP8_MAX
                input_scales[
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.input_scale"
                ] = torch.tensor(scale, dtype=torch.float32)
                input_scales[
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.input_scale"
                ] = torch.tensor(scale, dtype=torch.float32)

            # down_proj
            down_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj"
            if down_key in act_stats:
                stats = act_stats[down_key]
                absmax = max(abs(stats["min"]), abs(stats["max"]))
                scale = absmax / FP8_MAX
                input_scales[
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.input_scale"
                ] = torch.tensor(scale, dtype=torch.float32)

    return input_scales


def process_shard(shard_path, kv_scales, input_scales, output_dir, shard_idx, total_shards):
    """Process a single safetensors shard: copy weights and inject scales."""
    filename = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
    output_path = os.path.join(output_dir, filename)

    tensors = {}
    weight_map_entries = {}

    # Load all tensors from this shard
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for key in keys:
            tensor = f.get_tensor(key)

            # Skip shared_mlp weight_scale/weight_scale_2 (keep shared_mlp as bf16)
            if "shared_mlp" in key and ("weight_scale" in key):
                continue

            # For shared_mlp weights that were quantized in nvfp4 model,
            # we skip them here - they'll be taken as bf16 from the original
            # Actually the nvfp4 model has them quantized too, but we want bf16.
            # We handle this by checking: if shared_mlp .weight exists and
            # there's also weight_scale, we need the bf16 version instead.
            # This is handled at the caller level - we only include non-shared_mlp
            # quantized weights, or bf16 weights.

            tensors[key] = tensor
            weight_map_entries[key] = filename

    # Add KV scales that belong to layers in this shard
    for key, tensor in kv_scales.items():
        # Check if this shard contains weights from the same layer
        # We assign KV scales to the shard that contains the corresponding k_proj.weight
        layer_prefix = key.rsplit(".", 2)[0]  # model.layers.X.self_attn
        k_proj_weight = f"{layer_prefix}.k_proj.weight"
        if k_proj_weight in tensors:
            tensors[key] = tensor
            weight_map_entries[key] = filename

    # Add input_scales that belong to experts in this shard
    for key, tensor in input_scales.items():
        # Assign to same shard as the corresponding .weight
        # e.g., model.layers.1.mlp.experts.0.gate_proj.input_scale
        #     -> model.layers.1.mlp.experts.0.gate_proj.weight
        weight_key = key.replace(".input_scale", ".weight")
        if weight_key in tensors:
            tensors[key] = tensor
            weight_map_entries[key] = filename

    if tensors:
        save_file(tensors, output_path)

    return weight_map_entries


def build_quantization_config(num_layers):
    """Build the quantization_config matching the reference model format."""
    ignore_list = ["lm_head", "model.layers.0*"]
    for layer_idx in range(1, num_layers):
        ignore_list.append(f"model.layers.{layer_idx}.mlp.router*")
        ignore_list.append(f"model.layers.{layer_idx}.mlp.shared_mlp*")
        ignore_list.append(f"model.layers.{layer_idx}.self_attn*")

    return {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "weights": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "targets": ["Linear"],
            }
        },
        "ignore": ignore_list,
        "quant_algo": "NVFP4",
        "kv_cache_scheme": {
            "dynamic": False,
            "num_bits": 8,
            "type": "float",
        },
        "producer": {
            "name": "modelopt",
            "version": "angelslim",
        },
        "quant_method": "modelopt",
    }


def main():
    args = parse_args()
    bf16_model_path = args.bf16_model_path or args.nvfp4_w_path

    os.makedirs(args.output_path, exist_ok=True)

    # =========================================================================
    # 1. Load config to get model dimensions
    # =========================================================================
    config_path = os.path.join(bf16_model_path, "config.json")
    with open(config_path, "r") as f:
        model_config = json.load(f)

    num_layers = model_config["num_hidden_layers"]
    num_experts = model_config.get("num_experts", 0)
    print(f"Model: {num_layers} layers, {num_experts} experts")

    # =========================================================================
    # 2. Load calibration statistics
    # =========================================================================
    print(f"Loading statistics from: {args.statistics_path}")
    act_stats = load_activation_stats(args.statistics_path)
    print(f"  Loaded {len(act_stats)} stat entries")

    # =========================================================================
    # 3. Compute KV scales (optionally from a separate file)
    # =========================================================================
    kv_statistics_path = args.kv_statistics_path if args.kv_statistics_path else ""
    if kv_statistics_path:
        print(f"Loading KV cache statistics from: {kv_statistics_path}")
        with open(kv_statistics_path, "r") as f:
            kv_act_stats = json.load(f)
        print(f"  Loaded {len(kv_act_stats)} KV stat entries")
    else:
        kv_act_stats = act_stats

    print("Computing KV cache scales...")
    kv_scales = compute_kv_scales(kv_act_stats, num_layers)
    print(f"  Computed {len(kv_scales)} KV scale entries")

    # =========================================================================
    # 4. Compute expert input scales
    # =========================================================================
    # Expert layers start from layer 1 (layer 0 is dense)
    print("Computing expert input scales...")
    input_scales = compute_expert_input_scales(act_stats, num_layers, num_experts)
    print(f"  Computed {len(input_scales)} input_scale entries")

    # =========================================================================
    # 5. Load NVFP4 model index and determine shards to process
    # =========================================================================
    nvfp4_index_path = os.path.join(args.nvfp4_w_path, "model.safetensors.index.json")
    with open(nvfp4_index_path, "r") as f:
        nvfp4_index = json.load(f)

    nvfp4_weight_map = nvfp4_index["weight_map"]
    shard_files = sorted(set(nvfp4_weight_map.values()))
    total_shards = len(shard_files)
    print(f"Processing {total_shards} shards from NVFP4 model...")

    # =========================================================================
    # 6. Process shards: copy NVFP4 expert weights + bf16 non-expert weights,
    #    inject KV scales and input_scales
    # =========================================================================
    # We need to handle shared_mlp specially: nvfp4 model has them quantized
    # but we want them in bf16. We'll need to source bf16 shared_mlp from
    # the bf16 model if available, otherwise from the nvfp4 model's bf16 copy.
    #
    # Strategy: Process nvfp4 shards directly. The nvfp4 model contains:
    #   - Expert weights as NVFP4 (.weight, .weight_scale, .weight_scale_2) -> KEEP
    #   - shared_mlp as NVFP4 (.weight, .weight_scale, .weight_scale_2)
    #     -> DROP scale, keep .weight as-is
    #     Actually shared_mlp .weight in nvfp4 model is already quantized (uint8 packed).
    #     We need bf16 shared_mlp from the bf16 source model.
    #   - Attention/layernorm/embed in bf16 -> KEEP

    # Identify which keys need bf16 replacement (shared_mlp weights)
    shared_mlp_weight_keys = [
        k for k in nvfp4_weight_map if "shared_mlp" in k and k.endswith(".weight")
    ]
    needs_bf16_source = len(shared_mlp_weight_keys) > 0

    # Load bf16 model index if needed
    bf16_weight_map = {}
    if needs_bf16_source and bf16_model_path != args.nvfp4_w_path:
        bf16_index_path = os.path.join(bf16_model_path, "model.safetensors.index.json")
        if os.path.exists(bf16_index_path):
            with open(bf16_index_path, "r") as f:
                bf16_weight_map = json.load(f)["weight_map"]

    # Build full weight map for output
    full_weight_map = {}

    for shard_idx, shard_file in enumerate(shard_files, 1):
        shard_path = os.path.join(args.nvfp4_w_path, shard_file)
        output_filename = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
        output_path = os.path.join(args.output_path, output_filename)

        tensors = {}

        # Load tensors from nvfp4 shard
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                # Skip shared_mlp quantization artifacts (weight_scale, weight_scale_2)
                if "shared_mlp" in key and "weight_scale" in key:
                    continue
                # Skip shared_mlp .weight (it's quantized uint8, we need bf16)
                if "shared_mlp" in key and key.endswith(".weight"):
                    continue
                tensors[key] = f.get_tensor(key)
                full_weight_map[key] = output_filename

        # Load bf16 shared_mlp weights for this shard
        if needs_bf16_source and bf16_weight_map:
            for smk in shared_mlp_weight_keys:
                # Check if this key was originally in this shard
                if nvfp4_weight_map.get(smk) == shard_file:
                    # Load from bf16 model
                    bf16_shard = bf16_weight_map.get(smk)
                    if bf16_shard:
                        bf16_shard_path = os.path.join(bf16_model_path, bf16_shard)
                        with safe_open(bf16_shard_path, framework="pt", device="cpu") as bf:
                            if smk in bf.keys():
                                tensors[smk] = bf.get_tensor(smk)
                                full_weight_map[smk] = output_filename
        elif needs_bf16_source:
            # nvfp4_w_path == bf16_model_path, shared_mlp is already quantized
            # This case means we don't have a separate bf16 source.
            # The shared_mlp weights in the nvfp4 model are packed uint8.
            # We can't recover bf16 from them. User must provide --bf16_model_path.
            pass

        # Inject KV scales into appropriate shards
        for key, tensor in kv_scales.items():
            layer_prefix = key.rsplit(".", 2)[0]  # model.layers.X.self_attn
            k_proj_weight = f"{layer_prefix}.k_proj.weight"
            if k_proj_weight in tensors:
                tensors[key] = tensor
                full_weight_map[key] = output_filename

        # Inject input_scales into appropriate shards
        for key, tensor in input_scales.items():
            weight_key = key.replace(".input_scale", ".weight")
            if weight_key in tensors:
                tensors[key] = tensor
                full_weight_map[key] = output_filename

        # Save
        if tensors:
            save_file(tensors, output_path)
            print(
                f"  [{shard_idx}/{total_shards}] Saved {output_filename} ({len(tensors)} tensors)"
            )

    # =========================================================================
    # 7. Write model.safetensors.index.json
    # =========================================================================
    output_index = {
        "metadata": {"total_size": 0},  # placeholder
        "weight_map": dict(sorted(full_weight_map.items())),
    }
    index_path = os.path.join(args.output_path, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(output_index, f, indent=2)
    print(f"Saved index: {index_path}")

    # =========================================================================
    # 8. Write config.json with quantization_config
    # =========================================================================
    output_config = copy.deepcopy(model_config)
    # Remove old quantization_config if present
    output_config["quantization_config"] = build_quantization_config(num_layers)
    config_out_path = os.path.join(args.output_path, "config.json")
    with open(config_out_path, "w") as f:
        json.dump(output_config, f, indent=2)
    print(f"Saved config: {config_out_path}")

    # =========================================================================
    # 8b. Write hf_quant_config.json
    # =========================================================================
    exclude_modules = ["lm_head", "model.layers.0*"]
    for layer_idx in range(1, num_layers):
        exclude_modules.append(f"model.layers.{layer_idx}.mlp.router*")
        exclude_modules.append(f"model.layers.{layer_idx}.mlp.shared_mlp*")
        exclude_modules.append(f"model.layers.{layer_idx}.self_attn*")

    hf_quant_config = {
        "producer": {
            "name": "modelopt",
            "version": "angelslim",
        },
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": "FP8",
            "group_size": 16,
            "exclude_modules": sorted(exclude_modules),
        },
    }
    hf_quant_config_path = os.path.join(args.output_path, "hf_quant_config.json")
    with open(hf_quant_config_path, "w") as f:
        json.dump(hf_quant_config, f, indent=4)
    print(f"Saved hf_quant_config: {hf_quant_config_path}")

    # =========================================================================
    # 9. Copy tokenizer and other files
    # =========================================================================
    copy_patterns = [
        "tokenizer*",
        "special_tokens_map*",
        "generation_config*",
        "preprocessor_config*",
        "chat_template*",
    ]
    for pattern in copy_patterns:
        for src_file in glob.glob(os.path.join(bf16_model_path, pattern)):
            dst_file = os.path.join(args.output_path, os.path.basename(src_file))
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                print(f"  Copied {os.path.basename(src_file)}")

    print(f"\nDone! Merged model saved to: {args.output_path}")
    print(f"  - KV scales: {len(kv_scales)} entries")
    print(f"  - Input scales: {len(input_scales)} entries")
    print(f"  - Total weight map entries: {len(full_weight_map)}")


if __name__ == "__main__":
    main()
