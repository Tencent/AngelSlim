#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Merge NVFP4 expert weights + FP8 KV cache scales + activation input_scales
# into a single HF model directory for vLLM inference.  When the source HY3
# checkpoint contains appended MTP layers, their GEMM weights are serialized as
# static per-tensor FP8 and the final model uses ModelOpt MIXED_PRECISION.
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
from hy3_mtp_utils import (
    detect_hy3_mtp,
    fp8_scale_from_stats,
    is_mtp_fp8_weight,
    is_mtp_key,
    load_mtp_stats,
    module_name_from_weight,
    quantize_weight_per_tensor_fp8,
    read_weight_map,
    resolve_mtp_activation_key,
)
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
    parser.add_argument(
        "--mtp-fp8-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="MTP FP8 handling: auto=quantize detected MTP layers, on=require MTP, "
        "off=keep the existing non-FP8 behavior.",
    )
    parser.add_argument(
        "--allow-missing-mtp-stats",
        action="store_true",
        help="Keep MTP GEMM weights with missing activation statistics in BF16 "
        "instead of failing. Intended for debugging only.",
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


def compute_kv_scales(act_stats, layers):
    """
    Compute per-tensor FP8 KV cache scales from activation_stats.

    Keys in act_stats: "model.layers.{L}.self_attn.attn.k_cache" -> {"min": float, "max": float}
    Output keys: "model.layers.{L}.self_attn.k_proj.k_scale" -> float tensor
                 "model.layers.{L}.self_attn.v_proj.v_scale" -> float tensor
    """
    layer_ids = range(layers) if isinstance(layers, int) else layers
    kv_scales = {}
    for layer_idx in layer_ids:
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


def compute_mtp_kv_scales(mtp_stats, mtp_layer_ids):
    """Compute per-tensor MTP KV-cache scales from normalized draft stats."""

    return compute_kv_scales(mtp_stats, mtp_layer_ids)


def validate_mtp_stats(layout, mtp_stats, allow_missing=False):
    """Validate that every supported MTP FP8 weight has static input stats."""

    missing = []
    for weight_name in layout.weight_keys:
        if not is_mtp_fp8_weight(weight_name):
            continue
        stat_key = resolve_mtp_activation_key(weight_name)
        if stat_key not in mtp_stats:
            missing.append((weight_name, stat_key))

    if missing and not allow_missing:
        details = "\n".join(
            f"  weight={weight_name} expected_stat={stat_key}"
            for weight_name, stat_key in missing[:20]
        )
        suffix = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise ValueError(
            "Missing MTP activation statistics required for static FP8 quantization:\n"
            f"{details}{suffix}"
        )
    return missing


def write_mtp_fp8_shards(
    bf16_model_path,
    output_path,
    layout,
    mtp_stats,
    mtp_kv_scales,
    allow_missing_stats=False,
):
    """Write detected MTP tensors as dedicated FP8/BF16 safetensor shards."""

    source_weight_map = read_weight_map(bf16_model_path)
    keys_by_shard = {}
    for key in layout.weight_keys:
        source_shard = source_weight_map[key]
        keys_by_shard.setdefault(source_shard, []).append(key)

    missing = validate_mtp_stats(layout, mtp_stats, allow_missing=allow_missing_stats)
    missing_weights = {weight_name for weight_name, _ in missing}

    output_weight_map = {}
    fp8_modules = set()
    written_input_scales = set()
    kv_items = list(sorted(mtp_kv_scales.items()))

    for shard_idx, (source_shard, weight_names) in enumerate(
        sorted(keys_by_shard.items()), start=1
    ):
        output_filename = f"model-mtp-fp8-{shard_idx:05d}.safetensors"
        tensors = {}

        with safe_open(
            os.path.join(bf16_model_path, source_shard), framework="pt", device="cpu"
        ) as source:
            for weight_name in sorted(weight_names):
                weight = source.get_tensor(weight_name)
                should_quantize = (
                    is_mtp_fp8_weight(weight_name) and weight_name not in missing_weights
                )
                if not should_quantize:
                    tensors[weight_name] = weight
                    output_weight_map[weight_name] = output_filename
                    continue

                stat_key = resolve_mtp_activation_key(weight_name)
                module_name = module_name_from_weight(weight_name)
                quantized_weight, weight_scale = quantize_weight_per_tensor_fp8(weight)
                weight_scale_name = f"{weight_name}_scale"
                input_scale_name = f"{module_name}.input_scale"

                tensors[weight_name] = quantized_weight
                tensors[weight_scale_name] = weight_scale
                output_weight_map[weight_name] = output_filename
                output_weight_map[weight_scale_name] = output_filename

                if input_scale_name not in written_input_scales:
                    tensors[input_scale_name] = fp8_scale_from_stats(mtp_stats[stat_key])
                    output_weight_map[input_scale_name] = output_filename
                    written_input_scales.add(input_scale_name)

                fp8_modules.add(module_name)

        # KV-cache scales are small and can live in the first MTP shard.
        if shard_idx == 1:
            for key, tensor in kv_items:
                tensors[key] = tensor
                output_weight_map[key] = output_filename

        save_file(tensors, os.path.join(output_path, output_filename))
        print(
            f"  [MTP {shard_idx}/{len(keys_by_shard)}] Saved {output_filename} "
            f"({len(tensors)} tensors)"
        )

    return output_weight_map, fp8_modules


def build_quantized_layers(weight_map, fp8_modules):
    """Build vLLM ModelOpt per-layer algorithm metadata from output tensors."""

    quantized_layers = {}
    for key in weight_map:
        if not key.endswith(".weight_scale_2"):
            continue
        module_name = key[: -len(".weight_scale_2")]
        quantized_layers[module_name] = {"quant_algo": "NVFP4", "group_size": 16}

    for module_name in sorted(fp8_modules):
        quantized_layers[module_name] = {"quant_algo": "FP8"}

    return dict(sorted(quantized_layers.items()))


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


def build_ignore_list(num_layers, extra_ignored_prefixes=()):
    """Build ModelOpt wildcard exclusions for BF16 modules."""

    ignore_list = ["lm_head", "model.layers.0*"]
    for layer_idx in range(1, num_layers):
        ignore_list.append(f"model.layers.{layer_idx}.mlp.router*")
        ignore_list.append(f"model.layers.{layer_idx}.mlp.shared_mlp*")
        ignore_list.append(f"model.layers.{layer_idx}.self_attn*")
    ignore_list.extend(f"{prefix}*" for prefix in extra_ignored_prefixes)
    return ignore_list


def build_quantization_config(num_layers, extra_ignored_prefixes=()):
    """Build the quantization_config matching the reference model format."""
    ignore_list = build_ignore_list(num_layers, extra_ignored_prefixes)

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


def build_mixed_quantization_config(num_layers, quantized_layers):
    """Build ModelOpt metadata for NVFP4 main experts + FP8 MTP layers."""

    return {
        "group_size": 16,
        "ignore": build_ignore_list(num_layers),
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": quantized_layers,
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

    mtp_layout = detect_hy3_mtp(bf16_model_path)
    if args.mtp_fp8_mode == "on" and not mtp_layout.has_mtp:
        raise ValueError("--mtp-fp8-mode=on was requested, but no MTP layers were detected")
    mtp_fp8_enabled = args.mtp_fp8_mode != "off" and mtp_layout.has_mtp
    if mtp_layout.has_mtp:
        print(
            "MTP checkpoint layers detected: "
            f"{list(mtp_layout.layer_ids) or list(mtp_layout.explicit_prefixes)}; "
            f"FP8 mode={'enabled' if mtp_fp8_enabled else 'disabled'}"
        )
    else:
        print("No MTP checkpoint layers detected")

    # =========================================================================
    # 2. Load calibration statistics
    # =========================================================================
    print(f"Loading statistics from: {args.statistics_path}")
    act_stats = load_activation_stats(args.statistics_path)
    print(f"  Loaded {len(act_stats)} stat entries")

    mtp_stats = {}
    mtp_kv_scales = {}
    if mtp_fp8_enabled:
        mtp_stats = load_mtp_stats(args.statistics_path)
        if not mtp_stats:
            raise FileNotFoundError(
                "MTP layers were detected, but mtp_activation_stats.json / "
                "mtp_moe_expert_stats.json are missing or empty"
            )
        print(f"  Loaded {len(mtp_stats)} normalized MTP stat entries")
        mtp_kv_scales = compute_mtp_kv_scales(mtp_stats, mtp_layout.layer_ids)
        print(f"  Computed {len(mtp_kv_scales)} MTP KV scale entries")

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
        k
        for k in nvfp4_weight_map
        if "shared_mlp" in k
        and k.endswith(".weight")
        and not (mtp_fp8_enabled and is_mtp_key(k, mtp_layout))
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
                if mtp_fp8_enabled and is_mtp_key(key, mtp_layout):
                    continue
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

    fp8_mtp_modules = set()
    if mtp_fp8_enabled:
        print("Quantizing MTP layers to static per-tensor FP8...")
        mtp_weight_map, fp8_mtp_modules = write_mtp_fp8_shards(
            bf16_model_path=bf16_model_path,
            output_path=args.output_path,
            layout=mtp_layout,
            mtp_stats=mtp_stats,
            mtp_kv_scales=mtp_kv_scales,
            allow_missing_stats=args.allow_missing_mtp_stats,
        )
        full_weight_map.update(mtp_weight_map)
        print(
            f"  Added {len(mtp_weight_map)} MTP tensor entries across "
            f"{len(fp8_mtp_modules)} FP8 modules"
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
    quantized_layers = {}
    if mtp_fp8_enabled:
        quantized_layers = build_quantized_layers(full_weight_map, fp8_mtp_modules)
        output_config["quantization_config"] = build_mixed_quantization_config(
            num_layers, quantized_layers
        )
    else:
        extra_ignored = []
        if mtp_layout.has_mtp:
            extra_ignored = [
                *(f"model.layers.{layer_id}" for layer_id in mtp_layout.layer_ids),
                *mtp_layout.explicit_prefixes,
            ]
        output_config["quantization_config"] = build_quantization_config(
            num_layers, extra_ignored_prefixes=extra_ignored
        )
    config_out_path = os.path.join(args.output_path, "config.json")
    with open(config_out_path, "w") as f:
        json.dump(output_config, f, indent=2)
    print(f"Saved config: {config_out_path}")

    # =========================================================================
    # 8b. Write hf_quant_config.json
    # =========================================================================
    exclude_modules = build_ignore_list(num_layers)
    hf_quantization = {
        "quant_algo": "MIXED_PRECISION" if mtp_fp8_enabled else "NVFP4",
        "kv_cache_quant_algo": "FP8",
        "group_size": 16,
        "exclude_modules": sorted(exclude_modules),
    }
    if mtp_fp8_enabled:
        hf_quantization["quantized_layers"] = quantized_layers

    hf_quant_config = {
        "producer": {
            "name": "modelopt",
            "version": "angelslim",
        },
        "quantization": hf_quantization,
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
    print(f"  - MTP FP8 modules: {len(fp8_mtp_modules)}")
    print(f"  - Total weight map entries: {len(full_weight_map)}")


if __name__ == "__main__":
    main()
