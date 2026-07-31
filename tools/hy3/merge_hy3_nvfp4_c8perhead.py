#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Merge NVFP4 expert weights + FP8 KV cache scales + activation input_scales
# into a single HF model directory with fine-grained attention quantization format.
#
# This is equivalent to running merge_hy3_nvfp4_c8.py + convert_c8_new.py in one step.
#
# Differences from merge_hy3_nvfp4_c8.py:
#   - config.json: kv_cache_scheme = "static" + attn_quant_config
#   - Only v_cache scales are kept (as model.layers.*.self_attn.v_cache.scale)
#   - k_scale is NOT written (k_quant is dynamic per_token_per_head)
#
# Inputs:
#   --statistics_path: dir containing activation_stats.json & moe_expert_stats.json
#   --nvfp4_w_path: NVFP4 weight-only model dir
#   --output_path: where to write the merged model
#   --bf16_model_path: (optional) original bf16 model for config.json/tokenizer
#   --kv_statistics_path: (optional) separate activation_stats for KV cache scales

import argparse
import copy
import glob
import json
import multiprocessing as mp
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# FP8 E4M3 max value
FP8_MAX = 448.0

ATTN_QUANT_CONFIG = {
    "kv_cache_quant": {
        "dtype": "fp8_e4m3",
        "k_quant": {
            "scheme": "dynamic",
            "granularity": "per_token_per_head",
        },
        "v_quant": {
            "scheme": "static",
            "granularity": "per_head",
        },
    },
    "p_quant": {
        "dtype": "fp8_e4m3",
        "scheme": "static",
        "granularity": "per_head",
    },
    "q_quant": {
        "dtype": "fp8_e4m3",
        "scheme": "dynamic",
        "granularity": "per_token_per_head",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge NVFP4 weights + FP8 fine-grained attention quant (one-step)"
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
        "--kv_statistics_path",
        type=str,
        default="",
        help="Path to a separate activation_stats JSON file for KV cache scales. "
        "If not set, KV cache stats are loaded from statistics_path/activation_stats.json.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers for processing shards",
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

    act_stats.update(moe_stats)
    return act_stats


def compute_v_cache_scales(act_stats, num_layers):
    """
    Compute per-head FP8 V cache scales from activation_stats.
    Only V cache scales are needed (K is dynamic per_token_per_head).

    Output keys: "model.layers.{L}.self_attn.v_cache.scale" -> float tensor (per-head)
    """
    v_scales = {}
    for layer_idx in range(num_layers):
        keys = [
            f"model.layers.{layer_idx}.self_attn.attn.v_cache",
            f"model.layers.{layer_idx}.self_attn.v_proj.v_scale",
        ]
        key = next((k for k in keys if k in act_stats), None)
        if key is None:
            print(f"  [WARNING] Missing V cache stats for layer {layer_idx}, skipping")
            continue

        stats = act_stats[key]
        min_val = stats["min"]
        max_val = stats["max"]

        # Per-head: compute scale per head
        if isinstance(min_val, list):
            absmax = [max(abs(min_val[i]), abs(max_val[i])) * 2 for i in range(len(min_val))]
            scale = [ami / FP8_MAX for ami in absmax]
        else:
            absmax = max(abs(min_val), abs(max_val))
            scale = absmax / FP8_MAX

        out_key = f"model.layers.{layer_idx}.self_attn.v_cache.scale"
        v_scales[out_key] = torch.tensor(scale, dtype=torch.float32)

    return v_scales


def compute_expert_input_scales(act_stats, num_layers, num_experts):
    """
    Compute input_scale for each expert projection from moe_expert_stats.
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


def build_quantization_config(num_layers):
    """Build the quantization_config with kv_cache_scheme = 'static'."""
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
        "kv_cache_scheme": "static",
        "producer": {
            "name": "modelopt",
            "version": "angelslim",
        },
        "quant_method": "modelopt",
    }


def process_shard(
    shard_idx,
    shard_file,
    total_shards,
    nvfp4_w_path,
    output_path,
    nvfp4_weight_map,
    shared_mlp_weight_keys,
    needs_bf16_source,
    bf16_model_path,
    bf16_weight_map,
    v_scales,
    input_scales,
):
    """Process a single shard in a worker process. Returns weight_map entries."""
    shard_path = os.path.join(nvfp4_w_path, shard_file)
    output_filename = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
    out_path = os.path.join(output_path, output_filename)

    tensors = {}
    weight_map_entries = {}

    # Load tensors from nvfp4 shard
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if "shared_mlp" in key and "weight_scale" in key:
                continue
            if "shared_mlp" in key and key.endswith(".weight"):
                continue
            tensors[key] = f.get_tensor(key)
            weight_map_entries[key] = output_filename

    # Load bf16 shared_mlp weights
    if needs_bf16_source and bf16_weight_map:
        for smk in shared_mlp_weight_keys:
            if nvfp4_weight_map.get(smk) == shard_file:
                bf16_shard = bf16_weight_map.get(smk)
                if bf16_shard:
                    bf16_shard_path = os.path.join(bf16_model_path, bf16_shard)
                    with safe_open(bf16_shard_path, framework="pt", device="cpu") as bf:
                        if smk in bf.keys():
                            tensors[smk] = bf.get_tensor(smk)
                            weight_map_entries[smk] = output_filename

    # Inject V cache scales
    for key, tensor in v_scales.items():
        layer_prefix = key.rsplit(".", 2)[0]
        k_proj_weight = f"{layer_prefix}.k_proj.weight"
        if k_proj_weight in tensors:
            tensors[key] = tensor
            weight_map_entries[key] = output_filename

    # Inject input_scales
    for key, tensor in input_scales.items():
        weight_key = key.replace(".input_scale", ".weight")
        if weight_key in tensors:
            tensors[key] = tensor
            weight_map_entries[key] = output_filename

    if tensors:
        save_file(tensors, out_path)

    return weight_map_entries, output_filename, len(tensors)


def worker(
    worker_id,
    shard_items,
    nvfp4_w_path,
    output_path,
    nvfp4_weight_map,
    shared_mlp_weight_keys,
    needs_bf16_source,
    bf16_model_path,
    bf16_weight_map,
    v_scales,
    input_scales,
    return_dict,
):
    """Worker process that handles a subset of shards."""
    for shard_idx, shard_file, total_shards in tqdm(shard_items, desc=f"Worker {worker_id}"):
        entries, output_filename, n_tensors = process_shard(
            shard_idx,
            shard_file,
            total_shards,
            nvfp4_w_path,
            output_path,
            nvfp4_weight_map,
            shared_mlp_weight_keys,
            needs_bf16_source,
            bf16_model_path,
            bf16_weight_map,
            v_scales,
            input_scales,
        )
        return_dict[shard_file] = entries


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
    # 3. Compute V cache scales only (K is dynamic, no static scale needed)
    # =========================================================================
    kv_statistics_path = args.kv_statistics_path if args.kv_statistics_path else ""
    if kv_statistics_path:
        print(f"Loading KV cache statistics from: {kv_statistics_path}")
        with open(kv_statistics_path, "r") as f:
            kv_act_stats = json.load(f)
        print(f"  Loaded {len(kv_act_stats)} KV stat entries")
    else:
        kv_act_stats = act_stats

    print("Computing V cache scales (fine-grained attention quant)...")
    v_scales = compute_v_cache_scales(kv_act_stats, num_layers)
    print(f"  Computed {len(v_scales)} V cache scale entries")

    # =========================================================================
    # 4. Compute expert input scales
    # =========================================================================
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
    # 6. Identify shared_mlp keys that need bf16 source
    # =========================================================================
    shared_mlp_weight_keys = [
        k for k in nvfp4_weight_map if "shared_mlp" in k and k.endswith(".weight")
    ]
    needs_bf16_source = len(shared_mlp_weight_keys) > 0

    bf16_weight_map = {}
    if needs_bf16_source and bf16_model_path != args.nvfp4_w_path:
        bf16_index_path = os.path.join(bf16_model_path, "model.safetensors.index.json")
        if os.path.exists(bf16_index_path):
            with open(bf16_index_path, "r") as f:
                bf16_weight_map = json.load(f)["weight_map"]

    # =========================================================================
    # 7. Process shards in parallel (multiprocessing with file subsets)
    # =========================================================================
    full_weight_map = {}

    shard_items = [(idx, sf, total_shards) for idx, sf in enumerate(shard_files, 1)]
    num_workers = min(args.num_workers, total_shards)

    print(f"Processing {total_shards} shards with {num_workers} workers...")
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    file_subsets = [shard_items[i::num_workers] for i in range(num_workers)]
    processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker,
            args=(
                i,
                file_subsets[i],
                args.nvfp4_w_path,
                args.output_path,
                nvfp4_weight_map,
                shared_mlp_weight_keys,
                needs_bf16_source,
                bf16_model_path,
                bf16_weight_map,
                v_scales,
                input_scales,
                return_dict,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for entries in return_dict.values():
        full_weight_map.update(entries)

    # =========================================================================
    # 8. Write model.safetensors.index.json
    # =========================================================================
    output_index = {
        "metadata": {"total_size": 0},
        "weight_map": dict(sorted(full_weight_map.items())),
    }
    index_path = os.path.join(args.output_path, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(output_index, f, indent=2)
    print(f"Saved index: {index_path}")

    # =========================================================================
    # 9. Write config.json with quantization_config + attn_quant_config
    # =========================================================================
    output_config = copy.deepcopy(model_config)
    output_config["quantization_config"] = build_quantization_config(num_layers)
    output_config["attn_quant_config"] = ATTN_QUANT_CONFIG
    config_out_path = os.path.join(args.output_path, "config.json")
    with open(config_out_path, "w") as f:
        json.dump(output_config, f, indent=2)
    print(f"Saved config: {config_out_path}")

    # =========================================================================
    # 9b. Write hf_quant_config.json
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
    # 10. Copy tokenizer and other files
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
    print(f"  - V cache scales: {len(v_scales)} entries")
    print(f"  - Input scales: {len(input_scales)} entries")
    print(f"  - Total weight map entries: {len(full_weight_map)}")


if __name__ == "__main__":
    main()
