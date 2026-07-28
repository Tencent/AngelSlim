import json
import multiprocessing as mp
import os
import re
import shutil
from argparse import ArgumentParser
from copy import deepcopy

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

EXPERT_PATTERN = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$"
)

BF16_CONFIG_EXTRA_KEYS = {
    "mtp_loss_scaling_factor",
    "mtp_num_layers",
    "num_nextn_predict_layers",
    "num_nextn_predict_tokens",
}


def is_bf16_config_extra_key(key):
    """Return whether a missing bf16 config key is safe to append.

    The merge output is primarily an NVFP4 checkpoint: its tensor shapes must
    match the NVFP4 architecture config.  We therefore only supplement fields
    that are known metadata for optional MTP/nextn layers and never override an
    existing NVFP4 key.
    """

    key_lower = key.lower()
    return (
        key_lower in BF16_CONFIG_EXTRA_KEYS
        or key_lower.startswith("mtp_")
        or key_lower.startswith("multi_token")
        or "nextn" in key_lower
    )


def merge_model_configs(nvfp4_config, bf16_config):
    """Merge configs with NVFP4 as the source of truth.

    bf16 contributes only whitelisted keys that are absent from NVFP4.  This
    prevents bf16 architecture fields from silently changing layer dimensions
    while still allowing optional metadata such as MTP settings to be carried
    over.
    """

    config = deepcopy(nvfp4_config)
    for key, value in bf16_config.items():
        if key not in config and is_bf16_config_extra_key(key):
            config[key] = deepcopy(value)
    return config


def compute_expert_input_scales(statistics_path, num_hidden_layers):
    """Load moe_expert_stats.json and compute input_scale for each expert projection."""
    moe_path = os.path.join(statistics_path, "moe_expert_stats.json")
    if not os.path.exists(moe_path):
        return {}
    with open(moe_path, "r") as f:
        moe_stats = json.load(f)

    input_scales = {}
    for key, stats in moe_stats.items():
        m = re.match(r"(model\.layers\.(\d+)\.mlp\.experts\.(\d+))\.(gate_up_proj|down_proj)", key)
        if not m:
            continue
        prefix, layer_str, expert_str, proj_type = m.groups()
        layer_id = int(layer_str)
        if layer_id >= num_hidden_layers:
            continue

        absmax = max(abs(stats["min"]), abs(stats["max"]))
        scale = absmax / FP8_MAX

        if proj_type == "gate_up_proj":
            input_scales[f"{prefix}.gate_proj.input_scale"] = torch.tensor(
                scale, dtype=torch.float32
            )
            input_scales[f"{prefix}.up_proj.input_scale"] = torch.tensor(
                scale, dtype=torch.float32
            )
        else:
            input_scales[f"{prefix}.down_proj.input_scale"] = torch.tensor(
                scale, dtype=torch.float32
            )

    return input_scales


def process_shard(
    rank, file_names, nvfp4_path, output_path, num_hidden_layers, input_scales, return_dict
):
    """Worker: process a subset of safetensor files, injecting input_scale for experts."""
    for file_name in tqdm(file_names, desc=f"Worker {rank}"):
        state_dict = {}
        index = {}

        with safe_open(os.path.join(nvfp4_path, file_name), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for weight_name in keys:
                tensor = f.get_tensor(weight_name)
                state_dict[weight_name] = tensor
                index[weight_name] = file_name

                if weight_name.endswith(".weight"):
                    prefix = weight_name[: -len(".weight")]
                    if EXPERT_PATTERN.match(prefix):
                        input_scale_key = f"{prefix}.input_scale"
                        if input_scale_key in input_scales and input_scale_key not in state_dict:
                            state_dict[input_scale_key] = input_scales[input_scale_key]
                            index[input_scale_key] = file_name

        save_file(state_dict, os.path.join(output_path, file_name))
        del state_dict
        return_dict[file_name] = index


def copy_mtp_layers(bf16_path, nvfp4_weight_map, output_path, num_hidden_layers):
    """Copy MTP layers (layer_id >= num_hidden_layers) from bf16 model if not in nvfp4."""
    bf16_index_path = os.path.join(bf16_path, "model.safetensors.index.json")
    with open(bf16_index_path, "r") as f:
        bf16_index = json.load(f)
    bf16_weight_map = bf16_index["weight_map"]

    mtp_keys = [
        k
        for k in bf16_weight_map
        if re.match(r"model\.layers\.(\d+)\.", k)
        and int(re.match(r"model\.layers\.(\d+)\.", k).group(1)) >= num_hidden_layers
    ]

    already_in_nvfp4 = [k for k in mtp_keys if k in nvfp4_weight_map]
    if already_in_nvfp4:
        print(
            f"  MTP layers already in nvfp4 model"
            f" ({len(already_in_nvfp4)} keys), skipping MTP copy"
        )
        return {}

    if not mtp_keys:
        print("  No MTP layers found in bf16 model")
        return {}

    print(f"  Copying {len(mtp_keys)} MTP weight keys from bf16 model")

    mtp_files = sorted(set(bf16_weight_map[k] for k in mtp_keys))
    new_weight_map = {}

    for file_name in tqdm(mtp_files, desc="Copying MTP shards"):
        src_file = os.path.join(bf16_path, file_name)
        state_dict = {}
        with safe_open(src_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in mtp_keys:
                    state_dict[key] = f.get_tensor(key)
                    new_weight_map[key] = file_name

        if state_dict:
            save_file(state_dict, os.path.join(output_path, file_name))

    return new_weight_map


def build_quantization_config(num_hidden_layers):
    """Build quantization_config for config.json matching NVIDIA reference format."""
    ignore_list = ["lm_head"]

    for i in range(num_hidden_layers):
        if i < 3:
            ignore_list.append(f"model.layers.{i}*" if i == 0 else f"model.layers.{i}.*")
        else:
            ignore_list.append(f"model.layers.{i}.mlp.shared_experts*")
            ignore_list.append(f"model.layers.{i}.self_attn*")

    ignore_list.append(f"model.layers.{num_hidden_layers}*")

    ignore_list.sort()

    return {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                "targets": ["Linear"],
            }
        },
        "ignore": ignore_list,
        "quant_algo": "NVFP4",
        "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
        "producer": {
            "name": "modelopt",
        },
        "quant_method": "modelopt",
    }


def build_hf_quant_config(num_hidden_layers):
    """Build hf_quant_config.json matching NVIDIA reference format."""
    ignore_list = ["lm_head"]

    for i in range(num_hidden_layers):
        if i < 3:
            ignore_list.append(f"model.layers.{i}*" if i == 0 else f"model.layers.{i}.*")
        else:
            ignore_list.append(f"model.layers.{i}.mlp.shared_experts*")
            ignore_list.append(f"model.layers.{i}.self_attn*")

    ignore_list.append(f"model.layers.{num_hidden_layers}*")

    ignore_list.sort()

    return {
        "producer": {
            "name": "modelopt",
        },
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": "FP8",
            "group_size": 16,
            "exclude_modules": ignore_list,
        },
    }


def build_final_model_config(nvfp4_config, bf16_config):
    """Build config.json for the finalized ModelOpt-compatible checkpoint.

    Architecture fields remain sourced from the NVFP4 checkpoint, while the
    quantization schema is rebuilt unconditionally to describe the tensors
    produced by this merge (static NVFP4 activation scales and FP8 KV cache).
    The input checkpoint may contain an intermediate AngelSlim/GPTQ schema,
    which must not leak into the finalized deployment config.
    """

    config = merge_model_configs(nvfp4_config, bf16_config)
    config["quantization_config"] = build_quantization_config(
        config["num_hidden_layers"]
    )
    return config


def main():
    parser = ArgumentParser()
    parser.add_argument("--nvfp4_modelpath", type=str, required=True)
    parser.add_argument("--bf16_modelpath", type=str, required=True)
    parser.add_argument("--statistics_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    print(args)

    nvfp4_path = args.nvfp4_modelpath
    bf16_path = args.bf16_modelpath
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    # Merge config.json with NVFP4 as source of truth. 
    with open(os.path.join(nvfp4_path, "config.json"), "r") as f:
        nvfp4_config = json.load(f)
    with open(os.path.join(bf16_path, "config.json"), "r") as f:
        bf16_config = json.load(f)
    config = build_final_model_config(nvfp4_config, bf16_config)
    num_hidden_layers = config["num_hidden_layers"]
    print(f"num_hidden_layers: {num_hidden_layers}")

    # Read nvfp4 index
    index_path = os.path.join(nvfp4_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        nvfp4_index = json.load(f)
    nvfp4_weight_map = nvfp4_index["weight_map"]
    all_safetensor_files = sorted(set(nvfp4_weight_map.values()))
    safetensor_files = [
        f for f in all_safetensor_files if os.path.exists(os.path.join(nvfp4_path, f))
    ]
    if len(safetensor_files) < len(all_safetensor_files):
        print(
            f"WARNING: {len(all_safetensor_files) - len(safetensor_files)}"
            f" files in index not found on disk, skipping them"
        )
    print(f"Found {len(safetensor_files)} safetensor files to process")

    # Compute expert input scales
    print("Computing expert input scales...")
    input_scales = compute_expert_input_scales(args.statistics_path, num_hidden_layers)
    print(f"  Computed {len(input_scales)} input_scale entries")

    # Process nvfp4 shards with multiprocessing
    num_workers = min(args.num_workers, len(safetensor_files))
    file_subsets = [safetensor_files[i::num_workers] for i in range(num_workers)]

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=process_shard,
            args=(
                i,
                file_subsets[i],
                nvfp4_path,
                output_path,
                num_hidden_layers,
                input_scales,
                return_dict,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    new_weight_map = {}
    for result in return_dict.values():
        new_weight_map.update(result)

    # Copy MTP layers from bf16 model if needed
    print("Checking MTP layers...")
    mtp_weight_map = copy_mtp_layers(bf16_path, nvfp4_weight_map, output_path, num_hidden_layers)
    new_weight_map.update(mtp_weight_map)

    # Save model.safetensors.index.json
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {"metadata": {}, "weight_map": dict(sorted(new_weight_map.items()))}, f, indent=2
        )
    print("Saved model.safetensors.index.json")

    # Write config.json using the NVFP4 architecture plus a freshly generated
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("Saved config.json (nvfp4 architecture + ModelOpt quantization_config)")

    # Write hf_quant_config.json. 
    src_hf_quant_config = os.path.join(nvfp4_path, "hf_quant_config.json")
    dst_hf_quant_config = os.path.join(output_path, "hf_quant_config.json")
    if os.path.exists(src_hf_quant_config):
        shutil.copy2(src_hf_quant_config, dst_hf_quant_config)
        print("Copied hf_quant_config.json from nvfp4 model")
    else:
        hf_quant_config = build_hf_quant_config(num_hidden_layers)
        with open(dst_hf_quant_config, "w") as f:
            json.dump(hf_quant_config, f, indent=2)
        print("Saved generated hf_quant_config.json")

    # Copy other files from bf16_modelpath (excluding safetensors, index, config.json)
    print("Copying auxiliary files from bf16 model...")
    for item in os.listdir(bf16_path):
        if item.endswith(".safetensors"):
            continue
        if item in ("model.safetensors.index.json", "config.json", "hf_quant_config.json"):
            continue
        dst = os.path.join(output_path, item)
        if os.path.exists(dst):
            continue
        src = os.path.join(bf16_path, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"  Copied {item}")

    print(f"\nDone! Output: {output_path}")


if __name__ == "__main__":
    main()
