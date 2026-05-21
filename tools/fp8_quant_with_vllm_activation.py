import json
import math
import multiprocessing as mp
import os
import shutil
from argparse import ArgumentParser
from typing import Dict, List, Tuple

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

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


def create_quantized_param(param, weight_block_size=(128, 128)):
    """
    Quantizes weights to FP8 format using Block-wise quantization
    """
    # Get FP8 min/max values
    fp8_min = torch.finfo(torch.float8_e4m3fn).min
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    block_size_m, block_size_n = weight_block_size
    rows, cols = param.shape[-2:]

    # tensor-wise
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
    # Reshape back to matrix shape
    quantized_param = quantized_param.reshape(param_value_shape)[..., :rows, :cols]

    # Reshape scale_inv to match the number of blocks
    scale_inv = scale_inv.reshape(scale_orig_shape).squeeze().reciprocal()

    return quantized_param.contiguous(), scale_inv.contiguous()


def quantize_weight_per_tensor_fp8(
    tensor: torch.Tensor, scale: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    finfo = torch.finfo(torch.float8_e4m3fn)

    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float()
    return qweight, scale


def compute_scales(x, method="abs_max", group_size=-1):
    if method == "abs_max":
        quant_scale = torch.max(torch.abs(x.flatten()))
        quant_scale = 1e-8 if quant_scale == 0.0 else quant_scale
    return quant_scale


def process_safetensor(rank, file_name, bf16_path, fp8_path, block_size, ac_json_data):
    state_dict = {}
    index = {}
    count = 0
    with safe_open(os.path.join(bf16_path, file_name), framework="pt", device=f"cuda:{rank}") as f:
        print(f"Processing {file_name} with {len(f.keys())} weights")
        for weight_name in f.keys():
            weight = f.get_tensor(weight_name)
            if any(weight_name.endswith(suffix) for suffix in SUFFIX_TO_QUANT):
                # print(f"QDQ {weight_name}")
                scale_inv = weight.abs().max() / torch.finfo(torch.float8_e4m3fn).max
                quant_weight, scale_inv = quantize_weight_per_tensor_fp8(weight, scale_inv)
                # create_quantized_param(weight, block_size)
                # quant_weight, scale_inv = fake_quant_dequant(weight,
                #                                              bits=4)  # noqa: E501  # create_quantized_param(weight, block_size)
                # print(torch.norm(quant_weight-weight),scale_inv)
                scale_inv = scale_inv.view(-1) if scale_inv.ndim == 0 else scale_inv
                state_dict[weight_name] = quant_weight
                index[weight_name] = file_name
                if block_size[0] == -1 and block_size[1] == -1:  # tensor-wise
                    state_dict[f"{weight_name}_scale"] = scale_inv
                    index[f"{weight_name}_scale"] = file_name
                    # gate_up_proj
                    name_parts = weight_name.split(".")
                    if (
                        "mlp.experts" in weight_name
                        or "mlp.gate" in weight_name
                        or "mlp.up" in weight_name
                    ) and "down" not in weight_name:
                        prefix_name = ".".join(name_parts[:-2])
                        act_vllm_name = prefix_name + ".gate_up_proj"
                    elif (
                        "q_proj" in weight_name
                        or "k_proj" in weight_name
                        or "v_proj" in weight_name
                    ):
                        prefix_name = ".".join(name_parts[:-2])
                        act_vllm_name = prefix_name + ".qkv_proj"
                    else:
                        act_vllm_name = ".".join(name_parts[:-1])

                    act_save_name = f"{'.'.join(name_parts[:-1])}.input_scale"
                    input_scale = (
                        max(
                            abs(ac_json_data[act_vllm_name]["min"]),
                            abs(ac_json_data[act_vllm_name]["max"]),
                        )
                        / torch.finfo(torch.float8_e4m3fn).max
                    )
                    # input_scale = min(input_scal, 2.00)
                    if input_scale >= 2.00:
                        print(f"big weight_name:{weight_name} scale, {input_scale}")
                    tensor_input_scale = torch.tensor([input_scale])
                    state_dict[act_save_name] = tensor_input_scale
                    index[act_save_name] = file_name
                else:  # block-wise
                    raise AssertionError("block-wise FP8 path not supported yet")
                    state_dict[f"{weight_name}_scale_inv"] = scale_inv
                    index[f"{weight_name}_scale_inv"] = file_name
            else:
                print("###skip ", weight_name)
                state_dict[weight_name] = weight
                index[weight_name] = file_name
            count += 1

    new_safetensor_file = os.path.join(fp8_path, file_name)
    save_file(state_dict, new_safetensor_file)
    return index


def worker(i, file_names, bf16_path, fp8_path, block_size, return_dict, ac_json_data):
    world_size = torch.cuda.device_count()
    for file_name in tqdm(file_names, desc=f"Worker {i}"):
        index = process_safetensor(
            i % world_size, file_name, bf16_path, fp8_path, block_size, ac_json_data
        )
        return_dict[file_name] = index


def main(bf16_path, fp8_path, block_size, ac_json_data):
    os.makedirs(fp8_path, exist_ok=True)
    model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    safetensor_files = set(weight_map.values())
    safetensor_files = list(sorted(safetensor_files))
    print(f"Found {len(safetensor_files)} safetensor files")

    file_subsets = [safetensor_files[i :: args.num_workers] for i in range(args.num_workers)]
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    for i in range(args.num_workers):
        p = mp.Process(
            target=worker,
            args=(i, file_subsets[i], bf16_path, fp8_path, block_size, return_dict, ac_json_data),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    index = {}
    for result in return_dict.values():
        index.update(result)

    # Copy config / auxiliary files from the bf16 model dir
    for file in os.listdir(bf16_path):
        src_path = os.path.join(bf16_path, file)
        dst_path = os.path.join(fp8_path, file)
        if os.path.exists(dst_path):
            continue
        if os.path.isdir(src_path):
            print(f"cp -r {src_path} {dst_path}")
            shutil.copytree(src_path, dst_path)
        else:
            print(f"cp {src_path} {dst_path}")
            shutil.copy2(src_path, dst_path)

    # Quantization config consumed by vLLM at load time
    with open(os.path.join(fp8_path, "config.json"), "r") as f:
        config = json.load(f)
    config["quantization_config"] = {
        "activation_scheme": "static",
        "ignored_layers": ["lm_head", "model.embed_tokens"],
        "quant_method": "fp8",
    }
    if block_size[0] != -1 and block_size[1] != -1:
        config["quantization_config"]["weight_block_size"] = block_size

    kv_state_dict = {}
    kv_granularity = ""

    # Auto-detect kv_head_repeat from the model's real num_key_value_heads
    # vs the per-head stats vector length collected by AngelSlim.
    #
    # The new vllm_calibrate_utils.setup_kvcache_perhead_hooks already
    # all-reduces per-head stats to length == num_key_value_heads, so the
    # collected list is already de-duplicated and stride=1 is correct.
    # If a future / older calibration run produces a longer (replicated)
    # vector instead, ``collected_len // real_num_kv_heads`` is the
    # replication factor and we stride by it to recover one entry per
    # real KV head.
    real_num_kv_heads = None
    try:
        with open(os.path.join(bf16_path, "config.json"), "r", encoding="utf8") as _cf:
            _model_cfg = json.load(_cf)
        real_num_kv_heads = _model_cfg.get("num_key_value_heads")
    except Exception as _e:
        print(f"[WARN] could not read num_key_value_heads from config.json: {_e}")

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    for scale_name, stats in ac_json_data.items():
        if "cache" not in scale_name:
            continue
        act_save_name = f"{scale_name.replace('attn.attn', 'attn')}.scale"
        min_val = stats["min"]
        max_val = stats["max"]
        if isinstance(min_val, list):
            # per-head: compute one scale per head
            per_head_scales = [max(abs(mn), abs(mx)) / fp8_max for mn, mx in zip(min_val, max_val)]
            # Auto-detect replication factor: if the collected list is
            # longer than the model's real num_key_value_heads, each real
            # head was repeated ``replication`` consecutive times across
            # the list (the legacy collect path).  Stride-sample to keep
            # one copy per real head.  When replication == 1 (the new
            # collect path which already de-duplicates) this is a no-op.
            collected_len = len(per_head_scales)
            if (
                real_num_kv_heads is not None
                and real_num_kv_heads > 0
                and collected_len % real_num_kv_heads == 0
                and collected_len > real_num_kv_heads
            ):
                replication = collected_len // real_num_kv_heads
                print(
                    f"[KV-scale] {scale_name}: collected_len={collected_len}, "
                    f"num_key_value_heads={real_num_kv_heads}, "
                    f"replication={replication} -> striding [::{replication}]"
                )
                per_head_scales = per_head_scales[::replication]
            tensor_input_scale = torch.tensor(per_head_scales, dtype=torch.float32)
            kv_granularity = "per_head"
        else:
            # per-tensor: single scalar scale
            # input_scale = max(abs(min_val), abs(max_val)) / fp8_max
            # tensor_input_scale = torch.tensor([input_scale], dtype=torch.float32)
            tensor_input_scale = torch.tensor([1.0])
            kv_granularity = "per_tensor"
        print(f"{scale_name}  granularity={kv_granularity}  scale={tensor_input_scale}")
        kv_state_dict[act_save_name] = tensor_input_scale
        index[act_save_name] = "kv_cache_scales.safetensors"
    if len(kv_state_dict) > 0:
        kv_safetensor_file = os.path.join(fp8_path, "kv_cache_scales.safetensors")
        save_file(kv_state_dict, kv_safetensor_file)
        config["quantization_config"]["kv_cache_scheme"] = "static"
        config["attn_quant_config"] = {
            "kv_cache_quant": {
                "dtype": "fp8_e4m3",
                "k_quant": {"scheme": "static", "granularity": kv_granularity},
                "v_quant": {"scheme": "static", "granularity": kv_granularity},
            },
            "q_quant": {
                "dtype": "fp8_e4m3",
                "scheme": "dynamic",
                "granularity": "per_token_per_head",
            },
        }

    with open(os.path.join(fp8_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": index}, f, indent=2)

    print(f"quant config: {config['quantization_config']}")
    with open(os.path.join(fp8_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def merge_vllm_act_moe_jsonl(ac_json_data):
    def process_moe_values(data: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Process MoE expert values: for each layer's gate/up/down section,
        broadcast the maximum value across all experts in that section to
        every expert in the same section. Non-MoE linear-layer entries are
        left untouched.

        Args:
            data: original dict whose keys are "<layer>.<linear>" names and
                whose values are scalar numbers.

        Returns:
            The processed dict (modified in place and returned).
        """
        # Grouping store: group_key -> [(original_key, value), ...]
        groups: Dict[str, List[Tuple[str, float]]] = {}

        # Iterate over all key/value pairs
        for key, value in data.items():
            parts = key.split(".")
            # print(parts)
            # Look for the "expert" or "experts" keyword
            expert_idx = None
            for i, part in enumerate(parts):
                if part in ("expert", "experts"):
                    expert_idx = i
                    break

            if expert_idx is None:
                # No expert keyword, skip (non-MoE key)
                # print("skip key", key)
                continue

            # Check whether the segment after "expert" is a numeric index
            if expert_idx + 1 >= len(parts):
                continue

            int(parts[expert_idx + 1])  # try to parse as int

            # # Optionally also require gate/up/down before the expert idx
            # expert_types = {'gate', 'up', 'down'}
            # found_type = False
            # for j in range(expert_idx):
            #     if parts[j] in expert_types:
            #         found_type = True
            #         break
            # if not found_type:
            #     continue

            # Confirmed MoE expert key; build the group id (drop the numeric idx)
            group_parts = parts[: expert_idx + 1] + parts[expert_idx + 2 :]

            group_key = ".".join(group_parts)
            # print(group_key)

            # Add to its group
            groups.setdefault(group_key, []).append((key, value["max"]))

        # For each group, compute the max and broadcast it back
        for _group_key, items in groups.items():
            values = [v for _, v in items]
            max_val = max(values)
            for key, _ in items:
                data[key]["max"] = max_val

        return data

    return process_moe_values(ac_json_data)


if __name__ == "__main__":
    parser = ArgumentParser()
    # YAML config (values override argparse defaults; explicit CLI flags still win).
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. Keys must match argparse dest names "
        "(e.g. input_bf16_hf_path, output_fp8_hf_path, block_size). Values "
        "override argparse defaults; explicit command-line flags still take "
        "final precedence.",
    )
    parser.add_argument("--block-size", type=int, nargs=2, default=(-1, -1))
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--input_bf16_hf_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--input_vllm_ac_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output_fp8_hf_path",
        type=str,
        default="",
    )
    args = parser.parse_args()

    # Lazy-import _yaml_args (sibling module in tools/). Done here instead of
    # at module top so flake8 doesn't trip on a sys.path mutation between
    # imports.
    import sys

    _tools_dir = os.path.dirname(os.path.abspath(__file__))
    if _tools_dir not in sys.path:
        sys.path.insert(0, _tools_dir)
    from _yaml_args import apply_yaml_config

    apply_yaml_config(parser, args)

    # Validate required paths (may come from CLI or YAML).
    missing = [
        name
        for name in ("input_bf16_hf_path", "input_vllm_ac_json_path", "output_fp8_hf_path")
        if not getattr(args, name, "")
    ]
    if missing:
        parser.error(
            "the following arguments are required (via CLI or YAML config): "
            + ", ".join("--" + n for n in missing)
        )

    print(args)
    with open(os.path.join(args.input_bf16_hf_path, "config.json"), "r", encoding="utf8") as fp:
        json_data = json.load(fp)
        print(json_data)
    if "quantization_config" in json_data.keys():
        raise AssertionError("NOT SUPPORT FP8")
    with open(
        os.path.join(args.input_vllm_ac_json_path, "activation_stats.json"), "r", encoding="utf8"
    ) as fp:
        ac_json_data = json.load(fp)

    if os.path.isfile(os.path.join(args.input_vllm_ac_json_path, "moe_expert_stats.json")):
        with open(
            os.path.join(args.input_vllm_ac_json_path, "moe_expert_stats.json"),
            "r",
            encoding="utf8",
        ) as fp:
            moe_expert_stats = json.load(fp)
        ac_json_data.update(moe_expert_stats)
    else:
        print("no moe_expert_stats")
        raise AssertionError("moe_expert_stats file is required")
        # print(ac_json_data["model.layers.0.mlp.gate_up_proj"])
    # ac_json_data = merge_vllm_act_moe_jsonl(ac_json_data)
    print(ac_json_data["model.layers.11.mlp.experts.1.gate_up_proj"])
    print(ac_json_data["model.layers.11.mlp.experts.22.gate_up_proj"])
    print(ac_json_data["model.layers.11.mlp.experts.53.gate_up_proj"])
    if "quantization_config" in json_data.keys():
        print("NOT SUPPORT FP8 DS")
        raise AssertionError("FP8 DS quantization_config is not supported")

    main(args.input_bf16_hf_path, args.output_fp8_hf_path, args.block_size, ac_json_data)
    print(args.output_fp8_hf_path)
