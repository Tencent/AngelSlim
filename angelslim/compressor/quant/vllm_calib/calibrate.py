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

"""vLLM activation calibration: Linear / KV / MoE min/max, optional KV scale search, DP.
Text models read an angelslim TextDataset, VLMs rollout rows; ``--kv-only`` hooks attention only.
CLI: ``python -m angelslim.compressor.quant.vllm_calib.calibrate -c <yaml> [flags]``."""

import argparse
import json
import os
import re
from functools import partial

from vllm import LLM, SamplingParams

from ....utils.vllm_calibration_dp import (
    run_vllm_calibration_with_dp,
    validate_vllm_calibration_dp_args,
)
from . import image_embeds, worker_bootstrap
from .hooks import (
    get_activation_stats,
    get_moe_channel_stats,
    get_moe_stats,
    is_fp8_scale_only,
    setup_activation_hooks,
    setup_mtp_activation_hooks,
)
from .kv_search import (
    KVMSEProfileCollector,
    KVPerHeadMSEProfileCollector,
    KVScaleSearcher,
    KVScaleSearcherPerHead,
    get_kv_mse_profile_results,
    get_kv_mse_profile_results_perhead,
    get_kv_scale_search_results,
    get_kv_scale_search_results_perhead,
    remove_kv_scale_search_hooks,
    remove_kvcache_perhead_value_hooks,
    setup_kvcache_perhead_value_hooks,
    setup_kvcache_value_hooks,
)
from .kvcache import (
    get_kvcache_only_stats,
    get_kvcache_perhead_stats,
    remove_kvcache_perhead_hooks,
    setup_kvcache_only_hooks,
    setup_kvcache_perhead_hooks,
)
from .llm import (
    apply_on_draft_model,
    attention_backend,
    hf_config,
    parse_with_yaml,
    patch_hpc_version,
    remap_mtp_stats_to_checkpoint,
    store_true,
)

DEFAULTS = {
    "tp_size": 1,
    "gpu_memory_utilization": 0.75,
    "batch_size": 128,
    "num_samples": 512,
    "max_length": 16384,
    "vision_budget": 16384,
    "max_images_per_prompt": 4,
    "max_num_batched_tokens": 16384,
    "distributed_executor_backend": "ray",
    "placement_strategy": "PACK",
    "dp_size": 1,
    "dp_timeout": 7200,
    "num_speculative_tokens": 1,
    "kv_granularity": "per-tensor",
    "search_kv_num_samples": 64,
    "search_kv_min_multiplier": 0.8,
    "search_kv_max_multiplier": 16.0,
    "skip_weight_loading": False,
    "enable_mtp": False,
    "auto_detect_mtp": False,
    "kv_only": False,
    "per_head": False,
    "search_kv_scale": False,
}
FP8_MAX = 448.0
SAMPLING = dict(temperature=0.8, top_p=0.95)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-c", "--config", help="flat YAML whose keys are option names")
    p.add_argument("--model-path")
    p.add_argument("--ptq-data-path", help="TextDataset JSONL, or VLM rollout rows")
    p.add_argument("--output-dir")
    p.add_argument("--image-dir", help="VLM images (default: <data dir>/images)")
    p.add_argument("--image-embeds-dir", help="offline ViT cache; skips the vision tower")
    p.add_argument("--tp-size", type=int)
    p.add_argument("--gpu-memory-utilization", type=float)
    p.add_argument("--attention-backend", help="override the backend derived from config.json")
    store_true(p, "--skip-weight-loading", help="dummy weights")
    p.add_argument("--batch-size", type=int, help="max_num_seqs")
    p.add_argument("--num-samples", type=int, help="0 = all rows")
    p.add_argument("--max-length", type=int)
    p.add_argument("--vision-budget", type=int, help="VLM max_model_len headroom")
    p.add_argument("--max-images-per-prompt", type=int)
    p.add_argument("--max-num-batched-tokens", type=int)
    p.add_argument("--distributed-executor-backend", choices=["ray", "mp"])
    p.add_argument("--ray-address", help="DP launcher Ray address (default: RAY_ADDRESS/auto)")
    p.add_argument(
        "--placement-strategy", choices=["STRICT_PACK", "PACK", "SPREAD", "STRICT_SPREAD"]
    )
    p.add_argument("--dp-size", type=int, help="calibration data-parallel replicas")
    p.add_argument("--dp-timeout", type=int)
    store_true(p, "--enable-mtp", help="also calibrate the MTP draft block")
    store_true(p, "--auto-detect-mtp", help="enable MTP when the checkpoint has MTP layers")
    p.add_argument("--num-speculative-tokens", type=int)
    p.add_argument(
        "--mtp-calib-max-tokens",
        type=int,
        help="decode length with MTP; the drafter only runs after the first token",
    )
    p.add_argument("--kv-granularity", choices=["none", "per-tensor", "per-head"])
    store_true(p, "--kv-only", help="only KV-cache hooks: no Linear / MoE statistics")
    store_true(p, "--per-head", help="same as --kv-granularity per-head")
    store_true(p, "--search-kv-scale", help="grid-search the FP8 KV scale multiplier")
    p.add_argument("--search-kv-num-samples", type=int)
    p.add_argument("--search-kv-min-multiplier", type=float)
    p.add_argument("--search-kv-max-multiplier", type=float)
    p.add_argument("--search-kv-num-steps", type=int, help="default 50, or 100 with --kv-only")
    required = ("model_path", "ptq_data_path", "output_dir")
    args = parse_with_yaml(p, DEFAULTS, argv, required=required)
    if args.per_head:
        args.kv_granularity = "per-head"
    if args.auto_detect_mtp and not args.enable_mtp:
        args.enable_mtp = has_mtp_layers(args.model_path)
        print(f"[calibrate] auto-detect MTP: {'found' if args.enable_mtp else 'none'}")
    if args.search_kv_num_steps is None:
        args.search_kv_num_steps = 100 if args.kv_only else 50
    return validate_vllm_calibration_dp_args(p, args)


def has_mtp_layers(model_path):
    """Checkpoint keys beyond ``num_hidden_layers``, or under ``mtp.`` / ``mtp_layers.``."""
    cfg = _model_config(model_path)
    depth = int((cfg.get("text_config") or cfg).get("num_hidden_layers") or 0)
    index = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return False
    with open(index, encoding="utf-8") as f:
        keys = json.load(f)["weight_map"]
    for key in keys:
        m = re.match(r"^(?:language_model\.)?model\.layers\.(\d+)\.", key)
        if (m and int(m.group(1)) >= depth) or key.startswith("mtp.") or ".mtp_layers." in key:
            return True
    return False


def _model_config(model_path):
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def vlm_prompts(args):
    """VLM rollout rows -> ``(prompts, max_images)``; built before the engine needs it."""
    from transformers import AutoTokenizer

    from .llm import build_prompts, load_records

    cfg = hf_config(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    image_dir = args.image_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.ptq_data_path)), "images"
    )
    prompts, n_mm, max_images = build_prompts(
        load_records(args.ptq_data_path, args.num_samples), tokenizer, image_dir,
        int(cfg.image_token_id), args.max_length, args.max_images_per_prompt,
    )  # fmt: skip
    print(f"[calibrate] {len(prompts)} prompts, {n_mm} multimodal")
    return prompts, (max(1, max_images) if n_mm else 0)


def text_prompts(args, tokenizer):
    """The angelslim TextDataset, decoded back to prompt strings."""
    from types import SimpleNamespace

    from ....engine import Engine

    engine = Engine()
    engine.series = "LLM"
    engine.slim_model = SimpleNamespace(tokenizer=tokenizer, model=SimpleNamespace(device="cpu"))
    data = engine.prepare_data(
        data_path=args.ptq_data_path, max_length=args.max_length, num_samples=args.num_samples,
        shuffle=False, inference_settings=None, use_audio_in_video=False,
    )  # fmt: skip
    return [tokenizer.decode(d["input_ids"][0]) for d in data]


def build_llm(args, vlm, max_images):
    kwargs = dict(
        model=args.model_path, load_format="dummy" if args.skip_weight_loading else "auto",
        enforce_eager=True, enable_chunked_prefill=True, tensor_parallel_size=args.tp_size,
        distributed_executor_backend=args.distributed_executor_backend,
        enable_expert_parallel=False, max_num_seqs=args.batch_size,
        max_model_len=args.max_length + (args.vision_budget if vlm else 0) + 16,
    )  # fmt: skip
    if args.kv_only:
        kwargs.update(max_num_batched_tokens=65536, num_gpu_blocks_override=4096)
    else:
        kwargs.update(
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            attention_backend=attention_backend(args.model_path, args.attention_backend),
            # vllm_calib.hooks reads the expert inputs from TritonExperts.apply.
            moe_backend="triton",
        )
    if args.enable_mtp:
        kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    if max_images:
        kwargs.update(
            limit_mm_per_prompt={"image": max_images},
            mm_encoder_tp_mode="data",
            mm_encoder_attn_backend="TORCH_SDPA",
            **image_embeds.engine_kwargs(),
        )
    return LLM(**kwargs)


def shard_prompts(prompts, dp_rank, dp_size):
    if not prompts or len(prompts) < dp_size:
        raise ValueError(f"{len(prompts)} prompts cannot be sharded over dp_size={dp_size}")
    return prompts[dp_rank::dp_size]


def first(results):
    return next((r for r in results or [] if r), None)


def write(output_dir, payload):
    os.makedirs(output_dir, exist_ok=True)
    for name, data in payload.items():
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[calibrate] -> {os.path.join(output_dir, name)}")


def kv_scale_search(llm, prompts, stats, per_head, args):
    """Grid-search the FP8 KV scale multiplier; returns the multipliers and tuned scales."""
    if per_head:
        setup, remove = setup_kvcache_perhead_value_hooks, remove_kvcache_perhead_value_hooks
        searcher, results = KVScaleSearcherPerHead, get_kv_scale_search_results_perhead
    else:
        setup, remove = setup_kvcache_value_hooks, remove_kv_scale_search_hooks
        searcher, results = KVScaleSearcher, get_kv_scale_search_results
    llm.apply_model(setup)
    llm.generate(prompts, SamplingParams(max_tokens=1, **SAMPLING))
    multipliers = results(llm.apply_model(searcher(
        activation_stats=stats, min_multiplier=args.search_kv_min_multiplier,
        max_multiplier=args.search_kv_max_multiplier, num_steps=args.search_kv_num_steps,
    )))  # fmt: skip
    llm.apply_model(remove)

    tuned = {}
    for key, mult in multipliers.items():
        base = key.replace("attn.attn", "attn")
        if per_head:
            for h, m in enumerate(mult):
                amax = max(abs(stats[key]["min"][h]), abs(stats[key]["max"][h]))
                tuned[f"{base}.head_{h}.scale"] = (amax / FP8_MAX * 2.0 if amax else 1e-8) * m
        else:
            amax = max(abs(stats[key]["min"]), abs(stats[key]["max"]))
            tuned[f"{base}.scale"] = amax / FP8_MAX * 2.0 * mult
    suffix = "_per_head" if per_head else ""
    return {f"kv_scale_multipliers{suffix}.json": multipliers,
            f"kv_cache_tuned_scales{suffix}.json": tuned}  # fmt: skip


def kv_search_profiles(args, llm, prompts):
    """DP stage 2: this rank's KV-scale MSE profiles, merged by the DP launcher."""
    stats = getattr(args, "kv_search_activation_stats", None)
    if stats is None:
        with open(args.kv_search_activation_stats_path, encoding="utf-8") as f:
            stats = json.load(f)
    local = shard_prompts(prompts[: args.search_kv_num_samples], args.dp_rank, args.dp_size)
    per_head = args.kv_granularity == "per-head"
    if args.kv_granularity == "none":
        return {}
    if per_head:
        setup, remove = setup_kvcache_perhead_value_hooks, remove_kvcache_perhead_value_hooks
        collector, results = KVPerHeadMSEProfileCollector, get_kv_mse_profile_results_perhead
    else:
        setup, remove = setup_kvcache_value_hooks, remove_kv_scale_search_hooks
        collector, results = KVMSEProfileCollector, get_kv_mse_profile_results
    llm.apply_model(setup)
    llm.generate(local, SamplingParams(max_tokens=1, **SAMPLING))
    profiles = results(llm.apply_model(collector(
        activation_stats=stats, min_multiplier=args.search_kv_min_multiplier,
        max_multiplier=args.search_kv_max_multiplier, num_steps=args.search_kv_num_steps,
    )))  # fmt: skip
    llm.apply_model(remove)
    return {f"kv_scale_mse_profiles{'_per_head' if per_head else ''}.json": profiles}


def kv_only(args, llm, prompts):
    per_head = args.kv_granularity == "per-head"
    llm.apply_model(setup_kvcache_perhead_hooks if per_head else setup_kvcache_only_hooks)
    llm.generate(prompts, SamplingParams(max_tokens=1, **SAMPLING))
    stats = llm.apply_model(get_kvcache_perhead_stats if per_head else get_kvcache_only_stats)[0]
    payload = {"activation_stats_per_head.json" if per_head else "activation_stats.json": stats}
    if args.search_kv_scale:
        payload.update(kv_scale_search(llm, prompts[: args.search_kv_num_samples], stats,
                                       per_head, args))  # fmt: skip
    return payload


def calibrate(args, llm, prompts, vlm):
    rpc = llm.llm_engine.collective_rpc
    per_head = args.kv_granularity == "per-head"
    # A partial, not a lambda: the HPC fork pickles apply_model callables with stdlib pickle.
    print(llm.apply_model(partial(setup_activation_hooks, kv_granularity=args.kv_granularity))[0])
    if args.enable_mtp:
        print(rpc(lambda w: apply_on_draft_model(w, setup_mtp_activation_hooks))[0])
        if per_head:
            rpc(lambda w: apply_on_draft_model(w, setup_kvcache_perhead_hooks))

    local = shard_prompts(prompts, args.dp_rank, args.dp_size)
    # The drafter only runs in a decode step's propose phase, so MTP needs max_tokens > 1.
    max_tokens = 1
    if args.enable_mtp:
        default = max(8, 4 * args.num_speculative_tokens) if vlm else 1
        max_tokens = args.mtp_calib_max_tokens or default
        if max_tokens == 1:
            print("[calibrate] WARNING: max_tokens=1 never runs the MTP drafter; "
                  "set mtp_calib_max_tokens (e.g. 8) to collect MTP stats")  # fmt: skip
    llm.generate(local, SamplingParams(max_tokens=max_tokens, **SAMPLING))

    payload = {}
    if not any(llm.apply_model(is_fp8_scale_only)):  # FP8 checkpoints have no BF16 Linear/KV
        payload["activation_stats.json"] = llm.apply_model(get_activation_stats)[0]
    payload["moe_expert_stats.json"] = llm.apply_model(get_moe_stats)[0]
    if os.environ.get("VLLM_MOE_COLLECT_CHANNEL_STATS") == "1":
        payload["moe_down_proj_channel_stats.json"] = llm.apply_model(get_moe_channel_stats)[0]
    if args.enable_mtp:
        mtp_act = first(rpc(lambda w: apply_on_draft_model(w, get_activation_stats))) or {}
        mtp_moe = first(rpc(lambda w: apply_on_draft_model(w, get_moe_stats))) or {}
        if per_head:
            ph = first(rpc(lambda w: apply_on_draft_model(w, get_kvcache_perhead_stats)))
            mtp_act.update(ph or {})
            rpc(lambda w: apply_on_draft_model(w, remove_kvcache_perhead_hooks))
        if vlm:  # VLM stage 3 looks MTP stats up by checkpoint name (model.mtp_layers.i)
            depth = _model_config(args.model_path)["text_config"]["num_hidden_layers"]
            mtp_act = remap_mtp_stats_to_checkpoint(mtp_act, depth)
            mtp_moe = remap_mtp_stats_to_checkpoint(mtp_moe, depth)
        payload["mtp_activation_stats.json"] = mtp_act
        payload["mtp_moe_expert_stats.json"] = mtp_moe
    if per_head:
        payload.setdefault("activation_stats.json", {}).update(
            llm.apply_model(get_kvcache_perhead_stats)[0]
        )
        llm.apply_model(remove_kvcache_perhead_hooks)
    if args.search_kv_scale and args.kv_granularity != "none":
        stats = payload["activation_stats.json"]
        search = local[: args.search_kv_num_samples]
        payload.update(kv_scale_search(llm, search, stats, per_head, args))

    moe = payload["moe_expert_stats.json"] or {}
    expert_keys = [k for k, v in moe.items() if ".experts." in k]
    if expert_keys and all(moe[k]["min"] == moe[k]["max"] == 1.0 for k in expert_keys):
        raise RuntimeError("no MoE key calibrated: check moe_backend / VLLM_MOE_COLLECT_STATS")
    return payload


def run_one_calibration(args, llm=None, return_llm=False):
    """One calibration replica; also the DP launcher's worker (stage 1 and KV-search stage 2)."""
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # collective_rpc callables
    if args.image_embeds_dir:
        os.environ["VLLM_IMAGE_EMBEDS_DIR"] = args.image_embeds_dir
    vlm = "image_token_id" in _model_config(args.model_path)
    prompts, max_images = vlm_prompts(args) if vlm else (None, 0)
    if llm is None:
        llm = build_llm(args, vlm, max_images)
    llm.llm_engine.collective_rpc(worker_bootstrap())
    if prompts is None:
        prompts = text_prompts(args, llm.get_tokenizer())

    if getattr(args, "kv_search_only", False):
        payload = kv_search_profiles(args, llm, prompts)
    elif args.kv_only:
        payload = kv_only(args, llm, prompts)
    else:
        payload = calibrate(args, llm, prompts, vlm)
    write(args.output_dir, {k: v for k, v in payload.items() if v})
    return (payload, llm) if return_llm else payload


def main(argv=None):
    # moe_stats reads these per worker; keep any value the caller already exported.
    os.environ.setdefault("VLLM_MOE_COLLECT_STATS", "1")
    os.environ.setdefault("VLLM_MOE_COLLECT_PER_EXPERT_STATS", "1")
    patch_hpc_version()
    run_vllm_calibration_with_dp(parse_args(argv), run_one_calibration)


if __name__ == "__main__":
    main()
