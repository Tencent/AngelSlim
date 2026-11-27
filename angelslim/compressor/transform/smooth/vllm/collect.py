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

"""Smooth stat collection (phase 1): per-channel absmax / EMA of attention q / k / attn_out,
dense down_proj and FusedMoE down_proj inputs, plus optional alpha grid search.
CLI: ``python -m angelslim.compressor.transform.smooth.vllm.collect -c <yaml>``."""

import argparse
import gc
import json
import os
import random
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from vllm import LLM, SamplingParams

from .....utils.yaml_args import apply_yaml_config
from ....quant.vllm_calib import worker_bootstrap
from ....quant.vllm_calib.llm import apply_on_draft_model, patch_hpc_version
from ..config import SmoothAlphaSearchConfig
from . import (
    SmoothAlphaSearcher,
    get_smooth_stats,
    print_smooth_stats,
    remove_smooth_alpha_search_hooks,
    setup_smooth_alpha_search_hooks,
    setup_smooth_hooks,
)
from .hooks import remove_smooth_hooks

patch_hpc_version()
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # collective_rpc lambdas


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="vLLM SmoothQuant calibration: per-channel absmax/EMA statistics"
    )
    parser.add_argument("--model-name", help="Model adapter for offline Smooth")
    parser.add_argument("--profile", default=None, help=argparse.SUPPRESS)

    # Model configuration
    parser.add_argument("--model-path", type=str, help="Path to the model directory")
    parser.add_argument(
        "--hf-overrides", type=json.loads, help="Runtime HF config overrides (JSON)"
    )
    parser.add_argument(
        "--ptq-data-path",
        type=str,
        help="Path to the PTQ calibration data (JSONL format)",
    )
    parser.add_argument("--output-dir", type=str, help="Directory to save output JSON")

    # Model loading
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size (default: 1)")
    parser.add_argument(
        "--skip-weight-loading",
        action="store_true",
        help="Use dummy weights for fast debug (outputs will be random)",
    )

    # Dataset
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for inference (default: 128)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="Number of calibration samples (default: 512)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=16384,
        help="Maximum sequence length (default: 16384)",
    )

    # Distributed
    parser.add_argument(
        "--distributed-executor-backend",
        type=str,
        default="ray",
        choices=["ray", "mp"],
        help="Distributed executor backend (default: ray)",
    )

    # MTP (Multi-Token Prediction) configuration
    parser.add_argument(
        "--enable-mtp",
        action="store_true",
        help="Enable MTP (Multi-Token Prediction) speculative decoding and register "
        "smooth hooks on the MTP draft model",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=1,
        help="Number of speculative tokens for MTP "
        "(default: 1, only used when --enable-mtp is set)",
    )

    # EMA
    parser.add_argument(
        "--ema-momentum",
        type=float,
        default=0.9,
        help="EMA momentum for smooth stats (default: 0.9)",
    )

    # --- Scope control ---
    parser.add_argument(
        "--collect-attn",
        action="store_true",
        default=None,
        help="Collect attention q/k/attn_out smooth stats "
        "(default: on unless any --collect-* flag is given)",
    )
    parser.add_argument(
        "--collect-down-proj",
        action="store_true",
        default=None,
        help="Collect dense MLP down_proj input smooth stats",
    )
    parser.add_argument(
        "--collect-moe",
        action="store_true",
        default=None,
        help=(
            "Collect FusedMoE down_proj input smooth stats. "
            "Requires VLLM_MOE_COLLECT_SMOOTH_STATS=1 and vLLM kernel modification."
        ),
    )

    # Output
    parser.add_argument(
        "--output-filename",
        type=str,
        default="smooth_stats.json",
        help="Output JSON filename (default: smooth_stats.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed statistics after collection",
    )
    parser.add_argument(
        "--print-max-rows",
        type=int,
        default=40,
        help="Max rows to print in summary table (default: 40)",
    )

    # --- Alpha Grid Search ---
    parser.add_argument(
        "--enable-alpha-search",
        action="store_true",
        help="Enable per-layer alpha grid search for optimal SmoothQuant alpha (down_proj only)",
    )
    parser.add_argument(
        "--num-search-samples",
        type=int,
        default=16,
        help="Number of samples for alpha search (separate from smooth stat samples, default: 16)",
    )
    parser.add_argument(
        "--search-max-length",
        type=int,
        default=None,
        help="Max token length for alpha search data (default: same as --max-length). "
        "Set lower (e.g. 8192) to speed up search inference.",
    )
    parser.add_argument(
        "--alpha-min", type=float, default=0.3, help="Min alpha for grid search (default: 0.3)"
    )
    parser.add_argument(
        "--alpha-max", type=float, default=1.0, help="Max alpha for grid search (default: 1.0)"
    )
    parser.add_argument(
        "--alpha-steps", type=int, default=8, help="Number of grid search points (default: 8)"
    )
    parser.add_argument(
        "--alpha-act-quant-method",
        type=str,
        default="per_token",
        choices=["per_tensor", "per_token"],
        help="Activation quantization granularity for alpha search (default: per_token)",
    )
    parser.add_argument(
        "--alpha-act-quant-type",
        type=str,
        default="int8",
        choices=["int8", "fp8"],
        help="Activation quantization dtype for alpha search (default: int8)",
    )
    parser.add_argument(
        "--alpha-weight-quant-method",
        type=str,
        default="per_channel",
        choices=["per_tensor", "per_channel", "per_group", "per_block"],
        help="Weight quantization granularity for alpha search (default: per_channel)",
    )
    parser.add_argument(
        "--alpha-weight-quant-type",
        type=str,
        default="int8",
        choices=["int8", "int4", "fp8"],
        help="Weight quantization dtype for alpha search (default: int8)",
    )
    parser.add_argument(
        "--alpha-weight-quant-bits",
        type=int,
        default=8,
        help="Weight quant bit width (default: 8)",
    )
    parser.add_argument(
        "--alpha-weight-group-size",
        type=int,
        default=128,
        help="Group size for per_group weight quant (default: 128)",
    )
    parser.add_argument(
        "--alpha-max-tokens",
        type=int,
        default=4096,
        help="Max tokens per layer for alpha search (default: 4096)",
    )
    parser.add_argument(
        "--alpha-use-ema",
        action="store_true",
        help="Use EMA instead of absmax for smooth formula in alpha search",
    )
    parser.add_argument(
        "--alpha-smooth-search-mode",
        type=str,
        default="default",
        choices=["default", "per-tensor-act-first"],
        help="Alpha search strategy: 'default' (act^a/weight^(1-a)) or "
        "'per-tensor-act-first' (scale all channels to unified target absmax) (default: default)",
    )
    parser.add_argument(
        "--alpha-act-mul-min",
        type=float,
        default=0.1,
        help="per-tensor-act-first mode: min multiplier for target absmax (default: 0.1)",
    )
    parser.add_argument(
        "--alpha-act-mul-max",
        type=float,
        default=1.0,
        help="per-tensor-act-first mode: max multiplier for target absmax (default: 1.0)",
    )
    parser.add_argument(
        "--alpha-smooth-min",
        type=float,
        default=1e-6,
        help="per-tensor-act-first mode: smooth weight clamp lower bound (default: 1e-6)",
    )
    parser.add_argument(
        "--alpha-smooth-max",
        type=float,
        default=1e6,
        help="per-tensor-act-first mode: smooth weight clamp upper bound (default: 1e6)",
    )

    # YAML config support
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="YAML config file path. Values override argparse defaults; "
        "explicit CLI flags still take final precedence.",
    )

    args = parser.parse_args(argv)
    raw = {}
    if args.config:
        import yaml

        with open(args.config, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError("Top level of Smooth YAML must be a mapping")
    runtime = ((raw.get("compression") or {}).get("calibrate") or {}).get("args") or {}
    from ....quant.ckpt_convert import smooth_options
    from ..struct import resolve_model_name

    model_name = args.model_name or (raw.get("model") or {}).get("name") or raw.get("model_name")
    options = smooth_options(
        (raw.get("compression") or {}).get("quantization") or {}, runtime.get("smooth"), model_name
    )
    profile = (
        args.profile
        or (options or runtime.get("smooth") or {}).get("profile")
        or raw.get("profile")
    )
    if args.profile != "default" and (
        options or args.model_name or raw.get("model_name") or profile not in (None, "default")
    ):
        model_name = resolve_model_name(model_name, profile=profile)
        explicit = {item.split("=", 1)[0] for item in (sys.argv[1:] if argv is None else argv)}
        aliases = {
            "ptq_data_path": "data_path",
            "batch_size": "max_num_seqs",
            "output_dir": "run_root",
        }
        overrides = {
            aliases.get(action.dest, action.dest): getattr(args, action.dest)
            for action in parser._actions
            if explicit.intersection(action.option_strings)
            and action.dest not in ("config", "profile")
        }
        overrides["model_name"] = model_name
        args.offline_config = normalize_offline_config(raw, **overrides)
    else:
        apply_yaml_config(parser, args)
        if args.hf_overrides is not None and not isinstance(args.hf_overrides, dict):
            parser.error("hf_overrides must be a mapping")

    return args


# ---------------------------------------------------------------------------
# Scope-filtered setup / get helpers
# ---------------------------------------------------------------------------


class _ScopedSmoothHooks:
    """
    Pickle-safe callable that passes scope flags directly to setup_smooth_hooks.
    Required because llm.apply_model serialises the function via ray/mp.
    """

    def __init__(self, collect_attn, collect_down_proj, collect_moe, ema_momentum):
        self.collect_attn = collect_attn
        self.collect_down_proj = collect_down_proj
        self.collect_moe = collect_moe
        self.ema_momentum = ema_momentum

    def __call__(self, model):
        return setup_smooth_hooks(
            model,
            ema_momentum=self.ema_momentum,
            collect_attn=self.collect_attn,
            collect_down_proj=self.collect_down_proj,
            collect_moe=self.collect_moe,
        )


class _ScopedAlphaSearchHooks:
    """Pickle-safe callable for setup_smooth_alpha_search_hooks."""

    def __init__(self, max_tokens, collect_moe):
        self.max_tokens = max_tokens
        self.collect_moe = collect_moe

    def __call__(self, model):
        return setup_smooth_alpha_search_hooks(
            model,
            max_tokens=self.max_tokens,
            collect_moe=self.collect_moe,
        )


def _resolve_scopes(args):
    """
    If none of the --collect-* flags are given, enable all three scopes.
    If at least one is given, only enable the explicitly requested ones.
    """
    any_explicit = any([args.collect_attn, args.collect_down_proj, args.collect_moe])
    collect_attn = bool(args.collect_attn) if any_explicit else True
    collect_down_proj = bool(args.collect_down_proj) if any_explicit else True
    collect_moe = bool(args.collect_moe) if any_explicit else True
    return collect_attn, collect_down_proj, collect_moe


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def save_stats_to_json(stats_data, output_dir, filename, stats_type="statistics"):
    if isinstance(stats_data, list):
        if not stats_data or stats_data[0] is None:
            print(f"\nNo {stats_type} available.")
            return
        stats_data = stats_data[0]

    if stats_data is None:
        print(f"\nNo {stats_type} available.")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, filename)
    with open(output_file, "w") as f:
        json.dump(stats_data, f, indent=2)
    print(f"\n{stats_type.capitalize()} saved to: {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)
    if hasattr(args, "offline_config"):
        collect_offline_stats(args.offline_config)
        return search_offline_plan(args.offline_config)

    collect_attn, collect_down_proj, collect_moe = _resolve_scopes(args)

    # ------------------------------------------------------------------
    # Print configuration
    # ------------------------------------------------------------------
    print("\nConfiguration:")
    print(f"  Model:            {args.model_path}")
    print(f"  PTQ Data:         {args.ptq_data_path}")
    print(f"  Output Dir:       {args.output_dir}")
    print(f"  TP Size:          {args.tp_size}")
    print(f"  Batch Size:       {args.batch_size}")
    print(f"  Num Samples:      {args.num_samples}")
    print(f"  EMA Momentum:     {args.ema_momentum}")
    print(f"  Skip Weight Load: {args.skip_weight_loading}")
    print("\nCollection scope:")
    print(f"  Attention (q/k/attn_out): {'ON' if collect_attn else 'OFF'}")
    print(f"  Dense down_proj input:    {'ON' if collect_down_proj else 'OFF'}")
    print(f"  FusedMoE down_proj input: {'ON' if collect_moe else 'OFF'}")
    if collect_moe:
        moe_env = os.environ["VLLM_MOE_COLLECT_SMOOTH_STATS"] = "1"
        print(f"  VLLM_MOE_COLLECT_SMOOTH_STATS={moe_env}")
        if moe_env != "1":
            print(
                "  [WARNING] VLLM_MOE_COLLECT_SMOOTH_STATS is not set to 1. "
                "MoE smooth stats will NOT be collected."
            )

    # Configure MTP speculative decoding
    speculative_config = None
    if args.enable_mtp:
        speculative_config = {
            "method": "hunyuan_mtp",
            "num_speculative_tokens": args.num_speculative_tokens,
        }
        print(f"  MTP Enabled:      True (num_speculative_tokens={args.num_speculative_tokens})")
    else:
        print("  MTP Enabled:      False")

    if args.enable_alpha_search:
        print("\nAlpha Grid Search:")
        print(f"  Num Search Samples: {args.num_search_samples}")
        print(f"  Search Max Length:  {args.search_max_length or args.max_length}")
        print(
            f"  Alpha Range:        [{args.alpha_min}, {args.alpha_max}], steps={args.alpha_steps}"
        )
        print(f"  Act Quant:          {args.alpha_act_quant_method} / {args.alpha_act_quant_type}")
        print(
            f"  Weight Quant:       {args.alpha_weight_quant_method} / "
            f"{args.alpha_weight_quant_type} / {args.alpha_weight_quant_bits}bit"
        )
        print(f"  Max Tokens/Layer:   {args.alpha_max_tokens}")
        print(f"  Use EMA:            {args.alpha_use_ema}")

    # ------------------------------------------------------------------
    # Create LLM instance
    # ------------------------------------------------------------------
    llm = LLM(
        model=args.model_path,
        load_format="dummy" if args.skip_weight_loading else "auto",
        disable_log_stats=False,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.75,
        tensor_parallel_size=args.tp_size,
        distributed_executor_backend=args.distributed_executor_backend,
        enable_expert_parallel=False,
        max_num_seqs=args.batch_size,
        max_model_len=args.max_length + 16,
        speculative_config=speculative_config,
        # vllm_calib.hooks observes TritonExperts; other MoE backends bypass it.
        moe_backend="triton",
        **({"hf_overrides": args.hf_overrides} if args.hf_overrides else {}),
    )
    llm.llm_engine.collective_rpc(worker_bootstrap())

    if args.skip_weight_loading:
        print("\n" + "!" * 80)
        print("WARNING: Running with dummy weights (random values)!")
        print("Outputs will NOT make sense. This is for debugging only.")
        print("!" * 80 + "\n")

    # ------------------------------------------------------------------
    # Setup smooth hooks (main model)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Setting up smooth hooks...")
    print("=" * 80)
    setup_fn = _ScopedSmoothHooks(
        collect_attn=collect_attn,
        collect_down_proj=collect_down_proj,
        collect_moe=collect_moe,
        ema_momentum=args.ema_momentum,
    )
    hook_results = llm.apply_model(setup_fn)
    for i, result in enumerate(hook_results):
        print(f"Worker {i}: {result}")

    # ------------------------------------------------------------------
    # Setup smooth hooks (MTP draft model, if --enable-mtp is set)
    # ------------------------------------------------------------------
    if args.enable_mtp:
        print("\n" + "=" * 80)
        print("Setting up MTP draft model smooth hooks...")
        print("=" * 80)
        mtp_hook_results = llm.llm_engine.collective_rpc(
            lambda w: apply_on_draft_model(w, setup_fn)
        )
        for i, result in enumerate(mtp_hook_results):
            if result is not None:
                print(f"Worker {i}: {result}")
            else:
                print(f"Worker {i}: No MTP draft model available")

    # ------------------------------------------------------------------
    # Load dataset and prepare prompts (split into smooth + search groups)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Loading dataset and preparing prompts...")
    print("=" * 80)
    tokenizer = llm.get_tokenizer()

    num_search = args.num_search_samples if args.enable_alpha_search else 0
    total_samples = args.num_samples + num_search

    from .....engine import Engine

    slim_engine = Engine()
    slim_engine.slim_model = llm
    slim_engine.series = "LLM"
    slim_engine.slim_model.tokenizer = tokenizer
    slim_engine.slim_model.model = llm
    slim_engine.slim_model.model.device = "cpu"
    dataset = slim_engine.prepare_data(
        data_path=args.ptq_data_path,
        max_length=args.max_length,
        num_samples=total_samples,
        shuffle=False,
        inference_settings=None,
        use_audio_in_video=False,
    )

    # Materialise DataLoader into a list so we can slice it
    all_data = list(dataset)
    print(f"Total dataset samples loaded: {len(all_data)}")

    # Split: first num_samples for smooth stat, next num_search for alpha search
    smooth_dataset = all_data[: args.num_samples]
    smooth_prompts = [tokenizer.decode(data["input_ids"][0]) for data in smooth_dataset]
    print(f"Smooth prompts: {len(smooth_prompts)}")

    search_prompts = []
    if args.enable_alpha_search and num_search > 0:
        search_dataset = all_data[args.num_samples : args.num_samples + num_search]
        search_max_len = args.search_max_length or args.max_length
        for data in search_dataset:
            ids = data["input_ids"][0]
            if search_max_len < args.max_length:
                ids = ids[:search_max_len]
            search_prompts.append(tokenizer.decode(ids))
        print(f"Search prompts: {len(search_prompts)} (max_length={search_max_len})")

    # ------------------------------------------------------------------
    # Phase B: Forward pass — collect smooth stats (prefill only)
    # ------------------------------------------------------------------
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1)

    print("\n" + "=" * 80)
    print(f"[Phase B] Running smooth stat calibration ({len(smooth_prompts)} samples)...")
    print("=" * 80)
    outputs = llm.generate(smooth_prompts, sampling_params)
    print(f"Smooth stat sequences processed: {len(outputs)}")

    # ------------------------------------------------------------------
    # Phase C+D: Alpha grid search (if enabled)
    # ------------------------------------------------------------------
    if args.enable_alpha_search and search_prompts:
        print("\n" + "=" * 80)
        print(
            f"[Phase C] Setting up alpha search hooks + running search inference "
            f"({len(search_prompts)} samples)..."
        )
        print("=" * 80)

        alpha_setup_fn = _ScopedAlphaSearchHooks(
            max_tokens=args.alpha_max_tokens,
            collect_moe=collect_moe,
        )
        alpha_hook_results = llm.apply_model(alpha_setup_fn)
        for i, result in enumerate(alpha_hook_results):
            print(f"  Worker {i}: {result}")

        # ------ Phase C generate ------
        search_outputs = llm.generate(search_prompts, sampling_params)
        print(f"  Search sequences processed: {len(search_outputs)}")

        print("\n" + "=" * 80)
        print("[Phase D] Running alpha grid search...")
        print("=" * 80)

        # ------ Phase D search ------
        # Note: args.alpha_max_tokens is the per-layer raw-tensor cap used
        # at hook-registration time (see L620 above for setup_smooth_alpha_
        # search_hooks); it does NOT belong on SmoothAlphaSearchConfig — the
        # searcher consumes whatever tokens were already captured by the hooks.
        alpha_config = SmoothAlphaSearchConfig(
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            alpha_steps=args.alpha_steps,
            act_quant_method=args.alpha_act_quant_method,
            act_quant_type=args.alpha_act_quant_type,
            weight_quant_method=args.alpha_weight_quant_method,
            weight_quant_type=args.alpha_weight_quant_type,
            weight_quant_bits=args.alpha_weight_quant_bits,
            weight_group_size=args.alpha_weight_group_size,
            use_ema_for_absmax=args.alpha_use_ema,
            smooth_search_mode=args.alpha_smooth_search_mode,
            act_mul_min=args.alpha_act_mul_min,
            act_mul_max=args.alpha_act_mul_max,
            smooth_min=args.alpha_smooth_min,
            smooth_max=args.alpha_smooth_max,
        )
        alpha_output_path = os.path.join(args.output_dir, "smooth_alpha_search.json")
        searcher = SmoothAlphaSearcher(alpha_config, output_path=alpha_output_path)
        alpha_results_list = llm.apply_model(searcher)
        for i, summary in enumerate(alpha_results_list):
            print(f"  Worker {i}: {summary}")
        print(f"\nAlpha search results saved to: {alpha_output_path}")

        # Cleanup alpha search hooks
        llm.apply_model(remove_smooth_alpha_search_hooks)

    # ------------------------------------------------------------------
    # Phase E: Collect and save smooth statistics (all_gather, slow)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Phase E] Collecting smooth statistics (all_gather)...")
    print("=" * 80)

    if args.verbose:
        llm.apply_model(lambda model: print_smooth_stats(model, max_rows=args.print_max_rows))

    if args.enable_mtp and args.verbose:
        print("\n[MTP] Draft Model Smooth Statistics:")
        llm.llm_engine.collective_rpc(
            lambda w: apply_on_draft_model(
                w, lambda m: print_smooth_stats(m, max_rows=args.print_max_rows)
            )
        )

    # ------------------------------------------------------------------
    # Save statistics (main model)
    # ------------------------------------------------------------------
    smooth_stats_list = llm.apply_model(get_smooth_stats)
    save_stats_to_json(
        smooth_stats_list,
        args.output_dir,
        args.output_filename,
        stats_type="smooth statistics",
    )

    # ------------------------------------------------------------------
    # Save statistics (MTP draft model, if --enable-mtp is set)
    # ------------------------------------------------------------------
    if args.enable_mtp:
        mtp_smooth_stats_list = llm.llm_engine.collective_rpc(
            lambda w: apply_on_draft_model(w, get_smooth_stats)
        )
        # Derive MTP output filename: e.g. smooth_stats.json -> mtp_smooth_stats.json
        base, ext = os.path.splitext(args.output_filename)
        mtp_output_filename = f"mtp_{base}{ext}"
        save_stats_to_json(
            mtp_smooth_stats_list,
            args.output_dir,
            mtp_output_filename,
            stats_type="MTP smooth statistics",
        )

    print("\n" + "=" * 80)
    print("Smooth stat calibration completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


def normalize_offline_config(raw, model_path=None, output_dir=None, **overrides):
    """Resolve offline Smooth options; explicit CLI flags override YAML."""
    from ....quant.ckpt_convert import smooth_options
    from ....quant.vllm_calib.llm import flatten_angelslim_yaml
    from ..struct import resolve_model_name

    config = dict(
        num_samples=128,
        seed=42,
        sample_offset=0,
        max_length=16384,
        sample_tokens=128,
        alphas=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        smooth_min=1 / 12,
        smooth_max=2,
        minimum_gain=0.001,
        device="cuda:0",
        tp_size=32,
        max_num_seqs=4,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.70,
        vision_budget=8448,
        max_images_per_prompt=4,
        image_dir=None,
        image_embeds_dir=None,
        attention_backend=None,
    )
    if "compression" in raw:
        flat = flatten_angelslim_yaml(raw)
        # Calibration may use a no-draft card; search uses the full checkpoint.
        runtime = ((raw.get("compression") or {}).get("calibrate") or {}).get("args") or {}
        config["calibration_model_path"] = runtime.get("model_path")
        aliases = {"ptq_data_path": "data_path", "batch_size": "max_num_seqs"}
        for key in (
            "ptq_data_path",
            "max_length",
            "tp_size",
            "batch_size",
            "distributed_executor_backend",
            "image_dir",
            "image_embeds_dir",
            "vision_budget",
            "max_images_per_prompt",
            "max_num_batched_tokens",
            "gpu_memory_utilization",
            "attention_backend",
            "max_model_len",
            "hf_overrides",
        ):
            if key in flat:
                config[aliases.get(key, key)] = flat[key]
        config["model_path"] = (raw.get("model") or {}).get("model_path")
        config["model_name"] = (raw.get("model") or {}).get("name")
        config["run_root"] = (raw.get("global") or {}).get("save_path")
        # Smooth sample count is independent of dataset.num_samples.
        quant = (raw.get("compression") or {}).get("quantization") or {}
        config.update(
            smooth_options(
                quant, flat.get("smooth"), overrides.get("model_name") or config["model_name"]
            )
            or flat.get("smooth")
            or {}
        )
    else:
        config.update(raw)
        for key, alias in (
            ("ptq_data_path", "data_path"),
            ("batch_size", "max_num_seqs"),
            ("output_dir", "run_root"),
        ):
            if key in raw and alias not in raw:
                config[alias] = raw[key]
    if model_path is not None:
        config["model_path"] = model_path
    if output_dir is not None:
        config["run_root"] = output_dir
    config.update(overrides)
    if config.get("hf_overrides") is not None and not isinstance(config["hf_overrides"], dict):
        raise ValueError("hf_overrides must be a mapping")
    config["model_name"] = resolve_model_name(
        config.get("model_name"), profile=config.pop("profile", None)
    )
    for key in ("model_path", "data_path", "run_root"):
        if not config.get(key):
            raise ValueError(f"Offline Smooth requires {key}")
        config[key] = str(Path(config[key]).resolve())
    config["calibration_model_path"] = config.get("calibration_model_path") or config["model_path"]
    config.setdefault("distributed_executor_backend", "ray" if config["tp_size"] > 8 else "mp")
    config["image_dir"] = config["image_dir"] or str(Path(config["data_path"]).parent / "images")
    config["image_embeds_dir"] = config["image_embeds_dir"] or os.getenv("VLLM_IMAGE_EMBEDS_DIR")
    if config["image_embeds_dir"]:
        config["image_embeds_dir"] = str(Path(config["image_embeds_dir"]).resolve())
    for key in (
        "num_samples",
        "sample_tokens",
        "max_length",
        "tp_size",
        "max_num_seqs",
        "max_num_batched_tokens",
    ):
        if config[key] <= 0:
            raise ValueError(f"Offline Smooth {key} must be positive")
    if config["sample_offset"] < 0:
        raise ValueError("Offline Smooth sample_offset must be nonnegative")
    if config.get("skip_weight_loading") or config.get("enable_mtp"):
        raise ValueError("Offline Smooth requires real trunk weights and no speculative decoding")
    return config


def collect_offline_stats(config):
    """Collect seeded Smooth samples in a separate BF16 engine."""
    from transformers import AutoTokenizer

    from ....quant.vllm_calib import image_embeds
    from ....quant.vllm_calib.llm import (
        attention_backend,
        build_prompts,
        hf_config,
        load_records,
    )
    from ..core.stats_search import (
        TensorReader,
        input_name,
        offline_group_specs,
        save_smooth_stats,
        smooth_selection,
    )

    reader = TensorReader(config["model_path"], config["model_name"])
    specs = offline_group_specs(reader, **smooth_selection(config))
    expected = {input_name(key) for spec in specs for key in spec["targets"]}
    expected.update(input_name(spec["producer"]) for spec in specs if "producer" in spec)
    root = Path(config["run_root"]) / "smooth"
    root.mkdir(parents=True, exist_ok=True)
    if not specs:
        from safetensors.torch import save_file

        print("[smooth] no supported targets; skipping collection", flush=True)
        save_smooth_stats(dict(stats={}), root / "stats.json")
        save_file({}, str(root / "samples.safetensors"))
        return str(root / "stats.json")
    records = load_records(config["data_path"], 0)
    random.Random(config["seed"]).shuffle(records)
    offset, count = config["sample_offset"], config["num_samples"]
    if count <= 0 or offset < 0 or offset + count > len(records):
        raise ValueError(f"requested {offset}+{count} from {len(records)} usable records")
    records = records[offset : offset + count]
    metadata = hf_config(config["model_path"], config.get("hf_overrides"))
    image_pad_id = getattr(metadata, "image_token_id", None)
    vlm = image_pad_id is not None or getattr(metadata, "vision_config", None) is not None
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
    prompts, n_mm, max_images = build_prompts(
        records,
        tokenizer,
        config["image_dir"],
        image_pad_id,
        config["max_length"],
        config["max_images_per_prompt"],
    )
    if len(prompts) != count:
        raise RuntimeError(f"Smooth requires all {count} selected records; built {len(prompts)}")
    print(f"[smooth] {count} prompts, {n_mm} multimodal, seed={config['seed']}", flush=True)
    previous_cache = os.environ.get("VLLM_IMAGE_EMBEDS_DIR")
    llm = None
    hooks_started = False
    try:
        if config.get("image_embeds_dir"):
            os.environ["VLLM_IMAGE_EMBEDS_DIR"] = config["image_embeds_dir"]
        kwargs = dict(
            model=config["calibration_model_path"],
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=config["tp_size"],
            distributed_executor_backend=config["distributed_executor_backend"],
            enforce_eager=True,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            async_scheduling=False,
            enable_expert_parallel=False,
            moe_backend="triton",
            max_num_batched_tokens=config["max_num_batched_tokens"],
            max_num_seqs=config["max_num_seqs"],
            max_model_len=config.get("max_model_len")
            or (config["max_length"] + (config["vision_budget"] if vlm else 0) + 16),
            gpu_memory_utilization=config["gpu_memory_utilization"],
            attention_backend=attention_backend(
                config["model_path"], config.get("attention_backend")
            ),
        )
        if vlm or max_images:
            # VL checkpoints need vision options even for text-only samples.
            kwargs.update(
                limit_mm_per_prompt={"image": max(1, max_images)},
                mm_encoder_tp_mode="data",
                mm_encoder_attn_backend="TORCH_SDPA",
                **image_embeds.engine_kwargs(),
            )
        if config.get("hf_overrides"):
            kwargs["hf_overrides"] = config["hf_overrides"]
        llm = LLM(**kwargs)
        llm.llm_engine.collective_rpc(worker_bootstrap())
        hooks_started = True
        setup = llm.apply_model(
            partial(
                setup_smooth_hooks,
                model_name=config["model_name"],
                sample_tokens=config["sample_tokens"],
                seed=config["seed"],
                num_hidden_layers=int(reader.config["num_hidden_layers"]),
                stat_names=sorted(expected),
                image_embeds_dir=config.get("image_embeds_dir") or previous_cache,
            )
        )
        if len(setup) != config["tp_size"] or any(set(s["names"]) != expected for s in setup):
            raise RuntimeError("incomplete offline Smooth hook coverage")
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0, max_tokens=1, detokenize=False),
            use_tqdm=True,
        )
        if len(outputs) != count:
            raise RuntimeError("Smooth generation returned an incomplete sample set")
        results = llm.apply_model(
            partial(
                get_smooth_stats,
                model_name=config["model_name"],
                samples_output=str(root / "samples.safetensors"),
            )
        )
        stats = next((s for s in results if s is not None), None)
        if stats is None or set(stats) != expected:
            raise RuntimeError("incomplete offline Smooth statistics")
        save_smooth_stats(dict(stats=stats), root / "stats.json")
        return str(root / "stats.json")
    finally:
        try:
            if llm is not None:
                try:
                    if hooks_started:
                        llm.apply_model(
                            partial(remove_smooth_hooks, model_name=config["model_name"])
                        )
                finally:
                    llm.llm_engine.engine_core.shutdown()
                    del llm
        finally:
            if previous_cache is None:
                os.environ.pop("VLLM_IMAGE_EMBEDS_DIR", None)
            else:
                os.environ["VLLM_IMAGE_EMBEDS_DIR"] = previous_cache
            gc.collect()


def search_offline_plan(config):
    """Search scales and save the compensation plan."""
    from safetensors.torch import save_file

    from ..core.stats_search import (
        build_plan,
        save_smooth_stats,
        search_offline_scales,
        smooth_selection,
    )

    root = Path(config["run_root"]) / "smooth"
    report = search_offline_scales(
        SimpleNamespace(
            **{
                **config,
                "stats": root / "stats.json",
                "samples": root / "samples.safetensors",
                "output": root / "search.json",
            }
        )
    )
    plan, scales = build_plan(
        config["model_path"],
        json.loads((root / "stats.json").read_text())["stats"],
        report["selected_alpha"],
        config["smooth_min"],
        config["smooth_max"],
        selection=smooth_selection(config),
        model_name=config["model_name"],
    )
    (root / "plan").mkdir(parents=True, exist_ok=True)
    save_file(scales, str(root / "plan" / plan["scale_file"]))
    path = root / "plan" / "smooth_plan.json"
    save_smooth_stats(plan, path)
    print(f"[smooth] plan: {path}", flush=True)
    return str(path)


if __name__ == "__main__":
    main()
