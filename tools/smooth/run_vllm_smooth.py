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

"""
Offline SmoothQuant scale calibration tool.

Collects per-channel absmax / EMA statistics for:
  - Attention layers (q, k inputs and attn output = o_proj input)
  - Dense MLP down_proj input  (= silu(gate) * up)
  - FusedMoE down_proj input   (requires VLLM_MOE_COLLECT_SMOOTH_STATS=1
                                 and vLLM kernel source modification)

Control which statistics to collect via --collect-attn / --collect-down-proj /
--collect-moe flags (all enabled by default).
"""

import argparse
import json
import os
import platform

from vllm import LLM, SamplingParams

from angelslim.compressor.transform.smooth.config import SmoothAlphaSearchConfig
from angelslim.compressor.transform.smooth.vllm import (
    SmoothAlphaSearcher,
    get_smooth_stats,
    print_smooth_stats,
    remove_smooth_alpha_search_hooks,
    setup_smooth_alpha_search_hooks,
    setup_smooth_hooks,
)
from angelslim.engine import Engine

_original_python_version = platform.python_version


def _patched_python_version():
    return _original_python_version().rstrip("+")


platform.python_version = _patched_python_version


# ---------------------------------------------------------------------------
# Helper functions to access draft (MTP) model via collective_rpc
# ---------------------------------------------------------------------------


def _get_draft_model_from_worker(worker):
    """
    Extract the draft (MTP) model from a vLLM worker instance.
    Works by traversing: worker -> model_runner -> drafter -> model.

    This function is designed to be called inside collective_rpc, where the
    worker argument is a WorkerWrapperBase (mp executor) or WorkerBase subclass.
    """
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        return None
    drafter = getattr(model_runner, "drafter", None)
    if drafter is None:
        return None
    return getattr(drafter, "model", None)


def _apply_on_draft_model(worker, fn):
    """
    Apply a function on the draft model inside a worker.
    This is a collective_rpc-compatible callable: collective_rpc passes
    the worker as the first argument when method is a callable.

    Usage:
        llm.llm_engine.collective_rpc(
            lambda w: _apply_on_draft_model(w, some_fn)
        )
    """
    draft_model = _get_draft_model_from_worker(worker)
    if draft_model is not None:
        return fn(draft_model)
    return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM SmoothQuant Calibration Tool — collect per-channel absmax/EMA statistics"
    )

    # Model configuration
    parser.add_argument("--model-path", type=str, help="Path to the model directory")
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

    args = parser.parse_args()
    # Lazy-import _yaml_args (located in tools/). Done here instead of at
    # module top so flake8 doesn't trip on a sys.path mutation between
    # imports.
    import sys

    _tools_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _tools_dir not in sys.path:
        sys.path.insert(0, _tools_dir)
    from _yaml_args import apply_yaml_config

    apply_yaml_config(parser, args)

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


def main():
    args = parse_args()
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
        # Force TRITON MoE backend so that the AngelSlim collection hooks
        # injected in TritonExperts.apply() (fused_moe.py) are actually reached.
        # Default "auto" may select FlashInfer/CUTLASS backends which bypass
        # the injected code entirely.
        moe_backend="triton",
    )

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
            lambda w: _apply_on_draft_model(w, setup_fn)
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
            lambda w: _apply_on_draft_model(
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
            lambda w: _apply_on_draft_model(w, get_smooth_stats)
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


if __name__ == "__main__":
    main()
