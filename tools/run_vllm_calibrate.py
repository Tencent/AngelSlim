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


import argparse
import json
import os
import platform

from vllm import LLM, SamplingParams

from angelslim.compressor.quant import (  # Per-head KV-cache pipeline
    KVScaleSearcher,
    KVScaleSearcherPerHead,
    get_activation_stats,
    get_kv_scale_search_results,
    get_kv_scale_search_results_perhead,
    get_kvcache_perhead_stats,
    get_moe_stats,
    get_mtp_activation_stats,
    get_mtp_moe_stats,
    print_activation_stats,
    print_kvcache_perhead_stats,
    print_moe_stats,
    print_mtp_activation_stats,
    print_mtp_moe_stats,
    remove_kv_scale_search_hooks,
    remove_kvcache_perhead_hooks,
    remove_kvcache_perhead_value_hooks,
    setup_activation_hooks,
    setup_kvcache_perhead_value_hooks,
    setup_kvcache_value_hooks,
    setup_mtp_activation_hooks,
)
from angelslim.engine import Engine

# =============================================================================
# Helper functions to access draft (MTP) model via collective_rpc
# =============================================================================


def _get_draft_model_from_worker(worker):
    """
    Extract the draft (MTP) model from a vLLM worker instance.
    Works by traversing: worker -> model_runner -> drafter -> model.

    This function is designed to be called inside collective_rpc, where the
    worker argument is a WorkerWrapperBase (mp executor) or WorkerBase subclass.
    """
    # In mp executor, worker is WorkerWrapperBase, which proxies to real GPUWorker
    # via __getattr__. In uniproc executor, worker is the GPUWorker directly.
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
            lambda w: _apply_on_draft_model(w, setup_mtp_activation_hooks)
        )
    """
    draft_model = _get_draft_model_from_worker(worker)
    if draft_model is not None:
        return fn(draft_model)
    return None


_original_python_version = platform.python_version


def _patched_python_version():
    return _original_python_version().rstrip("+")


platform.python_version = _patched_python_version


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="vLLM Calibration Tool - Collect activation and MoE expert statistics"
    )

    # YAML config (values override argparse defaults; explicit CLI flags still win)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. Keys must match argparse dest names "
        "(e.g. model_path, tp_size, search_kv_scale). Values override argparse "
        "defaults; explicit command-line flags still take final precedence.",
    )

    # Model configuration
    # NOTE: required=False because these can also come from the YAML config.
    # apply_yaml_config() runs after parse_args(); we validate required fields
    # manually below.
    parser.add_argument("--model-path", type=str, default=None, help="Path to the model directory")
    parser.add_argument(
        "--ptq-data-path",
        type=str,
        default=None,
        help="Path to the PTQ calibration data (JSONL format)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save output statistics"
    )

    # Model loading configuration
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size (default: 1)")
    parser.add_argument(
        "--skip-weight-loading",
        action="store_true",
        help="Use dummy weights for fast debug mode (outputs will be random)",
    )

    # Dataset configuration
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for inference (default: 128)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="Number of samples to process from dataset (default: 512)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=16384,
        help="Maximum sequence length for tokenization (default: 16384)",
    )

    # Distributed configuration
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
        help="Enable MTP (Multi-Token Prediction) speculative decoding with hunyuan_mtp method",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=1,
        help="Number of speculative tokens for MTP "
        "(default: 1, only used when --enable-mtp is set)",
    )

    # Debug options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging (print detailed statistics during all-reduce)",
    )

    # KV-cache granularity
    parser.add_argument(
        "--kv-granularity",
        type=str,
        default="per-tensor",
        choices=["none", "per-tensor", "per-head"],
        help="KV-cache calibration granularity: "
        "'none' = skip KV scale calibration (activation/MoE stats only); "
        "'per-tensor' = per-layer scale (default, same as original behaviour); "
        "'per-head' = per-head scale (finer grained).",
    )

    # KV cache scale search options
    parser.add_argument(
        "--search-kv-scale",
        action="store_true",
        help="After calibration, search for the best KV-cache scale multiplier "
        "(per-layer when --kv-granularity=per-tensor; "
        "per-head when --kv-granularity=per-head). "
        "Has no effect when --kv-granularity=none.",
    )
    parser.add_argument(
        "--search-kv-num-samples",
        type=int,
        default=64,
        help="Number of samples used for KV-cache scale search (default: 64). "
        "These are taken from --ptq-data-path (the same dataset).",
    )
    parser.add_argument(
        "--search-kv-min-multiplier",
        type=float,
        default=0.8,
        help="Lower bound of the scale multiplier search range (default: 0.8).",
    )
    parser.add_argument(
        "--search-kv-max-multiplier",
        type=float,
        default=16.0,
        help="Upper bound of the scale multiplier search range (default: 16.0).",
    )
    parser.add_argument(
        "--search-kv-num-steps",
        type=int,
        default=50,
        help="Number of grid points for the scale multiplier search (default: 50). "
        "Candidates are sampled on a log-uniform grid.",
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

    # Validate fields that used to be ``required=True`` but now may come
    # from either the CLI or the YAML config.
    missing = [
        name
        for name in ("model_path", "ptq_data_path", "output_dir")
        if getattr(args, name, None) in (None, "")
    ]
    if missing:
        parser.error(
            "the following arguments are required (via CLI or YAML config): "
            + ", ".join("--" + n.replace("_", "-") for n in missing)
        )

    return args


def save_stats_to_json(
    stats_data, output_dir: str, filename: str, stats_type: str = "statistics"
) -> None:
    """
    Save statistics to JSON file.

    Args:
        stats_data: Statistics data (can be dict or list)
        output_dir: Directory to save the file
        filename: Output filename
        stats_type: Type of statistics for error messages
    """
    # Handle list input - extract first worker's data
    if isinstance(stats_data, list):
        if not stats_data or stats_data[0] is None:
            print(f"\nNo {stats_type} available.")
            return
        stats_data = stats_data[0]

    # Check if data is None
    if stats_data is None:
        print(f"\nNo {stats_type} available.")
        if "moe" in stats_type.lower():
            print("Make sure VLLM_MOE_COLLECT_STATS=1 is set and the model has MoE layers.")
        return

    # Save to file
    output_file = os.path.join(output_dir, filename)
    with open(output_file, "w") as f:
        json.dump(stats_data, f, indent=2)
    print(f"\n{stats_type.capitalize()} saved to: {output_file}")


def main():
    """Main function to run calibration."""
    args = parse_args()

    # Verify environment variables are set
    print(f"VLLM_MOE_COLLECT_STATS: {os.environ.get('VLLM_MOE_COLLECT_STATS')}")
    print("\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  PTQ Data: {args.ptq_data_path}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  TP Size: {args.tp_size}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Num Samples: {args.num_samples}")
    print(f"  Skip Weight Loading: {args.skip_weight_loading}")
    print(f"  KV Granularity: {args.kv_granularity}")

    # Configure MTP speculative decoding
    speculative_config = None
    if args.enable_mtp:
        speculative_config = {
            "method": "hunyuan_mtp",
            "num_speculative_tokens": args.num_speculative_tokens,
        }
        print(f"  MTP Enabled: True (num_speculative_tokens={args.num_speculative_tokens})")
    else:
        print("  MTP Enabled: False")

    # Create LLM instance
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
        # Force the Triton MoE backend so the AngelSlim fused_moe.py patch
        # (which inserts collect_fused_moe_internal_stats hooks inside
        # fused_experts_impl) is actually exercised. Without this vLLM may
        # auto-select FlashInfer CUTLASS / TRTLLM, which run the entire
        # gate_up -> activation -> down_proj pipeline inside a single
        # opaque C++ kernel and bypass our Python-level hooks.
        moe_backend="triton",
    )

    if args.skip_weight_loading:
        print("\n" + "!" * 80)
        print("WARNING: Running with dummy weights (random values)!")
        print("Outputs will NOT make sense. This is for debugging only.")
        print("Use --skip-weight-loading flag to enable this mode.")
        print("!" * 80 + "\n")

    # Setup activation hooks on all workers
    # kv_granularity controls which KV hooks are registered alongside Linear hooks:
    #   'none'       -> no KV hooks
    #   'per-tensor' -> per-layer KV min/max hooks (default)
    #   'per-head'   -> per-head KV min/max hooks (shares the same forward pass)
    print("\n" + "=" * 80)
    print(f"Setting up activation hooks (kv_granularity={args.kv_granularity})...")
    print("=" * 80)
    hook_results = llm.apply_model(
        lambda model: setup_activation_hooks(model, kv_granularity=args.kv_granularity)
    )
    for i, result in enumerate(hook_results):
        print(f"Worker {i}: {result}")

    # Setup MTP draft model activation hooks (if MTP is enabled)
    # NOTE: We use collective_rpc with a callable to directly access the draft
    # model inside each worker, bypassing the need to modify vllm internals.
    if args.enable_mtp:
        print("\n" + "=" * 80)
        print("Setting up MTP draft model activation hooks...")
        print("=" * 80)
        mtp_hook_results = llm.llm_engine.collective_rpc(
            lambda w: _apply_on_draft_model(w, setup_mtp_activation_hooks)
        )
        for i, result in enumerate(mtp_hook_results):
            if result is not None:
                print(f"Worker {i}: {result}")
            else:
                print(f"Worker {i}: No MTP draft model available")

    # Load dataset and prepare prompts
    print("\n" + "=" * 80)
    print("Loading dataset and preparing prompts...")
    print("=" * 80)
    tokenizer = llm.get_tokenizer()

    slim_engine = Engine()
    slim_engine.slim_model = llm
    slim_engine.series = "LLM"
    slim_engine.slim_model.tokenizer = tokenizer
    slim_engine.slim_model.model = llm
    slim_engine.slim_model.model.device = "cpu"
    dataset = slim_engine.prepare_data(
        data_path=args.ptq_data_path,
        max_length=args.max_length,
        num_samples=args.num_samples,
        shuffle=False,
        inference_settings=None,
        use_audio_in_video=False,
    )

    prompts = [tokenizer.decode(data["input_ids"][0]) for data in dataset]
    print(f"Loaded {len(prompts)} prompts from dataset")

    # Create sampling params (fixed values for calibration)
    # When MTP is enabled, we need to generate more tokens to trigger
    # speculative decoding so the MTP draft model layers also get activated.
    calibration_max_tokens = 1
    if args.enable_mtp:
        # Generate enough tokens to trigger multiple rounds of speculative
        # decoding, ensuring MTP layers produce activations for calibration.
        # The draft model runs during each decode step's propose phase,
        # so we need max_tokens > 1 to enter decode and trigger at least
        # one propose call. Using max(8, ...) gives several rounds of
        # drafter execution for more robust activation statistics.
        calibration_max_tokens = 1  # max(8, args.num_speculative_tokens * 4)
        print(
            f"  MTP calibration: setting max_tokens={calibration_max_tokens} "
            f"to trigger speculative decoding"
        )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=calibration_max_tokens,
    )

    # Generate outputs
    print("\n" + "=" * 80)
    print("Generating outputs...")
    print("=" * 80)
    outputs = llm.generate(prompts, sampling_params)

    # Print sample outputs
    print("\n" + "=" * 80)
    print("Sample Generated Outputs (first 5):")
    print("=" * 80)
    for i, output in enumerate(outputs[:5]):
        generated_text = output.outputs[0].text
        print(f"[{i + 1}] Output: {generated_text!r}")
    print(f"\nTotal outputs generated: {len(outputs)}")

    # Collect and save statistics
    print("\n" + "=" * 80)
    print("Collecting Statistics...")
    print("=" * 80)

    # Print activation stats from all workers
    print("\nActivation Statistics:")
    llm.apply_model(print_activation_stats)

    # Print MoE expert statistics
    print("\nMoE Expert Statistics:")
    llm.apply_model(lambda model: print_moe_stats(model, verbose=args.verbose))

    # Print MTP draft model statistics (if MTP is enabled)
    if args.enable_mtp:
        print("\n[MTP] Draft Model Activation Statistics:")
        llm.llm_engine.collective_rpc(
            lambda w: _apply_on_draft_model(w, print_mtp_activation_stats)
        )

        print("\n[MTP] Draft Model MoE Expert Statistics:")
        llm.llm_engine.collective_rpc(
            lambda w: _apply_on_draft_model(
                w, lambda m: print_mtp_moe_stats(m, verbose=args.verbose)
            )
        )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save activation statistics
    stats_list = llm.apply_model(get_activation_stats)
    save_stats_to_json(
        stats_list, args.output_dir, "activation_stats.json", stats_type="activation statistics"
    )

    # Save MoE expert statistics
    moe_stats_dict = llm.apply_model(get_moe_stats)
    save_stats_to_json(
        moe_stats_dict,
        args.output_dir,
        "moe_expert_stats.json",
        stats_type="MoE expert statistics",
    )

    # Save MTP draft model statistics (if MTP is enabled)
    if args.enable_mtp:
        mtp_stats_list = llm.llm_engine.collective_rpc(
            lambda w: _apply_on_draft_model(w, get_mtp_activation_stats)
        )
        save_stats_to_json(
            mtp_stats_list,
            args.output_dir,
            "mtp_activation_stats.json",
            stats_type="MTP activation statistics",
        )

        mtp_moe_stats_dict = llm.llm_engine.collective_rpc(
            lambda w: _apply_on_draft_model(w, get_mtp_moe_stats)
        )
        save_stats_to_json(
            mtp_moe_stats_dict,
            args.output_dir,
            "mtp_moe_expert_stats.json",
            stats_type="MTP MoE expert statistics",
        )

    print("\n" + "=" * 80)
    print("Calibration completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Per-tensor KV-cache scale search  (--kv-granularity per-tensor)
    # -------------------------------------------------------------------------
    if args.kv_granularity == "per-tensor" and args.search_kv_scale:
        print("\n" + "=" * 80)
        print("Starting KV-cache per-tensor scale search...")
        print(f"  Search samples  : {args.search_kv_num_samples}")
        print(
            f"  Multiplier range: "
            f"[{args.search_kv_min_multiplier}, {args.search_kv_max_multiplier}]"
        )
        print(f"  Grid steps      : {args.search_kv_num_steps}")
        print("=" * 80)

        # Load activation stats that were just saved (contains k/v cache min/max)
        activation_stats_path = os.path.join(args.output_dir, "activation_stats.json")
        with open(activation_stats_path, "r") as f:
            activation_stats = json.load(f)

        # Register value-capture hooks on all workers
        print("\nRegistering KV-value capture hooks...")
        hook_results = llm.apply_model(setup_kvcache_value_hooks)
        for i, r in enumerate(hook_results):
            print(f"  Worker {i}: {r}")

        # Run a short forward pass to collect raw k/v tensors
        search_prompts = prompts[: args.search_kv_num_samples]
        print(f"\nRunning {len(search_prompts)} forward passes for KV-value collection...")
        llm.generate(search_prompts, SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1))

        # Run search inside each worker; take rank-0 result
        print("\nSearching best multiplier per layer...")
        searcher = KVScaleSearcher(
            activation_stats=activation_stats,
            min_multiplier=args.search_kv_min_multiplier,
            max_multiplier=args.search_kv_max_multiplier,
            num_steps=args.search_kv_num_steps,
        )
        search_results_list = llm.apply_model(searcher)
        kv_multipliers = get_kv_scale_search_results(search_results_list)

        # Clean up value-capture hooks
        llm.apply_model(remove_kv_scale_search_hooks)

        # Save multipliers to JSON
        multipliers_path = os.path.join(args.output_dir, "kv_scale_multipliers.json")
        with open(multipliers_path, "w") as f:
            json.dump(kv_multipliers, f, indent=2)
        print(f"\nKV-cache scale multipliers saved to: {multipliers_path}")

        # Also save the final (scaled) kv cache scales for direct use
        fp8_max = 448.0  # torch.finfo(torch.float8_e4m3fn).max
        tuned_kv_scales = {}
        for stats_key, multiplier in kv_multipliers.items():
            stats = activation_stats[stats_key]
            abs_max = max(abs(stats["min"]), abs(stats["max"]))
            base_scale = abs_max / fp8_max * 2.0
            tuned_scale = base_scale * multiplier
            save_key = f"{stats_key.replace('attn.attn', 'attn')}.scale"
            tuned_kv_scales[save_key] = tuned_scale
        tuned_scales_path = os.path.join(args.output_dir, "kv_cache_tuned_scales.json")
        with open(tuned_scales_path, "w") as f:
            json.dump(tuned_kv_scales, f, indent=2)
        print(f"Tuned KV-cache scales saved to: {tuned_scales_path}")

        print("\n" + "=" * 80)
        print("KV-cache per-tensor scale search completed!")
        print("=" * 80)

    # -------------------------------------------------------------------------
    # Per-head KV-cache collect stats + optional scale search
    # (--kv-granularity per-head)
    # Per-head hooks were already registered before the main forward pass, so
    # we only need to collect and save the results here — no extra forward pass.
    # -------------------------------------------------------------------------
    if args.kv_granularity == "per-head":
        print("\n" + "=" * 80)
        print("Collecting per-head kv-cache statistics (from main forward pass)...")
        print("=" * 80)
        llm.apply_model(print_kvcache_perhead_stats)

        stats_list_ph = llm.apply_model(get_kvcache_perhead_stats)
        if not stats_list_ph or stats_list_ph[0] is None:
            print(
                "\nERROR: No per-head kv-cache statistics collected. Aborting per-head pipeline."
            )
        else:
            activation_stats_perhead = stats_list_ph[0]  # rank-0; all-gather done inside
            # Merge per-head stats into activation_stats.json
            merged_stats_path = os.path.join(args.output_dir, "activation_stats.json")
            if os.path.exists(merged_stats_path):
                with open(merged_stats_path, "r") as f:
                    merged_stats = json.load(f)
            else:
                merged_stats = {}
            merged_stats.update(activation_stats_perhead)
            with open(merged_stats_path, "w") as f:
                json.dump(merged_stats, f, indent=2)
            print(f"\nKV-cache per-head statistics merged into: {merged_stats_path}")

            # Remove per-head min/max hooks before (optionally) registering value hooks
            llm.apply_model(remove_kvcache_perhead_hooks)

            print("\n" + "=" * 80)
            print("KV-cache per-head calibration completed successfully!")
            print(f"Results saved to: {args.output_dir}")
            print("=" * 80)

            # ------------------------------------------------------------------
            # Optional: per-head scale search
            # ------------------------------------------------------------------
            if args.search_kv_scale:
                print("\n" + "=" * 80)
                print("Starting KV-cache per-head scale search...")
                print(f"  Search samples  : {args.search_kv_num_samples}")
                print(
                    f"  Multiplier range: "
                    f"[{args.search_kv_min_multiplier}, {args.search_kv_max_multiplier}]"
                )
                print(f"  Grid steps      : {args.search_kv_num_steps}")
                print("=" * 80)

                print("\nRegistering per-head KV-value capture hooks...")
                hook_results = llm.apply_model(setup_kvcache_perhead_value_hooks)
                for i, r in enumerate(hook_results):
                    print(f"  Worker {i}: {r}")

                search_prompts_ph = prompts[: args.search_kv_num_samples]
                print(
                    f"\nRunning {len(search_prompts_ph)} forward passes "
                    f"for per-head KV-value collection..."
                )
                llm.generate(
                    search_prompts_ph, SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1)
                )

                print("\nSearching best multiplier per head per layer...")
                searcher_ph = KVScaleSearcherPerHead(
                    activation_stats=activation_stats_perhead,
                    min_multiplier=args.search_kv_min_multiplier,
                    max_multiplier=args.search_kv_max_multiplier,
                    num_steps=args.search_kv_num_steps,
                )
                search_results_list_ph = llm.apply_model(searcher_ph)
                kv_multipliers_perhead = get_kv_scale_search_results_perhead(
                    search_results_list_ph
                )

                llm.apply_model(remove_kvcache_perhead_value_hooks)

                multipliers_ph_path = os.path.join(
                    args.output_dir, "kv_scale_multipliers_per_head.json"
                )
                with open(multipliers_ph_path, "w") as f:
                    json.dump(kv_multipliers_perhead, f, indent=2)
                print(f"\nKV-cache per-head scale multipliers saved to: {multipliers_ph_path}")

                # Compute and save final tuned per-head scales
                fp8_max = 448.0
                tuned_kv_scales_perhead = {}
                for stats_key, multipliers in kv_multipliers_perhead.items():
                    stats = activation_stats_perhead[stats_key]
                    min_vals = stats["min"]
                    max_vals = stats["max"]
                    for head_idx, multiplier in enumerate(multipliers):
                        abs_max = max(abs(min_vals[head_idx]), abs(max_vals[head_idx]))
                        base_scale = abs_max / fp8_max * 2.0 if abs_max != 0 else 1e-8
                        tuned_scale = base_scale * multiplier
                        base_key = stats_key.replace("attn.attn", "attn")
                        save_key = f"{base_key}.head_{head_idx}.scale"
                        tuned_kv_scales_perhead[save_key] = tuned_scale
                tuned_ph_path = os.path.join(
                    args.output_dir, "kv_cache_tuned_scales_per_head.json"
                )
                with open(tuned_ph_path, "w") as f:
                    json.dump(tuned_kv_scales_perhead, f, indent=2)
                print(f"Tuned per-head KV-cache scales saved to: {tuned_ph_path}")

                print("\n" + "=" * 80)
                print("KV-cache per-head scale search completed!")
                print("=" * 80)


if __name__ == "__main__":
    main()
