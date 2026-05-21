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
Standalone KV-cache calibration + scale search tool.

Unlike run_vllm_calibrate.py this script registers **only** Attention hooks,
so it skips all weight / activation / MoE statistics collection.  This makes
it faster to start up and uses less CPU memory.

Workflow (per-tensor, default)
------------------------------
1. Load model with vLLM.
2. Register kv-cache min/max hooks  (setup_kvcache_only_hooks).
3. Run a forward pass on calibration data  (llm.generate).
4. Collect & save kv-cache min/max stats  -> activation_stats.json
   (same key format as run_vllm_calibrate.py so downstream tools are compatible).
5. Optionally search for the best per-layer KV-scale multiplier  -> kv_scale_multipliers.json
   and the final tuned scales  -> kv_cache_tuned_scales.json.

Workflow (per-head, --per-head flag)
-------------------------------------
Same as above but with per-head granularity (one scale per KV head per layer):
4. Collect & save per-head stats    -> activation_stats_per_head.json
5. Optionally search per-head multipliers -> kv_scale_multipliers_per_head.json
   and the final tuned scales  -> kv_cache_tuned_scales_per_head.json.
"""

import argparse
import json
import os
import platform

from vllm import LLM, SamplingParams

from angelslim.compressor.quant import (  # Per-tensor pipeline; Per-head pipeline
    KVScaleSearcher,
    KVScaleSearcherPerHead,
    get_kv_scale_search_results,
    get_kv_scale_search_results_perhead,
    get_kvcache_only_stats,
    get_kvcache_perhead_stats,
    print_kvcache_only_stats,
    print_kvcache_perhead_stats,
    remove_kv_scale_search_hooks,
    remove_kvcache_perhead_value_hooks,
    setup_kvcache_only_hooks,
    setup_kvcache_perhead_hooks,
    setup_kvcache_perhead_value_hooks,
    setup_kvcache_value_hooks,
)
from angelslim.engine import Engine

_original_python_version = platform.python_version


def _patched_python_version():
    return _original_python_version().rstrip("+")


platform.python_version = _patched_python_version


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone KV-cache calibration and scale search tool. "
        "Only registers Attention hooks – no weight/activation/MoE overhead."
    )

    # YAML config (values override argparse defaults; explicit CLI flags still win)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. Keys must match argparse dest names "
        "(e.g. model_path, tp_size, per_head, search_kv_scale). Values override "
        "argparse defaults; explicit command-line flags still take final precedence.",
    )

    # Model configuration
    # NOTE: required=False because these can also come from the YAML config.
    parser.add_argument("--model-path", type=str, default=None, help="Path to the model directory")
    parser.add_argument(
        "--ptq-data-path",
        type=str,
        default=None,
        help="Path to the PTQ calibration data (JSONL / JSON format)",
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
        help="Number of samples for kv-cache min/max calibration (default: 512)",
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

    # Granularity
    parser.add_argument(
        "--per-head",
        action="store_true",
        help="Collect and search per KV-head scales instead of per-layer (per-tensor) scales.",
    )

    # KV cache scale search options
    parser.add_argument(
        "--search-kv-scale",
        action="store_true",
        help="After calibration, search for the best KV-cache scale multiplier "
        "(per-layer in default mode; per-head when --per-head is set).",
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
        default=100,
        help="Number of grid points for the scale multiplier search (default: 100). "
        "Candidates are sampled on a log-uniform grid.",
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


def save_json(data, output_dir: str, filename: str, label: str = "data") -> str:
    """Save *data* as JSON and return the full path."""
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n{label} saved to: {path}")
    return path


def _run_pertensor(args, llm, prompts):
    """Per-tensor (per-layer) calibration + optional scale search."""
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1)

    # ------------------------------------------------------------------
    # 2. Register per-tensor kv-cache hooks
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Setting up kv-cache-only hooks (per-tensor, no weight/activation/MoE hooks)...")
    print("=" * 80)
    hook_results = llm.apply_model(setup_kvcache_only_hooks)
    for i, result in enumerate(hook_results):
        print(f"  Worker {i}: {result}")

    # ------------------------------------------------------------------
    # 3. Forward pass
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Running forward pass for kv-cache min/max collection...")
    print("=" * 80)
    outputs = llm.generate(prompts, sampling_params)
    print(f"Total outputs generated: {len(outputs)}")

    # ------------------------------------------------------------------
    # 4. Collect and save per-tensor stats
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Collecting kv-cache statistics...")
    print("=" * 80)
    llm.apply_model(print_kvcache_only_stats)

    os.makedirs(args.output_dir, exist_ok=True)

    stats_list = llm.apply_model(get_kvcache_only_stats)
    if not stats_list or stats_list[0] is None:
        print("\nERROR: No kv-cache statistics collected. Aborting.")
        return

    activation_stats = stats_list[0]  # rank-0 result; all-reduce already done inside
    save_json(
        activation_stats,
        args.output_dir,
        "activation_stats.json",
        label="KV-cache min/max statistics (per-tensor)",
    )

    print("\n" + "=" * 80)
    print("KV-cache per-tensor calibration completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 5. Optional: per-tensor scale search
    # ------------------------------------------------------------------
    if not args.search_kv_scale:
        return

    print("\n" + "=" * 80)
    print("Starting KV-cache per-tensor scale search...")
    print(f"  Search samples  : {args.search_kv_num_samples}")
    print(
        f"  Multiplier range: [{args.search_kv_min_multiplier}, {args.search_kv_max_multiplier}]"
    )
    print(f"  Grid steps      : {args.search_kv_num_steps}")
    print("=" * 80)

    print("\nRegistering KV-value capture hooks...")
    hook_results = llm.apply_model(setup_kvcache_value_hooks)
    for i, r in enumerate(hook_results):
        print(f"  Worker {i}: {r}")

    search_prompts = prompts[: args.search_kv_num_samples]
    print(f"\nRunning {len(search_prompts)} forward passes for KV-value collection...")
    llm.generate(search_prompts, SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1))

    print("\nSearching best multiplier per layer...")
    searcher = KVScaleSearcher(
        activation_stats=activation_stats,
        min_multiplier=args.search_kv_min_multiplier,
        max_multiplier=args.search_kv_max_multiplier,
        num_steps=args.search_kv_num_steps,
    )
    search_results_list = llm.apply_model(searcher)
    kv_multipliers = get_kv_scale_search_results(search_results_list)

    llm.apply_model(remove_kv_scale_search_hooks)

    save_json(
        kv_multipliers,
        args.output_dir,
        "kv_scale_multipliers.json",
        label="KV-cache scale multipliers (per-tensor)",
    )

    # Compute and save final (scaled) kv cache scales
    fp8_max = 448.0  # torch.finfo(torch.float8_e4m3fn).max
    tuned_kv_scales = {}
    for stats_key, multiplier in kv_multipliers.items():
        stats = activation_stats[stats_key]
        abs_max = max(abs(stats["min"]), abs(stats["max"]))
        base_scale = abs_max / fp8_max * 2.0
        tuned_scale = base_scale * multiplier
        # Use the ".scale" suffix to match kv_cache_scales.safetensors key naming
        save_key = f"{stats_key.replace('attn.attn', 'attn')}.scale"
        tuned_kv_scales[save_key] = tuned_scale
    save_json(
        tuned_kv_scales,
        args.output_dir,
        "kv_cache_tuned_scales.json",
        label="Tuned KV-cache scales (per-tensor)",
    )

    print("\n" + "=" * 80)
    print("KV-cache per-tensor scale search completed!")
    print("=" * 80)


def _run_perhead(args, llm, prompts):
    """Per-head calibration + optional per-head scale search."""
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1)

    # ------------------------------------------------------------------
    # 2. Register per-head kv-cache hooks
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Setting up kv-cache per-head hooks (no weight/activation/MoE hooks)...")
    print("=" * 80)
    hook_results = llm.apply_model(setup_kvcache_perhead_hooks)
    for i, result in enumerate(hook_results):
        print(f"  Worker {i}: {result}")

    # ------------------------------------------------------------------
    # 3. Forward pass
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Running forward pass for per-head kv-cache min/max collection...")
    print("=" * 80)
    outputs = llm.generate(prompts, sampling_params)
    print(f"Total outputs generated: {len(outputs)}")

    # ------------------------------------------------------------------
    # 4. Collect and save per-head stats
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Collecting per-head kv-cache statistics...")
    print("=" * 80)
    llm.apply_model(print_kvcache_perhead_stats)

    os.makedirs(args.output_dir, exist_ok=True)

    stats_list = llm.apply_model(get_kvcache_perhead_stats)
    if not stats_list or stats_list[0] is None:
        print("\nERROR: No per-head kv-cache statistics collected. Aborting.")
        return

    activation_stats_perhead = stats_list[0]  # rank-0 result
    save_json(
        activation_stats_perhead,
        args.output_dir,
        "activation_stats_per_head.json",
        label="KV-cache min/max statistics (per-head)",
    )

    print("\n" + "=" * 80)
    print("KV-cache per-head calibration completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 5. Optional: per-head scale search
    # ------------------------------------------------------------------
    if not args.search_kv_scale:
        return

    print("\n" + "=" * 80)
    print("Starting KV-cache per-head scale search...")
    print(f"  Search samples  : {args.search_kv_num_samples}")
    print(
        f"  Multiplier range: [{args.search_kv_min_multiplier}, {args.search_kv_max_multiplier}]"
    )
    print(f"  Grid steps      : {args.search_kv_num_steps}")
    print("=" * 80)

    print("\nRegistering per-head KV-value capture hooks...")
    hook_results = llm.apply_model(setup_kvcache_perhead_value_hooks)
    for i, r in enumerate(hook_results):
        print(f"  Worker {i}: {r}")

    search_prompts = prompts[: args.search_kv_num_samples]
    print(f"\nRunning {len(search_prompts)} forward passes for per-head KV-value collection...")
    llm.generate(search_prompts, SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1))

    print("\nSearching best multiplier per head per layer...")
    searcher = KVScaleSearcherPerHead(
        activation_stats=activation_stats_perhead,
        min_multiplier=args.search_kv_min_multiplier,
        max_multiplier=args.search_kv_max_multiplier,
        num_steps=args.search_kv_num_steps,
    )
    search_results_list = llm.apply_model(searcher)
    kv_multipliers_perhead = get_kv_scale_search_results_perhead(search_results_list)

    llm.apply_model(remove_kvcache_perhead_value_hooks)

    save_json(
        kv_multipliers_perhead,
        args.output_dir,
        "kv_scale_multipliers_per_head.json",
        label="KV-cache scale multipliers (per-head)",
    )

    # Compute and save final (scaled) per-head kv cache scales
    # Output format:  {"<layer>.k_cache.head_0.scale": float, ...}
    fp8_max = 448.0  # torch.finfo(torch.float8_e4m3fn).max
    tuned_kv_scales_perhead = {}
    for stats_key, multipliers in kv_multipliers_perhead.items():
        stats = activation_stats_perhead[stats_key]
        min_vals = stats["min"]  # list[float], length H
        max_vals = stats["max"]  # list[float], length H
        for head_idx, multiplier in enumerate(multipliers):
            abs_max = max(abs(min_vals[head_idx]), abs(max_vals[head_idx]))
            if abs_max == 0:
                base_scale = 1e-8
            else:
                base_scale = abs_max / fp8_max * 2.0
            tuned_scale = base_scale * multiplier
            # Key naming convention: "<layer_key>.head_<H>.scale"
            base_key = stats_key.replace("attn.attn", "attn")
            save_key = f"{base_key}.head_{head_idx}.scale"
            tuned_kv_scales_perhead[save_key] = tuned_scale
    save_json(
        tuned_kv_scales_perhead,
        args.output_dir,
        "kv_cache_tuned_scales_per_head.json",
        label="Tuned KV-cache scales (per-head)",
    )

    print("\n" + "=" * 80)
    print("KV-cache per-head scale search completed!")
    print("=" * 80)


def main():
    args = parse_args()

    print("\nConfiguration:")
    print(f"  Model             : {args.model_path}")
    print(f"  PTQ Data          : {args.ptq_data_path}")
    print(f"  Output Dir        : {args.output_dir}")
    print(f"  TP Size           : {args.tp_size}")
    print(f"  Batch Size        : {args.batch_size}")
    print(f"  Num Samples       : {args.num_samples}")
    print(f"  Skip Wgt Loading  : {args.skip_weight_loading}")
    print(f"  Per-Head Mode     : {args.per_head}")
    print(f"  Search KV Scale   : {args.search_kv_scale}")

    # ------------------------------------------------------------------
    # 1. Create LLM instance
    # ------------------------------------------------------------------
    llm = LLM(
        model=args.model_path,
        load_format="dummy" if args.skip_weight_loading else "auto",
        disable_log_stats=False,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=65536,
        num_gpu_blocks_override=4096,
        tensor_parallel_size=args.tp_size,
        distributed_executor_backend=args.distributed_executor_backend,
        enable_expert_parallel=False,
        max_num_seqs=args.batch_size,
        max_model_len=args.max_length + 16,
    )

    if args.skip_weight_loading:
        print("\n" + "!" * 80)
        print("WARNING: Running with dummy weights (random values)!")
        print("Outputs will NOT make sense. This is for debugging only.")
        print("!" * 80 + "\n")

    # ------------------------------------------------------------------
    # 2. Load dataset and prepare prompts
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Dispatch to per-tensor or per-head pipeline
    # ------------------------------------------------------------------
    if args.per_head:
        _run_perhead(args, llm, prompts)
    else:
        _run_pertensor(args, llm, prompts)


if __name__ == "__main__":
    main()
