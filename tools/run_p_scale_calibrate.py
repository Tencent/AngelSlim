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
P-matrix (attention probability) FP8 scale search tool.

Workflow
--------
1. Load model with vLLM.
2. Register P-matrix scale hooks  (setup_p_matrix_scale_hooks).
3. Run a **single prefill-only** forward pass on calibration data
   (max_tokens=1 so no decode steps are generated).
4. Collect per-head NMSE statistics and pick the best scale per Q-head.
   By default, the first ``mse_skip_first_cols`` (default: 2) sample-local
   key columns are **excluded** from the NMSE objective so attention-sink
   columns do not dominate the error signal.  NMSE = sum(diff²) / sum(P²)
   computed over the same scope.
5. Save two JSON files:
     - p_matrix_search_results_per_head.json   (full NMSE table + best scale)
     - p_matrix_tuned_scales_per_head.json     (flat key→scale dict, same
       naming convention as kv_cache_tuned_scales_per_head.json)

Granularity: per Q-head.
Under TP=N each rank holds H_q_total/N Q-heads; the workload is naturally
partitioned with no redundancy.  No base-scale calibration pass is needed –
the user supplies the candidate scale list directly via --scale-list.
"""

import argparse
import json
import os
import platform

from vllm import LLM, SamplingParams

from angelslim.compressor.quant import (
    PMatrixScaleSearcher,
    remove_p_matrix_scale_hooks,
    setup_p_matrix_scale_hooks,
)
from angelslim.engine import Engine

_original_python_version = platform.python_version


def _patched_python_version():
    return _original_python_version().rstrip("+")


platform.python_version = _patched_python_version


def parse_args():
    parser = argparse.ArgumentParser(
        description="P-matrix (attention probability) FP8 scale search tool. "
        "Registers Attention hooks, runs one prefill pass, and "
        "picks the best FP8 scale per Q-head via MSE minimisation."
    )

    # Model configuration
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model directory."
    )
    parser.add_argument(
        "--ptq-data-path",
        type=str,
        required=True,
        help="Path to the PTQ calibration data (JSONL / JSON format).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save output JSON files.",
    )

    # Model loading
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1).",
    )
    parser.add_argument(
        "--skip-weight-loading",
        action="store_true",
        help="Use dummy weights for fast debug mode (outputs will be random).",
    )

    # Dataset
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of calibration samples (default: 32).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=32768,
        help="Maximum sequence length for tokenisation (default: 32768).",
    )

    # Distributed
    parser.add_argument(
        "--distributed-executor-backend",
        type=str,
        default="ray",
        choices=["ray", "mp"],
        help="Distributed executor backend (default: ray).",
    )

    # P-matrix scale search
    parser.add_argument(
        "--scale-list",
        type=str,
        default="0.05,0.1,0.2,0.4,0.8,1.0",
        help="Comma-separated list of candidate FP8 scales to evaluate "
        "(default: '0.05,0.1,0.2,0.4,0.8,1.0').",
    )
    parser.add_argument(
        "--q-block-size",
        type=int,
        default=256,
        help="Number of Q tokens processed per block during P re-computation. "
        "Larger values are faster but use more GPU memory (default: 256).",
    )
    parser.add_argument(
        "--no-collect-p-dist",
        action="store_true",
        help="Disable per-Q-head P-value distribution collection "
        "(histogram / mean / std / max).  Speeds up calibration and "
        "skips writing p_matrix_distribution_*.json outputs.",
    )
    parser.add_argument(
        "--mse-skip-first-cols",
        type=int,
        default=2,
        help="Number of leading sample-local key columns to *exclude* from "
        "the FP8 NMSE objective (attention-sink columns).  Set to 0 "
        "to disable masking.  Default: 2.",
    )

    return parser.parse_args()


def save_json(data, output_dir: str, filename: str, label: str = "data") -> str:
    """Save *data* as JSON and return the full path."""
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n{label} saved to: {path}")
    return path


def main():
    args = parse_args()

    # Parse scale list from CLI string.
    scale_list = [float(s.strip()) for s in args.scale_list.split(",") if s.strip()]
    if not scale_list:
        raise ValueError("--scale-list must contain at least one value.")

    print("\nConfiguration:")
    print(f"  Model             : {args.model_path}")
    print(f"  PTQ Data          : {args.ptq_data_path}")
    print(f"  Output Dir        : {args.output_dir}")
    print(f"  TP Size           : {args.tp_size}")
    print(f"  Batch Size        : {args.batch_size}")
    print(f"  Num Samples       : {args.num_samples}")
    print(f"  Max Length        : {args.max_length}")
    print(f"  Skip Wgt Loading  : {args.skip_weight_loading}")
    print(f"  Scale List        : {scale_list}")
    print(f"  Q Block Size      : {args.q_block_size}")
    print(f"  Collect P Dist    : {not args.no_collect_p_dist}")
    print(f"  MSE Skip Cols     : {args.mse_skip_first_cols} (NMSE objective)")

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
        num_gpu_blocks_override=8192,
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

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Register P-matrix hooks
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Setting up P-matrix scale hooks...")
    print("=" * 80)

    _scale_list = scale_list
    _q_block_size = args.q_block_size
    _collect_dist = not args.no_collect_p_dist
    _mse_skip = int(max(0, args.mse_skip_first_cols))

    def _setup(model):
        return setup_p_matrix_scale_hooks(
            model,
            _scale_list,
            _q_block_size,
            collect_dist=_collect_dist,
            mse_skip_first_cols=_mse_skip,
        )

    hook_results = llm.apply_model(_setup)
    for i, result in enumerate(hook_results):
        print(f"  Worker {i}: {result}")

    # ------------------------------------------------------------------
    # 4. Single prefill-only forward pass  (max_tokens=1 → no decode)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Running prefill-only forward pass for P-matrix MSE collection...")
    print("=" * 80)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1)
    outputs = llm.generate(prompts, sampling_params)
    print(f"Total outputs generated: {len(outputs)}")

    # ------------------------------------------------------------------
    # 5. Collect results and remove hooks
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Collecting P-matrix scale statistics...")
    print("=" * 80)

    searcher = PMatrixScaleSearcher()
    results_list = llm.apply_model(searcher)

    llm.apply_model(remove_p_matrix_scale_hooks)

    # results_list[0] is rank-0's result; other ranks return None.
    p_search_results = None
    for r in results_list:
        if r is not None:
            p_search_results = r
            break

    if not p_search_results:
        print("\nERROR: No P-matrix statistics collected. Aborting.")
        return

    # ------------------------------------------------------------------
    # 6. Save full search results
    # ------------------------------------------------------------------
    save_json(
        p_search_results,
        args.output_dir,
        "p_matrix_search_results_per_head.json",
        label="P-matrix search results (per Q-head)",
    )

    # ------------------------------------------------------------------
    # 7. Build and save flat scale dict  (same naming as kv_cache_tuned_scales_per_head.json)
    # ------------------------------------------------------------------
    # Key format:  "model.layers.L.self_attn.p_matrix.head_H.scale"
    # (mirrors:    "model.layers.L.self_attn.attn.k_cache.head_H.scale")
    tuned_scales = {}
    for layer_key, info in p_search_results.items():
        best_scales = info["best_scale_per_head"]
        # Normalise key: remove duplicate "attn.attn" if present.
        base_key = layer_key.replace("attn.attn", "attn")
        for head_idx, scale_val in enumerate(best_scales):
            save_key = f"{base_key}.p_matrix.head_{head_idx}.scale"
            tuned_scales[save_key] = scale_val

    save_json(
        tuned_scales,
        args.output_dir,
        "p_matrix_tuned_scales_per_head.json",
        label="Tuned P-matrix scales (per Q-head)",
    )

    # ------------------------------------------------------------------
    # 8. Extract and save P-value distribution statistics.
    #    Two outputs:
    #      - p_matrix_distribution_per_head.json : full per-head hist + stats
    #      - p_matrix_distribution_summary.json  : cross-head aggregated
    #        summary per layer (easier to eyeball).
    # ------------------------------------------------------------------
    p_dist_per_head = {}
    p_dist_summary = {}
    hist_edges_ref = None

    for layer_key, info in p_search_results.items():
        p_dist = info.get("p_dist")
        if not p_dist:
            continue
        hist_edges_ref = p_dist["hist_edges"]
        p_dist_per_head[layer_key] = p_dist

        # Aggregate across heads for quick summary.
        hist_counts = p_dist["hist_counts_per_head"]  # list[list[int]]
        means = p_dist["mean_per_head"]
        stds = p_dist["std_per_head"]
        maxes = p_dist["max_per_head"]

        num_heads = len(means)
        # Layer-level histogram = sum of per-head counts.
        if hist_counts:
            layer_hist_counts = [sum(col) for col in zip(*hist_counts)]
            total = sum(layer_hist_counts) or 1
            layer_hist_prob = [c / total for c in layer_hist_counts]
        else:
            layer_hist_counts = []
            layer_hist_prob = []

        p_dist_summary[layer_key] = {
            "num_heads": num_heads,
            "mean_of_head_means": sum(means) / max(len(means), 1),
            "mean_of_head_stds": sum(stds) / max(len(stds), 1),
            "max_across_heads": max(maxes) if maxes else 0.0,
            "min_of_head_maxes": min(maxes) if maxes else 0.0,
            "layer_hist_counts": layer_hist_counts,
            "layer_hist_prob": layer_hist_prob,
        }

    if p_dist_per_head:
        save_json(
            {
                "hist_edges": hist_edges_ref,
                "per_layer_per_head": p_dist_per_head,
            },
            args.output_dir,
            "p_matrix_distribution_per_head.json",
            label="P-matrix value distribution (per Q-head)",
        )
        save_json(
            {
                "hist_edges": hist_edges_ref,
                "per_layer": p_dist_summary,
            },
            args.output_dir,
            "p_matrix_distribution_summary.json",
            label="P-matrix value distribution (layer summary)",
        )

        # Print a concise human-readable summary.
        print("\n" + "=" * 80)
        print("P-value distribution summary (aggregated across heads per layer):")
        print("=" * 80)
        if hist_edges_ref is not None:
            edge_strs = [f"{e:.0e}" if e > 0 and e < 1e-2 else f"{e:.3g}" for e in hist_edges_ref]
            bin_labels = [
                f"[{edge_strs[i]},{edge_strs[i + 1]})" for i in range(len(edge_strs) - 1)
            ]
            header = f"{'layer':60s}  {'mean':>8s}  {'std':>8s}  {'max':>8s}"
            print(header)
            print("-" * len(header))
            # Print first / middle / last layer only (save log space).
            layer_names = list(p_dist_summary.keys())
            show_idx = sorted(
                set(
                    [
                        0,
                        len(layer_names) // 4,
                        len(layer_names) // 2,
                        3 * len(layer_names) // 4,
                        len(layer_names) - 1,
                    ]
                )
            )
            for li in show_idx:
                if li >= len(layer_names):
                    continue
                ln = layer_names[li]
                s = p_dist_summary[ln]
                print(
                    f"{ln[:60]:60s}  "
                    f"{s['mean_of_head_means']:.2e}  "
                    f"{s['mean_of_head_stds']:.2e}  "
                    f"{s['max_across_heads']:.2e}"
                )
            # Show the aggregated histogram for the first layer as a
            # representative example.
            ln0 = layer_names[0]
            print(f"\nExample bin probabilities for layer '{ln0}':")
            probs = p_dist_summary[ln0]["layer_hist_prob"]
            for label, p in zip(bin_labels, probs):
                bar = "#" * int(p * 80)
                print(f"  {label:24s}  {p * 100:6.2f}%  {bar}")

    print("\n" + "=" * 80)
    print("P-matrix scale search completed!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
