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
P-matrix first-N-columns analysis tool.

Independent of the FP8 scale search: this tool only computes P = softmax(QK^T)
and accumulates per-Q-head, per-column statistics on the first ``--num-cols``
columns.  Designed to answer questions like:

    "Is the very first key column much larger than the rest of the leading
     columns? If so, by how much?"

Outputs (rank-0 only):
    - p_first_cols_per_head.json       : full per-layer / per-head / per-column
    - p_first_cols_layer_summary.json  : cross-head aggregated per-layer view

The tool is **prefill-only** (max_tokens=1), so a single forward pass over a
small number of calibration prompts is enough.
"""

import argparse
import json
import os
import platform

from vllm import LLM, SamplingParams

from angelslim.compressor.quant import (
    PMatrixScaleSearcher,
    remove_p_matrix_scale_hooks,
    setup_p_first_cols_hooks,
)
from angelslim.engine import Engine

_original_python_version = platform.python_version


def _patched_python_version():
    return _original_python_version().rstrip("+")


platform.python_version = _patched_python_version


def parse_args():
    parser = argparse.ArgumentParser(
        description="P-matrix first-N-columns analysis tool. "
        "Skips FP8 scale search; only collects per-column stats."
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model directory."
    )
    parser.add_argument(
        "--ptq-data-path",
        type=str,
        required=True,
        help="Path to the PTQ calibration data (JSONL/JSON).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save output JSON files."
    )

    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallel size (default: 1)."
    )
    parser.add_argument(
        "--skip-weight-loading", action="store_true", help="Use dummy weights for fast debug mode."
    )

    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for inference (default: 4)."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of calibration samples (default: 4 - "
        "this analysis is cheap, a handful of prompts is "
        "usually enough).",
    )
    parser.add_argument(
        "--max-length", type=int, default=32768, help="Maximum sequence length (default: 32768)."
    )

    parser.add_argument(
        "--distributed-executor-backend",
        type=str,
        default="ray",
        choices=["ray", "mp"],
        help="Distributed executor backend (default: ray).",
    )

    parser.add_argument(
        "--num-cols",
        type=int,
        default=64,
        help="Number of leading P columns to analyse " "(default: 64).",
    )
    parser.add_argument(
        "--q-block-size",
        type=int,
        default=2048,
        help="Q-token block size for memory-bounded P " "computation (default: 2048).",
    )
    parser.add_argument(
        "--no-per-head-detail",
        action="store_true",
        help="Skip per-Q-head per-column mean/std/max in "
        "the output JSON.  Reduces file size from "
        "O(H_q*N*num_layers) to O(N*num_layers).",
    )

    return parser.parse_args()


def save_json(data, output_dir: str, filename: str, label: str = "data") -> str:
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n{label} saved to: {path}")
    return path


def main():
    args = parse_args()

    print("\nConfiguration:")
    print(f"  Model             : {args.model_path}")
    print(f"  PTQ Data          : {args.ptq_data_path}")
    print(f"  Output Dir        : {args.output_dir}")
    print(f"  TP Size           : {args.tp_size}")
    print(f"  Batch Size        : {args.batch_size}")
    print(f"  Num Samples       : {args.num_samples}")
    print(f"  Max Length        : {args.max_length}")
    print(f"  Skip Wgt Loading  : {args.skip_weight_loading}")
    print(f"  Num Cols          : {args.num_cols}")
    print(f"  Q Block Size      : {args.q_block_size}")
    print(f"  Per-head Detail   : {not args.no_per_head_detail}")

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
    # 3. Register first-N-cols hooks (no scale search, no histogram)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Setting up P-matrix first-N-columns hooks...")
    print("=" * 80)

    _num_cols = args.num_cols
    _q_block_size = args.q_block_size
    _per_head_detail = not args.no_per_head_detail

    def _setup(model):
        return setup_p_first_cols_hooks(
            model,
            num_cols=_num_cols,
            q_block_size=_q_block_size,
            per_head_detail=_per_head_detail,
        )

    hook_results = llm.apply_model(_setup)
    for i, result in enumerate(hook_results):
        print(f"  Worker {i}: {result}")

    # ------------------------------------------------------------------
    # 4. Single prefill-only pass (max_tokens=1)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Running prefill-only forward pass...")
    print("=" * 80)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1)
    outputs = llm.generate(prompts, sampling_params)
    print(f"Total outputs generated: {len(outputs)}")

    # ------------------------------------------------------------------
    # 5. Collect results and remove hooks
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Collecting first-N-columns statistics...")
    print("=" * 80)

    searcher = PMatrixScaleSearcher()  # same callable; layer dict will only
    # contain "p_first_cols" because
    # scale_list is empty.
    results_list = llm.apply_model(searcher)
    llm.apply_model(remove_p_matrix_scale_hooks)

    p_results = None
    for r in results_list:
        if r is not None:
            p_results = r
            break

    if not p_results:
        print("\nERROR: No first-cols statistics collected. Aborting.")
        return

    # ------------------------------------------------------------------
    # 6. Save full per-head results
    # ------------------------------------------------------------------
    save_json(
        p_results,
        args.output_dir,
        "p_first_cols_per_head.json",
        label="P-matrix first-N-cols stats (per Q-head)",
    )

    # ------------------------------------------------------------------
    # 7. Build a compact per-layer summary
    # ------------------------------------------------------------------
    summary = {}
    for layer_key, info in p_results.items():
        fc = info.get("p_first_cols")
        if not fc:
            continue
        mean_per_col_layer = fc["mean_per_col_layer"]  # (N,)
        max_per_col_layer = fc["max_per_col_layer"]  # (N,)
        col_gain_layer = fc.get("col_gain_layer", [])
        sink_ratios = fc.get("sink_ratios", {})
        sink_K_best = fc.get("sink_K_best", 1)
        sink_ratio_best = fc.get("sink_ratio_best", 1.0)
        top_ratio = fc.get("top_col_ratio_layer", 1.0)

        # Per-head sink ratios using the layer's best K, so we can see if
        # this is a property of every head or only a few.
        means_hh = fc.get("mean_per_head_per_col")  # (H_q_total, N) or None
        head_sink_ratios = []
        if means_hh:
            K = sink_K_best
            for row in means_hh:
                if len(row) > K:
                    head_part = sum(row[:K]) / max(K, 1)
                    tail_part = sum(row[K:]) / max(len(row) - K, 1)
                    head_sink_ratios.append(
                        head_part / tail_part if tail_part > 0 else float("inf")
                    )
                else:
                    head_sink_ratios.append(1.0)

        summary[layer_key] = {
            "num_cols": fc["num_cols"],
            "mean_per_col_layer": mean_per_col_layer,
            "max_per_col_layer": max_per_col_layer,
            "col_gain_layer": col_gain_layer,
            "sink_ratios": sink_ratios,
            "sink_K_best": sink_K_best,
            "sink_ratio_best": sink_ratio_best,
            "top_col_ratio_layer": top_ratio,
        }
        if head_sink_ratios:
            summary[layer_key]["head_sink_ratios_min"] = min(head_sink_ratios)
            summary[layer_key]["head_sink_ratios_mean"] = sum(head_sink_ratios) / len(
                head_sink_ratios
            )
            summary[layer_key]["head_sink_ratios_max"] = max(head_sink_ratios)

    save_json(
        summary,
        args.output_dir,
        "p_first_cols_layer_summary.json",
        label="P-matrix first-N-cols stats (layer summary)",
    )

    # ------------------------------------------------------------------
    # 8. Print a concise human-readable table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("First-N-cols summary (selected layers):")
    print("=" * 80)
    layer_names = list(summary.keys())
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
    header = (
        f"{'layer':52s}  {'K_best':>6s}  {'ratio@K':>10s}  "
        f"{'mean[0]':>10s}  {'mean[K..]':>10s}  {'max[0]':>10s}"
    )
    print(header)
    print("-" * len(header))
    for li in show_idx:
        if li >= len(layer_names):
            continue
        ln = layer_names[li]
        s = summary[ln]
        col_means = s["mean_per_col_layer"]
        col_maxes = s["max_per_col_layer"]
        K = s["sink_K_best"]
        if not col_means:
            continue
        m0 = col_means[0]
        m_tail = sum(col_means[K:]) / max(len(col_means) - K, 1) if len(col_means) > K else 0.0
        print(
            f"{ln[:52]:52s}  {K:>6d}  {s['sink_ratio_best']:10.3f}  "
            f"{m0:10.3e}  {m_tail:10.3e}  {col_maxes[0]:10.3e}"
        )

    # Show full first-K_best+4 columns mean/gain for the first layer.
    if layer_names:
        ln0 = layer_names[0]
        s0 = summary[ln0]
        col_means = s0["mean_per_col_layer"]
        col_maxes = s0["max_per_col_layer"]
        col_gains = s0.get("col_gain_layer", [])
        n_show = min(max(s0["sink_K_best"] + 4, 8), len(col_means))
        print(
            f"\nLayer '{ln0}': K_best={s0['sink_K_best']}, "
            f"sink ratios by K = {s0['sink_ratios']}"
        )
        print(f"  {'col':>4s}  {'mean':>12s}  {'max':>12s}  {'gain_vs_tail':>14s}")
        for j in range(n_show):
            g = col_gains[j] if j < len(col_gains) else float("nan")
            print(f"  {j:>4d}  {col_means[j]:12.3e}  {col_maxes[j]:12.3e}  " f"{g:14.3f}")

    print("\n" + "=" * 80)
    print("First-N-columns analysis completed!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
