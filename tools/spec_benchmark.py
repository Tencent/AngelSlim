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
import random

import numpy as np
import torch

from angelslim.engine import SpecEngine


def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for SpecEngine"""
    parser = argparse.ArgumentParser(
        description="SpecEngine: Speculative Decoding Benchmark Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--base-model-path", type=str, required=True, help="Path to base model"
    )
    parser.add_argument(
        "--eagle-model-path", type=str, required=True, help="Path to Eagle model"
    )
    parser.add_argument("--model-id", type=str, required=True, help="Model identifier")

    # Benchmark configuration
    parser.add_argument(
        "--bench-name", type=str, default="mt_bench", help="Benchmark dataset name"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eagle", "baseline", "both"],
        default="both",
        help="Benchmark execution mode",
    )
    parser.add_argument("--output-dir", type=str, help="Output directory for results")

    # Generation parameters
    parser.add_argument(
        "--num-choices", type=int, default=1, help="Number of completion choices"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-new-token", type=int, default=1024, help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--total-token", type=int, default=60, help="Total nodes in draft tree"
    )
    parser.add_argument("--depth", type=int, default=5, help="Tree depth")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k sampling")

    # Hardware configuration
    parser.add_argument(
        "--num-gpus-per-model", type=int, default=1, help="Number of GPUs per model"
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="Total number of GPUs"
    )
    parser.add_argument("--max-gpu-memory", type=str, help="Maximum GPU memory per GPU")

    # Question range (for debugging)
    parser.add_argument("--question-begin", type=int, help="Begin index of questions")
    parser.add_argument("--question-end", type=int, help="End index of questions")

    # Other settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-metrics", action="store_true", help="Skip automatic metrics calculation"
    )
    parser.add_argument(
        "--early-stop-method", type=str, default=None, help="Early stopping method"
    )

    return parser.parse_args()


def main():
    """Main entry point for command-line usage"""
    args = parse_args()

    # Set random seed
    setup_seed(args.seed)

    # Create SpecEngine instance
    engine = SpecEngine()

    # Setup benchmark configuration
    config = engine.setup_benchmark(
        base_model_path=args.base_model_path,
        eagle_model_path=args.eagle_model_path,
        model_id=args.model_id,
        bench_name=args.bench_name,
        output_dir=args.output_dir,
        num_choices=args.num_choices,
        temperature=args.temperature,
        max_new_token=args.max_new_token,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        question_begin=args.question_begin,
        question_end=args.question_end,
        calculate_metrics=not args.no_metrics,
        early_stop_method=args.early_stop_method,
    )

    print("Starting benchmark with configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Base Model: {args.base_model_path}")
    print(f"  Eagle Model: {args.eagle_model_path}")
    print(f"  Output Directory: {config.output_dir}")

    # Run benchmark based on mode
    if args.mode == "eagle":
        results = engine.run_eagle_benchmark()
    elif args.mode == "baseline":
        results = engine.run_baseline_benchmark()
    else:  # both
        results = engine.run_full_benchmark()

    # Print performance report
    print("\n" + engine.get_performance_report())

    return results


if __name__ == "__main__":
    main()
