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

from vllm import LLM, SamplingParams

from angelslim.compressor.quant import (
    get_activation_stats,
    get_moe_stats,
    print_activation_stats,
    print_moe_stats,
    setup_activation_hooks,
)
from angelslim.engine import Engine


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="vLLM Calibration Tool - Collect activation and MoE expert statistics"
    )

    # Model configuration
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model directory"
    )
    parser.add_argument(
        "--ptq-data-path",
        type=str,
        required=True,
        help="Path to the PTQ calibration data (JSONL format)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save output statistics"
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

    # Debug options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging (print detailed statistics during all-reduce)",
    )

    return parser.parse_args()


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

    # Create LLM instance
    llm = LLM(
        model=args.model_path,
        load_format="dummy" if args.skip_weight_loading else "auto",
        disable_log_stats=False,
        enforce_eager=True,
        enable_chunked_prefill=False,
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
        print("Use --skip-weight-loading flag to enable this mode.")
        print("!" * 80 + "\n")

    # Setup activation hooks on all workers
    print("\n" + "=" * 80)
    print("Setting up activation hooks...")
    print("=" * 80)
    hook_results = llm.apply_model(setup_activation_hooks)
    for i, result in enumerate(hook_results):
        print(f"Worker {i}: {result}")

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
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1,
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
        print(f"[{i+1}] Output: {generated_text!r}")
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

    print("\n" + "=" * 80)
    print("Calibration completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
