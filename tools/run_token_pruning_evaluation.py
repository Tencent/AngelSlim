# Copyright 2026 Tencent Inc. All Rights Reserved.
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
Universal Metadata-Driven Pruning Evaluation Script.
"""

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List

# Import lmms-eval components
import torch
import yaml
from lmms_eval import evaluator
from loguru import logger as eval_logger

# Import Transformers components
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from angelslim.compressor.token_compressor.adapter import UniversalPruningAdapter

# Import AngelSlim core components
from angelslim.compressor.token_compressor.base.config import TokenCompressorConfig
from angelslim.compressor.token_compressor.utils.eval_utils import PruningModel


def calculate_theoretical_speedup(
    total_layers: int, config: TokenCompressorConfig
) -> Dict[str, Any]:
    """
    Estimates theoretical computation reduction based on the TokenCompressorConfig.

    Args:
        total_layers (int): The total number of layers in the LLM backbone.
        config (TokenCompressorConfig): The pruning configuration object.

    Returns:
        Dict[str, Any]: Metadata containing cost_ratio and speedup_factor.
    """
    cost_ratio = 1.0
    note = ""
    strategies = config.strategies

    # Case 1: Global Compression (Occurs once before the backbone)
    if "global" in strategies:
        strategy = strategies["global"]
        ratio = strategy.params.get("ratio", 0.0)
        cost_ratio = 1.0 - ratio
        note = f"Estimated via global ratio: {ratio}"

    # Case 2: Layer-wise Pruning (Progressive reduction)
    else:
        pruning_points = []
        for stage, strat in strategies.items():
            if isinstance(stage, int):
                r = strat.params.get("ratio", 0.0)
                pruning_points.append((stage, r))

        if pruning_points:
            pruning_points.sort()
            remaining_comp = 0.0
            current_token_ratio = 1.0
            last_layer = -1

            for layer_idx, r in pruning_points:
                # Computation for layers before this pruning point
                layers_count = layer_idx - last_layer
                remaining_comp += layers_count * current_token_ratio
                # Apply pruning factor for subsequent layers
                current_token_ratio *= 1.0 - r
                last_layer = layer_idx

            # Final segment computation
            remaining_comp += (total_layers - (last_layer + 1)) * current_token_ratio
            cost_ratio = remaining_comp / total_layers if total_layers > 0 else 1.0
            note = f"Estimated via {len(pruning_points)} pruning stages"

    speedup = 1.0 / cost_ratio if cost_ratio > 0 else float("inf")

    return {
        "cost_ratio": round(cost_ratio, 4),
        "speedup_factor": round(speedup, 2),
        "note": note,
    }


def run_single_config_eval(
    model: torch.nn.Module,
    processor: Any,
    tokenizer: Any,
    config_path: str,
    tasks: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Runs evaluation for a single YAML configuration.

    Args:
        model: Loaded HF Model.
        processor: Loaded HF Processor.
        tokenizer: Loaded HF Tokenizer.
        config_path (str): Path to the strategy YAML.
        tasks (List[str]): List of lmms-eval task names.
        args: Command line arguments.

    Returns:
        Dict[str, Any]: Evaluation results and config metadata.
    """
    eval_logger.info(f"Processing configuration: {config_path}")

    # 1. Parse YAML metadata and strategy config
    with open(config_path, "r", encoding="utf-8") as f:
        raw_yaml = yaml.safe_load(f)

    mapping_data = raw_yaml.get("model_mapping", [])
    if not mapping_data:
        raise ValueError(f"Missing 'model_mapping' in YAML: {config_path}")

    strategy_config = TokenCompressorConfig.from_yaml(config_path)

    # 2. Activate the Universal Adapter (Sequential Wrapping)
    adapter = UniversalPruningAdapter(
        model=model, strategy_config=strategy_config, raw_map_data=mapping_data
    )
    wrapped_model = adapter.wrap_model()

    # 3. Calculate metrics for report
    total_llm_layers = getattr(model.config, "num_hidden_layers", 0)
    accel_info = calculate_theoretical_speedup(total_llm_layers, strategy_config)

    # 4. Initialize the general PruningModel wrapper for lmms-eval
    lmm_obj = PruningModel(
        model_instance=wrapped_model,
        processor_instance=processor,
        tokenizer_instance=tokenizer,
        batch_size=args.batch_size,
        use_cache=True,
    )

    # 5. Execute benchmark
    try:
        results = evaluator.simple_evaluate(
            model=lmm_obj,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            log_samples=args.log_samples,
            device="cuda",
        )

        return {
            "config_path": config_path,
            "results": results["results"],
            "acceleration_info": accel_info,
            "strategies": str(strategy_config.strategies),
        }

    finally:
        # 6. Safety Restoration: Restore original module pointers
        eval_logger.info("Reverting model to standard state...")
        adapter.unwrap_model()


def main():
    """Main entry point for the evaluator."""
    parser = argparse.ArgumentParser(description="AngelSlim Universal Pruning Evaluator")

    # Model parameters
    parser.add_argument("--model_path", type=str, required=True, help="HF model ID or local path")
    parser.add_argument("--configs", nargs="+", required=True, help="List of strategy YAML paths")

    # Evaluation parameters
    parser.add_argument("--tasks", nargs="+", default=["textvqa"], help="List of lmms-eval tasks")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (only 1 is recommended)"
    )
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument(
        "--output_dir", type=str, default="./eval_results", help="Directory for results"
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Whether to log detailed sample outputs",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Initialize base model and components once to save time during serial runs
    eval_logger.info(f"Loading base architecture: {args.model_path}")
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            # Pruning hooks require eager implementation to capture attention maps
            attn_implementation="sdpa",
        ).eval()

        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        eval_logger.success("Base architecture loaded successfully.")
    except Exception as e:
        eval_logger.error(f"Critical failure during model loading: {e}")
        raise e

    final_report = {}

    # 2. Iterate through provided strategy configurations
    for i, cfg_path in enumerate(args.configs):
        eval_logger.info(f"Starting Experiment {i+1}/{len(args.configs)}: {cfg_path}")

        try:
            config_result = run_single_config_eval(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                config_path=cfg_path,
                tasks=args.tasks,
                args=args,
            )

            # Map result by filename
            exp_id = os.path.basename(cfg_path)
            final_report[exp_id] = config_result

            # Incremental save to prevent data loss
            live_path = os.path.join(
                args.output_dir, f"results_checkpoint_{int(time.time())}.json"
            )
            with open(live_path, "w", encoding="utf-8") as f:
                json.dump(final_report, f, indent=4, ensure_ascii=False)

        except Exception as e:
            eval_logger.error(f"Execution failed for {cfg_path}: {e}")
            eval_logger.error(traceback.format_exc())
            # Principle 6: Fail fast on internal errors
            continue

    # 3. Save final summary report
    summary_path = os.path.join(args.output_dir, f"final_summary_{int(time.time())}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    eval_logger.success(f"Evaluation complete. Summary saved to: {summary_path}")


if __name__ == "__main__":
    # Custom logging format aligned with project standard
    eval_logger.remove()
    eval_logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{message}</cyan>",
        level="INFO",
    )
    main()
