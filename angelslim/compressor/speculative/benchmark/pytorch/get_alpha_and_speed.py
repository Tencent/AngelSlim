#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark utilities for speculative decoding performance analysis.
Includes functions for calculating acceptance length and speedup ratio.
"""

import argparse
import json

import numpy as np
from transformers import AutoTokenizer


def calculate_acceptance_length(input_file: str) -> float:
    """
    Calculate average acceptance length from benchmark results.

    Args:
        input_file: Path to JSONL file containing benchmark results

    Returns:
        Average acceptance length
    """
    with open(input_file, "r") as f:
        lines = f.readlines()

    print(f"Number of samples: {len(lines)}")
    avg_accept_length = 0.0

    for line in lines:
        data = json.loads(line)
        accept_lengths = data["choices"][0]["accept_length"]
        avg_accept_length += sum(accept_lengths) / len(accept_lengths) + 1

    avg_accept_length /= len(lines)
    return avg_accept_length


def calculate_speedup_ratio(
    model_path: str, baseline_json: str, eagle_json: str
) -> float:
    """
    Calculate speedup ratio between baseline and speculative decoding.

    Args:
        model_path: Path to HuggingFace model for tokenization
        baseline_json: Path to baseline benchmark results
        eagle_json: Path to speculative decoding benchmark results

    Returns:
        Speedup ratio (eagle speed / baseline speed)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Process speculative decoding results
    eagle_speeds = []
    with open(eagle_json, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            tokens = sum(data["choices"][0]["new_tokens"])
            times = sum(data["choices"][0]["wall_time"])
            eagle_speeds.append(tokens / times)

    # Process baseline results
    baseline_speeds = []
    total_time = 0.0
    total_token = 0
    with open(baseline_json, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            answers = data["choices"][0]["turns"]
            tokens = sum(len(tokenizer(ans).input_ids) - 1 for ans in answers)
            times = sum(data["choices"][0]["wall_time"])
            baseline_speeds.append(tokens / times)
            total_time += times
            total_token += tokens

    return np.array(eagle_speeds).mean() / np.array(baseline_speeds).mean()


def main():
    """Main entry point for the benchmark analysis tool."""
    parser = argparse.ArgumentParser(
        description="Benchmark analysis for speculative decoding performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Acceptance length subcommand
    accept_parser = subparsers.add_parser(
        "acceptance", help="Calculate average acceptance length"
    )
    accept_parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL file with benchmark results",
    )

    # Speedup ratio subcommand
    speed_parser = subparsers.add_parser(
        "speedup",
        help="Calculate speedup ratio between baseline and speculative decoding",
    )
    speed_parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to HuggingFace model for tokenization",
    )
    speed_parser.add_argument(
        "--baseline_json",
        type=str,
        required=True,
        help="Baseline benchmark results JSON file",
    )
    speed_parser.add_argument(
        "--eagle_json",
        type=str,
        required=True,
        help="Speculative decoding benchmark results JSON file",
    )

    args = parser.parse_args()

    if args.command == "acceptance":
        avg_length = calculate_acceptance_length(args.input_file)
        print(f"Average acceptance length: {avg_length:.2f}")
    elif args.command == "speedup":
        ratio = calculate_speedup_ratio(
            args.model_path, args.baseline_json, args.eagle_json
        )
        print(f"Speedup ratio: {ratio:.2f}x")


if __name__ == "__main__":
    main()
