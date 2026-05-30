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
usage:
# 使用 benchmark 名称（自动从 dataset/ 目录加载）
python3 vllm_spec_benchmark.py \
    --target_model "$MODEL_DIR" \
    --draft_model "$EAGLE_DIR" \
    --dataset "gsm8k" \
    --output_file "$OUTPUT_FILE" \
    --method eagle3 \
    --output_len 1024 \
    --max_num_seqs "$BATCH_SIZE"

# 使用本地文件路径
python3 vllm_spec_benchmark.py \
    --target_model "$MODEL_DIR" \
    --draft_model "$EAGLE_DIR" \
    --dataset "/path/to/local/question.jsonl" \
    --output_file "$OUTPUT_FILE" \
    --method eagle3 \
    --output_len 1024 \
    --max_num_seqs "$BATCH_SIZE"

# 多个数据集（逗号分隔，支持混合使用）
python3 vllm_spec_benchmark.py \
    --target_model "$MODEL_DIR" \
    --draft_model "$EAGLE_DIR" \
    --dataset "mt_bench,gsm8k,/path/to/local/question.jsonl" \
    --output_file "$OUTPUT_FILE" \
    --method eagle3 \
    --output_len 1024 \
    --max_num_seqs "$BATCH_SIZE"
"""

import json
import os
import time
from datetime import datetime

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.v1.metrics.reader import Counter, Vector

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser


def _extract_text_content(content):
    """
    从 content 字段中提取纯文本内容。

    兼容两种格式：
    - 字符串格式: "xxx"
    - 列表格式: [{"type": "text", "text": "xxx"}, ...]

    Args:
        content: 字符串或列表格式的 content

    Returns:
        提取出的纯文本字符串
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 拼接所有 type=text 的文本片段
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "".join(text_parts)
    return str(content)


def _extract_prompts(ds, num_prompts):
    """
    从数据集中提取 prompt 列表，自动识别数据格式。

    支持的格式：
    1. turns 格式: {"turns": ["xxx"], "reference": ["xxx"]}
       - 取第一个 turn 作为 prompt，返回纯文本列表
    2. conversations 格式:
       {"conversations": [
          {"role": "system", "content": "xxx"},
          {"role": "user", "content": "xxx"},
          {"role": "assistant", ...}]
       }
       {"conversations": [{"role": "user", "content": [{"type": "text", "text": "xxx"}]}, ...]}
       - 提取 system + 第一个 user 消息，构建完整的 messages 列表

    Args:
        ds: HuggingFace Dataset 对象
        num_prompts: 最多提取的 prompt 数量

    Returns:
        prompts: 纯文本 prompt 列表（turns 格式）或 messages 列表的列表（conversations 格式）
        format_type: 数据格式类型，"turns" 或 "conversations"
    """
    columns = ds.column_names

    if "turns" in columns:
        # turns 格式：取每条数据的第一个 turn
        return [q[0] for q in ds["turns"][:num_prompts]], "turns"
    elif "conversations" in columns:
        # conversations 格式：提取 system prompt + 第一个 user 消息，构建完整 messages
        prompts = []
        for item in ds[:num_prompts]["conversations"]:
            messages = []
            for msg in item:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    messages.append({"role": "system", "content": _extract_text_content(content)})
                elif role == "user":
                    messages.append({"role": "user", "content": _extract_text_content(content)})
                    break  # 只取到第一个 user 消息
            if messages and messages[-1]["role"] == "user":
                prompts.append(messages)
        return prompts, "conversations"
    elif "messages" in columns:
        # messages 格式（sharegpt_gpt4_qwen 等）：与 conversations 格式类似
        prompts = []
        for item in ds[:num_prompts]["messages"]:
            messages = []
            for msg in item:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    messages.append({"role": "system", "content": _extract_text_content(content)})
                elif role == "user":
                    messages.append({"role": "user", "content": _extract_text_content(content)})
                    break
            if messages and messages[-1]["role"] == "user":
                prompts.append(messages)
        return prompts, "conversations"
    else:
        raise ValueError(
            f"Unsupported dataset format. Expected 'turns', 'conversations' or 'messages' column, "
            f"but found: {columns}"
        )


def load_benchmark_dataset(dataset, num_prompts, tokenizer, dataset_base_dir):
    """
    Load dataset based on dataset name or local file path

    Args:
        dataset: Benchmark name (folder name under dataset directory) or local JSONL file path.
                 If the path exists as a file, it will be loaded directly;
                 otherwise it is treated as a benchmark name under dataset_base_dir.
        num_prompts: Number of prompts to load
        tokenizer: Tokenizer for encoding prompts
        dataset_base_dir: Base directory containing all benchmark datasets

    Returns:
        prompt_ids: List of tokenized prompt ids
        prompts: List of original prompt texts
    """

    # 判断是本地文件路径还是 benchmark 名称
    if os.path.exists(dataset):
        dataset_file = dataset
        print(f"Loading dataset from local path: {dataset_file}")
    else:
        dataset_file = os.path.join(dataset_base_dir, dataset, "question.jsonl")

    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    # Load dataset
    ds = load_dataset("json", data_files=dataset_file)["train"]
    prompts, format_type = _extract_prompts(ds, num_prompts)

    # Tokenize prompts
    if format_type == "conversations":
        # conversations/messages 格式：使用 apply_chat_template 构建包含 system prompt 的完整 prompt
        prompt_ids = []
        for messages in prompts:
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            prompt_ids.append(token_ids)
        # 将 messages 列表转为可读文本用于日志
        prompts_text = [
            next(
                (msg["content"] for msg in msgs if msg["role"] == "user"),
                "",
            )
            for msgs in prompts
        ]
    else:
        # turns 格式：直接编码纯文本
        # add_special_tokens is False to avoid adding bos twice when using chat templates
        prompt_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
        prompts_text = prompts

    print(f"Loaded {len(prompt_ids)} prompts from: {dataset_file} (dataset={dataset})")
    print(
        "Average prompt length: " f"{sum(len(p) for p in prompt_ids) / len(prompt_ids):.2f} tokens"
    )

    return prompt_ids, prompts_text


def save_stats_to_jsonl(stats, output_file):
    """
    Save benchmark statistics to a jsonl file

    Args:
        stats: Dictionary containing benchmark statistics
        output_file: Path to the output jsonl file
    """
    # Create directory if it doesn't exist
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )

    # Append stats to jsonl file
    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Stats saved to: {output_file}")


def parse_args():
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help="Benchmark name or local JSONL file path. Supports comma-separated values "
        "for multiple datasets (e.g. 'mt_bench,gsm8k,/path/to/local.jsonl'). "
        "If the value is an existing file path, it will be loaded directly; "
        "otherwise it is treated as a benchmark name under the dataset/ directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=80,
        help="Number of prompts to load from the dataset",
    )

    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--method",
        type=str,
        default="eagle3",
        choices=["ngram", "eagle", "eagle3", "mtp", "ar"],
    )
    parser.add_argument("--num_spec_tokens", type=int, default=2)
    parser.add_argument("--prompt_lookup_max", type=int, default=5)
    parser.add_argument("--prompt_lookup_min", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--enable_chunked_prefill", action="store_true")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--print_output", action="store_true")
    parser.add_argument("--output_len", type=int, default=1024)
    parser.add_argument("--max_num_seqs", type=int, default=1)
    parser.add_argument(
        "--output_file",
        type=str,
        default="benchmark_stats.jsonl",
        help="Path to save benchmark statistics in jsonl format",
    )
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--draft_model", type=str, default="AngelSlim/Qwen3-1.7B_eagle3")
    return parser.parse_args()


def run_benchmark(args, dataset_name, llm, tokenizer, sampling_params):
    """
    Run benchmark for a single dataset

    Args:
        args: Command line arguments
        dataset_name: Benchmark name or local file path
        llm: Initialized LLM model
        tokenizer: Tokenizer instance
        sampling_params: Sampling parameters

    Returns:
        stats: Dictionary containing all benchmark statistics
    """
    print("\n" + "=" * 80)
    print(f"Running benchmark: {dataset_name}")
    print("=" * 80)

    # Load dataset
    current_file_path = os.path.abspath(__file__)
    dataset_base_dir = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), "dataset")
    prompt_ids, _ = load_benchmark_dataset(
        dataset=dataset_name,
        num_prompts=args.num_prompts,
        tokenizer=tokenizer,
        dataset_base_dir=dataset_base_dir,
    )

    # Generate outputs
    tic = time.perf_counter()
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=x) for x in prompt_ids],
        sampling_params=sampling_params,
    )
    latency = time.perf_counter() - tic

    # Print generated text if requested
    if args.print_output:
        for i, output in enumerate(outputs):
            print("-" * 50)
            print(f"Prompt {i + 1}: {output.prompt}")
            print(f"Generated text: {output.outputs[0].text}")
            print("-" * 50)

    # Get and process metrics
    try:
        metrics = llm.get_metrics()
    except AssertionError:
        print("Metrics are not supported in the V0 engine.")
        return None

    total_num_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * args.num_spec_tokens

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    total_num_input_tokens = sum(len(p) for p in prompt_ids)
    num_prompts = len(prompt_ids)
    output_throughput = total_num_output_tokens / latency
    request_throughput = num_prompts / latency
    avg_time_per_sample = latency / num_prompts if num_prompts > 0 else 0
    avg_input_tokens = total_num_input_tokens / num_prompts if num_prompts > 0 else 0
    avg_output_tokens = total_num_output_tokens / num_prompts if num_prompts > 0 else 0
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1

    # Calculate acceptance rate at each position
    acceptance_rates = {}
    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        acceptance_rates[f"acceptance_rate_pos_{i}"] = round(acceptance_rate, 4)

    # Prepare statistics dictionary
    stats = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "model_dir": args.target_model,
        "eagle_dir": args.draft_model if args.method in ["eagle", "eagle3"] else None,
        "method": args.method,
        "num_spec_tokens": args.num_spec_tokens,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "temperature": args.temp,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "total_num_input_tokens": total_num_input_tokens,
        "total_num_output_tokens": total_num_output_tokens,
        "latency_seconds": round(latency, 2),
        "output_throughput_tokens_per_sec": round(output_throughput, 2),
        "request_throughput_per_sec": round(request_throughput, 2),
        "avg_time_per_sample": round(avg_time_per_sample, 4),
        "avg_input_tokens": round(avg_input_tokens, 2),
        "avg_output_tokens": round(avg_output_tokens, 2),
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "mean_acceptance_length": round(acceptance_length, 4),
        **acceptance_rates,
    }

    # Print statistics
    print("-" * 50)
    print(f"Dataset: {dataset_name}")
    print(f"total_num_input_tokens: {total_num_input_tokens}")
    print(f"total_num_output_tokens: {total_num_output_tokens}")
    print(f"latency: {latency:.2f} s")
    print(f"output_throughput: {output_throughput:.2f} tokens/s")
    print(f"request_throughput: {request_throughput:.2f} requests/s")
    print(f"avg_time_per_sample: {avg_time_per_sample:.4f} s")
    print(f"avg_input_tokens: {avg_input_tokens:.2f}")
    print(f"avg_output_tokens: {avg_output_tokens:.2f}")
    print(f"num_drafts: {num_drafts}")
    print(f"num_draft_tokens: {num_draft_tokens}")
    print(f"num_accepted_tokens: {num_accepted_tokens}")
    print(f"mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    # Print acceptance rate at each token position
    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        print(f"acceptance at token {i}: {acceptance_rate:.2f}")

    return stats


def calculate_average_stats(all_stats, args):
    """
    Calculate average statistics from multiple benchmark results

    Args:
        all_stats: List of statistics dictionaries from all benchmarks
        args: Command line arguments

    Returns:
        avg_stats: Dictionary containing average statistics
    """
    # Define numeric fields to calculate average
    numeric_fields = [
        "total_num_input_tokens",
        "total_num_output_tokens",
        "latency_seconds",
        "output_throughput_tokens_per_sec",
        "request_throughput_per_sec",
        "avg_time_per_sample",
        "avg_input_tokens",
        "avg_output_tokens",
        "num_drafts",
        "num_draft_tokens",
        "num_accepted_tokens",
        "mean_acceptance_length",
    ]

    # Add acceptance rate fields
    if all_stats[0].get("acceptance_rate_pos_0") is not None:
        for i in range(args.num_spec_tokens):
            numeric_fields.append(f"acceptance_rate_pos_{i}")

    # Initialize average stats dictionary
    avg_stats = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "AVERAGE",
        "model_dir": args.target_model,
        "eagle_dir": args.draft_model if args.method in ["eagle", "eagle3"] else None,
        "method": args.method,
        "num_spec_tokens": args.num_spec_tokens,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "temperature": args.temp,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_benchmarks": len(all_stats),
    }

    # Calculate average for each numeric field
    for field in numeric_fields:
        values = [s[field] for s in all_stats if field in s]
        if values:
            avg_value = sum(values) / len(values)
            avg_stats[field] = (
                round(avg_value, 4) if isinstance(values[0], float) else int(avg_value)
            )

    return avg_stats


def print_average_statistics(avg_stats):
    """
    Print average statistics in a formatted way

    Args:
        avg_stats: Dictionary containing average statistics
    """
    print("\n" + "=" * 80)
    print("AVERAGE STATISTICS")
    print("=" * 80)
    print(f"Number of benchmarks: {avg_stats['num_benchmarks']}")
    print(
        "Average output throughput: "
        f"{avg_stats['output_throughput_tokens_per_sec']:.2f} tokens/s"
    )
    print(f"Average acceptance length: {avg_stats['mean_acceptance_length']:.4f}")
    print("=" * 80)


def print_summary(all_stats, output_file):
    """
    Print summary of all benchmark results

    Args:
        all_stats: List of statistics dictionaries from all benchmarks
        output_file: Path to the output file
    """
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for stats in all_stats:
        print(f"{stats['dataset']}: " f"acceptance_length = {stats['mean_acceptance_length']:.2f}")
    print("=" * 80)
    print(f"\nAll statistics saved to: {output_file}")


def main(args):
    # Set endpoint type for dataset processing
    args.endpoint_type = "openai-chat"

    # Set default model directory
    model_dir = args.target_model
    if args.target_model is None:
        model_dir = "meta-llama/Llama-3.1-8B-Instruct"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Configure speculative decoding
    if args.method == "eagle" or args.method == "eagle3":
        eagle_dir = args.draft_model
        if args.method == "eagle" and eagle_dir is None:
            eagle_dir = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
        elif args.method == "eagle3" and eagle_dir is None:
            eagle_dir = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"

        speculative_config = {
            "method": args.method,
            "model": eagle_dir,
            "num_speculative_tokens": args.num_spec_tokens,
        }
    elif args.method == "ngram":
        speculative_config = {
            "method": "ngram",
            "num_speculative_tokens": args.num_spec_tokens,
            "prompt_lookup_max": args.prompt_lookup_max,
            "prompt_lookup_min": args.prompt_lookup_min,
        }
    elif args.method == "mtp":
        speculative_config = {
            "method": "mtp",
            "num_speculative_tokens": args.num_spec_tokens,
        }
    elif args.method == "ar":
        speculative_config = None
    else:
        raise ValueError(f"unknown method: {args.method}")

    # Initialize LLM
    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=0.8,
        speculative_config=speculative_config,
        disable_log_stats=False,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_cudagraph_capture_size=24,
    )

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.output_len,
    )

    # Parse dataset names (support multiple datasets separated by comma)
    dataset_names = [name.strip() for name in args.dataset.split(",")]

    # Run benchmarks
    all_stats = []
    for dataset_name in dataset_names:
        stats = run_benchmark(
            args=args,
            dataset_name=dataset_name,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
        )
        if stats is not None:
            all_stats.append(stats)
            # Save stats to jsonl file
            save_stats_to_jsonl(stats, args.output_file)

    # Calculate and save average statistics if multiple benchmarks
    if len(all_stats) > 1:
        avg_stats = calculate_average_stats(all_stats, args)
        save_stats_to_jsonl(avg_stats, args.output_file)
        print_average_statistics(avg_stats)

    # Print summary
    print_summary(all_stats, args.output_file)

    return all_stats


if __name__ == "__main__":
    args = parse_args()
    results = main(args)
