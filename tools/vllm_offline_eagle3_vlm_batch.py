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
for task in "Lin-Chen/MMStar" "HuggingFaceH4/MATH-500" "MMMU/MMMU" "/path/to/local.jsonl"; do
    python3 ./tools/vllm_offline_eagle3_vlm_batch.py \
        --target_model "$MODEL_DIR" \
        --draft_model "$EAGLE_DIR" \
        --use_eagle \
        --num_spec_tokens 4 \
        --dataset  "$task" \
        --num_prompts 80 \
        --temp 0 \
        --max_num_seqs 1 \
        --output_len 1024 \
        --output_file "$OUTPUT_FILE"
done
"""

import argparse
import base64
import json
import os
import time
from io import BytesIO

from datasets import load_dataset
from transformers.image_utils import load_image
from vllm import LLM, SamplingParams


def pil_to_base64(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--draft_model", type=str, default=None, help="Path to draft model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="lmms-lab/textvqa",
        help="Dataset to use: HuggingFace dataset name or local JSONL file path",
    )
    parser.add_argument(
        "--use_eagle",
        action="store_true",
        help="Enable speculative decoding with Eagle",
    )
    parser.add_argument(
        "--output_file", type=str, default="results/qwen3-vl-4b-eagle3-results.jsonl"
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=None,
        help="Number of prompts to run (default: all)",
    )
    parser.add_argument(
        "--num_spec_tokens", type=int, default=2, help="Number of speculative tokens"
    )
    parser.add_argument("--max_num_seqs", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=32768, help="Maximum model length")
    parser.add_argument("--temp", type=float, default=0, help="Number of speculative tokens")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--output_len", type=int, default=1024)
    parser.add_argument(
        "--limit_mm_per_prompt_image",
        type=int,
        default=1,
        help="Maximum number of images per prompt",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=None,
        help="Maximum pixels for image processing (e.g. 602112)",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=1024,
        help="Minimum pixels for image processing (e.g. 1024)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    is_local_dataset = os.path.exists(args.dataset)
    if is_local_dataset:
        print(f"Loading dataset from local path: {args.dataset}")
        ds = load_dataset(
            path="json", data_files=args.dataset, split="train", trust_remote_code=True
        )
    else:
        print(f"Loading {args.dataset} dataset...")
        if args.dataset == "MMMU/MMMU":
            ds = load_dataset(args.dataset, "History", split="test", trust_remote_code=True)
        elif args.dataset == "Lin-Chen/MMStar":
            ds = load_dataset(args.dataset, split="val", trust_remote_code=True)
        elif args.dataset == "opendatalab/OmniDocBench":
            ds = load_dataset(args.dataset, split="train", trust_remote_code=True)
        else:
            ds = load_dataset(args.dataset, split="test", trust_remote_code=True)
    if args.num_prompts is not None:
        ds = ds.select(range(min(args.num_prompts, len(ds))))
    if len(ds) == 0:
        raise ValueError(f"Dataset {args.dataset} is empty")

    print(f"Loaded {len(ds)} samples.")

    prompts = []
    if is_local_dataset:
        for item in ds:
            user_messages = [msg for msg in item["conversations"] if msg.get("role") == "user"]
            if user_messages:
                user_content = user_messages[0].get("content", [])

                prompt_content = []
                for content_item in user_content:
                    if content_item.get("type") == "text":
                        prompt_content.append(
                            {"type": "text", "text": content_item.get("text", "")}
                        )
                    elif content_item.get("type") == "image":
                        image_path = content_item.get("image", "")
                        img = load_image(image_path)
                        image_url = pil_to_base64(img)
                        prompt_content.append(
                            {"type": "image_url", "image_url": {"url": image_url}}
                        )

                if prompt_content:
                    prompts.append([{"role": "user", "content": prompt_content}])

    elif args.dataset == "lmms-lab/textvqa":
        for item in ds:
            # Convert PIL image to base64
            image_url = pil_to_base64(item["image"])
            prompts.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": item["question"]},
                        ],
                    }
                ]
            )
    elif args.dataset == "MMMU/MMMU":
        for item in ds:
            # Convert PIL image to base64
            image_url = pil_to_base64(item["image_1"])
            question = item["question"].replace("<image 1>", "")
            prompts.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": question},
                        ],
                    }
                ]
            )
    elif args.dataset == "Lin-Chen/MMStar":
        for item in ds:
            # Convert PIL image to base64
            image_url = pil_to_base64(item["image"])
            question = item["question"].replace("<image>", "")
            prompts.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": question},
                        ],
                    }
                ]
            )
    elif args.dataset == "HuggingFaceH4/MATH-500":
        for item in ds:
            prompts.append([{"role": "user", "content": item["problem"]}])
    elif args.dataset == "opendatalab/OmniDocBench":
        for item in ds:
            image_url = pil_to_base64(item["image"])
            prompts.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": "提取并识别图片中的文本。"},
                        ],
                    }
                ]
            )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    speculative_config = None
    if args.use_eagle:
        if args.draft_model:
            speculative_config = {
                "method": "eagle3",
                "model": args.draft_model,
                "num_speculative_tokens": args.num_spec_tokens,
            }
        else:
            print(
                "Warning: use_eagle is set but no draft_model provided. "
                "Running without speculative decoding."
            )

    print(
        f"Initializing LLM with target_model={args.target_model}, "
        f"speculative_config={speculative_config}"
    )

    # Build mm_processor_kwargs based on model type (Qwen3-VL vs others)
    mm_processor_kwargs = None
    if args.max_pixels is not None:
        model_name_lower = args.target_model.lower()
        if "qwen3" in model_name_lower:
            # Qwen3-VL requires both max_pixels and size with shortest_edge/longest_edge
            mm_processor_kwargs = {
                "max_pixels": args.max_pixels,
                "size": {
                    "shortest_edge": (
                        args.min_pixels if args.min_pixels is not None else args.max_pixels
                    ),
                    "longest_edge": args.max_pixels,
                },
            }
        else:
            mm_processor_kwargs = {"max_pixels": args.max_pixels}
        if args.min_pixels is not None and "min_pixels" not in (mm_processor_kwargs or {}):
            mm_processor_kwargs["min_pixels"] = args.min_pixels
        print(f"mm_processor_kwargs: {mm_processor_kwargs}")

    llm = LLM(
        model=args.target_model,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.9,
        speculative_config=speculative_config,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
        disable_log_stats=False,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": args.limit_mm_per_prompt_image},
        mm_processor_kwargs=mm_processor_kwargs,
        disable_chunked_mm_input=False,
    )

    sampling_params = SamplingParams(temperature=args.temp, max_tokens=args.output_len)

    print("Starting generation...")
    start_time = time.perf_counter()
    outputs = llm.chat(prompts, sampling_params=sampling_params)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Generation finished in {total_time:.2f} seconds.")

    # Process results
    results_data = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        if is_local_dataset:
            results_data.append(
                {
                    "index": i,
                    "input_data": ds[i] if i < len(ds) else {},
                    "generated_text": generated_text,
                }
            )
        elif args.dataset == "lmms-lab/textvqa":
            results_data.append(
                {
                    "question_id": ds[i]["question_id"],
                    "image_id": ds[i]["image_id"],
                    "question": ds[i]["question"],
                    "generated_text": generated_text,
                    "answers": ds[i]["answers"],
                }
            )
        elif args.dataset == "HuggingFaceH4/MATH-500":
            results_data.append(
                {
                    "problem": ds[i]["problem"],
                    "generated_text": generated_text,
                    "solution": ds[i].get("solution", ""),
                    "answer": ds[i].get("answer", ""),
                }
            )
        elif args.dataset == "MMMU/MMMU":
            results_data.append(
                {
                    "id": ds[i]["id"],
                    "question": ds[i]["question"],
                    "generated_text": generated_text,
                    "answer": ds[i]["answer"],
                }
            )
        elif args.dataset == "Lin-Chen/MMStar":
            results_data.append(
                {
                    "id": ds[i]["index"],
                    "question": ds[i]["question"],
                    "generated_text": generated_text,
                    "answer": ds[i]["answer"],
                }
            )
        elif args.dataset == "opendatalab/OmniDocBench":
            results_data.append(
                {
                    "generated_text": generated_text,
                }
            )

    total_num_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    total_num_input_tokens = sum(len(output.prompt_token_ids) for output in outputs)

    num_prompts = len(prompts)
    output_throughput = total_num_output_tokens / total_time
    request_throughput = num_prompts / total_time
    avg_input_tokens = total_num_input_tokens / num_prompts if num_prompts > 0 else 0
    avg_output_tokens = total_num_output_tokens / num_prompts if num_prompts > 0 else 0
    metrics_info = {
        "total_time": total_time,
        "avg_time_per_sample": total_time / num_prompts if num_prompts > 0 else 0,
        "use_eagle": args.use_eagle,
        "output_throughput": output_throughput,
        "request_throughput": request_throughput,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
    }

    if args.use_eagle and speculative_config:
        try:
            metrics = llm.get_metrics()

            total_num_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            num_drafts = 0
            num_accepted_tokens = 0
            acceptance_counts = [0] * args.num_spec_tokens

            for metric in metrics:
                if metric.name == "vllm:spec_decode_num_drafts":
                    num_drafts += metric.value
                elif metric.name == "vllm:spec_decode_num_accepted_tokens":
                    num_accepted_tokens += metric.value
                elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
                    for pos in range(len(metric.values)):
                        acceptance_counts[pos] += metric.values[pos]

            acceptance_rates = {}
            for i in range(len(acceptance_counts)):
                acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
                acceptance_rates[f"acceptance_rate_pos_{i}"] = round(acceptance_rate, 4)
            acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
            metrics_info["mean_acceptance_length"] = acceptance_length
            metrics_info["num_drafts"] = num_drafts
            metrics_info["num_accepted_tokens"] = num_accepted_tokens
            metrics_info["acceptance_rates"] = acceptance_rates

            print(f"Mean acceptance length: {acceptance_length:.2f}")
            print(f"output_throughput: {output_throughput:.2f} tokens/s")
            print(f"request_throughput: {request_throughput:.2f} requests/s")
            print(f"avg_input_tokens: {avg_input_tokens:.1f}")
            print(f"avg_output_tokens: {avg_output_tokens:.1f}")
            print(f"acceptance rates: {acceptance_rates}")
        except Exception as e:
            print(f"Error getting metrics: {e}")

    # Save to file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(
            {"metrics": metrics_info, "results": results_data}, f, indent=4, ensure_ascii=False
        )

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
