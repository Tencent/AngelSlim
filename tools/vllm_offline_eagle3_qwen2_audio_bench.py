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
python3 ./tools/vllm_offline_eagle3_qwen2_audio_bench.py \
    --target_model "$MODEL_DIR" \
    --draft_model "$EAGLE_DIR" \
    --use_eagle \
    --num_spec_tokens 4 \
    --num_prompts 10 \
    --temp 0 \
    --max_num_seqs 1 \
    --output_len 1024 \
    --output_file "$OUTPUT_FILE"
"""

import argparse
import os
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, NamedTuple

from mistral_common.audio import Audio
from vllm import LLM, EngineArgs, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str | None = None
    prompt_token_ids: dict[str, list[int]] | None = None
    multi_modal_data: dict[str, Any] | None = None
    stop_token_ids: list[int] | None = None


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.


# Qwen2-Audio
def run_qwen2_audio(args, question: str, audio_count: int) -> ModelRequestData:
    num_spec_tokens = args.num_spec_tokens

    speculative_config = None
    if args.use_eagle:
        if args.draft_model:
            speculative_config = {
                "method": "eagle3",
                "model": args.draft_model,
                "num_speculative_tokens": num_spec_tokens,
                "prefill_token_shift": False,
            }
        else:
            print(
                "Warning: use_eagle is set but no draft_model provided. "
                "Running without speculative decoding."
            )

    engine_args = EngineArgs(
        model=args.target_model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"audio": audio_count},
        speculative_config=speculative_config,
        enforce_eager=True,
    )

    audio_in_prompt = "".join(
        [
            f"Audio {idx + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
            for idx in range(audio_count)
        ]
    )

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{audio_in_prompt}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_model", type=str, default=None, help="Path to target model"
    )
    parser.add_argument(
        "--draft_model", type=str, default=None, help="Path to draft model"
    )
    parser.add_argument(
        "--use_eagle",
        action="store_true",
        help="Enable speculative decoding with Eagle",
    )
    parser.add_argument(
        "--num_spec_tokens", type=int, default=2, help="Number of speculative tokens"
    )
    parser.add_argument("--max_num_seqs", type=int, default=1)
    parser.add_argument(
        "--max_model_len", type=int, default=16384, help="Maximum model length"
    )
    parser.add_argument(
        "--num_prompts", type=int, default=100, help="Number of prompts to run"
    )
    parser.add_argument("--output_file", type=str, default="None", help="Output file")
    parser.add_argument("--temp", type=float, default=0, help="./results")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set the seed when initializing `vllm.LLM`.",
    )
    parser.add_argument("--output_len", type=int, default=1024)
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size to override the model's default setting. ",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="dataset/librispeech_test/librispeech_eval_10_test.jsonl",
        help="Dataset to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tp is not None and args.tp < 1:
        raise ValueError(
            f"tensor_parallel_size must be a positive integer, " f"got {args.tp}"
        )
    audio_count = 1

    req_data = run_qwen2_audio(
        args, "Transcribe speech to text. <|en|>", audio_count=audio_count
    )

    # Disable other modalities to save memory
    default_limits = {"image": 0, "video": 0, "audio": 0}
    req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
        req_data.engine_args.limit_mm_per_prompt or {}
    )

    engine_args = asdict(req_data.engine_args) | {"seed": args.seed}
    if args.tp is not None:
        engine_args["tensor_parallel_size"] = args.tp
    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        temperature=args.temp,
        max_tokens=args.output_len,
        stop_token_ids=req_data.stop_token_ids,
    )

    import json

    inputs_list = []
    num_prompts = args.num_prompts
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if line_num > num_prompts:
                break
            data_line = json.loads(line)

            mm_data = req_data.multi_modal_data
            if not mm_data:
                mm_data = {}
                if audio_count > 0:
                    audio_path = data_line["conversations"][0]["content"][0]["audio"]
                    audio_path = os.path.join(
                        os.path.dirname(args.test_data_path), audio_path
                    )
                    mm_data = {"audio": [Audio.from_file(audio_path).audio_array]}
            inputs = {"multi_modal_data": mm_data}

            if req_data.prompt:
                inputs["prompt"] = req_data.prompt
            else:
                inputs["prompt_token_ids"] = req_data.prompt_token_ids

            inputs_list.append(inputs)

    tic = time.perf_counter()
    outputs = llm.generate(
        inputs_list,
        sampling_params=sampling_params,
    )
    latency = time.perf_counter() - tic

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

    try:
        metrics = llm.get_metrics()
    except AssertionError:
        print("Metrics are not supported in the V0 engine.")
        return None

    total_num_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs
    )
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    num_spec_tokens = args.num_spec_tokens
    acceptance_counts = [0] * num_spec_tokens

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

    output_throughput = total_num_output_tokens / latency
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1

    # Calculate acceptance rate at each position
    acceptance_rates = {}
    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        acceptance_rates[f"acceptance_rate_pos_{i}"] = round(acceptance_rate, 4)

    # Prepare statistics dictionary
    stats = {
        "timestamp": datetime.now().isoformat(),
        "num_spec_tokens": num_spec_tokens,
        "total_num_output_tokens": total_num_output_tokens,
        "latency_seconds": round(latency, 2),
        "output_throughput_tokens_per_sec": round(output_throughput, 2),
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "mean_acceptance_length": round(acceptance_length, 4),
        **acceptance_rates,
    }

    # Print statistics
    print("-" * 50)
    print(f"total_num_output_tokens: {total_num_output_tokens}")
    print(f"latency: {latency:.2f} s")
    print(f"output_throughput: {output_throughput:.2f} tokens/s")
    print(f"num_drafts: {num_drafts}")
    print(f"num_draft_tokens: {num_draft_tokens}")
    print(f"num_accepted_tokens: {num_accepted_tokens}")
    print(f"mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    # Print acceptance rate at each token position
    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        print(f"acceptance at token {i}: {acceptance_rate:.2f}")

    if args.output_file != "None":
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
