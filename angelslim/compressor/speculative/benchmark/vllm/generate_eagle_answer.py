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

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from angelslim.utils.lazy_imports import fastchat, shortuuid, vllm

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, racist, sexist, "
        "toxic, dangerous, or illegal content. Please ensure that your responses are "
        "socially unbiased and positive in nature.\n\nIf a question does not make any "
        "sense, or is not factually coherent, explain why instead of answering "
        "something not correct. If you don't know the answer to a question, please "
        "don't share false information."
    ),
}


def calculate_acceptance_length(llm) -> float | None:
    """Calculate average acceptance length from vLLM metrics."""
    try:
        metrics = llm.get_metrics()
    except AssertionError as e:
        print(f"Metrics are not supported: {e}")
        return None

    num_drafts = 0
    num_accepted_tokens = 0

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            print(f"Num drafts: {metric.value}")
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            print(f"Num accepted tokens: {metric.value}")
            num_accepted_tokens += metric.value

    if num_drafts > 0:
        avg_accept_length = 1 + (num_accepted_tokens / num_drafts)
    else:
        avg_accept_length = 1.0

    return avg_accept_length


class EvaluationConfig:
    """Container for evaluation configuration"""

    def __init__(self, args: argparse.Namespace):
        self.base_model_path = args.base_model_path
        self.eagle_model_path = args.eagle_model_path
        self.model_id = f"{args.model_id}-temperature-{args.temperature}"
        self.question_file = self._get_question_file_path(args)
        self.answer_file = self._get_answer_file_path(args)
        self.num_choices = args.num_choices
        self.temperature = args.temperature
        self.max_tokens = args.max_new_token
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.depth = args.depth
        self.batch_size = args.batch_size

    def _get_question_file_path(self, args: argparse.Namespace) -> str:
        """Get question file path"""
        current_file = os.path.abspath(__file__)
        project_root = current_file.split("/AngelSlim/")[0] + "/AngelSlim"
        return os.path.join(project_root, "dataset", args.bench_name, "question.jsonl")

    def _get_answer_file_path(self, args: argparse.Namespace) -> str:
        print(f"Args answer file: {args.answer_file}")
        if args.answer_file:
            return args.answer_file

        current_file = os.path.abspath(__file__)
        project_root = current_file.split("/AngelSlim/")[0] + "/AngelSlim"
        answer_file = os.path.join(
            project_root, "output", args.bench_name, self.model_id, ".jsonl"
        )
        print(f"Answer file path: {answer_file}")
        return answer_file


def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def initialize_model(config: EvaluationConfig, args: argparse.Namespace):
    """Initialize and return the vLLM model with speculative decoding"""
    speculative_config = {
        "method": "eagle3",
        "model": config.eagle_model_path,
        "num_speculative_tokens": config.depth,
    }
    llm = vllm.LLM(
        model=config.base_model_path,
        tensor_parallel_size=args.num_gpus_per_model,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        disable_log_stats=False,
        speculative_config=speculative_config,
    )
    print(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES")}')
    return llm


def process_conversation_turn(
    llm,
    tokenizer: Any,
    conv: List[Dict[str, str]],
    qs: str,
    **kwargs,
) -> Dict[str, Any]:
    """Process a single conversation turn"""
    conv.append({"role": "user", "content": qs})
    conversation = tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True
    )

    sampling_params = vllm.SamplingParams(**kwargs)

    start_time = time.time()
    outputs = llm.generate([conversation], sampling_params)
    total_time = time.time() - start_time

    output = outputs[0].outputs[0].text.strip()
    new_token = len(outputs[0].outputs[0].token_ids)

    # Extract acceptance length from metrics if available
    accept_length_list = []
    if hasattr(outputs[0], "metrics") and outputs[0].metrics:
        # vLLM provides speculative decoding metrics
        metrics = outputs[0].metrics
        if hasattr(metrics, "num_spec_tokens_accepted"):
            accept_length_list = [metrics.num_spec_tokens_accepted]

    conv.append({"role": "assistant", "content": output})

    return {
        "output": output,
        "new_token": new_token,
        "wall_time": total_time,
        "accept_length_list": accept_length_list,
    }


def generate_answer_for_question(
    llm: vllm.LLM,
    tokenizer: Any,
    question: Dict[str, Any],
    num_choices: int,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Generate answers for a single question with multiple choices"""
    choices = []
    for i in range(num_choices):
        torch.manual_seed(i)
        conv = [SYSTEM_PROMPT]
        turns = []
        new_tokens = []
        wall_time = []

        for qs in question["turns"]:
            result = process_conversation_turn(llm, tokenizer, conv, qs, **kwargs)
            turns.append(result["output"])
            new_tokens.append(result["new_token"])
            wall_time.append(result["wall_time"])

        choices.append(
            {
                "index": i,
                "turns": turns,
                "new_tokens": new_tokens,
                "wall_time": wall_time,
            }
        )

    return choices


def warmup_model(
    llm: vllm.LLM,
    tokenizer: Any,
    question: Dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> None:
    """Warm up the model before actual evaluation"""
    for _ in tqdm(range(3), desc="Warming up model"):
        torch.manual_seed(0)
        conv = [SYSTEM_PROMPT]
        for qs in question["turns"]:
            process_conversation_turn(
                llm, tokenizer, conv, qs, temperature=temperature, max_tokens=max_tokens
            )
    print("Warmup done")


def get_model_answers(
    model_id: str,
    questions: List[Dict[str, Any]],
    answer_file: str,
    num_choices: int,
    temperature: float,
    args: argparse.Namespace,
    lock: Optional[mp.Lock] = None,
    results_list: Optional[List] = None,
    device_list: Optional[List] = None,
) -> float | None:
    """Generate answers for a batch of questions."""
    config = EvaluationConfig(args)
    if device_list:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_list))

    llm = initialize_model(config, args)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)

    if questions:
        warmup_model(llm, tokenizer, questions[0], temperature, config.max_tokens)

    print(
        f"Generating {len(questions)} answers to {answer_file}, "
        f"batch_size={config.batch_size}"
    )
    print(
        f"SamplingParams: "
        f"temperature={temperature}, "
        f"max_tokens={config.max_tokens}, "
        f"top_k={config.top_k}, "
        f"top_p={config.top_p}"
    )

    # Group questions by the number of turns
    questions_by_turn_count = {}
    for q in questions:
        turn_count = len(q["turns"])
        if turn_count not in questions_by_turn_count:
            questions_by_turn_count[turn_count] = []
        questions_by_turn_count[turn_count].append(q)

    for turn_count, turn_questions in questions_by_turn_count.items():
        conversation_states = [
            [
                {
                    "conv": [SYSTEM_PROMPT],
                    "turns": [],
                    "new_tokens": [],
                    "wall_time": [],
                }
                for _ in range(num_choices)
            ]
            for _ in turn_questions
        ]

        for i in range(turn_count):
            for batch_start in tqdm(
                range(0, len(turn_questions), args.batch_size),
                desc=f"Generating answers (turn {i + 1}/{turn_count})",
            ):
                batch_end = batch_start + args.batch_size
                batch_questions = turn_questions[batch_start:batch_end]
                batch_states = conversation_states[batch_start:batch_end]

                prompts = []
                for q_idx, question in enumerate(batch_questions):
                    qs = question["turns"][i]
                    for c_idx in range(num_choices):
                        conv = batch_states[q_idx][c_idx]["conv"]
                        # Create a temporary conversation to apply the template
                        temp_conv = conv + [{"role": "user", "content": qs}]
                        prompt = tokenizer.apply_chat_template(
                            temp_conv, tokenize=False, add_generation_prompt=True
                        )
                        prompts.append(prompt)

                sampling_params = vllm.SamplingParams(
                    temperature=temperature,
                    max_tokens=config.max_tokens,
                    top_k=config.top_k,
                    top_p=config.top_p,
                )

                start_time = time.time()
                outputs = llm.generate(prompts, sampling_params)
                total_time = time.time() - start_time

                output_idx = 0
                for q_idx, question in enumerate(batch_questions):
                    qs = question["turns"][i]
                    for c_idx in range(num_choices):
                        state = batch_states[q_idx][c_idx]
                        output = outputs[output_idx].outputs[0].text.strip()
                        new_token = len(outputs[output_idx].outputs[0].token_ids)

                        state["conv"].append({"role": "user", "content": qs})
                        state["conv"].append({"role": "assistant", "content": output})
                        state["turns"].append(output)
                        state["new_tokens"].append(new_token)
                        state["wall_time"].append(total_time / len(prompts))
                        output_idx += 1

        # After all turns, write answers to file
        for q_idx, question in enumerate(turn_questions):
            choices = []
            for c_idx in range(num_choices):
                state = conversation_states[q_idx][c_idx]
                choices.append(
                    {
                        "index": c_idx,
                        "turns": state["turns"],
                        "new_tokens": state["new_tokens"],
                        "wall_time": state["wall_time"],
                    }
                )

            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }

            if lock:
                with lock:
                    with open(os.path.expanduser(answer_file), "a") as fout:
                        fout.write(json.dumps(ans_json) + "\n")
            else:
                with open(os.path.expanduser(answer_file), "a") as fout:
                    fout.write(json.dumps(ans_json) + "\n")

    avg_accept_length = calculate_acceptance_length(llm)
    if results_list is not None:
        results_list.append(avg_accept_length)

    return avg_accept_length


def run_evaluation(config: EvaluationConfig, args: argparse.Namespace) -> List[Any]:
    """Run the evaluation. Standalone execution is single-process."""
    questions = fastchat.llm_judge.common.load_questions(
        config.question_file, args.question_begin, args.question_end
    )

    result = get_model_answers(
        config.model_id,
        questions,
        config.answer_file,
        config.num_choices,
        config.temperature,
        args,
    )
    return [result]


def reorg_answer_file(answer_file: str) -> None:
    """Sort answers by question id and remove duplicates"""
    answers = {}
    with open(answer_file, "r") as fin:
        for line in fin:
            qid = json.loads(line)["question_id"]
            answers[qid] = line

    with open(answer_file, "w") as fout:
        for qid in sorted(answers.keys()):
            fout.write(answers[qid])


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eagle-model-path", type=str, required=True, help="Path to Eagle draft model"
    )
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="")
    parser.add_argument(
        "--bench-name", type=str, default="mt_bench", help="Benchmark question set name"
    )
    parser.add_argument(
        "--question-begin", type=int, help="Begin index of questions (debug)"
    )
    parser.add_argument(
        "--question-end", type=int, help="End index of questions (debug)"
    )
    parser.add_argument("--answer-file", type=str, help="Output answer file path")
    parser.add_argument(
        "--max-new-token", type=int, default=1024, help="Max new generated tokens"
    )
    parser.add_argument(
        "--num-choices", type=int, default=1, help="Number of completion choices"
    )
    parser.add_argument(
        "--num-gpus-per-model", type=int, default=1, help="GPUs per model"
    )
    parser.add_argument("--num-gpus-total", type=int, default=1, help="Total GPUs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Batch size in vLLM offline generation",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--speculative-draft-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for draft model",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function"""
    args = parse_args()
    setup_seed(args.seed)

    config = EvaluationConfig(args)
    os.makedirs(os.path.dirname(config.answer_file), exist_ok=True)
    print(f"Output to {config.answer_file}")

    run_evaluation(config, args)
    reorg_answer_file(config.answer_file)


if __name__ == "__main__":
    main()
