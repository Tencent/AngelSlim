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
import random
import time
from typing import Any, Dict, List

import numpy as np
import ray
import shortuuid
import torch
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

from angelslim.compressor.speculative.inference.models import Eagle3Model

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
        self.total_token = args.total_token
        self.depth = args.depth
        self.top_k = args.top_k
        self.early_stop_method = args.early_stop_method

    def _get_question_file_path(self, args: argparse.Namespace) -> str:
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(os.path.dirname(script_dir))
        return f"{parent_dir}/data/{args.bench_name}/question.jsonl"

    def _get_answer_file_path(self, args: argparse.Namespace) -> str:
        if args.answer_file:
            return args.answer_file

        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(os.path.dirname(script_dir))
        return f"{parent_dir}/output/{args.bench_name}/{self.model_id}.jsonl"


def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def initialize_model(config: EvaluationConfig) -> Eagle3Model:
    """Initialize and return the Eagle3 model"""
    model = Eagle3Model.from_pretrained(
        base_model_path=config.base_model_path,
        eagle_model_path=config.eagle_model_path,
        total_token=config.total_token,
        depth=config.depth,
        top_k=config.top_k,
        device_map="auto",
        torch_dtype="auto",
        early_stop_method=config.early_stop_method,
    )
    model.eval()
    print(f"Model training state: {model.training}")
    print(f'CUDA VISIBLE DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES")}')
    return model


def process_conversation_turn(
    model: Eagle3Model,
    tokenizer: Any,
    conv: List[Dict[str, str]],
    qs: str,
    temperature: float,
) -> Dict[str, Any]:
    """Process a single conversation turn"""
    conv.append({"role": "user", "content": qs})
    conversation = tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )

    input_ids = tokenizer(
        conversation, return_tensors="pt", max_length=2048, add_special_tokens=False
    ).input_ids

    torch.cuda.synchronize()
    start_time = time.time()

    output_ids, new_token, idx, accept_length_list = model.eagle_generate(
        torch.as_tensor(input_ids).cuda(), temperature=temperature, log=True
    )

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    output_ids = output_ids[0][len(input_ids[0]) :]

    output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for special_tok in special_token:
                output = output.replace(special_tok, "")
        else:
            output = output.replace(special_token, "")
    output = output.strip()

    conv.append({"role": "assistant", "content": output})

    return {
        "output": output,
        "idx": int(idx),
        "new_token": int(new_token),
        "wall_time": total_time,
        "accept_length_list": accept_length_list,
    }


def generate_answer_for_question(
    model: Eagle3Model,
    tokenizer: Any,
    question: Dict[str, Any],
    num_choices: int,
    temperature: float,
) -> List[Dict[str, Any]]:
    """Generate answers for a single question with multiple choices"""
    choices = []
    for i in range(num_choices):
        torch.manual_seed(i)
        conv = [SYSTEM_PROMPT]
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        accept_length_lists = []

        for qs in question["turns"]:
            result = process_conversation_turn(model, tokenizer, conv, qs, temperature)
            turns.append(result["output"])
            idxs.append(result["idx"])
            new_tokens.append(result["new_token"])
            wall_time.append(result["wall_time"])
            accept_length_lists += result["accept_length_list"]

        choices.append(
            {
                "index": i,
                "turns": turns,
                "idxs": idxs,
                "new_tokens": new_tokens,
                "wall_time": wall_time,
                "accept_length": accept_length_lists,
            }
        )

    return choices


def warmup_model(
    model: Eagle3Model, tokenizer: Any, question: Dict[str, Any], temperature: float
) -> None:
    """Warm up the model before actual evaluation"""
    for _ in range(3):
        torch.manual_seed(0)
        conv = [SYSTEM_PROMPT]
        for qs in question["turns"]:
            process_conversation_turn(model, tokenizer, conv, qs, temperature)
    print("Warmup done")


@torch.inference_mode()
def get_model_answers(
    model_id: str,
    questions: List[Dict[str, Any]],
    answer_file: str,
    num_choices: int,
    temperature: float,
    args: argparse.Namespace,
) -> None:
    """Generate answers for a batch of questions"""
    config = EvaluationConfig(args)
    model = initialize_model(config)
    tokenizer = model.get_tokenizer()

    if questions:
        warmup_model(model, tokenizer, questions[0], temperature)

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    for question in tqdm(questions):
        choices = generate_answer_for_question(
            model, tokenizer, question, num_choices, temperature
        )

        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def run_evaluation(config: EvaluationConfig, args: argparse.Namespace) -> None:
    """Run the evaluation with optional distributed processing"""
    questions = load_questions(
        config.question_file, args.question_begin, args.question_end
    )

    use_ray = args.num_gpus_total // args.num_gpus_per_model > 1
    get_answers_func = (
        ray.remote(num_gpus=args.num_gpus_per_model)(get_model_answers).remote
        if use_ray
        else get_model_answers
    )

    chunk_size = len(questions) // (args.num_gpus_total // args.num_gpus_per_model)
    ans_handles = [
        get_answers_func(
            config.model_id,
            questions[i : i + chunk_size],
            config.answer_file,
            config.num_choices,
            config.temperature,
            args,
        )
        for i in range(0, len(questions), chunk_size)
    ]

    if use_ray:
        ray.get(ans_handles)


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
        "--eagle-model-path",
        type=str,
        default="down_checkpoints/LC70B",
        help="Path to the weights (local folder or Hugging Face repo ID)",
    )
    parser.add_argument("--base-model-path", type=str, default="")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
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
        "--total-token", type=int, default=60, help="Total nodes in draft tree"
    )
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--num-choices", type=int, default=1, help="Number of completion choices"
    )
    parser.add_argument(
        "--num-gpus-per-model", type=int, default=1, help="GPUs per model"
    )
    parser.add_argument("--num-gpus-total", type=int, default=1, help="Total GPUs")
    parser.add_argument("--max-gpu-memory", type=str, help="Max GPU memory per GPU")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-method", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Main execution function"""
    args = parse_args()
    setup_seed(args.seed)

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        ray.init()

    config = EvaluationConfig(args)
    os.makedirs(os.path.dirname(config.answer_file), exist_ok=True)
    print(f"Output to {config.answer_file}")

    run_evaluation(config, args)
    reorg_answer_file(config.answer_file)


if __name__ == "__main__":
    main()
