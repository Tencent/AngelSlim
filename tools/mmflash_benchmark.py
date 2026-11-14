"""MMFlash end-to-end speculative decoding benchmark for vision-language models.

MMFlash (https://github.com/fscdc/MMFlash) drafts use the DFlash draft architecture and are
loaded with ``QwenDFlashDraftModel``; the decoding loop for vision-language targets
(multimodal prefill, multimodal RoPE, rollback of Qwen3.5's linear-attention states) is
``angelslim.compressor.speculative.inference.mmflash.MMFlashGenerator``.

Every question is decoded twice with the same loop: block size 1 (the target alone,
autoregressive) and the draft's block size (MMFlash). The script reports the decoding
speedup, the average acceptance length and its histogram, like ``tools/dflash_benchmark.py``
does for text-only models. Supports torchrun for multi-GPU sharded evaluation.

Usage (single GPU)::

    python tools/mmflash_benchmark.py \\
        --model-name-or-path Qwen/Qwen3.5-4B \\
        --draft-name-or-path FSCCS/Qwen3.5-4B-MMFlash \\
        --dataset chartqa --max-samples 64

One image and prompt::

    python tools/mmflash_benchmark.py \\
        --model-name-or-path Qwen/Qwen3.5-4B \\
        --draft-name-or-path FSCCS/Qwen3.5-4B-MMFlash \\
        --image /path/to/image.png --prompt "Describe this image."

Usage (8 GPUs)::

    torchrun --nproc_per_node=8 --master_port=29600 tools/mmflash_benchmark.py \\
        --model-name-or-path Qwen/Qwen3.5-4B \\
        --draft-name-or-path FSCCS/Qwen3.5-4B-MMFlash \\
        --dataset mathvista --max-samples 256
"""

from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from itertools import chain
from typing import Any, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch import distributed as torch_dist
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from angelslim.compressor.speculative.inference.mmflash import MMFlashGenerator
from angelslim.compressor.speculative.train.models.draft.qwen_dflash import (
    QwenDFlashDraftModel,
)


# ---------------------------------------------------------------------------
# Distributed helpers (same as tools/dflash_benchmark.py).
# ---------------------------------------------------------------------------
def _dist_init() -> None:
    if "RANK" not in os.environ:
        warnings.warn(
            "Environment variable `RANK` is not set; running single-process.",
            stacklevel=2,
        )
        return
    torch_dist.init_process_group(backend="nccl", init_method="env://")


def _dist_is_initialized() -> bool:
    return torch_dist.is_initialized()


def _dist_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def _dist_rank() -> int:
    return int(os.environ.get("RANK", 0))


def _dist_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _dist_is_main() -> bool:
    return _dist_rank() == 0


def _dist_gather(obj: Any, dst: int = 0) -> Optional[List[Any]]:
    if not _dist_is_initialized():
        return [obj]
    if _dist_is_main():
        objs: List[Any] = [None for _ in range(_dist_size())]
        torch_dist.gather_object(obj, objs, dst=dst)
        return objs
    torch_dist.gather_object(obj, dst=dst)
    return None


# ---------------------------------------------------------------------------
# Datasets: (image, question) pairs with the prompts of the MMFlash evaluation.
# ---------------------------------------------------------------------------
STEP_BY_STEP_BOXED = (
    'Please solve the problem step by step and put your answer in one "\\boxed{}".'
)

# lm-evaluation-harness `chartqa` prompt (from mistralai/mistral-evals), verbatim
CHARTQA_INSTRUCTION = """Analyze the image and question carefully, using step-by-step reasoning.
First, describe any image provided in detail. Then, present your reasoning. And finally your final answer in this format:
Final Answer: <answer>
where <answer> follows the following instructions:
- <answer> should should be a single phrase or number.
- <answer> should not paraphrase or reformat the text in the image.
- If <answer> is a ratio, it should be a decimal value like 0.25 instead of 1:4.
- If the question is a Yes/No question, <answer> should be Yes/No.
- If <answer> is a number, it should not contain any units.
- If <answer> is a percentage, it should include a % sign.
- If <answer> is an entity, it should include the full label from the graph.
IMPORTANT: Remember, to end your answer with Final Answer: <answer>."""  # noqa: E501


def _mmstar_prompt(row) -> str:
    question = row["question"].strip().replace(" Please answer yes or no.", "")
    return f"{question}\n{STEP_BY_STEP_BOXED}"


def _mathvista_prompt(row) -> str:
    query = row["query"]
    head, separator, tail = query.partition("\n")
    if separator and head.startswith("Hint: "):  # replaced by the boxed instruction
        query = tail
    return f"{query.strip()}\n\n{STEP_BY_STEP_BOXED}"


# name -> (Hugging Face dataset, split, image column, prompt builder)
DATASETS = {
    "chartqa": (
        "HuggingFaceM4/ChartQA",
        "test",
        "image",
        lambda row: f"{row['query']}\n{CHARTQA_INSTRUCTION}",
    ),
    "mmstar": ("Lin-Chen/MMStar", "val", "image", _mmstar_prompt),
    "mathvista": ("AI4Math/MathVista", "testmini", "decoded_image", _mathvista_prompt),
}


class Questions:
    """(image, prompt) pairs of a dataset, decoded on access."""

    def __init__(self, name: str, max_samples: Optional[int]):
        hub_id, split, self.image_column, self.build_prompt = DATASETS[name]
        dataset = load_dataset(hub_id, split=split)
        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.shuffle(seed=0).select(range(max_samples))
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> dict:
        row = self.dataset[i]
        return {"image": row[self.image_column], "prompt": self.build_prompt(row), "index": i}


def build_inputs(processor, image: Optional[Image.Image], prompt: str, enable_thinking: bool):
    content = [] if image is None else [{"type": "image", "image": image.convert("RGB")}]
    content.append({"type": "text", "text": prompt})
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    )


def stop_token_ids(target, processor) -> List[int]:
    """The generation config's EOS ids plus the tokenizer's (``<|im_end|>`` for Qwen)."""
    eos = target.generation_config.eos_token_id
    ids = {eos} if isinstance(eos, int) else set(eos or [])
    tokenizer = processor.tokenizer
    ids.add(tokenizer.eos_token_id)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tokenizer.unk_token_id:
        ids.add(im_end)
    return sorted(i for i in ids if i is not None)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path", required=True, help="Target VLM, e.g. Qwen/Qwen3.5-4B."
    )
    parser.add_argument(
        "--draft-name-or-path",
        required=True,
        help="MMFlash (or DFlash) draft, e.g. FSCCS/Qwen3.5-4B-MMFlash.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Speculative block size. Defaults to the draft model's config value.",
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    parser.add_argument("--image", default=None, help="Run one image + --prompt instead.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Untimed short generations per method before measuring (kernel compilation).",
    )
    parser.add_argument("--output-file", default=None, help="Write per-sample results (JSON).")
    args = parser.parse_args()
    if (args.dataset is None) == (args.prompt is None):
        parser.error("pass either --dataset or --prompt (optionally with --image)")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    # PyTorch may run SDPA through cuDNN on Hopper GPUs, which rebuilds its execution plan
    # whenever the key/value length changes, i.e. at every decoding step (on an H200 it made
    # autoregressive decoding about 8x and MMFlash decoding about 28x slower).
    torch.backends.cuda.enable_cudnn_sdp(False)

    _dist_init()
    torch.cuda.set_device(_dist_local_rank())
    device = torch.device(f"cuda:{_dist_local_rank()}")

    target = (
        AutoModelForImageTextToText.from_pretrained(
            args.model_name_or_path,
            attn_implementation=args.attn_implementation,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    draft = (
        QwenDFlashDraftModel.from_pretrained(
            args.draft_name_or_path,
            attn_implementation=args.attn_implementation,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    block_size = args.block_size if args.block_size is not None else draft.block_size
    generator = MMFlashGenerator(target, draft)
    stop_ids = stop_token_ids(target, processor)

    if args.dataset is not None:
        questions = Questions(args.dataset, args.max_samples)
    else:
        image = Image.open(args.image) if args.image else None
        questions = [{"image": image, "prompt": args.prompt, "index": 0}]
    tag = f"[{args.dataset or 'prompt'}]"

    def run(question, max_new_tokens):
        inputs = build_inputs(
            processor, question["image"], question["prompt"], args.enable_thinking
        )
        return {
            bs: generator.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_ids,
                temperature=args.temperature,
                block_size=bs,
            )
            for bs in (1, block_size)
        }

    indices = list(range(_dist_rank(), len(questions), _dist_size()))
    for _ in range(args.warmup if indices else 0):
        run(questions[indices[0]], min(64, args.max_new_tokens))

    records = []
    for idx in tqdm(indices, disable=not _dist_is_main()):
        question = questions[idx]
        response = run(question, args.max_new_tokens)
        record = {"index": question["index"]}
        for name, bs in (("ar", 1), ("mmflash", block_size)):
            r = response[bs]
            record[name] = {
                "text": processor.decode(
                    r.output_ids[0, r.num_input_tokens :], skip_special_tokens=True
                ),
                "num_input_tokens": r.num_input_tokens,
                "num_output_tokens": r.num_output_tokens,
                "time_to_first_token": r.time_to_first_token,
                "decode_time": r.decode_time,
                "time_per_output_token": r.time_per_output_token,
                "acceptance_lengths": r.acceptance_lengths,
            }
        records.append(record)

    if _dist_size() > 1:
        gathered = _dist_gather(records, dst=0)
        if not _dist_is_main():
            return
        records = sorted(chain(*gathered), key=lambda r: r["index"])
    if not records:
        return

    if args.prompt is not None:
        for name in ("ar", "mmflash"):
            print(f"{tag} {name} output:\n{records[0][name]['text']}\n")

    ar = [r["ar"] for r in records]
    mm = [r["mmflash"] for r in records]
    t1 = np.mean([r["time_per_output_token"] for r in ar])
    tb = np.mean([r["time_per_output_token"] for r in mm])
    print(f"{tag} Decoding speedup: {t1 / tb:.2f}")
    for name, rs in (("AR", ar), ("MMFlash", mm)):
        tokens = sum(r["num_output_tokens"] for r in rs)
        seconds = sum(r["decode_time"] for r in rs)
        print(f"{tag} {name} decode throughput: {tokens / seconds:.1f} tok/s")
    tau = np.mean([np.mean(r["acceptance_lengths"]) for r in mm if r["acceptance_lengths"]])
    print(f"{tag} Average acceptance length: {tau:.2f}")
    lengths = list(chain(*[r["acceptance_lengths"] for r in mm]))
    histogram = [lengths.count(b) / len(lengths) for b in range(1, block_size + 1)]
    print(
        f"{tag} Acceptance length histogram (1..{block_size}): "
        f"{[f'{x * 100:.1f}%' for x in histogram]}"
    )
    if args.temperature < 1e-5:
        same = sum(r["ar"]["text"] == r["mmflash"]["text"] for r in records)
        print(f"{tag} Outputs identical to autoregressive decoding: {same}/{len(records)}")
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(records, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
