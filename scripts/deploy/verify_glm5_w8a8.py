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

"""Offline smoke-test for the AngelSlim quantized checkpoint (W8A8C16).

This script is the fastest way to verify the quantized GLM-5 checkpoint
can "actually speak".  It uses vLLM's offline ``LLM`` API (no HTTP
server), which natively understands the ``compressed-tensors`` format
written by AngelSlim's ``PTQSaveVllmHF`` saver
(``quantization_config.quant_method = "compressed-tensors"``,
``format = "int-quantized"``).

Why vLLM and not raw transformers ?
  * transformers can load ``compressed-tensors`` only through the
    ``compressed-tensors`` package plus a matching int8 kernel; on a
    700B-class MoE model that path is fragile and, more importantly,
    won't fit on a single node without vLLM's expert/tensor parallel
    scheduler.
  * vLLM has first-class support for ``int-quantized`` weights: it will
    (de)quantize per channel + per-token dynamic activation on the fly
    exactly the way the AngelSlim W8A8C16 pipeline emitted them.

Usage
-----
    # single node, 8 GPUs (default), pipeline_parallel=1
    python3 scripts/deploy/verify_glm5_w8a8.py \
        --model-path /apdcephfs_zwfy2/share_300532381/harviexu/project/AngelSlim/output_glm5_w8a8c16/glm5_w8a8c16_low_memory

    # customise decoding
    python3 scripts/deploy/verify_glm5_w8a8.py \
        --model-path <path> \
        --max-tokens 256 --temperature 0.7 --top-p 0.9

    # supply your own prompts (one per line, plain text — no chat template
    # is applied unless --apply-chat-template is passed)
    python3 scripts/deploy/verify_glm5_w8a8.py \
        --model-path <path> --prompt-file my_prompts.txt

    # apply the model's chat template (recommended for chat-tuned
    # checkpoints; will format each prompt as a single-turn user message)
    python3 scripts/deploy/verify_glm5_w8a8.py \
        --model-path <path> --apply-chat-template
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List


DEFAULT_PROMPTS: List[str] = [
    # A short factual prompt — easy to eyeball.
    "Explain what tensor parallelism is in one short paragraph.",
    # A Chinese prompt — ChatGLM tokenizer coverage sanity check.
    "请用一句话介绍你自己。",
    # A tiny reasoning prompt — spot obvious quantization damage
    # (garbled tokens, immediate EOS, repeat loops).
    "If a train leaves at 9:15 and arrives at 11:05, how long is the trip?",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an AngelSlim-quantized checkpoint through vLLM and "
            "generate a few smoke-test completions."
        )
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Directory containing the AngelSlim output (config.json, "
        "model-*.safetensors, tokenizer files, quantization_config).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=int(os.environ.get("TP", 8)),
        help="vLLM tensor parallel size (default: 8, i.e. single-node 8-GPU).",
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="vLLM pipeline parallel size (default: 1).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of each GPU's memory vLLM may use (default: 0.9).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max context length (default: 4096).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Activation dtype (default: bfloat16 — matches the calibration).",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        help=(
            "KV cache dtype.  For W8A8C16 the KV path stays in bf16, so "
            "'auto' is correct.  Pass 'fp8' only if you actually calibrated "
            "an fp8 KV scheme."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="max_new_tokens for each prompt (default: 128).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature.  Default 0.0 = greedy — gives "
            "reproducible smoke-test output.  Set > 0 with --top-p to "
            "check for degenerate sampling."
        ),
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="nucleus top-p.")
    parser.add_argument("--top-k", type=int, default=-1, help="top-k (-1 = disabled).")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="repetition_penalty (default: 1.0 = off).",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional file of prompts (one per line). Overrides the built-in set.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help=(
            "Wrap each prompt as a single-turn user message via the "
            "tokenizer's chat template (recommended for chat-tuned "
            "checkpoints like ChatGLM 5.2)."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Passed through to vLLM/transformers (default: True — GLM-5 needs it).",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help=(
            "Disable CUDA graph capture.  Useful if you see hangs during "
            "warmup; slightly slower but easier to debug."
        ),
    )
    return parser.parse_args()


def _load_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompt_file is None:
        return DEFAULT_PROMPTS
    p = Path(args.prompt_file)
    if not p.is_file():
        raise FileNotFoundError(f"--prompt-file not found: {p}")
    lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in lines if line]
    if not prompts:
        raise ValueError(f"--prompt-file is empty after strip: {p}")
    return prompts


def _maybe_apply_chat_template(
    prompts: List[str], model_path: str, enable: bool
) -> List[str]:
    """Optionally wrap each prompt as a single-turn chat message.

    We import ``transformers`` lazily so the script still works even if
    only the vLLM stack is installed.
    """
    if not enable:
        return prompts
    from transformers import AutoTokenizer  # local import — see docstring

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.chat_template is None:
        print(
            "[verify] WARN: tokenizer has no chat_template — "
            "falling back to raw prompts.",
            file=sys.stderr,
        )
        return prompts
    out: List[str] = []
    for p in prompts:
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        out.append(formatted)
    return out


def _validate_checkpoint(model_path: str) -> None:
    """Cheap sanity check: does the directory actually contain a saved model?

    This is the #1 source of confusion when the PTQ pipeline crashes at
    the very end (e.g. ``generation_config`` strict-validate bug): the
    directory exists but only has ``config.json``, no weight shards, so
    vLLM bombs with an obscure "no safetensors found" error.
    """
    root = Path(model_path)
    if not root.is_dir():
        raise FileNotFoundError(f"--model-path is not a directory: {root}")
    shards = list(root.glob("*.safetensors")) + list(root.glob("*.bin"))
    if not shards:
        raise FileNotFoundError(
            f"No *.safetensors / *.bin found under {root} — the PTQ save step "
            "did not complete.  Check the tail of the calibration log for "
            "an exception around `save_pretrained` / `generation_config`."
        )
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"{root}/config.json is missing.")
    qcfg = root / "config.json"
    text = qcfg.read_text(encoding="utf-8")
    if "quantization_config" not in text:
        print(
            "[verify] WARN: config.json does not contain a "
            "`quantization_config` field — is this really an AngelSlim "
            "PTQ output?",
            file=sys.stderr,
        )


def main() -> int:
    args = _parse_args()
    _validate_checkpoint(args.model_path)

    # Import vLLM lazily so ``--help`` works even if vLLM isn't installed.
    from vllm import LLM, SamplingParams

    prompts = _load_prompts(args)
    prompts = _maybe_apply_chat_template(
        prompts, args.model_path, args.apply_chat_template
    )

    print(
        f"[verify] loading model from: {args.model_path}\n"
        f"[verify] TP={args.tensor_parallel_size}, "
        f"PP={args.pipeline_parallel_size}, "
        f"dtype={args.dtype}, kv_cache_dtype={args.kv_cache_dtype}, "
        f"max_model_len={args.max_model_len}, "
        f"enforce_eager={args.enforce_eager}",
        flush=True,
    )

    t0 = time.time()
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
    )
    print(f"[verify] LLM ready in {time.time() - t0:.1f}s", flush=True)

    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    t1 = time.time()
    outputs = llm.generate(prompts, sampling)
    dt = time.time() - t1

    total_out_tokens = 0
    print("\n" + "=" * 72)
    for i, out in enumerate(outputs):
        gen = out.outputs[0]
        total_out_tokens += len(gen.token_ids)
        print(f"[prompt #{i}] {out.prompt!r}")
        print(f"[reply   ] {gen.text}")
        print(
            f"[stats   ] out_tokens={len(gen.token_ids)}, "
            f"finish_reason={gen.finish_reason}"
        )
        print("-" * 72)
    print(
        f"[verify] {len(outputs)} prompts, {total_out_tokens} generated "
        f"tokens in {dt:.2f}s "
        f"({total_out_tokens / max(dt, 1e-6):.1f} tok/s)"
    )
    print("=" * 72)

    # Fail-fast health check: if EVERY reply is empty, the checkpoint is
    # almost certainly broken (bad KV path / bad dequant / wrong dtype).
    if all(len(o.outputs[0].token_ids) == 0 for o in outputs):
        print(
            "[verify] FAIL: every reply is empty — quantization likely "
            "damaged the model.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
