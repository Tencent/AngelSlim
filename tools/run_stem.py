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
Stem sparse attention inference script.

Usage:
  python tools/run_stem.py --mode stem --model-path /path/to/Qwen3-8B --prompt-file prompt.txt
  python tools/run_stem.py --mode dense --model-path /path/to/Qwen3-8B --prompt-file prompt.txt
  python tools/run_stem.py --mode stem --stem-backend hpc \
      --hpc-dtype fp8 --model-path /path/to/Qwen3-8B --prompt-file prompt.txt
"""

import argparse
import sys
import time
import traceback

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from angelslim.compressor.sparsity.stem import StemInference

DEFAULT_MODEL_PATH = "Qwen/Qwen3-8B"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_MAX_MODEL_LEN = 131072


def build_stem_alpha_schedule(
    num_layers: int,
    warmup_layers: int = 5,
    warmup_alpha: float = 1.0,
    steady_alpha: float = 0.7,
) -> list[float]:
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if warmup_layers < 0 or warmup_layers > num_layers:
        raise ValueError(f"warmup_layers must be in [0, {num_layers}], got {warmup_layers}")
    return [warmup_alpha] * warmup_layers + [steady_alpha] * (num_layers - warmup_layers)


def build_prompt(tokenizer, raw_prompt: str) -> str:
    messages = [{"role": "user", "content": raw_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stem sparse attention inference: Dense vs Stem on Qwen3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="stem",
        choices=["dense", "stem"],
        help="Attention mode: 'dense' (no patch) or 'stem' (Stem sparse).",
    )
    parser.add_argument(
        "--stem-backend",
        type=str,
        default="torch",
        choices=["torch", "hpc"],
        help="Stem backend: 'torch' (PyTorch + Triton) or 'hpc' (HPC C++ extension).",
    )
    parser.add_argument(
        "--hpc-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp8"],
        help="HPC backend precision.",
    )
    parser.add_argument(
        "--hpc-fp8-path",
        type=str,
        default="paged",
        choices=["paged", "varlen"],
        help="HPC FP8 execution path.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the model directory or HuggingFace model ID.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name (used to determine rope_scaling config).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help="Maximum position embeddings length.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="Path to a text file containing the input prompt.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Maximum number of new tokens to generate."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[Env] Python executable: {sys.executable}")
    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model config ...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if "Qwen" in args.model_name or "qwen" in args.model_name.lower():
        rope_theta = (
            config.rope_scaling.get("rope_theta", 1000000) if config.rope_scaling else 1000000
        )
        config.rope_scaling = {
            "rope_type": "yarn",
            "rope_theta": rope_theta,
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        }
    config.max_position_embeddings = args.max_model_len

    print("Loading model weights ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if args.mode == "stem":
        num_layers = int(model.config.num_hidden_layers)
        stem_alpha = build_stem_alpha_schedule(
            num_layers=num_layers,
            warmup_layers=5,
            warmup_alpha=1.0,
            steady_alpha=0.7,
        )
        minf = StemInference(
            attn_kwargs={
                "backend": args.stem_backend,
                "hpc_dtype": args.hpc_dtype,
                "hpc_fp8_path": args.hpc_fp8_path,
                "stem_alpha": stem_alpha,
            },
        )
        model = minf(model)
        msg = (
            f"[Stem] Patch applied. backend={args.stem_backend}, "
            f"hpc_dtype={args.hpc_dtype}, num_layers={num_layers}"
        )
        print(msg)
    else:
        print("[Dense] No Stem patch applied. Using standard flash_attention_2.")

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        raw_prompt = f.read()
    print(f"Loaded prompt from: {args.prompt_file}")

    prompt = build_prompt(tokenizer, raw_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = int(inputs.input_ids.shape[1])
    print(f"[Input Stats] token_length={input_len}")

    print("Generating ...")
    start = time.time()
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        end = time.time()

        gen_ids = out[0][input_len:]
        gen_text = tokenizer.decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        print("=" * 80)
        print(f"Mode: {args.mode}")
        print(f"Time Taken: {end - start:.3f} s")
        print(f"Generated Tokens: {len(gen_ids)}")
        print("Model Output:")
        print(gen_text.strip())
        print("=" * 80)
    except Exception as e:
        print(f"[Error]: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
