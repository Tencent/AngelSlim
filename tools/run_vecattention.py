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
VecAttention sparse attention inference script for Vision-Language Models.

Usage:
  # Text-only query on Qwen2.5-VL
  python tools/run_vecattention.py --mode vecattention \
      --model-path Qwen/Qwen2.5-VL-3B-Instruct \
      --prompt "What is the capital of France?"

  # Image (URL) + text query
  python tools/run_vecattention.py --mode vecattention \
      --model-path Qwen/Qwen2.5-VL-3B-Instruct \
      --prompt "Describe this image in detail." \
      --image https://inews.gtimg.com/news_bt/OQSQBp_mW8TxXv7UsR55mi2DMfWW4D2aJJ-jsFphE5YD8AA/1000

  # Image (local file) + text query
  python tools/run_vecattention.py --mode vecattention \
      --model-path Qwen/Qwen2.5-VL-3B-Instruct \
      --prompt "What's in this photo?" \
      --image /path/to/local/image.jpg

  # Video (local file) + text query
  python tools/run_vecattention.py --mode vecattention \
      --model-path Qwen/Qwen2.5-VL-3B-Instruct \
      --prompt "Summarize what happens in this video." \
      --video /path/to/video.mp4 --nframes 24

  # Video (URL) + text query
  python tools/run_vecattention.py --mode vecattention \
      --model-path Qwen/Qwen2.5-VL-3B-Instruct \
      --prompt "Describe the key events in this video." \
      --video https://example.com/video.mp4 --nframes 16

  # Dense baseline for comparison
  python tools/run_vecattention.py --mode dense \
      --model-path Qwen/Qwen2.5-VL-3B-Instruct \
      --prompt "Describe this image." \
      --image https://example.com/image.jpg
"""

import argparse
import sys
import time
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import AutoProcessor

from angelslim.compressor.sparsity.vecattention import VecAttentionInference

# Default test image
DEFAULT_IMAGE_URL = (
    "https://inews.gtimg.com/news_bt/" "OQSQBp_mW8TxXv7UsR55mi2DMfWW4D2aJJ-jsFphE5YD8AA/1000"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(image_source: str) -> Image.Image:
    """Load an image from a local path or HTTP(S) URL.

    Args:
        image_source: Local file path or URL starting with http:// or https://

    Returns:
        PIL Image in RGB mode.
    """
    if image_source.startswith(("http://", "https://")):
        response = requests.get(image_source, timeout=15)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_source)
    return image.convert("RGB")


def parse_args():
    parser = argparse.ArgumentParser(
        description="VecAttention VLM inference: Dense vs VecAttention on Qwen2.5-VL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="vecattention",
        choices=["dense", "vecattention"],
        help="Attention mode: 'dense' (no patch) or 'vecattention' (sparse prefill).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Path to the VLM model directory or HuggingFace model ID.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the content of this image in one short sentence.",
        help="Text prompt for the model.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Image source: local file path or HTTP(S) URL. "
        "If not provided, runs text-only inference.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Video source: local file path or HTTP(S) URL.",
    )
    parser.add_argument(
        "--nframes",
        type=int,
        default=24,
        help="Number of frames to sample from video.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    # VecAttention parameters
    parser.add_argument("--threshold", type=float, default=0.1, help="MinP threshold.")
    parser.add_argument(
        "--block-size-q", type=int, default=64, choices=[64, 128], help="Q pooling block size."
    )
    parser.add_argument("--block-size-k", type=int, default=16, help="K local block size.")
    parser.add_argument(
        "--group-k-block", type=int, default=16, help="K block grouping (default 16 for VLM)."
    )
    parser.add_argument("--chunk-size", type=int, default=64 * 1024, help="Prefill chunk size.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[Env] Python: {sys.executable}, Device: {DEVICE}")
    print(f"[Config] mode={args.mode}, model={args.model_path}")
    if args.image:
        print(f"[Config] image={args.image}")
    if args.video:
        print(f"[Config] video={args.video}, nframes={args.nframes}")
    if args.mode == "vecattention":
        print(
            f"[Config] threshold={args.threshold}, block_size_q={args.block_size_q}, "
            f"block_size_k={args.block_size_k}, group_k_block={args.group_k_block}, "
            f"chunk_size={args.chunk_size}"
        )

    # --- 1. Load model and processor ---
    print("Loading model...")
    # Use the same loading pattern as test_token_pruning.py
    from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if DEVICE != "cpu" else torch.float32,
        device_map=DEVICE,
        trust_remote_code=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    num_layers = getattr(
        model.config,
        "num_hidden_layers",
        getattr(model.config, "text_config", model.config).num_hidden_layers,
    )
    print(f"Model: {model.config.model_type}, {num_layers} layers")

    # --- 2. Prepare inputs (before patch, so we can detect vision positions) ---
    print("Preparing inputs...")
    image = None
    if args.video:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": args.video, "nframes": args.nframes},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        text_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=[text_prompt], videos=video_inputs, padding=True, return_tensors="pt"
        ).to(model.device)
    elif args.image:
        image = load_image(args.image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        ).to(model.device)
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(text=[text_prompt], padding=True, return_tensors="pt").to(model.device)

    input_len = inputs.input_ids.shape[1]
    print(f"[Input] token_length={input_len}")

    # --- 3. Detect vision token positions from input_ids ---
    vision_start_position = None
    vision_end_position = None
    if args.image or args.video:
        input_ids = inputs.input_ids[0]
        VISION_START_TOKEN_ID = 151652  # <|vision_start|>
        VISION_END_TOKEN_ID = 151653  # <|vision_end|>

        vision_start_indices = (input_ids == VISION_START_TOKEN_ID).nonzero(as_tuple=True)[0]
        vision_end_indices = (input_ids == VISION_END_TOKEN_ID).nonzero(as_tuple=True)[0]

        if len(vision_start_indices) > 0:
            vision_start_position = int(vision_start_indices[0].item())
        if len(vision_end_indices) > 0:
            # vision_end_position is one past the last vision token
            vision_end_position = int(vision_end_indices[-1].item()) + 1

        print(f"[Vision] start={vision_start_position}, end={vision_end_position}")

    # --- 4. Apply VecAttention patch ---
    if args.mode == "vecattention":
        print("Applying VecAttention patch...")
        attn_kwargs = {
            "threshold": args.threshold,
            "block_size_q": args.block_size_q,
            "block_size_k": args.block_size_k,
            "group_k_block": args.group_k_block,
            "chunk_size": args.chunk_size,
        }
        # Pass vision positions so VecAttention only applies to vision region
        if vision_start_position is not None:
            attn_kwargs["vision_start_position"] = vision_start_position
        if vision_end_position is not None:
            attn_kwargs["vision_end_position"] = vision_end_position

        vec = VecAttentionInference(attn_kwargs=attn_kwargs)
        model = vec(model)
        print(f"[VecAttention] Patched {num_layers} attention layers.")
    else:
        print("[Dense] No patch applied. Using standard attention.")

    # --- 5. Generate ---
    print("Generating...")
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # --- 6. Decode and display ---
    generated_ids_trimmed = generated_ids[:, input_len:]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Input tokens: {input_len}")
    print(f"Generated tokens: {generated_ids_trimmed.shape[1]}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Tokens/sec: {generated_ids_trimmed.shape[1] / (elapsed + 1e-9):.1f}")
    print("-" * 80)
    print("Output:")
    print(output_text.strip())
    print("=" * 80)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
