# Copyright 2026 Tencent Inc. All Rights Reserved.
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
import traceback
from io import BytesIO

import requests
import torch
import yaml
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

from angelslim.compressor.token_compressor.adapter import UniversalPruningAdapter

# Import components from the 'token_compressor' package
from angelslim.compressor.token_compressor.base.config import TokenCompressorConfig

# Constants for test data
DEFAULT_IMAGE_PATH = (
    "https://inews.gtimg.com/news_bt/" "OQSQBp_mW8TxXv7UsR55mi2DMfWW4D2aJJ-jsFphE5YD8AA/1000"
)
MAX_NEW_TOKENS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(image_source: str) -> Image.Image:
    """
    Loads an image from either a local filesystem path or a remote network URL.

    Args:
        image_source (str): The source identifier,
        can be a local path or an HTTP(S) URL.

    Returns:
        Image.Image: The processed PIL Image object converted to RGB mode.
    """
    if image_source.startswith(("http://", "https://")):
        # Fetch the binary image data from the remote server
        response = requests.get(image_source, timeout=10)
        # Raise an exception if the request failed (e.g., 404 or 500)
        response.raise_for_status()

        # Wrap the byte content in a file-like object for PIL decoder
        image = Image.open(BytesIO(response.content))
    else:
        # Load the image directly from the local storage
        image = Image.open(image_source)

    # Convert to RGB to ensure compatibility with vision encoders that expect 3 channels
    return image.convert("RGB")


def main():
    parser = argparse.ArgumentParser(
        description="Tailored Smoke Test for DART and Universal Architecture"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the strategy YAML " "(e.g., configs/qwen2_5_vl/pruning/dart_r0.75.yaml)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace model path or local path",
    )
    args = parser.parse_args()

    print(f"🚀 Initializing Smoke Test for Model: {args.model_path}")

    # 1. Parse YAML to extract Model Mapping and Strategy Configuration
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            full_yaml = yaml.safe_load(f)

        # Mapping used by UniversalPruningAdapter to identify paths and wrappers
        mapping_data = full_yaml.get("model_mapping", [])
        # Strategy requirements and stage-wise parameters
        strategy_config = TokenCompressorConfig.from_yaml(args.config)

        if not mapping_data:
            print(f"❌ Error: 'model_mapping' not found in {args.config}")
            return

    except Exception as e:
        print(f"❌ YAML Parsing Failed: {e}")
        return

    # 2. Load the base model and its processor
    try:
        print(f"📥 Loading base model into {DEVICE}...")
        model = AutoModelForMultimodalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if DEVICE != "cpu" else torch.float32,
            attn_implementation="sdpa",
            device_map=DEVICE,
            trust_remote_code=True,
        ).eval()

        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        print("✅ Base architecture loaded successfully.")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        return

    # 3. Prepare visual and textual inputs
    try:
        image = load_image(DEFAULT_IMAGE_PATH)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": "Describe the content of this image " "in one short sentence.",
                    },
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        ).to(model.device)
        print("✅ Multimodal inputs prepared.")
    except Exception as e:
        print(f"❌ Input preparation failed: {e}")
        return

    # 4. Instantiate the Universal Adapter and apply wrappers
    adapter = None
    try:
        print("🛠️  Applying UniversalPruningAdapter using steps from YAML...")
        adapter = UniversalPruningAdapter(
            model=model, strategy_config=strategy_config, raw_map_data=mapping_data
        )
        # Execute the sequential wrapping based on 'model_mapping' order
        model = adapter.wrap_model()
        print("✅ Pruning wrappers successfully injected.")
    except Exception as e:
        print(f"❌ Transformation failed: {e}")
        traceback.print_exc()
        return

    print(f"model after wrapping: {model}")

    # 5. Execute Autoregressive Generation with Pruning Support
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        # Post-process results
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1] :]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print("\n" + "=" * 60)
        print(f"📝 PRUNED MODEL OUTPUT:\n{output_text}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        traceback.print_exc()

    finally:
        # 6. Safety Cleanup: Restore original pointers and clear GPU cache
        if adapter:
            print("🧹 Reverting model to standard state...")
            adapter.unwrap_model()

        torch.cuda.empty_cache()
        print("✅ Done.")


if __name__ == "__main__":
    main()
