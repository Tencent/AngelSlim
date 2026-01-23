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
Generate VLM responses for Eagle3 training.

This script loads a target VLM model (e.g., Qwen3-VL-30B-A3B) and generates
responses for image+text inputs, saving the conversations in ShareGPT format.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VLMResponseGenerator:
    """Generates responses using a vision-language model."""

    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype: str = "bfloat16",
        device: str = "cuda",
        trust_remote_code: bool = True,
    ):
        """
        Initialize the VLM response generator.

        Args:
            model_name_or_path: Path or name of the target VLM model
            torch_dtype: Data type for model weights
            device: Device to load model on
            trust_remote_code: Whether to trust remote code
        """
        self.device = device
        logger.info(f"Loading model from {model_name_or_path}...")

        # Map string dtype to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

        # Load model
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        logger.info("Model loaded successfully")

    def generate_response(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> Optional[str]:
        """
        Generate a response for an image+question input.

        Args:
            image_path: Path to the image file
            question: Question about the image
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling

        Returns:
            Generated response or None if failed
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Prepare conversation format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            # Apply chat template
            text_input = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(
                text=[text_input],
                images=[image],
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                )

            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return response.strip()

        except Exception as e:
            logger.error(f"Failed to generate response for {image_path}: {e}")
            return None

    def generate_batch_responses(
        self,
        samples: List[Dict[str, Any]],
        batch_size: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> List[Optional[str]]:
        """
        Generate responses for a batch of samples.

        Args:
            samples: List of samples with 'img_path' and 'question' fields
            batch_size: Number of samples to process in parallel
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            List of generated responses
        """
        responses = []

        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]

            try:
                # Load images
                images = []
                texts = []
                for sample in batch:
                    img = Image.open(sample["img_path"]).convert("RGB")
                    images.append(img)

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": sample["question"]},
                            ],
                        }
                    ]
                    text = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True
                    )
                    texts.append(text)

                # Process batch
                inputs = self.processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                    )

                # Decode
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                batch_responses = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                responses.extend([r.strip() for r in batch_responses])

            except Exception as e:
                logger.error(f"Failed to process batch {i}-{i+batch_size}: {e}")
                # Add None for failed samples
                responses.extend([None] * len(batch))

        return responses


def load_dataset(
    input_data_path: str,
    images_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load dataset from JSONL file.

    Args:
        input_data_path: Path to input JSONL file
        images_dir: Directory containing images (if None, uses same dir as data)

    Returns:
        List of samples with absolute image paths
    """
    if images_dir is None:
        images_dir = os.path.join(os.path.dirname(input_data_path), "images")

    samples = []
    with open(input_data_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)

            # Make image path absolute
            if "img_path" in sample:
                if not os.path.isabs(sample["img_path"]):
                    sample["img_path"] = os.path.join(images_dir, sample["img_path"])

            samples.append(sample)

    logger.info(f"Loaded {len(samples)} samples from {input_data_path}")
    return samples


def save_conversation(
    output_file: str,
    sample: Dict[str, Any],
    response: str,
) -> None:
    """
    Save a conversation in ShareGPT format.

    Args:
        output_file: Output file path
        sample: Original sample data
        response: Generated response
    """
    conversation = {
        "conversations": [
            {
                "from": "human",
                "value": f"<image>{sample['question']}",
            },
            {
                "from": "gpt",
                "value": response,
            },
        ],
        "img_path": sample.get("img_path", ""),
    }

    # Append to file
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(conversation, ensure_ascii=False) + "\n")


def generate_vlm_responses(
    model_name_or_path: str,
    input_data_path: str,
    output_data_path: str,
    images_dir: Optional[str] = None,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    torch_dtype: str = "bfloat16",
    resume: bool = True,
    save_interval: int = 100,
):
    """
    Generate VLM responses for a dataset.

    Args:
        model_name_or_path: Path to the VLM model
        input_data_path: Path to input JSONL file with questions
        output_data_path: Path to output JSONL file for conversations
        images_dir: Directory containing images
        batch_size: Batch size for generation
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        torch_dtype: Model data type
        resume: Whether to resume from existing output
        save_interval: Save checkpoint every N samples
    """
    # Load dataset
    samples = load_dataset(input_data_path, images_dir)

    # Check for existing output
    start_idx = 0
    if resume and os.path.exists(output_data_path):
        with open(output_data_path, "r", encoding="utf-8") as f:
            start_idx = sum(1 for _ in f)
        logger.info(f"Resuming from sample {start_idx}")

    if start_idx >= len(samples):
        logger.info("All samples already processed!")
        return

    # Initialize generator
    generator = VLMResponseGenerator(
        model_name_or_path=model_name_or_path,
        torch_dtype=torch_dtype,
    )

    # Process samples
    logger.info(f"Generating responses for {len(samples) - start_idx} samples...")

    for i in tqdm(range(start_idx, len(samples)), desc="Generating"):
        sample = samples[i]

        # Generate response
        response = generator.generate_response(
            image_path=sample["img_path"],
            question=sample["question"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        if response:
            save_conversation(output_data_path, sample, response)

        # Periodic logging
        if (i + 1) % save_interval == 0:
            logger.info(f"Processed {i + 1}/{len(samples)} samples")

    logger.info(f"Generation complete! Output saved to {output_data_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate VLM responses for Eagle3 training"
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path or name of the target VLM model",
    )
    parser.add_argument(
        "--input_data_path",
        type=str,
        required=True,
        help="Path to input JSONL file with questions",
    )
    parser.add_argument(
        "--output_data_path",
        type=str,
        required=True,
        help="Path to output JSONL file for conversations",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Directory containing images (default: same dir as input_data_path/images)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation (default: 4)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model data type (default: bfloat16)",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Don't resume from existing output",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Log progress every N samples (default: 100)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    generate_vlm_responses(
        model_name_or_path=args.model_name_or_path,
        input_data_path=args.input_data_path,
        output_data_path=args.output_data_path,
        images_dir=args.images_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        torch_dtype=args.torch_dtype,
        resume=not args.no_resume,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
