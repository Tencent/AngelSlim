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

import json
import os
from typing import Dict, List, Union

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import ProcessorMixin

from ..utils.lazy_imports import qwen_vl_utils
from .base_dataset import BaseDataset


class MultiModalDataset(BaseDataset):
    """Dataset for multimodal (text + image) data"""

    def __init__(
        self,
        processor: ProcessorMixin,
        device: str = "cpu",
        max_length: int = 4096,
        num_samples: int = -1,
        data_source: Union[str, Dict] = None,
        is_hf_dataset: bool = False,
        model_name: str = None,
    ):
        super().__init__(processor, device, max_length)
        self.is_hf_dataset = is_hf_dataset
        self.model_name = model_name

        if is_hf_dataset:
            self._load_hf_dataset(data_source, num_samples)
        else:
            self._load_file_based_dataset(data_source, num_samples)

    def _load_file_based_dataset(self, data_path: str, num_samples: int):
        """Load dataset from local file system"""
        image_dir = os.path.join(os.path.dirname(data_path), "images")
        line_count = 0

        with open(data_path, "r") as f:
            for line in f:
                if num_samples > 0 and line_count >= num_samples:
                    break

                data = json.loads(line.strip())
                image_path = os.path.join(image_dir, data["img_path"])

                # Prepare chat messages with image
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {
                                "type": "text",
                                "text": data["question"].replace("<image>", ""),
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": data["answer"]}],
                    },
                ]

                self._process_and_append(messages)
                line_count += 1

    def _load_hf_dataset(self, dataset: str, num_samples: int):
        """Load dataset from Hugging Face format"""
        dataset = load_dataset(dataset, split="test")
        total_samples = (
            min(num_samples, len(dataset["query"]))
            if num_samples > 0
            else len(dataset["query"])
        )

        for i in tqdm(range(total_samples), desc="Processing HF Dataset"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": dataset["image"][i]},
                        {"type": "text", "text": dataset["query"][i]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": dataset["label"][i][0]}],
                },
            ]
            self._process_and_append(messages)

    def _process_and_append(self, messages: List[Dict]):
        """Process messages and append to dataset"""
        if self.model_name in ["Qwen3VL"]:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Extract vision info
            image_inputs, video_inputs = qwen_vl_utils.process_vision_info(messages)

            # Process inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )

        labels = inputs["input_ids"].roll(shifts=-1, dims=-1)
        labels[:, -1] = -100
        inputs["labels"] = labels

        self.data.append(inputs)

    def _extract_vision_info(self, messages: List[Dict]) -> tuple:
        """Extract image and video paths from messages"""
        image_paths = []
        video_paths = []

        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue

            for item in content:
                if item.get("type") == "image":
                    # Handle both file paths and PIL images
                    if isinstance(item["image"], str):
                        try:
                            img = Image.open(item["image"])
                            image_paths.append(img)
                        except ValueError as e:
                            raise ValueError(
                                f"Could not open image file: {item['image']}, {e}"
                            )
                    elif isinstance(item["image"], Image.Image):
                        image_paths.append(item["image"])
                elif item.get("type") == "video":
                    video_paths.append(item["video"])

        return image_paths, video_paths
