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

import requests
from transformers import ProcessorMixin
from transformers.pipelines.audio_utils import ffmpeg_read

from .base_dataset import BaseDataset


class AudioDataset(BaseDataset):
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

        self._load_file_based_dataset(data_source, num_samples)

    def _load_file_based_dataset(self, data_path: str, num_samples: int):
        """Load dataset from local file system"""
        audio_dir = os.path.join(os.path.dirname(data_path), "audios")
        line_count = 0

        with open(data_path, "r") as f:
            for line in f:
                if num_samples > 0 and line_count >= num_samples:
                    break

                data = json.loads(line.strip())
                if data["audio_path"].startswith("http://") or data[
                    "audio_path"
                ].startswith("https://"):
                    audio_path = data["audio_path"]
                else:
                    audio_path = os.path.join(audio_dir, data["audio_path"])

                # Prepare chat messages with image
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": audio_path},
                            {
                                "type": "text",
                                "text": data["question"].replace("<audio>", ""),
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

    def _process_and_append(self, messages: List[Dict]):
        """Process messages and append to dataset"""

        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_audios = self._extract_audio_info(messages)

        # Process inputs
        inputs = self.processor(
            text=input_text,
            audio=input_audios,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        self.data.append(inputs)

    @staticmethod
    def read_audio(audio_path):
        if audio_path.startswith("http://") or audio_path.startswith("https://"):
            # We need to actually check for a real protocol,
            # otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            inputs = requests.get(audio_path).content
        else:
            with open(audio_path, "rb") as f:
                inputs = f.read()
        return inputs

    def _extract_audio_info(self, messages: List[Dict]) -> tuple:
        """Extract Audio paths from messages"""
        audio_paths = []
        sampling_rate = self.processor.feature_extractor.sampling_rate

        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue

            for item in content:
                if item.get("type") == "audio":
                    # Handle both file paths and PIL images
                    if isinstance(item["audio_url"], str):
                        try:
                            audio_paths.append(
                                ffmpeg_read(
                                    self.read_audio(item["audio_url"]),
                                    sampling_rate=sampling_rate,
                                )
                            )
                        except ValueError as e:
                            raise ValueError(
                                f"Could not open audio file: {item['audio_url']}, {e}"
                            )
        return audio_paths
