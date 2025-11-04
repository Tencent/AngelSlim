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
from pathlib import Path
from typing import Dict, List, Union

from transformers import ProcessorMixin

from ..utils.lazy_imports import qwen_omni_utils
from .base_dataset import BaseDataset


class OmniDataset(BaseDataset):
    """Dataset for multimodal (text + image) data"""

    def __init__(
        self,
        processor: ProcessorMixin,
        device: str = "cpu",
        max_length: int = 4096,
        num_samples: int = -1,
        data_source: Union[str, Dict] = None,
        is_hf_dataset: bool = False,
        use_audio_in_video: bool = False,
    ):
        super().__init__(processor, device, max_length)
        self.is_hf_dataset = is_hf_dataset
        self.use_audio_in_video = use_audio_in_video

        self._load_file_based_dataset(data_source, num_samples)

    def _load_file_based_dataset(self, data_path: str, num_samples: int):
        """Load dataset from local file system"""
        path_obj = Path(data_path)
        data_dir = path_obj.parent

        line_count = 0
        with open(data_path, "r") as f:
            for line in f:
                if num_samples > 0 and line_count >= num_samples:
                    break
                data = json.loads(line.strip())
                video_path = None
                audio_path = None
                image_path = None

                if "video_path" in data:
                    video_path = os.path.normpath(
                        os.path.join(data_dir, data["video_path"])
                    )
                if "audio_path" in data:
                    audio_path = os.path.normpath(
                        os.path.join(data_dir, data["audio_path"])
                    )
                if "image_path" in data:
                    image_path = os.path.normpath(
                        os.path.join(data_dir, data["image_path"])
                    )

                ms = data.get("messages")

                conversation = []
                for m in ms:
                    if m["role"] == "system":
                        conversation.append(
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": m["content"]}],
                            }
                        )
                    elif m["role"] == "user":
                        content = []
                        text_content = m["content"]
                        text_content = (
                            text_content.replace("<video>", "")
                            .replace("<audio>", "")
                            .replace("<image>", "")
                        )
                        content.append({"type": "text", "text": text_content})
                        if video_path:
                            content.append({"type": "video", "video": video_path})
                        if audio_path:
                            content.append({"type": "audio", "audio": audio_path})
                        if image_path:
                            content.append({"type": "image", "image": image_path})
                        conversation.append(
                            {
                                "role": "user",
                                "content": content,
                            }
                        )
                self._process_and_append(conversation)
                line_count += 1

    def _process_and_append(self, messages: List[Dict]):
        """Process messages and append to dataset"""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = qwen_omni_utils.process_mm_info(
            messages, use_audio_in_video=self.use_audio_in_video
        )

        # Process inputs
        inputs = self.processor(
            text=text,
            images=images,
            audios=audios,
            videos=videos,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=self.use_audio_in_video,
        )
        self.data.append(inputs)
