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

from .base_dataset import BaseDataset


class Text2ImageDataset(BaseDataset):
    """Dataset for text-only data in Parquet or JSONL formats"""

    def __init__(
        self,
        data_path: str,
        num_samples: int = -1,
        inference_settings: dict = None,
    ):
        self.data = []
        self.seed = inference_settings["seed"]
        self.height = inference_settings["height"]
        self.width = inference_settings["width"]
        self.guidance_scale = inference_settings["guidance_scale"]
        self.num_inference_steps = inference_settings["num_inference_steps"]
        self.max_sequence_length = inference_settings["max_sequence_length"]
        self._load_data(data_path, num_samples)

    def _load_data(self, data_path: str, num_samples: int):
        if data_path.endswith(".jsonl"):
            self._load_jsonl_data(data_path, num_samples)
        else:
            raise ValueError("Unsupported file format. Only JSONL is supported.")

    def _load_jsonl_data(self, data_path: str, num_samples: int):
        line_count = 0
        with open(data_path, "r") as f:
            for line in f:
                if num_samples > 0 and line_count >= num_samples:
                    break

                data = json.loads(line)

                # Validate format
                assert "input" in data, "JSON format error"

                self.data.append(
                    {
                        "input": data["input"],
                        "seed": self.seed,
                        "height": self.height,
                        "width": self.width,
                        "guidance_scale": self.guidance_scale,
                        "num_inference_steps": self.num_inference_steps,
                        "max_sequence_length": self.max_sequence_length,
                    }
                )

                line_count += 1
