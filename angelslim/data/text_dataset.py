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
from typing import Dict, List

import pyarrow.parquet as pq
import torch
from transformers import ProcessorMixin

from .base_dataset import BaseDataset


class TextDataset(BaseDataset):
    """Dataset for text-only data in Parquet or JSONL formats"""

    def __init__(
        self,
        data_path: str,
        processor: ProcessorMixin,
        device: str = "cpu",
        max_length: int = 4096,
        num_samples: int = -1,
    ):
        super().__init__(processor, device, max_length)
        self._load_data(data_path, num_samples)

    def _load_data(self, data_path: str, num_samples: int):
        if ".parquet" in data_path.lower():
            self._load_parquet_data(data_path, num_samples)
        else:
            self._load_jsonl_data(data_path, num_samples)

    def _load_parquet_data(self, data_path: str, num_samples: int):
        table = pq.read_table(data_path)
        df = table.to_pandas()

        # Handle sample limits
        total_samples = min(num_samples, len(df)) if num_samples > 0 else len(df)

        for i in range(total_samples):
            text = df["text"].iloc[i]
            model_inputs = self.processor(
                [text],
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )

            # Handle potential labels
            if "labels" in df.columns:
                labels = torch.tensor(df["labels"].iloc[i]).unsqueeze(0)
            else:
                labels = model_inputs["input_ids"].roll(shifts=-1, dims=-1)
                labels[:, -1] = -100

            data_item = {
                "input_ids": model_inputs["input_ids"].to(self.device),
                "attention_mask": model_inputs["attention_mask"].to(self.device),
                "labels": labels.to(self.device),
            }
            self.data.append(data_item)

    def _load_jsonl_data(self, data_path: str, num_samples: int):
        line_count = 0
        with open(data_path, "r") as f:
            for line in f:
                if num_samples > 0 and line_count >= num_samples:
                    break

                data = json.loads(line)

                # Validate format
                assert (
                    "messages" in data or "input" in data or "conversations" in data
                ), "JSON format error"

                # Prepare messages
                messages = self._prepare_messages(data)

                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                thinking_data = False
                for dic in messages:
                    if dic["role"] == "assistant":
                        if "<think>" and "</think>" in dic["content"]:
                            thinking_data = True
                            break
                if thinking_data:
                    text = (
                        self.processor.bos_token
                        if self.processor.bos_token is not None
                        else ""
                    )
                    for dic in messages:
                        if dic["role"] == "system":
                            text += dic["content"]
                        elif dic["role"] == "user":
                            text = (
                                text + "<｜User｜>" + dic["content"] + "<｜Assistant｜>"
                            )
                        elif dic["role"] == "assistant":
                            text = text + dic["content"] + self.processor.eos_token

                model_inputs = self.processor(
                    [text],
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                )

                labels = model_inputs["input_ids"].roll(shifts=-1, dims=-1)
                labels[:, -1] = -100

                self.data.append(
                    {
                        "input_ids": model_inputs["input_ids"].to(self.device),
                        "attention_mask": model_inputs["attention_mask"].to(
                            self.device
                        ),
                        "labels": labels.to(self.device),
                    }
                )

                line_count += 1

    def _prepare_messages(self, data: Dict) -> List[Dict]:
        """Prepare chat messages from data entry"""
        if "messages" in data:
            messages = data["messages"]
            # Add system prompt if available
            if (
                "system_prompt" in data
                and data["system_prompt"]
                and messages[0]["role"] != "system"
            ):
                messages = [
                    {"role": "system", "content": data["system_prompt"]}
                ] + messages
        elif "conversations" in data:
            share_gpt_data = data["conversations"]
            messages = [
                {"role": "user", "content": share_gpt_data[0]["value"]},
                {"role": "assistant", "content": share_gpt_data[1]["value"]},
            ]
            if "system" in data and data["system"]:
                messages = [
                    {"role": "system", "content": data["system_prompt"]}
                ] + messages
        else:
            messages = [
                {"role": "user", "content": data["input"]},
                {"role": "assistant", "content": data["output"]},
            ]
            if "system_prompt" in data and data["system_prompt"]:
                messages = [
                    {"role": "system", "content": data["system_prompt"]}
                ] + messages

        # Normalize role names
        for item in messages:
            if "role" not in item and "from" in item:
                item["role"] = item["from"]
            if "content" not in item and "value" in item:
                item["content"] = item["value"]
            role = item["role"]
            if "human" in role:
                item["role"] = "user"
            elif "gpt" in role:
                item["role"] = "assistant"

        return messages
