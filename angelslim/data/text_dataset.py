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
from typing import Dict, List

import pyarrow.parquet as pq
import torch
from datasets import load_dataset
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
        is_sft_data: bool = False,
    ):
        super().__init__(processor, device, max_length)
        self.is_sft_data = is_sft_data
        self._load_data(data_path, num_samples)

    def _load_data(self, data_path: str, num_samples: int):
        if not os.path.isfile(data_path):
            self._load_hf_dataset(data_path, num_samples)
            return

        if ".parquet" in data_path.lower():
            self._load_parquet_data(data_path, num_samples)
        else:
            self._load_jsonl_data(data_path, num_samples)

    def _load_hf_dataset(self, data_path: str, num_samples: int, block_size: int = 2048):
        parts = data_path.split(",")
        dataset = load_dataset(*parts)["train"]
        total_samples = (
            min(num_samples, len(dataset["text"])) if num_samples > 0 else len(dataset["text"])
        )

        concatenated = {}
        for sample in dataset:
            tokenized = self.processor(sample["text"])
            for key in tokenized.keys():
                if key not in concatenated:
                    concatenated[key] = []
                concatenated[key].extend(tokenized[key])

        total = len(concatenated["input_ids"])
        if total >= block_size:
            total = (total // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total, block_size)]
            for k, t in concatenated.items()
        }
        for i in range(total_samples):
            inputs = {
                "input_ids": torch.tensor(result["input_ids"][i]).unsqueeze(0).to(self.device)
            }
            # HF CausalLM models shift labels internally; feed labels == input_ids.
            inputs["labels"] = inputs["input_ids"].clone()
            inputs["attention_mask"] = torch.tensor(result["attention_mask"][i]).to(self.device)
            self.data.append(inputs)

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
                # HF CausalLM models shift labels internally; feed labels == input_ids.
                labels = model_inputs["input_ids"].clone()

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

                # Find the LAST assistant turn — loss is computed ONLY on
                # this reply. Everything before it (system + user(s) +
                # earlier assistant(s)) serves as prompt context.
                last_assistant_idx = None
                for idx, item in enumerate(messages):
                    if item["role"] == "assistant":
                        last_assistant_idx = idx
                if last_assistant_idx is None:
                    # No assistant turn -> nothing to supervise; skip.
                    continue
                prompt_messages = messages[:last_assistant_idx]
                assistant_msg = messages[last_assistant_idx]

                # Tokenize the prompt (up to the generation marker) and the
                # full conversation separately so we know exactly where the
                # assistant reply starts.
                prompt_text = self.processor.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                full_messages = prompt_messages + [assistant_msg]
                full_text = self.processor.apply_chat_template(
                    full_messages, tokenize=False, add_generation_prompt=False
                )

                # Legacy branch: thinking-style data without a chat template.
                thinking_data = any(
                    m["role"] == "assistant"
                    and "<think>" in m.get("content", "")
                    and "</think>" in m.get("content", "")
                    for m in messages
                )
                if thinking_data:
                    bos = self.processor.bos_token or ""
                    prompt_text = bos
                    for m in prompt_messages:
                        if m["role"] == "system":
                            prompt_text += m["content"]
                        elif m["role"] == "user":
                            prompt_text += "<｜User｜>" + m["content"] + "<｜Assistant｜>"
                        elif m["role"] == "assistant":
                            prompt_text += m["content"] + self.processor.eos_token
                    full_text = prompt_text + assistant_msg["content"] + self.processor.eos_token

                # Token-level prompt length: count tokens in ``prompt_text``
                # without special-token insertion so it aligns with the
                # prefix of the tokenization of ``full_text``.
                prompt_ids = self.processor(
                    prompt_text,
                    add_special_tokens=False,
                    return_tensors=None,
                )["input_ids"]
                prompt_len = len(prompt_ids)

                model_inputs = self.processor(
                    text=[full_text],
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                )

                # Build labels: HF CausalLM shifts labels internally, so
                # the label at position ``t`` supervises the prediction of
                # ``input_ids[t+1]``. Positions before (and at) the end of
                # the prompt are set to -100 so they contribute no loss.
                input_ids = model_inputs["input_ids"]
                attention_mask = model_inputs["attention_mask"]
                labels = input_ids.clone()
                if self.is_sft_data:
                    labels[:, :prompt_len] = -100
                # Also mask padding tokens.
                labels[attention_mask == 0] = -100

                self.data.append(
                    {
                        "input_ids": input_ids.to(self.device),
                        "attention_mask": attention_mask.to(self.device),
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
                messages = [{"role": "system", "content": data["system_prompt"]}] + messages
        elif "conversations" in data:
            share_gpt_data = data["conversations"]
            messages = [
                {"role": "user", "content": share_gpt_data[0]["value"]},
                {"role": "assistant", "content": share_gpt_data[1]["value"]},
            ]
            if "system" in data and data["system"]:
                messages = [{"role": "system", "content": data["system_prompt"]}] + messages
        else:
            messages = [
                {"role": "user", "content": data["input"]},
                {"role": "assistant", "content": data["output"]},
            ]
            if "system_prompt" in data and data["system_prompt"]:
                messages = [{"role": "system", "content": data["system_prompt"]}] + messages

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
