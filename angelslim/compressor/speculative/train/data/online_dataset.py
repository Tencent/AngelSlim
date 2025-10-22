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

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .chat_templates import (
    ChatTemplateType,
    string_to_chat_template_type,
    template_manager,
)


class DatasetBuilder:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        shuffle_seed: int = 42,
        chat_template_type: ChatTemplateType = ChatTemplateType.QWEN3,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_seed = shuffle_seed
        self.chat_template_type = chat_template_type

        # Get chat template
        template = template_manager.get_template_dict(chat_template_type)
        self.user_header = template["user_header"]
        self.assistant_header = template["assistant_header"]

    def build_dataset(self, datapath: str, num_proc: int = 8) -> Dataset:
        try:
            # Load and shuffle dataset
            ds = load_dataset("json", data_files=datapath)
            ds = ds["train"].shuffle(seed=self.shuffle_seed)

            # Store original columns for removal
            original_columns = ds.column_names

            # Apply preprocessing
            processed_ds = ds.map(
                self._preprocess_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=original_columns,
                load_from_cache_file=False,
                desc="Processing conversations",
            )

            # Filter out None results
            processed_ds = processed_ds.filter(lambda x: x["input_ids"] is not None)
            processed_ds.set_format(type="torch")

            return processed_ds

        except Exception as e:
            raise RuntimeError(f"Dataset building failed for {datapath}") from e

    def _preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        new_examples = {"input_ids": [], "attention_mask": [], "loss_mask": []}

        for i in range(len(examples["id"])):
            try:
                processed_example = self._process_single_conversation(
                    examples["conversations"][i]
                )

                if processed_example is not None:
                    for key, value in processed_example.items():
                        if key in new_examples:
                            new_examples[key].append(value)

            except Exception as e:
                # TODO: rank0 print
                print(f"Error processing example: {e}")
                # Add None placeholders to maintain batch consistency
                for key in new_examples:
                    new_examples[key].append(None)

        return new_examples

    def _process_single_conversation(
        self, conversation_data: List[Dict]
    ) -> Optional[Dict]:
        if not conversation_data or not isinstance(conversation_data, list):
            return None

        try:
            # Build messages with system prompt
            messages = self._build_messages(conversation_data)
            if not messages:
                return None

            # Apply chat template
            conversation = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Tokenize conversation
            encoding = self.tokenizer(
                conversation,
                return_offsets_mapping=True,
                max_length=self.max_length,
                truncation=True,
                padding=False,
            )

            input_ids = encoding.input_ids
            offsets = encoding.offset_mapping

            # Create loss mask for assistant responses
            loss_mask = self._create_loss_mask_from_offsets(conversation, offsets)
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.ones_like(input_ids)

            return {
                "input_ids": input_ids[None, :],
                "attention_mask": attention_mask[None, :],
                "loss_mask": loss_mask[None, :],
            }

        except Exception as e:
            # TODO: rank0 print
            print(f"Error processing conversation: {e}")
            return None

    # Copied from https://github.com/NickL77/BaldEagle/blob/master/generate_data/generate_data.py # noqa: E501
    def _create_loss_mask_from_offsets(
        self, conversation: str, offsets: torch.Tensor
    ) -> torch.Tensor:
        loss_mask = torch.zeros(len(offsets), dtype=torch.long)

        # Find all assistant response spans
        assistant_pattern = (
            re.escape(self.assistant_header)
            + r"(.*?)(?="
            + re.escape(self.user_header)
            + "|$)"
        )

        for match in re.finditer(assistant_pattern, conversation, re.DOTALL):
            # Get the actual response content (excluding header)
            response_start = match.start(1)
            response_end = match.end(1)

            # Mark tokens that overlap with assistant response
            for idx, (token_start, token_end) in enumerate(offsets):

                # Check if token overlaps with assistant response span
                if not (token_end <= response_start or token_start > response_end):
                    loss_mask[idx] = 1

        return loss_mask

    def _build_messages(self, source: List[Dict]) -> List[Dict]:
        # System message
        messages = [{"role": "system", "content": self._get_system_prompt()}]

        # Role mapping
        role_mapping = {"human": "user", "gpt": "assistant"}
        expected_roles = ["user", "assistant"]

        # Ensure conversation starts with user
        if source[0]["from"] != "human":
            source = source[1:]

        # Filter and validate conversation turns
        valid_turns = []
        for turn in source:
            # TODO: 数据集改成openai格式
            if not isinstance(turn, dict) or "from" not in turn or "value" not in turn:
                continue

            role = role_mapping.get(turn["from"])
            if role and turn["value"].strip():
                valid_turns.append({"role": role, "content": turn["value"].strip()})

        # Validate alternating pattern
        for i, turn in enumerate(valid_turns):
            expected_role = expected_roles[i % 2]
            if turn["role"] != expected_role:
                break
            messages.append(turn)

        return messages if len(messages) > 1 else []

    def _get_system_prompt(self) -> str:
        """Get the system prompt for conversations."""
        return (
            "You are a helpful, respectful and honest assistant. "
            "Always answer as helpfully as possible, while being safe. "
            "Your answers should not include any harmful, unethical, racist, "
            "sexist, toxic, dangerous, or illegal content. Please ensure that "
            "your responses are socially unbiased and positive in nature.\n\n"
            "If a question does not make any sense, or is not factually coherent, "
            "explain why instead of answering something not correct. If you don't "
            "know the answer to a question, please don't share false information."
        )


class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [
                self.paddingtensor2D(item["attention_mask"], max_length)
                for item in features
            ]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


class DatasetManager:
    """
    Simplified DatasetManager for train_eagle3_online.py.

    This manager is designed to work with DataArguments from train_eagle3_online.py
    and provides a simple interface to create train and eval datasets.
    """

    def __init__(
        self,
        data_args,
        tokenizer: AutoTokenizer,
        model_max_length: int = 2048,
        chat_template_type: Optional[Union[str, ChatTemplateType]] = None,
    ):
        """
        Initialize DatasetManager with DataArguments.

        Args:
            data_args: DataArguments object from train_eagle3_online.py
            tokenizer: Tokenizer for the model
            model_max_length: Maximum sequence length
            chat_template_type: Chat template type. Can be:
                - ChatTemplateType enum value (e.g., ChatTemplateType.QWEN3)
                - String (e.g., "llama", "qwen")
                - None (will default to LLAMA)
        """
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

        # Convert chat_template_type to ChatTemplateType enum
        if chat_template_type is None:
            # Default to QWEN3
            chat_template_type = ChatTemplateType.QWEN3
        elif isinstance(chat_template_type, str):
            # Convert string to enum
            chat_template_type = string_to_chat_template_type(chat_template_type)

        # Create dataset builder
        self.dataset_builder = DatasetBuilder(
            tokenizer=tokenizer,
            max_length=model_max_length,
            shuffle_seed=data_args.shuffle_seed,
            chat_template_type=chat_template_type,
        )

    def create_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Create train and eval datasets based on DataArguments.

        Returns:
            Tuple of (train_dataset, eval_dataset)
            eval_dataset will be None if eval_data_path is not provided
        """
        # Determine number of processes
        num_proc = self.data_args.num_proc
        if self.data_args.preprocessing_num_workers is not None:
            num_proc = self.data_args.preprocessing_num_workers

        # Create train dataset
        train_dataset = self.dataset_builder.build_dataset(
            self.data_args.train_data_path, num_proc=num_proc
        )

        # Create eval dataset if path is provided
        eval_dataset = None
        if self.data_args.eval_data_path is not None:
            eval_dataset = self.dataset_builder.build_dataset(
                self.data_args.eval_data_path, num_proc=num_proc
            )

        return train_dataset, eval_dataset
