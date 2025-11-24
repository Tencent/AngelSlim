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
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from angelslim.utils import rank0_print

from ..chat_templates import ChatTemplateType, template_manager


class DatasetBuilder(metaclass=ABCMeta):
    @abstractmethod
    def build_dataset(
        self, datapath: str, num_proc: int = 8, shuffle: bool = True, **kwargs
    ) -> Dataset:
        pass

    @abstractmethod
    def get_data_collator(self) -> Any:
        pass


class OnlineDatasetBuilder(DatasetBuilder):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        shuffle_seed: int = 42,
        chat_template_type: ChatTemplateType = ChatTemplateType.QWEN3,
        display: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_seed = shuffle_seed
        self.chat_template_type = chat_template_type
        self.display = display
        self.display_count = 0  # Track how many samples have been displayed

        # Get chat template
        template = template_manager.get_template_dict(chat_template_type)
        self.user_header = template["user_header"]
        self.assistant_header = template["assistant_header"]
        self.system_prompt = template["system_prompt"]

    def _visualize_loss_mask(
        self, input_ids: torch.Tensor, loss_mask: torch.Tensor, conversation: str
    ) -> None:
        """
        Visualize loss_mask with color-coded output.

        Args:
            input_ids: Token IDs
            loss_mask: Loss mask tensor (1 for training, 0 for ignoring)
            conversation: Original conversation text
        """
        # ANSI color codes
        RED = "\033[91m"  # For masked out tokens (loss_mask=0)
        GREEN = "\033[92m"  # For training tokens (loss_mask=1)
        RESET = "\033[0m"  # Reset color
        BOLD = "\033[1m"

        rank0_print("\n" + "=" * 80)
        rank0_print(f"{BOLD}Loss Mask Visualization{RESET}")
        rank0_print("=" * 80)

        # Display legend
        rank0_print(f"\n{BOLD}Legend:{RESET}")
        rank0_print(f"{GREEN}■ Green: Training tokens (loss_mask=1){RESET}")
        rank0_print(f"{RED}■ Red: Ignored tokens (loss_mask=0){RESET}")

        # Display statistics
        total_tokens = len(loss_mask)
        training_tokens = loss_mask.sum().item()
        ignored_tokens = total_tokens - training_tokens
        training_ratio = training_tokens / total_tokens * 100 if total_tokens > 0 else 0

        rank0_print(f"\n{BOLD}Statistics:{RESET}")
        rank0_print(f"Total tokens: {total_tokens}")
        rank0_print(f"Training tokens: {training_tokens} ({training_ratio:.2f}%)")
        rank0_print(f"Ignored tokens: {ignored_tokens} ({100-training_ratio:.2f}%)")

        # Display token-by-token visualization
        rank0_print(f"\n{BOLD}Token-by-token visualization:{RESET}")
        rank0_print("-" * 80)

        decoded_tokens = []
        for token_id, mask_value in zip(input_ids, loss_mask):
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)

            # Choose color based on mask value
            color = GREEN if mask_value == 1 else RED

            # Format token with color
            colored_token = f"{color}{token_text}{RESET}"
            decoded_tokens.append(colored_token)

        # Print all tokens directly
        rank0_print("".join(decoded_tokens))

        # Display original conversation for reference
        rank0_print(f"\n{BOLD}Original conversation:{RESET}")
        rank0_print("-" * 80)
        rank0_print(conversation)
        rank0_print("=" * 80 + "\n")

    def build_dataset(
        self,
        datapath: str,
        num_proc: int = 8,
        shuffle: bool = True,
        sample_num: Optional[int] = None,
    ) -> Dataset:
        try:
            # Load dataset
            ds = load_dataset("json", data_files=datapath)

            # Conditionally shuffle dataset
            if shuffle:
                ds = ds["train"].shuffle(seed=self.shuffle_seed)
            else:
                ds = ds["train"]

            if sample_num is not None and 0 < sample_num < len(ds):
                ds = ds.select(range(sample_num))

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

            # Filter out None results with multiprocessing support
            processed_ds = processed_ds.filter(
                lambda batch: [ids is not None for ids in batch["input_ids"]],
                batched=True,
                num_proc=num_proc,
                desc="Filtering empty input_ids",
            )
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
                rank0_print(f"Error processing example: {e}")
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

            # Visualize loss mask if display mode is enabled
            if self.display and self.display_count == 0:
                self._visualize_loss_mask(input_ids, loss_mask, conversation)
                self.display_count += 1

            return {
                "input_ids": input_ids[None, :],
                "attention_mask": attention_mask[None, :],
                "loss_mask": loss_mask[None, :],
            }

        except Exception as e:
            rank0_print(f"Error processing conversation: {e}")
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
        if source[0]["role"] != "system":
            messages = [{"role": "system", "content": self.system_prompt}]
        else:
            messages = [{"role": "system", "content": source[0]["content"]}]
            source = source[1:]

        # Role mapping
        expected_roles = ["user", "assistant"]

        # Ensure conversation starts with user
        if source[0]["role"] != "user":
            source = source[1:]

        # Filter and validate conversation turns
        valid_turns = []
        for turn in source:
            if (
                not isinstance(turn, dict)
                or "role" not in turn
                or "content" not in turn
            ):
                continue

            role = turn["role"]
            content = turn["content"]
            if isinstance(content, str):
                content = content.strip()
            if role and content:
                valid_turns.append({"role": role, "content": content})

        # Validate alternating pattern
        for i, turn in enumerate(valid_turns):
            expected_role = expected_roles[i % 2]
            if turn["role"] != expected_role:
                break
            messages.append(turn)

        return messages if len(messages) > 1 else []
