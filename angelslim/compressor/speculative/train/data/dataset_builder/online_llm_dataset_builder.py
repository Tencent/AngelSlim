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

from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoProcessor, AutoTokenizer

from angelslim.utils import rank0_print

from ..chat_templates import ChatTemplateType
from ..data_utils import DataCollatorWithPadding
from .base_dataset_builder import OnlineDatasetBuilder
from .dataset_builder_factory import DatasetBuilderFactory


@DatasetBuilderFactory.register("online", "LLM")
class OnlineLLMDatasetBuilder(OnlineDatasetBuilder):
    def __init__(
        self,
        tokenizer: Union[AutoTokenizer, AutoProcessor],
        max_length: int = 2048,
        shuffle_seed: int = 42,
        chat_template_type: ChatTemplateType = ChatTemplateType.QWEN3,
        display: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            tokenizer,
            max_length,
            shuffle_seed,
            chat_template_type,
            display,
        )

    def get_data_collator(self) -> Any:
        return DataCollatorWithPadding()

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

            # Check if tokenizer supports offset_mapping
            is_fast_tokenizer = (
                hasattr(self.tokenizer, "is_fast") and self.tokenizer.is_fast
            )

            # Tokenize conversation
            if is_fast_tokenizer:
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
            else:
                # For Python tokenizers, use alternative approach
                encoding = self.tokenizer(
                    conversation,
                    max_length=self.max_length,
                    truncation=True,
                    padding=False,
                )
                input_ids = torch.tensor(encoding.input_ids)
                # Create loss mask without offsets (alternative implementation needed)
                loss_mask = self._create_loss_mask_without_offsets(
                    conversation, input_ids
                )

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

    def _create_loss_mask_without_offsets(self, conversation, input_ids):
        # Implement alternative loss mask creation logic for Python tokenizers
        loss_mask = torch.ones_like(input_ids)

        turns = conversation.split(self.user_header)
        if len(turns) == 1:
            # Handle single-turn conversations
            parts = turns[0].split(self.assistant_header)
            instruction_part = parts[0] + self.assistant_header
            instruction_len = len(self.tokenizer(instruction_part).input_ids)
            loss_mask[:instruction_len] = 0
        else:
            # Handle multi-turn conversations
            cur_len = 0
            user_header_len = len((self.tokenizer(self.user_header)).input_ids)

            for _, turn in enumerate(turns):
                parts = turn.split(self.assistant_header)
                instruction_part = parts[0] + self.assistant_header

                instruction_len = len(self.tokenizer(instruction_part).input_ids)
                loss_mask[cur_len : cur_len + instruction_len] = 0

                turn_len = len(self.tokenizer(turn).input_ids)
                cur_len += turn_len
                cur_len += user_header_len

                loss_mask[cur_len - user_header_len : cur_len] = 0

        return loss_mask
