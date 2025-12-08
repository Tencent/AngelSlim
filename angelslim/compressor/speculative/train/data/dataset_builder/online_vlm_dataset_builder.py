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
from datasets import Features, Value, load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer

from angelslim.utils import rank0_print

from ..chat_templates import ChatTemplateType
from ..data_utils import VLMDataCollatorWithPadding
from .base_dataset_builder import OnlineDatasetBuilder
from .dataset_builder_factory import DatasetBuilderFactory


@DatasetBuilderFactory.register("online", "VLM")
class OnlineVLMDatasetBuilder(OnlineDatasetBuilder):
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

    def build_dataset(
        self,
        datapath: str,
        num_proc: int = 8,
        shuffle: bool = True,
        sample_num: Optional[int] = None,
    ) -> Dataset:
        try:
            # Load dataset
            features = Features(
                {
                    "id": Value("string"),
                    "conversations": [
                        {
                            "role": Value("string"),
                            "content": [
                                {
                                    "type": Value("string"),
                                    "text": Value("string"),
                                    "image": Value("string"),
                                    "video": Value("string"),
                                }
                            ],
                        }
                    ],
                }
            )
            ds = load_dataset("json", data_files=datapath, features=features)

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

    def get_data_collator(self) -> Any:
        return VLMDataCollatorWithPadding()

    def _preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        new_examples = {
            "input_ids": [],
            "attention_mask": [],
            "loss_mask": [],
            "pixel_values": [],
            "video_pixel_values": [],
            "image_grid_thw": [],
            "video_grid_thw": [],
        }

        for i in range(len(examples["id"])):
            try:
                processed_example = self._process_single_conversation(
                    examples["conversations"][i]
                )

                if processed_example is not None:
                    for key in new_examples.keys():
                        if key not in processed_example:
                            new_examples[key].append(None)
                        else:
                            new_examples[key].append(processed_example[key])

            except Exception as e:
                rank0_print(f"Error processing example: {e}")
                # Add None placeholders to maintain batch consistency
                for key in new_examples:
                    new_examples[key].append(None)

        cleaned_new_examples = {}
        for key, value in new_examples.items():
            if any(v is not None for v in value):
                cleaned_new_examples[key] = value

        return new_examples

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
        input_ids = input_ids.view(-1)
        return super()._visualize_loss_mask(input_ids, loss_mask, conversation)

    def _create_loss_mask_from_offsets(
        self, conversation: str, offsets: torch.Tensor
    ) -> torch.Tensor:
        if offsets.ndim == 3:
            offsets = offsets[0]
        return super()._create_loss_mask_from_offsets(conversation, offsets)

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
            assert isinstance(
                messages, list
            ), f"type(messages)={type(messages)} is not list"
            for message in messages:
                if isinstance(message["content"], str):
                    continue
                assert isinstance(
                    message["content"], list
                ), f"content={type(message['content'])} is not str or list"
                new_content = []
                for item in message["content"]:
                    new_item = {"type": item["type"], item["type"]: item[item["type"]]}
                    new_content.append(new_item)
                del message["content"]
                message["content"] = new_content

            encoding = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                return_offsets_mapping=True,
                max_length=self.max_length,
                truncation=True,
                padding=False,
            )

            input_ids = encoding["input_ids"]
            offsets = encoding["offset_mapping"]

            conversation = self.tokenizer.decode(
                input_ids[0], skip_special_tokens=False
            )

            # Create loss mask for assistant responses
            try:
                loss_mask = self._create_loss_mask_from_offsets(conversation, offsets)
            except Exception as e:
                rank0_print(f"Error creating loss mask: {e}")
                rank0_print(f"offsets: {offsets}")
                raise e
            attention_mask = torch.ones_like(input_ids)

            # Visualize loss mask if display mode is enabled
            if self.display and self.display_count == 0:
                try:
                    self._visualize_loss_mask(input_ids, loss_mask, conversation)
                except Exception as e:
                    rank0_print(f"Error visualizing loss mask: {e}")
                    rank0_print(f"input_ids: {input_ids}, loss_mask: {loss_mask}")
                    raise e
                self.display_count += 1

            result_dict = {
                "input_ids": input_ids.view(1, -1),
                "attention_mask": attention_mask.view(1, -1),
                "loss_mask": loss_mask.view(1, -1),
            }

            if "pixel_values" in encoding:
                result_dict["pixel_values"] = encoding["pixel_values"].unsqueeze(0)
            if "video_pixel_values" in encoding:
                result_dict["video_pixel_values"] = encoding[
                    "video_pixel_values"
                ].unsqueeze(0)
            if "image_grid_thw" in encoding:
                result_dict["image_grid_thw"] = encoding["image_grid_thw"].unsqueeze(0)
            if "video_grid_thw" in encoding:
                result_dict["video_grid_thw"] = encoding["video_grid_thw"].unsqueeze(0)

            return result_dict

        except Exception as e:
            rank0_print(f"Error processing conversation: {e}")
            return None
