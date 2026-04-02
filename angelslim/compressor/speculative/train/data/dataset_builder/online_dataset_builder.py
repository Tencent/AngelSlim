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
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
import torch
from datasets import Features, Value, load_dataset
from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from transformers.image_utils import load_image
from transformers.pipelines.audio_utils import ffmpeg_read

from angelslim.utils import rank0_print

from ......utils.lazy_imports import onnxruntime, torchaudio, whisper
from ......utils.utils import decide_device_for_distributed
from ....inference.models.eagle3.target.modeling_cosyvoice3_kv import mel_spectrogram
from ..chat_templates import ChatTemplateType
from ..data_utils import (
    AudioDataCollatorWithPadding,
    CosyVoice3DataCollatorWithPadding,
    DataCollatorWithPadding,
    VLMDataCollatorWithPadding,
    VLMHunyuanDataCollatorWithPadding,
    build_image_processor_kwargs,
)
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

    def _process_single_conversation(self, conversation_data):
        """
        compatible with two format:
        {"id": "0", "conversations": [
            {"role": "user", "content": "xxx"},
            {"role": "assistant", "content": "xxx"}
        ]}
        {"id": "0", "conversations": [
            {"role": "user", "content": [{"type": "text", "text": "xxx"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "xxx"}]}
        ]}
        """
        if conversation_data:
            for message in conversation_data:
                content = message.get("content")
                if isinstance(content, list):
                    message["content"] = self._normalize_content(content)
        return super()._process_single_conversation(conversation_data)


@DatasetBuilderFactory.register("online", "VLM", "qwen2_5_vl")
@DatasetBuilderFactory.register("online", "VLM", "qwen3_vl")
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
        _max_pixels = os.environ.get("MAX_PIXELS")
        _min_pixels = os.environ.get("MIN_PIXELS", "1024")
        self.max_pixels = int(_max_pixels) if _max_pixels is not None else None
        self.min_pixels = int(_min_pixels) if _min_pixels is not None else None
        rank0_print(f"max_pixels: {self.max_pixels}, min_pixels: {self.min_pixels}")

    def build_dataset(
        self,
        datapath: str,
        num_proc: int = 8,
        shuffle: bool = True,
        sample_num: Optional[int] = None,
        min_loss_tokens: Optional[int] = None,
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
            if min_loss_tokens is not None:
                processed_ds = processed_ds.filter(
                    lambda batch: [
                        sum(sum(x) if isinstance(x, list) else x for x in m) >= min_loss_tokens
                        for m in batch["loss_mask"]
                    ],
                    batched=True,
                    num_proc=num_proc,
                    desc=f"Filtering sequences with loss tokens < {min_loss_tokens}",
                )

            torch_columns = [c for c in processed_ds.column_names if c != "image_paths"]
            processed_ds.set_format(type="torch", columns=torch_columns, output_all_columns=True)

            return processed_ds

        except Exception as e:
            raise RuntimeError(f"Dataset building failed for {datapath}") from e

    def get_data_collator(self) -> Any:
        # for online vlm training: dynamically compute pixel_values during collate stage
        image_processor_kwargs = {}
        if self.max_pixels is not None:
            image_processor_kwargs["max_pixels"] = self.max_pixels
        if self.min_pixels is not None:
            image_processor_kwargs["min_pixels"] = self.min_pixels
        return VLMDataCollatorWithPadding(
            processor=self.tokenizer,
            image_processor_kwargs=image_processor_kwargs or None,
        )

    def _preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        new_examples = {
            "input_ids": [],
            "attention_mask": [],
            "loss_mask": [],
            "image_paths": [],
        }

        for i in range(len(examples["id"])):
            try:
                processed_example = self._process_single_conversation(examples["conversations"][i])

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

        return cleaned_new_examples

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

    def _get_image_size(self, image_source: str) -> tuple:
        """
        Get image (width, height), compatible with local file paths, URLs, base64, etc.

        For local file paths, uses PIL.Image.open() to quickly read the file header
        for dimensions (without decoding pixels), and manually handles EXIF orientation
        to match the exif_transpose behavior of load_image().
        For URLs/base64 and other non-local sources, uses
        transformers.image_utils.load_image to fully load the image
        (which internally calls exif_transpose + convert("RGB")), then reads dimensions.
        """
        if os.path.isfile(image_source):
            # Local file: quickly read file header for dimensions without decoding pixels
            with Image.open(image_source) as img:
                width, height = img.size
                # Check EXIF orientation; swap width/height if 90°/270° rotation is needed
                # Mimics the exif_transpose behavior in load_image()
                try:
                    exif = img.getexif()
                    orientation = exif.get(0x0112)  # EXIF Orientation tag
                    # orientation 5,6,7,8 involve 90°/270° rotation, need to swap width/height
                    if orientation is not None and orientation >= 5:
                        width, height = height, width
                except Exception:
                    pass
            return width, height
        else:
            # URL/base64/other formats: use load_image to fully load the image
            img = load_image(image_source)
            return img.size

    def _process_single_conversation(self, conversation_data: List[Dict]) -> Optional[Dict]:
        if not conversation_data or not isinstance(conversation_data, list):
            return None

        try:
            # Build messages with system prompt
            messages = self._build_messages(conversation_data)
            if not messages:
                return None

            # extract image paths before apply_chat_template modifies messages in-place
            image_paths = []
            for message in messages:
                content = message.get("content", [])
                if not isinstance(content, list):
                    continue
                for item in content:
                    if item.get("type") == "image" and item.get("image"):
                        image_paths.append(item["image"])

            # Apply chat template
            assert isinstance(messages, list), f"type(messages)={type(messages)} is not list"
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

            # ====================================================================
            # Performance optimization: avoid loading full images during preprocessing
            #
            # The original apply_chat_template(tokenize=True) internally calls
            # image_processor to fully load and process each image (resize, compute
            # pixel_values, etc.), which is very time-consuming.
            # However, the preprocessing stage only needs input_ids/loss_mask, not
            # pixel_values (which are computed in VLMDataCollatorWithPadding's
            # collate stage).
            #
            # Optimization approach:
            #  1. apply_chat_template(tokenize=False) renders the Jinja template,no image loading
            #  2. PIL.Image.open().size to quickly get image dimensions
            #  3. _get_num_multimodal_tokens to compute token count per image based on dimensions
            #  4. Manually expand <|image_pad|> placeholders to the correct count in the text
            #  5. Use tokenizer for tokenization (bypassing processor's image loading pipeline)
            # ====================================================================

            # Step 1: Get formatted text (no tokenization, no image loading)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Step 2 & 3: If images exist, get dimensions and compute token count per image
            image_token = getattr(self.tokenizer, "image_token", "<|image_pad|>")
            if image_paths and hasattr(self.tokenizer, "image_processor"):
                image_processor = self.tokenizer.image_processor
                merge_size = getattr(image_processor, "merge_size", 2)

                # Build kwargs required by get_number_of_image_patches
                # Note: get_number_of_image_patches expects "min_pixels"/"max_pixels" keys,
                # cannot use build_image_processor_kwargs (Qwen3-VL returns "size" format)
                patches_kwargs = {}
                if self.max_pixels is not None:
                    patches_kwargs["max_pixels"] = self.max_pixels
                if self.min_pixels is not None:
                    patches_kwargs["min_pixels"] = self.min_pixels

                # Step 4: Replace image_pad placeholder with the correct number of tokens per image
                for img_path in image_paths:
                    # Get image dimensions, compatible with local files/URLs/base64 etc.
                    width, height = self._get_image_size(img_path)
                    num_patches = image_processor.get_number_of_image_patches(
                        height, width, patches_kwargs
                    )
                    num_tokens = num_patches // (merge_size**2)
                    # Replace a single <|image_pad|> with num_tokens <|image_pad|> tokens
                    text = text.replace(image_token, "<|placeholder|>" * num_tokens, 1)
                text = text.replace("<|placeholder|>", image_token)

            # Step 5: Tokenize with tokenizer (bypassing processor's image loading pipeline)
            encoding = self.tokenizer.tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                add_special_tokens=False,
            )

            input_ids = encoding["input_ids"]
            offsets = encoding["offset_mapping"]

            conversation = self.tokenizer.tokenizer.decode(input_ids[0], skip_special_tokens=False)

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
                "image_paths": json.dumps(image_paths),
            }

            return result_dict

        except Exception as e:
            rank0_print(f"Error processing conversation: {e}")
            return None


@DatasetBuilderFactory.register("online", "VLM", "hunyuan_vl")
class OnlineVLMHunyuanVLDatasetBuilder(OnlineDatasetBuilder):
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
        _max_pixels = os.environ.get("MAX_PIXELS")
        _min_pixels = os.environ.get("MIN_PIXELS", "1024")
        self.max_pixels = int(_max_pixels) if _max_pixels is not None else None
        self.min_pixels = int(_min_pixels) if _min_pixels is not None else None
        rank0_print(f"max_pixels: {self.max_pixels}, min_pixels: {self.min_pixels}")

    def build_dataset(
        self,
        datapath: str,
        num_proc: int = 8,
        shuffle: bool = True,
        sample_num: Optional[int] = None,
        min_loss_tokens: Optional[int] = None,
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
            if min_loss_tokens is not None:
                processed_ds = processed_ds.filter(
                    lambda batch: [
                        sum(sum(x) if isinstance(x, list) else x for x in m) >= min_loss_tokens
                        for m in batch["loss_mask"]
                    ],
                    batched=True,
                    num_proc=num_proc,
                    desc=f"Filtering sequences with loss tokens < {min_loss_tokens}",
                )
            torch_columns = [c for c in processed_ds.column_names if c != "image_paths"]
            processed_ds.set_format(type="torch", columns=torch_columns, output_all_columns=True)

            return processed_ds

        except Exception as e:
            raise RuntimeError(f"Dataset building failed for {datapath}") from e

    def get_data_collator(self) -> Any:
        # for online training, we need to use VLMHunyuanDataCollatorWithPadding
        image_processor_kwargs = {}
        if self.max_pixels is not None:
            image_processor_kwargs["max_pixels"] = self.max_pixels
        if self.min_pixels is not None:
            image_processor_kwargs["min_pixels"] = self.min_pixels
        return VLMHunyuanDataCollatorWithPadding(
            processor=self.tokenizer,
            image_processor_kwargs=image_processor_kwargs or None,
        )

    def _preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        new_examples = {
            "input_ids": [],
            "attention_mask": [],
            "loss_mask": [],
            "image_paths": [],
            "input_position_ids": [],
        }
        for i in range(len(examples["id"])):
            try:
                processed_example = self._process_single_conversation(examples["conversations"][i])
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
        return cleaned_new_examples

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

    def _process_single_conversation(self, conversation_data: List[Dict]) -> Optional[Dict]:
        if not conversation_data or not isinstance(conversation_data, list):
            return None

        try:
            for message in conversation_data:
                # adapt to hunyuan_vl
                if message["role"] == "assistant" or message["role"] == "system":
                    message["content"] = message["content"][0]["text"]
                else:
                    for content in message["content"]:
                        if "image" in content and content["image"] is None:
                            content.pop("image")
                        if "text" in content and content["text"] is None:
                            content.pop("text")

            # Build messages with system prompt
            messages = self._build_messages(conversation_data)
            if not messages:
                return None

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, _ = self._extract_vision_info(messages)

            image_kwargs = {}
            if image_inputs and hasattr(self.tokenizer, "image_processor"):
                image_kwargs = build_image_processor_kwargs(
                    self.tokenizer.image_processor, self.max_pixels, self.min_pixels
                )
            encoding = self.tokenizer(
                text=[text],
                images=image_inputs,
                return_tensors="pt",
                return_offsets_mapping=True,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                **image_kwargs,
            )
            input_ids = encoding["input_ids"]
            offsets = encoding["offset_mapping"]
            input_position_ids = encoding["position_ids"]
            conversation = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)

            # Create loss mask for assistant responses
            try:
                # loss_mask = torch.tensor(conversation_data['loss_mask']).unsqueeze(0)
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
                "input_position_ids": input_position_ids,
            }

            # get image_paths
            image_paths = []
            for message in messages:
                content = message.get("content", [])
                if not isinstance(content, list):
                    continue
                for item in content:
                    if item.get("type") == "image" and item.get("image"):
                        image_paths.append(item["image"])
            result_dict["image_paths"] = json.dumps(image_paths)

            return result_dict

        except Exception as e:
            rank0_print(f"Error processing conversation: {e}")
            return None

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
                            raise ValueError(f"Could not open image file: {item['image']}, {e}")
                    elif isinstance(item["image"], Image.Image):
                        image_paths.append(item["image"])
                elif item.get("type") == "video":
                    video_paths.append(item["video"])

        return image_paths, video_paths


@DatasetBuilderFactory.register("online", "Audio", "qwen2_audio")
class OnlineAudioDatasetBuilder(OnlineDatasetBuilder):
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
        min_loss_tokens: Optional[int] = None,
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
                                    "audio": Value("string"),
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

            if min_loss_tokens is not None:
                processed_ds = processed_ds.filter(
                    lambda batch: [
                        sum(sum(x) if isinstance(x, list) else x for x in m) >= min_loss_tokens
                        for m in batch["loss_mask"]
                    ],
                    batched=True,
                    num_proc=num_proc,
                    desc=f"Filtering sequences with loss tokens < {min_loss_tokens}",
                )

            processed_ds.set_format(type="torch")

            return processed_ds

        except Exception as e:
            raise RuntimeError(f"Dataset building failed for {datapath}") from e

    def get_data_collator(self) -> Any:
        return AudioDataCollatorWithPadding()

    def read_audio(self, audio_path):
        if audio_path.startswith("http://") or audio_path.startswith("https://"):
            inputs = requests.get(audio_path).content
        else:
            with open(audio_path, "rb") as f:
                inputs = f.read()
        return inputs

    def _preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        new_examples = {
            "input_ids": [],
            "attention_mask": [],
            "loss_mask": [],
            "input_features": [],
            "feature_attention_mask": [],
        }

        for i in range(len(examples["id"])):
            try:
                processed_example = self._process_single_conversation(examples["conversations"][i])

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

        return cleaned_new_examples

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

    def _extract_audio_info(self, messages: List[Dict]) -> tuple:
        """Extract Audio paths from messages"""
        audio_paths = []

        sampling_rate = self.tokenizer.feature_extractor.sampling_rate
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue

            for item in content:
                if item.get("type") == "audio":
                    # Handle both file paths and PIL images
                    if isinstance(item["audio"], str):
                        try:
                            audio_paths.append(
                                ffmpeg_read(
                                    self.read_audio(item["audio"]),
                                    sampling_rate=sampling_rate,
                                )
                            )
                        except ValueError as e:
                            raise ValueError(f"Could not open audio file: {item['audio']}, {e}")
        return audio_paths

    def _process_single_conversation(self, conversation_data: List[Dict]) -> Optional[Dict]:
        if not conversation_data or not isinstance(conversation_data, list):
            return None

        try:
            # Build messages with system prompt
            messages = self._build_messages(conversation_data)
            if not messages:
                return None

            # Apply chat template
            assert isinstance(messages, list), f"type(messages)={type(messages)} is not list"
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

            input_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            input_audios = self._extract_audio_info(messages)

            # cannot set max_length,
            # otherwise the input_ids audio token length will be aligned(missing)
            encoding = self.tokenizer(
                text=input_text,
                audio=input_audios,
                sampling_rate=self.tokenizer.feature_extractor.sampling_rate,
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=True,
                padding=False,
            )
            input_ids = encoding["input_ids"]
            offsets = encoding["offset_mapping"]

            conversation = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)

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

            if "input_features" in encoding:
                result_dict["input_features"] = encoding["input_features"]
            if "feature_attention_mask" in encoding:
                result_dict["feature_attention_mask"] = encoding["feature_attention_mask"]

            return result_dict

        except Exception as e:
            rank0_print(f"Error processing conversation: {e}")
            return None


@DatasetBuilderFactory.register("online", "TTS")
class OnlineTTSDatasetBuilder(OnlineDatasetBuilder):
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
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.global_rank = int(os.getenv("RANK", -1))
        self.output_dir = kwargs["output_dir"]
        self.device = decide_device_for_distributed()

        self.model_path = kwargs["target_model_name_or_path"]
        if not os.path.exists(self.model_path):
            self.model_path = snapshot_download(self.model_path)

        if os.path.exists(os.path.join(self.model_path, "cosyvoice3.yaml")):
            self.model_name = "cosyvoice3"
            onnx_path = os.path.join(self.model_path, "speech_tokenizer_v3.onnx")
            self._init_audio_tokenizer_cosyvoice3(onnx_path)
            self.feat_extractor = partial(
                mel_spectrogram,
                n_fft=1920,
                num_mels=80,
                sampling_rate=24000,
                hop_size=480,
                win_size=1920,
                fmin=0,
                fmax=None,
                center=False,
            )

    def _init_audio_tokenizer_cosyvoice3(self, onnx_path) -> None:
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ["CUDAExecutionProvider"]
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            onnx_path, sess_options=option, providers=providers
        )

    def get_data_collator(self) -> Any:
        if self.model_name == "cosyvoice3":
            return CosyVoice3DataCollatorWithPadding()

    def read_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        data = []
        try:
            for file in file_path:
                with open(file, "r", encoding="utf-8") as f:
                    for line in tqdm(
                        f,
                        desc=f"read data file {os.path.basename(file)}",
                        disable=self.global_rank > 0,
                    ):
                        try:
                            item = json.loads(line.strip())
                            if isinstance(item, dict):
                                data.append(item)
                        except json.JSONDecodeError as e:
                            rank0_print(f"JSON extract error: {e}, line: {line[:100]}...")
                            continue
        except Exception as e:
            rank0_print(f"read data file {file_path} failed: {e}")
        return data

    def build_dataset(
        self,
        datapath: str,
        num_proc: int = 8,
        shuffle: bool = True,
        sample_num: Optional[int] = None,
        min_loss_tokens: Optional[int] = None,
    ) -> Dataset:
        try:
            if not isinstance(datapath, list):
                datapath = [datapath]
            data_name = "_"
            for path in datapath:
                data_name += os.path.basename(path)[:-6]
            os.makedirs(self.output_dir, exist_ok=True)
            cache_path = os.path.join(self.output_dir, f"processed{data_name}_merged_cache.jsonl")

            if not os.path.exists(cache_path):
                raw_data = self.read_jsonl_file(datapath)
                chunk_size = len(raw_data) // self.world_size
                start_idx = self.global_rank * chunk_size
                end_idx = (
                    start_idx + chunk_size
                    if self.global_rank < self.world_size - 1
                    else len(raw_data)
                )
                rank_data = raw_data[start_idx:end_idx]
                processed_data = []
                count = 0
                for item in tqdm(
                    rank_data,
                    desc=f"Rank {self.global_rank} process data",
                    disable=self.global_rank > 0,
                ):
                    if sample_num is not None and count == sample_num // self.world_size:
                        break
                    text = item.get("text", "")
                    audio_tokens = item.get("audio_tokens", None)
                    audio_path = item.get("audio_path", "")
                    instruct = item.get("instruct", "")
                    instruct_audio_path = item.get("instruct_audio_path", "")

                    if self.model_name == "cosyvoice3":
                        processed = self._process_single_item_cosyvoice3(
                            text,
                            audio_tokens,
                            audio_path,
                            instruct,
                            instruct_audio_path,
                        )
                    else:
                        raise NotImplementedError("This model is not implemented")

                    processed_data.append(processed)
                    count += 1

                # save for each rank
                rank_file = os.path.join(
                    self.output_dir,
                    f"processed{data_name}_rank_{self.global_rank}.jsonl",
                )
                with open(rank_file, "w", encoding="utf-8") as f:
                    for item in processed_data:
                        f.write(json.dumps(item, ensure_ascii=True) + "\n")
                done_file = os.path.join(
                    self.output_dir,
                    f"processed{data_name}_rank_{self.global_rank}.done",
                )
                Path(done_file).touch()
                self._wait_for_all_ranks_done(self.output_dir, data_name, self.world_size)

                # merge processed data on rank 0
                merge_done_file = os.path.join(
                    self.output_dir, f"processed{data_name}_merged_cache.done"
                )
                if self.global_rank == 0:
                    all_processed_data = []
                    for rank in range(self.world_size):
                        rank_data = []
                        rank_tmp_file = os.path.join(
                            self.output_dir, f"processed{data_name}_rank_{rank}.jsonl"
                        )
                        with open(rank_tmp_file, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    rank_data.append(json.loads(line.strip()))
                        all_processed_data.extend(rank_data)

                    with open(cache_path, "w", encoding="utf-8") as f:
                        for item in all_processed_data:
                            f.write(json.dumps(item, ensure_ascii=True) + "\n")

                    for rank in range(self.world_size):
                        rank_tmp_file = os.path.join(
                            self.output_dir, f"processed{data_name}_rank_{rank}.jsonl"
                        )
                        rank_done_file = os.path.join(
                            self.output_dir, f"processed{data_name}_rank_{rank}.done"
                        )
                        if os.path.exists(rank_tmp_file):
                            os.remove(rank_tmp_file)
                        if os.path.exists(rank_done_file):
                            os.remove(rank_done_file)

                    with open(merge_done_file, "w") as f:
                        f.write("Merged done")
                    rank0_print("Rank 0: Created merge completion marker")
                else:
                    merge_done = False
                    while not merge_done:
                        if os.path.exists(merge_done_file):
                            merge_done = True
                            break

                # Load dataset
                processed_ds = load_dataset("json", data_files=cache_path)

                # Conditionally shuffle dataset
                if shuffle:
                    processed_ds = processed_ds["train"].shuffle(seed=self.shuffle_seed)
                else:
                    processed_ds = processed_ds["train"]

                # Filter out None results with multiprocessing support
                processed_ds = processed_ds.filter(
                    lambda batch: [ids is not None for ids in batch["speech_token"]],
                    batched=True,
                    num_proc=num_proc,
                    desc="Filtering empty speech_token",
                )

                processed_ds.set_format(type="torch")
                return processed_ds

            else:
                # Load dataset
                rank0_print(f"Loading cache data from {cache_path}")
                ds = load_dataset("json", data_files=cache_path)

                # Conditionally shuffle dataset
                if shuffle:
                    ds = ds["train"].shuffle(seed=self.shuffle_seed)
                else:
                    ds = ds["train"]

                # Filter out None results with multiprocessing support
                ds = ds.filter(
                    lambda batch: [ids is not None for ids in batch["speech_token"]],
                    batched=True,
                    num_proc=num_proc,
                    desc="Filtering empty speech_token",
                )

                ds.set_format(type="torch")
                return ds

        except Exception as e:
            raise RuntimeError(f"Dataset building failed for {datapath}") from e

    def _wait_for_all_ranks_done(self, output_dir, data_name, world_size):
        all_done = False
        while not all_done:
            done_count = 0
            for rank in range(world_size):
                done_file = os.path.join(output_dir, f"processed{data_name}_rank_{rank}.done")
                if os.path.exists(done_file):
                    done_count += 1

            if done_count == world_size:
                all_done = True
                break

    def _process_single_item_cosyvoice3(
        self,
        text: str,
        audio_tokens: Optional[list],
        audio_path: str,
        instruct: Dict[str, Any],
        instruct_audio_path: str,
    ) -> Optional[Dict[str, Any]]:
        text_token = self.tokenizer.encode(text)
        instruct_token = self.tokenizer.encode(instruct)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(instruct_audio_path)
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(
            instruct_audio_path
        )

        resample_rate = 24000
        if resample_rate == 24000:
            token_len = min(int(prompt_speech_feat.shape[1] / 2), prompt_speech_token.shape[1])
            prompt_speech_feat, prompt_speech_feat_len[:] = (
                prompt_speech_feat[:, : 2 * token_len],
                2 * token_len,
            )
            prompt_speech_token, prompt_speech_token_len[:] = (
                prompt_speech_token[:, :token_len],
                token_len,
            )

        if audio_tokens is not None:
            return {
                "text": text_token,
                "text_len": len(text_token),
                "speech_token": audio_tokens,
                "speech_token_len": len(audio_tokens),
                "prompt_speech_token": prompt_speech_token.squeeze(0).tolist(),
                "prompt_speech_token_len": prompt_speech_token_len.item(),
                "prompt_text": instruct_token,
                "prompt_text_len": len(instruct_token),
            }

        speech_token, speech_token_len = self._extract_speech_token(audio_path)
        return {
            "text": text_token,
            "text_len": len(text_token),
            "speech_token": speech_token.squeeze(0).tolist(),
            "speech_token_len": speech_token_len.item(),
            "prompt_speech_token": prompt_speech_token.squeeze(0).tolist(),
            "prompt_speech_token_len": prompt_speech_token_len.item(),
            "prompt_text": instruct_token,
            "prompt_text_len": len(instruct_token),
        }

    def _extract_speech_token(self, wav):
        speech = self.load_wav(wav, 16000)
        assert (
            speech.shape[1] / 16000 <= 30
        ), "do not support extract speech token for audio longer than 30s"
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = (
            self.speech_tokenizer_session.run(
                None,
                {
                    self.speech_tokenizer_session.get_inputs()[0]
                    .name: feat.detach()
                    .cpu()
                    .numpy(),
                    self.speech_tokenizer_session.get_inputs()[1].name: np.array(
                        [feat.shape[2]], dtype=np.int32
                    ),
                },
            )[0]
            .flatten()
            .tolist()
        )
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len

    def _extract_speech_feat(self, wav):
        speech = self.load_wav(wav, 24000)
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def load_wav(self, wav, target_sr, min_sr=16000):
        speech, sample_rate = torchaudio.load(wav, backend="soundfile")
        speech = speech.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            assert sample_rate >= min_sr, "wav sample rate {} must be greater than {}".format(
                sample_rate, target_sr
            )
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(
                speech
            )
        return speech
