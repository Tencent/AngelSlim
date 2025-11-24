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

from typing import Any, Optional, Tuple, Union

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .chat_templates import ChatTemplateType, string_to_chat_template_type
from .dataset_builder import DatasetBuilderFactory


class DatasetManager:
    """
    Unified DatasetManager for EAGLE3 training.

    This manager supports creating datasets for:
    - Offline mode: Loads pre-computed hidden states from .ckpt files for training
    - Online mode: Processes raw conversation data on-the-fly

    Can create both types of datasets simultaneously when needed.
    """

    def __init__(
        self,
        data_args,
        tokenizer: Optional[AutoTokenizer] = None,
        model_max_length: int = 2048,
        chat_template_type: Optional[Union[str, ChatTemplateType]] = None,
        display: bool = False,
        cache_in_memory: bool = False,
    ):
        """
        Initialize DatasetManager with DataArguments.

        Args:
            data_args: DataArguments object containing data paths and configurations
            tokenizer: Tokenizer for the model (required for online dataset processing)
            model_max_length: Maximum sequence length
            chat_template_type: Chat template type for conversation formatting. Can be:
                - ChatTemplateType enum value (e.g., ChatTemplateType.QWEN3)
                - String (e.g., "llama", "qwen")
                - None (will default to QWEN3)
            display: Whether to display loss mask visualization for the first sample
            cache_in_memory: Whether to cache all data in memory for offline datasets
        """
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.display = display
        self.cache_in_memory = cache_in_memory

        # Convert chat_template_type to ChatTemplateType enum
        if chat_template_type is None:
            chat_template_type = ChatTemplateType.QWEN3
        elif isinstance(chat_template_type, str):
            chat_template_type = string_to_chat_template_type(chat_template_type)

        self.chat_template_type = chat_template_type

        # Create dataset builder for online processing
        self.online_dataset_builder = None
        self.offline_dataset_builder = None
        if tokenizer is not None:
            self.online_dataset_builder = DatasetBuilderFactory.create(
                training_mode="online",
                modal_type=data_args.modal_type,
                tokenizer=tokenizer,
                max_length=model_max_length,
                shuffle_seed=data_args.shuffle_seed,
                chat_template_type=chat_template_type,
                display=display,
            )
        if data_args.training_mode == "offline":
            self.offline_dataset_builder = DatasetBuilderFactory.create(
                training_mode="offline",
                modal_type=data_args.modal_type,
                cache_in_memory=cache_in_memory,
            )

    def create_all_datasets(
        self,
    ) -> Tuple[Dataset, Optional[Dataset], Dataset, Optional[Dataset]]:
        """
        Create all required datasets: offline and online datasets.

        Returns:
            Tuple of (offline_train_dataset, offline_eval_dataset,
                online_train_dataset, online_eval_dataset)
            - offline_train_dataset: Offline training dataset from .ckpt files
            - offline_eval_dataset: Offline evaluation dataset (None if not provided)
            - online_train_dataset: Online training dataset from raw conversation data
            - online_eval_dataset: Online evaluation dataset (None if not provided)

        Raises:
            ValueError: If required paths are not provided
        """
        # Create offline datasets (from .ckpt files)
        offline_train_dataset, offline_eval_dataset, offline_data_collator = (
            self._create_offline_datasets()
        )

        # Create online datasets (from raw JSON data) if tokenizer is provided
        online_train_dataset, online_eval_dataset, online_data_collator = (
            self._create_online_datasets()
        )

        if self.data_args.training_mode == "online":
            data_collator = online_data_collator
        else:
            data_collator = offline_data_collator

        return (
            offline_train_dataset,
            offline_eval_dataset,
            online_train_dataset,
            online_eval_dataset,
            data_collator,
        )

    def create_offline_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Create offline datasets only.

        Returns:
            Tuple of (train_dataset, eval_dataset)
            eval_dataset will be None if eval_hidden_path is not provided
        """
        return self._create_offline_datasets()

    def create_online_datasets(
        self,
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Any]:
        """
        Create online datasets only.

        Returns:
            Tuple of (train_dataset, eval_dataset)
            Both will be None if tokenizer not provided
        """
        if self.tokenizer is None or self.online_dataset_builder is None:
            return None, None, None
        return self._create_online_datasets()

    def _create_online_datasets(
        self,
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Any]:
        """
        Create online datasets from raw conversation data.

        Returns:
            Tuple of (train_dataset, eval_dataset)
            eval_dataset will be None if eval_data_path is not provided
        """
        # Determine number of processes
        num_proc = self.data_args.num_proc
        if self.display:
            num_proc = None

        # Create training dataset
        train_dataset = None
        if self.data_args.train_data_path is not None:
            train_dataset = self.online_dataset_builder.build_dataset(
                self.data_args.train_data_path,
                num_proc=num_proc,
                shuffle=True,
                sample_num=self.data_args.sample_num,
            )

        # Create evaluation dataset
        eval_dataset = None
        if self.data_args.eval_data_path is not None:
            eval_dataset = self.online_dataset_builder.build_dataset(
                self.data_args.eval_data_path,
                num_proc=num_proc,
                shuffle=False,
                sample_num=self.data_args.sample_num,
            )

        data_collator = self.online_dataset_builder.get_data_collator()

        return train_dataset, eval_dataset, data_collator

    def _create_offline_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Create offline datasets from pre-computed .ckpt files.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        # Create train dataset
        train_dataset = self.offline_dataset_builder.build_dataset(
            self.data_args.train_hidden_path
        )

        # Create eval dataset if path is provided
        eval_dataset = None
        if self.data_args.eval_hidden_path is not None:
            eval_dataset = self.offline_dataset_builder.build_dataset(
                self.data_args.eval_hidden_path
            )

        data_collator = self.offline_dataset_builder.get_data_collator()

        return train_dataset, eval_dataset, data_collator
