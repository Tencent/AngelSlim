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
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from angelslim.utils import rank0_print

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
        self, datapath: str, num_proc: int = 8, shuffle: bool = True
    ) -> Dataset:
        try:
            # Load dataset
            ds = load_dataset("json", data_files=datapath)

            # Conditionally shuffle dataset
            if shuffle:
                ds = ds["train"].shuffle(seed=self.shuffle_seed)
            else:
                ds = ds["train"]

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
                lambda x: x["input_ids"] is not None, num_proc=num_proc
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
            if role and turn["content"].strip():
                valid_turns.append({"role": role, "content": turn["content"].strip()})

        # Validate alternating pattern
        for i, turn in enumerate(valid_turns):
            expected_role = expected_roles[i % 2]
            if turn["role"] != expected_role:
                break
            messages.append(turn)

        return messages if len(messages) > 1 else []


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
        self.dataset_builder = None
        if tokenizer is not None:
            self.dataset_builder = DatasetBuilder(
                tokenizer=tokenizer,
                max_length=model_max_length,
                shuffle_seed=data_args.shuffle_seed,
                chat_template_type=chat_template_type,
                display=display,
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
        offline_train_dataset, offline_eval_dataset = self._create_offline_datasets()

        # Create online datasets (from raw JSON data) if tokenizer is provided
        online_train_dataset, online_eval_dataset = None, None
        if self.tokenizer is not None and self.dataset_builder is not None:
            online_train_dataset, online_eval_dataset = self._create_online_datasets()

        return (
            offline_train_dataset,
            offline_eval_dataset,
            online_train_dataset,
            online_eval_dataset,
        )

    def create_offline_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Create offline datasets only.

        Returns:
            Tuple of (train_dataset, eval_dataset)
            eval_dataset will be None if eval_hidden_path is not provided
        """
        return self._create_offline_datasets()

    def create_online_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset]]:
        """
        Create online datasets only.

        Returns:
            Tuple of (train_dataset, eval_dataset)
            Both will be None if tokenizer not provided
        """
        if self.tokenizer is None or self.dataset_builder is None:
            return None, None
        return self._create_online_datasets()

    def _create_online_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset]]:
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
            train_dataset = self.dataset_builder.build_dataset(
                self.data_args.train_data_path, num_proc=num_proc, shuffle=True
            )

        # Create evaluation dataset
        eval_dataset = None
        if self.data_args.eval_data_path is not None:
            eval_dataset = self.dataset_builder.build_dataset(
                self.data_args.eval_data_path, num_proc=num_proc, shuffle=False
            )

        return train_dataset, eval_dataset

    def _create_offline_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Create offline datasets from pre-computed .ckpt files.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        # Create train dataset
        train_dataset = OfflineEagle3Dataset(
            data_dir=self.data_args.train_hidden_path,
            file_pattern="*.ckpt",
            cache_in_memory=self.cache_in_memory,
        )

        # Create eval dataset if path is provided
        eval_dataset = None
        if self.data_args.eval_hidden_path is not None:
            eval_dataset = OfflineEagle3Dataset(
                data_dir=self.data_args.eval_hidden_path,
                file_pattern="*.ckpt",
                cache_in_memory=self.cache_in_memory,
            )

        return train_dataset, eval_dataset


class OfflineEagle3Dataset(Dataset):
    """
    Offline Dataset for EAGLE3 training.

    Loads pre-computed hidden states, logits, and other data from .ckpt files.
    Each .ckpt file contains a dictionary with keys: input_ids, target_logits,
    hidden_states, and loss_mask.
    """

    def __init__(
        self, data_dir: str, file_pattern: str = "*.ckpt", cache_in_memory: bool = False
    ):
        """
        Initialize the OfflineEagle3Dataset.

        Args:
            data_dir: Directory containing .ckpt files
                (will search recursively in subdirectories)
            file_pattern: Pattern to match checkpoint files (default: "*.ckpt")
            cache_in_memory: Whether to cache all data in memory (default: False)
        """
        self.data_dir = Path(data_dir)
        self.cache_in_memory = cache_in_memory

        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        # Recursively find all checkpoint files in subdirectories
        self.ckpt_files = sorted(list(self.data_dir.rglob(file_pattern)))

        if len(self.ckpt_files) == 0:
            raise ValueError(
                f"No checkpoint files found in {data_dir} "
                f"(including subdirectories) with pattern {file_pattern}"
            )

        rank0_print(
            f"Found {len(self.ckpt_files)} checkpoint files "
            f"in {data_dir} (including subdirectories)"
        )

        # Track valid indices (files that can be loaded successfully)
        self.valid_indices = list(range(len(self.ckpt_files)))

        # Cache data in memory if requested
        self.cached_data: Optional[List[Dict[str, torch.Tensor]]] = None
        if self.cache_in_memory:
            rank0_print("Caching all data in memory...")
            self.cached_data = []
            failed_count = 0
            for i in range(len(self.ckpt_files)):
                data = self._load_ckpt(i)
                if data is not None:
                    self.cached_data.append(data)
                else:
                    failed_count += 1

            # Update valid indices based on successful loads
            self.valid_indices = list(range(len(self.cached_data)))

            if failed_count > 0:
                rank0_print(
                    f"Data caching completed. "
                    f"Successfully loaded {len(self.cached_data)} files, "
                    f"failed to load {failed_count} files"
                )
            else:
                rank0_print("Data caching completed")

    def _load_ckpt(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load a checkpoint file.

        Args:
            idx: Index of the checkpoint file

        Returns:
            Dictionary containing input_ids, target_hiddens,
                hidden_states, and loss_mask, or None if loading fails
        """
        ckpt_path = self.ckpt_files[idx]

        try:
            data = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            warnings.warn(
                f"Failed to load checkpoint {ckpt_path}: {e}. Skipping this file.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        # Validate required keys
        required_keys = [
            "input_ids",  # B, N
            "target_hiddens",  # B, N, D
            "hidden_states",  # B, N, 3*D
            "loss_mask",  # B, N
        ]
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            warnings.warn(
                f"Checkpoint {ckpt_path} is missing required keys: {missing_keys}. "
                f"Skipping this file.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        # Validate tensor types
        for key in required_keys:
            if not isinstance(data[key], torch.Tensor):
                warnings.warn(
                    f"Value for key '{key}' in {ckpt_path} is not a torch.Tensor. "
                    f"Skipping this file.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None

        attention_mask = torch.ones_like(data["input_ids"])
        data["attention_mask"] = attention_mask  # B, N
        return data

    def __len__(self) -> int:
        """Return the number of valid samples in the dataset."""
        if self.cached_data is not None:
            return len(self.cached_data)
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing:
                - input_ids: Token IDs (torch.Tensor)
                - target_logits: Pre-computed logits from target
                    model (torch.Tensor)
                - hidden_states: Pre-computed hidden states from
                    target model (torch.Tensor)
                - loss_mask: Mask for loss computation (torch.Tensor)
        """
        if self.cached_data is not None:
            return self.cached_data[idx]
        else:
            # Try to load the checkpoint, retry with next valid index if fails
            max_retries = len(self.valid_indices)
            for _attempt in range(max_retries):
                actual_idx = self.valid_indices[idx % len(self.valid_indices)]
                data = self._load_ckpt(actual_idx)
                if data is not None:
                    return data
                else:
                    # Remove failed index from valid_indices
                    self.valid_indices.remove(actual_idx)
                    if len(self.valid_indices) == 0:
                        raise RuntimeError(
                            "All checkpoint files failed to load. "
                            "Cannot continue training."
                        )
                    # Try next index
                    idx += 1

            # If all retries failed, raise error
            raise RuntimeError(
                f"Failed to load any valid checkpoint after {max_retries} attempts"
            )
