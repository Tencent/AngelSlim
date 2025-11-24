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

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from angelslim.utils import rank0_print

from ..data_utils import VLMDataCollatorWithPadding
from .base_dataset_builder import DatasetBuilder
from .dataset_builder_factory import DatasetBuilderFactory


class OfflineVLMEagle3Dataset(Dataset):
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
        Initialize the OfflineVLMEagle3Dataset.

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
            "inputs_embeds",  # B, N, D
            "position_ids",  # B, N
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


@DatasetBuilderFactory.register("offline", "VLM")
class OfflineVLMDatasetBuilder(DatasetBuilder):
    def __init__(
        self, file_pattern: str = "*.ckpt", cache_in_memory: bool = False, **kwargs: Any
    ):
        self.file_pattern = file_pattern
        self.cache_in_memory = cache_in_memory

    def build_dataset(self, datapath: str, **kwargs: Any) -> Dataset:
        """
        Create offline datasets from pre-computed .ckpt files.
        """
        return OfflineVLMEagle3Dataset(
            data_dir=datapath,
            file_pattern=self.file_pattern,
            cache_in_memory=self.cache_in_memory,
        )

    def get_data_collator(self) -> Any:
        return VLMDataCollatorWithPadding()
