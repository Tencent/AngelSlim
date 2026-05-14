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

"""Hidden-state transform utilities for offline EAGLE3 training.

Usage in a training script::

    from angelslim.compressor.speculative.train.data.noise_transforms import (
        GaussianNoise,
        TransformDataset,
    )

    train_dataset = TransformDataset(base_train_dataset, GaussianNoise(std=0.05))
    # eval_dataset is passed through unchanged

Calibrating noise std
---------------------
To verify the noise/signal ratio before committing to a full run, use
tools/test_dataloader.py::

    python3 tools/test_dataloader.py \\
        --train-hidden-path $TRAIN_HIDDEN_PATH \\
        --eval-hidden-path  $EVAL_HIDDEN_PATH  \\
        --target-model      $TARGET_MODEL_NAME_OR_PATH \\
        --hidden-noise-std  0.05
"""

import torch
from torch.utils.data import Dataset


class TransformDataset(Dataset):
    """Wraps an offline dataset and applies a callable transform at ``__getitem__`` time.

    The transform is applied to a shallow copy of each sample so the underlying
    cache is not mutated when ``cache_in_memory=True`` is used.

    Args:
        dataset: The underlying offline dataset (e.g. OfflineVLMEagle3Dataset).
        transform: Callable that takes a sample dict and returns a modified dict.
    """

    def __init__(self, dataset: Dataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        # Shallow copy prevents mutating the underlying cache when the inner
        # dataset uses cache_in_memory=True.
        sample = dict(self.dataset[idx])
        return self.transform(sample)


class GaussianNoise:
    """Adds Gaussian noise to a tensor field of a sample dict.

    Noise is sampled fresh each call, so the model sees a different perturbation
    of every sample on every epoch — effectively multiplying dataset diversity
    without generating new hidden states.

    Args:
        std: Standard deviation of the Gaussian noise. Set to 0.0 to disable.
        field: Key of the tensor field to perturb (default: ``"hidden_states"``).
    """

    def __init__(self, std: float, field: str = "hidden_states") -> None:
        self.std = std
        self.field = field

    def __call__(self, sample: dict) -> dict:
        sample[self.field] = sample[self.field] + torch.randn_like(sample[self.field]) * self.std
        return sample
