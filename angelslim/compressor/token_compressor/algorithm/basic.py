# Copyright 2026 Tencent Inc. All Rights Reserved.
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


import torch

from ..base.context import PruningContext
from .utils.utils import _extract_and_validate_vision_token_info


def baseline_pruning(context: PruningContext, **kwargs) -> torch.Tensor:
    """
    Baseline strategy: Keep all tokens without any pruning.

    Args:
        context (PruningContext): The execution context.

    Returns:
        torch.Tensor: All-True boolean mask.
    """
    input_ids = context.input_ids
    if input_ids is None:
        raise ValueError("[TokenCompressor Error] 'input_ids' not found in context.")

    return torch.ones_like(input_ids, dtype=torch.bool)


def override_pruning(context: PruningContext, **kwargs) -> torch.Tensor:
    """
    Manual override strategy: Apply a pre-defined mask to vision tokens.

    Args:
        context (PruningContext): The execution context.
        **kwargs:
            mask (torch.Tensor): A boolean tensor for vision tokens only.

    Returns:
        torch.Tensor: Global boolean keep_mask.
    """
    partial_mask = kwargs.get("mask")
    if partial_mask is None:
        raise ValueError("[TokenCompressor Error] 'override_pruning' requires 'mask' in params.")

    input_ids = context.input_ids
    if input_ids is None:
        raise ValueError("[TokenCompressor Error] 'input_ids' not found in context.")

    if input_ids.shape[0] != 1:
        raise NotImplementedError(
            "[TokenCompressor Error] override_pruning only supports batch_size=1."
        )

    device = input_ids.device
    partial_mask = partial_mask.to(device)

    # Identify vision token indices and the vision_token_mask from context
    # attribute
    vision_indices, _, vision_token_mask, _ = _extract_and_validate_vision_token_info(context)

    # Validate mask length against identified vision tokens
    if len(partial_mask) != len(vision_indices):
        if partial_mask.dim() > 1:
            partial_mask = partial_mask.flatten()
        if len(partial_mask) != len(vision_indices):
            raise ValueError(
                "[TokenCompressor Error] "
                f"Provided mask length ({len(partial_mask)}) mismatch "
                f"with vision tokens ({len(vision_indices)})."
            )

    # Build full mask: Keep all non-vision tokens, apply partial_mask to
    # vision positions
    full_keep_mask = ~vision_token_mask
    if len(vision_indices) > 0:
        full_keep_mask[vision_indices] = partial_mask

    return full_keep_mask.unsqueeze(0)


def random_pruning(context: PruningContext, **kwargs) -> torch.Tensor:
    """
    Random pruning strategy: Randomly discard vision tokens based on a ratio.

    Args:
        context (PruningContext): The execution context.
        **kwargs:
            ratio (float): Ratio of vision tokens to prune [0.0, 1.0].

    Returns:
        torch.Tensor: Boolean keep_mask of shape [B, seq_len].
    """
    try:
        ratio = kwargs["ratio"]
    except KeyError:
        raise ValueError("[TokenCompressor Error] 'random_pruning' requires 'ratio' in params.")

    input_ids = context.input_ids
    if input_ids is None:
        raise ValueError("[TokenCompressor Error] 'input_ids' not found in context.")

    bsz, seq_len = input_ids.shape
    device = input_ids.device
    if bsz != 1:
        raise NotImplementedError(
            "[TokenCompressor Error] random_pruning only supports batch_size=1."
        )

    batch_keep_masks = []
    for i in range(bsz):
        input_ids_single = input_ids[i]

        # Extract vision/non-vision indices using context
        vision_indices, non_vision_indices, _, _ = _extract_and_validate_vision_token_info(context)

        num_vision_tokens = len(vision_indices)
        if num_vision_tokens == 0:
            batch_keep_masks.append(torch.ones_like(input_ids_single, dtype=torch.bool))
            continue

        num_to_keep = int(round(num_vision_tokens * (1 - ratio)))
        # Retain at least one token if ratio < 1.0
        if ratio < 1.0 and num_to_keep == 0 and num_vision_tokens > 0:
            num_to_keep = 1

        kept_vision_indices = torch.tensor([], dtype=torch.long, device=device)
        if num_to_keep > 0:
            shuffled_indices = vision_indices[torch.randperm(num_vision_tokens, device=device)]
            kept_vision_indices = shuffled_indices[:num_to_keep]

        # Combine non-vision tokens with kept vision tokens
        final_kept_indices = torch.cat([non_vision_indices.to(device), kept_vision_indices])

        keep_mask = torch.zeros_like(input_ids_single, dtype=torch.bool)
        if len(final_kept_indices) > 0:
            keep_mask[final_kept_indices] = True
        batch_keep_masks.append(keep_mask)

    return torch.stack(batch_keep_masks, dim=0)
