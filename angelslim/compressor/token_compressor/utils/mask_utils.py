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

from typing import Any, List, Optional, Tuple, Union

import torch


def apply_pruning_mask(
    hidden_states: Optional[torch.Tensor],
    keep_mask: torch.Tensor,
    context: Optional[Any] = None,
    position_ids: Optional[torch.Tensor] = None,
    text_position_ids: Optional[torch.Tensor] = None,
    causal_mask: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.Tensor] = None,
    stage_key: Union[str, int] = "global",
    past_key_values: Any = None,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Apply a boolean keep_mask to synchronize tensors during prefill.
    Aligned with Adapter: uses end-truncation
    for position_ids and re-generates cache_position.
    """
    if keep_mask is None:
        return (
            hidden_states,
            position_ids,
            causal_mask,
            cache_position,
        )

    mask_1d = keep_mask.view(-1)
    num_kept = mask_1d.sum().item()

    # 1. Physical slicing for hidden states [Batch, Seq, Dim]
    # Slicing is performed only if hidden_states is provided and not yet
    # compressed (e.g., Pruning)
    if hidden_states is not None:
        print(f"Hidden States Shape Before Pruning: {hidden_states.shape}")
        hidden_states = hidden_states[:, mask_1d, :]
        print(f"Hidden States Shape After Pruning: {hidden_states.shape}")

    # 2. Position IDs: Aligned with Adapter's end-truncation logic
    if position_ids is not None:
        position_ids = position_ids[..., mask_1d]
    if text_position_ids is not None:
        text_position_ids = text_position_ids[..., :num_kept]

    # 3. Causal Mask: Symmetrical slicing for prefill stage [B, 1, Seq, Seq]
    if causal_mask is not None and causal_mask.ndim == 4:
        causal_mask = causal_mask[:, :, mask_1d, :][:, :, :, mask_1d]
    elif causal_mask is not None and causal_mask.ndim == 2:
        causal_mask = causal_mask[:, mask_1d]

    # 4. Cache Position: Re-generate continuous indices starting from 0
    # (Adapter style)
    if cache_position is not None:
        cache_position = torch.arange(num_kept, device=mask_1d.device)

    # 5. Register mask into KV cache for future decoding compensation
    if hasattr(past_key_values, "set_pruning_mask_for_layer"):
        past_key_values.set_pruning_mask_for_layer(stage_key, mask_1d)

    # 6. Synchronize context information
    if context is not None:
        if hasattr(context, "input_ids") and context.input_ids is not None:
            context.input_ids = context.input_ids[:, mask_1d]

    return (
        hidden_states,
        position_ids,
        text_position_ids,
        causal_mask,
        cache_position,
    )


def apply_token_merging(
    hidden_states: torch.Tensor,
    merge_weights: List[torch.Tensor],
    segment_sizes: List[int],
    is_vision_flags: List[bool],
    fake_mask: torch.Tensor,
    context: Optional[Any] = None,
    position_ids: Optional[torch.Tensor] = None,
    text_position_ids: Optional[torch.Tensor] = None,
    causal_mask: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.Tensor] = None,
    stage_key: Union[str, int] = "global",
    past_key_values: Any = None,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Merge tokens via weighted aggregation for hidden states.
    Metadata are synchronized using fake_mask via apply_pruning_mask.
    """
    if isinstance(segment_sizes, torch.Tensor):
        segment_sizes = segment_sizes.tolist()

    # 1. Weighted feature aggregation
    hidden_splits = torch.split(hidden_states, segment_sizes, dim=1)
    merged_list = []
    weight_idx = 0

    for segment, is_vision in zip(hidden_splits, is_vision_flags):
        if is_vision:
            w = merge_weights[weight_idx].to(device=segment.device, dtype=segment.dtype)
            if w.ndim == 2:
                w = w.unsqueeze(0)
            merged_list.append(torch.matmul(w, segment))
            weight_idx += 1
        else:
            merged_list.append(segment)

    merged_hidden = torch.cat(merged_list, dim=1)

    print(f"Hidden States Shape Before Merging: {hidden_states.shape}")
    print(f"Hidden States Shape After Merging: {merged_hidden.shape}")

    # 2. Metadata synchronization using fake_mask (Truncation/Re-generation logic)
    # Pass hidden_states=None to avoid redundant slicing as it is already
    # compressed
    (
        _,
        position_ids,
        text_position_ids,
        causal_mask,
        cache_position,
    ) = apply_pruning_mask(
        hidden_states=None,
        keep_mask=fake_mask,
        context=context,
        position_ids=position_ids,
        text_position_ids=text_position_ids,
        causal_mask=causal_mask,
        cache_position=cache_position,
        stage_key=stage_key,
        past_key_values=past_key_values,
    )

    return (
        merged_hidden,
        position_ids,
        text_position_ids,
        causal_mask,
        cache_position,
    )


def compensate_decoding_state(
    position_ids: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    causal_mask: Optional[torch.Tensor],
    stage_key: Union[str, int],
    past_key_values: Any,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Adjust indices and masks during decoding to compensate for past pruning.
    Performs numerical offset for IDs and optional Key-dimension slicing for the mask.
    """
    if not hasattr(past_key_values, "get_pruning_mask_for_layer"):
        return position_ids, cache_position, causal_mask

    is_pruned, keep_mask, pruned_count = past_key_values.get_pruning_mask_for_layer(stage_key)

    if not is_pruned or pruned_count <= 0:
        return position_ids, cache_position, causal_mask

    # 1. Coordinate compensation (Numerical shift)
    if position_ids is not None:
        position_ids = position_ids - pruned_count
    if cache_position is not None:
        cache_position = cache_position - pruned_count

    # 2. Key-dimension mask slicing (only if causal_mask is provided)
    if causal_mask is not None:
        if causal_mask.shape[-1] > keep_mask.shape[-1]:
            padding = torch.ones(
                causal_mask.shape[-1] - keep_mask.shape[-1],
                dtype=keep_mask.dtype,
                device=keep_mask.device,
            )
            effective_mask = torch.cat([keep_mask, padding], dim=-1)
        else:
            effective_mask = keep_mask
        causal_mask = causal_mask[:, :, :, effective_mask]

    return position_ids, cache_position, causal_mask
