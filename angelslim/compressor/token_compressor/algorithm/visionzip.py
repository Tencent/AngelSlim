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

"""
VisionZip Merging Strategy module.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F

from ..base.context import PruningContext
from .utils.merging_utils import get_dialogue_masks
from .utils.utils import _recompute_attention_maps_for_all_images


def visionzip(
    context: PruningContext, **kwargs
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Executes VisionZip merging to compress vision tokens via weighted aggregation.

    Args:
        context (PruningContext): The execution context with ViT Q/K and embeds.
        **kwargs:
            ratio (float): Global compression ratio.
            zip_ratio (float): Ratio of Target tokens relative to the total kept tokens.
            layer_idx (int): ViT layer index used for Q/K recomputation.

    Returns:
        Tuple containing:
            - List[torch.Tensor]:
                Merge weight matrices [M_i, N_i] for each vision segment.
            - torch.Tensor: Segment sizes on GPU.
            - torch.Tensor: Vision flag mask on GPU.
            - torch.Tensor: Representative fake_mask for metadata synchronization.
    """
    # 1. Parse strategy parameters
    try:
        ratio = kwargs["ratio"]
        layer_idx = kwargs["layer_idx"]
        zip_ratio = kwargs.get("zip_ratio", 10.0 / 64.0)
    except KeyError as e:
        raise ValueError(f"[TokenCompressor Error] 'visionzip' missing required parameter: {e}")

    keep_ratio = 1.0 - ratio

    # 2. Access context attributes directly
    input_ids = context.input_ids
    inputs_embeds = context.inputs_embeds

    if input_ids is None or inputs_embeds is None:
        raise ValueError(
            "[TokenCompressor Error] 'input_ids' or 'inputs_embeds' missing in context."
        )

    # Retrieve Q/K for the specific layer from LayerTensorMap via attribute
    # access
    q_tensor_fine = context.vit_q[layer_idx]
    k_tensor_fine = context.vit_k[layer_idx]

    if q_tensor_fine is None or k_tensor_fine is None:
        raise ValueError(
            "[TokenCompressor Error] " f"VisionZip requires ViT Q/K from layer {layer_idx}."
        )

    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    # 3. Recompute attention maps and keys
    final_scores_list, final_keys_list = _recompute_attention_maps_for_all_images(
        q_tensor_fine, k_tensor_fine, context
    )

    # 4. Analyze dialogue structure
    (
        _,
        _,
        _,
        sizes_list_cpu,
        is_vision_list_cpu,
    ) = get_dialogue_masks(context)

    sizes_list_gpu = torch.tensor(sizes_list_cpu, device=device)
    is_vision_list_gpu = torch.tensor(is_vision_list_cpu, device=device)

    # Split hidden states
    hidden_split_list = torch.split(inputs_embeds, sizes_list_cpu, dim=1)

    # 5. Segment iteration
    merge_weight_list = []
    fake_mask_segments = []

    scores_iter = iter(final_scores_list)
    keys_iter = iter(final_keys_list)

    for hidden_part, is_vision in zip(hidden_split_list, is_vision_list_cpu):
        if not is_vision:
            l_j = hidden_part.shape[1]
            fake_mask_segments.append(torch.ones(l_j, dtype=torch.bool, device=device))
            continue

        num_vision_tokens = hidden_part.shape[1]
        try:
            scores = next(scores_iter).squeeze(0)
            keys = next(keys_iter).squeeze(0)
        except StopIteration:
            raise RuntimeError(
                "[TokenCompressor Error] VisionZip: Segment mismatch during iteration."
            )

        # Execute VisionZip core logic for current vision segment
        num_to_keep_total = int(round(num_vision_tokens * keep_ratio))
        if keep_ratio > 0.0 and num_to_keep_total == 0 and num_vision_tokens > 0:
            num_to_keep_total = 1

        if num_to_keep_total >= num_vision_tokens:
            merge_weight_list.append(torch.eye(num_vision_tokens, dtype=dtype, device=device))
            fake_mask_segments.append(
                torch.ones(num_vision_tokens, dtype=torch.bool, device=device)
            )
            continue

        num_target = max(1, int(round(num_to_keep_total * zip_ratio)))
        num_dominant = max(0, num_to_keep_total - num_target)

        # Select indices
        if num_dominant > 0:
            _, dominant_indices = torch.topk(scores, k=num_dominant)
        else:
            dominant_indices = torch.tensor([], dtype=torch.long, device=device)

        dominant_mask = torch.zeros(num_vision_tokens, dtype=torch.bool, device=device)
        if dominant_indices.numel() > 0:
            dominant_mask[dominant_indices] = True

        candidate_indices = torch.where(~dominant_mask)[0]
        if num_target > 0 and len(candidate_indices) > 0:
            step = max(1, len(candidate_indices) // num_target)
            target_indices = candidate_indices[
                torch.arange(0, len(candidate_indices), step, device=device)[:num_target]
            ]
        else:
            target_indices = torch.tensor([], dtype=torch.long, device=device)

        target_mask = torch.zeros(num_vision_tokens, dtype=torch.bool, device=device)
        if target_indices.numel() > 0:
            target_mask[target_indices] = True

        contextual_indices = torch.where(~(dominant_mask | target_mask))[0]
        kept_indices = torch.cat([dominant_indices, target_indices]).sort().values

        # Metadata sync mask
        segment_mask = torch.zeros(num_vision_tokens, dtype=torch.bool, device=device)
        if kept_indices.numel() > 0:
            segment_mask[kept_indices] = True
        fake_mask_segments.append(segment_mask)

        # Build merging matrix
        index_map = torch.full((num_vision_tokens,), -1, dtype=torch.long, device=device)
        index_map[kept_indices] = torch.arange(len(kept_indices), device=device)
        merge_mat = torch.zeros(
            len(kept_indices),
            num_vision_tokens,
            dtype=dtype,
            device=device,
        )
        merge_mat.scatter_(1, kept_indices.unsqueeze(1), 1.0)

        if len(contextual_indices) > 0 and len(target_indices) > 0:
            ctx_feats = F.normalize(keys[contextual_indices].float(), p=2, dim=1)
            tgt_feats = F.normalize(keys[target_indices].float(), p=2, dim=1)
            sim = torch.mm(ctx_feats, tgt_feats.t())
            _, best_target_local = torch.max(sim, dim=1)
            target_row_indices = index_map[target_indices[best_target_local]]
            merge_mat.index_put_(
                (target_row_indices, contextual_indices),
                torch.ones(
                    len(contextual_indices),
                    dtype=dtype,
                    device=device,
                ),
                accumulate=True,
            )

        merge_mat = merge_mat / (merge_mat.sum(dim=1, keepdim=True) + 1e-8)
        merge_weight_list.append(merge_mat.to(dtype))

    return (
        merge_weight_list,
        sizes_list_gpu,
        is_vision_list_gpu,
        torch.cat(fake_mask_segments, dim=0),
    )
