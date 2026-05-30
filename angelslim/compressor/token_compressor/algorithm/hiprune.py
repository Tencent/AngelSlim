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
HiPrune (Hierarchical Pruning) Strategy.
"""

import math

import torch

from ..base.context import PruningContext
from .utils.utils import (
    _extract_and_validate_vision_token_info,
    _recompute_attention_maps_for_all_images,
    identify_model_architecture,
)


def hiprune_pruning(context: PruningContext, **kwargs) -> torch.Tensor:
    """
    Executes HiPrune by combining attention from shallow and deep layers.

    Args:
        context (PruningContext): The execution context with ViT Q/K and grid metadata.
        **kwargs:
            ratio (float): Global pruning ratio.
            object_layer (int): Index of the shallow ViT layer for spatial anchors.
            last_vit_layer (int): Index of the deep ViT layer for semantic info.
            alpha (float): Proportion of budget for shallow anchor expansion.

    Returns:
        torch.Tensor: Boolean keep_mask of shape [B, seq_len].
    """
    # 1. Parse parameters
    try:
        ratio = kwargs["ratio"]
        object_layer = kwargs["object_layer"]
        last_vit_layer = kwargs["last_vit_layer"]
        alpha = kwargs["alpha"]
    except KeyError as e:
        raise ValueError(f"[TokenCompressor Error] HiPrune missing required parameter: {e}")

    input_ids = context.input_ids
    if input_ids is None:
        raise ValueError("[TokenCompressor Error] 'input_ids' missing in context.")

    device = input_ids.device

    # Extract grid info from context attributes
    grid_thw = (
        context.image_grid_thw if context.image_grid_thw is not None else context.video_grid_thw
    )

    spatial_merge_size = context.spatial_merge_size
    model_type = identify_model_architecture(context)

    # 2. Extract vision token distribution
    (
        vision_indices_global,
        non_vision_indices_global,
        _,
        num_tokens_per_image,
    ) = _extract_and_validate_vision_token_info(context)

    if len(vision_indices_global) == 0:
        return torch.ones_like(input_ids, dtype=torch.bool)

    # 3. Retrieve and recompute attention scores for dual layers
    # Using vit_q/vit_k LayerTensorMap from PruningContext
    q_shallow = context.vit_q[object_layer]
    k_shallow = context.vit_k[object_layer]
    q_deep = context.vit_q[last_vit_layer]
    k_deep = context.vit_k[last_vit_layer]

    if any(x is None for x in [q_shallow, k_shallow, q_deep, k_deep]):
        raise ValueError(
            f"[TokenCompressor Error] "
            f"HiPrune requires Q/K from layers {object_layer} and {last_vit_layer}."
        )

    shallow_scores_list, _ = _recompute_attention_maps_for_all_images(
        q_shallow, k_shallow, context
    )
    deep_scores_list, _ = _recompute_attention_maps_for_all_images(q_deep, k_deep, context)

    # Align scores with logical image segments
    from .utils.utils import _regroup_tensors_by_count

    shallow_scores_list, _ = _regroup_tensors_by_count(
        shallow_scores_list, num_tokens_per_image, None
    )
    deep_scores_list, _ = _regroup_tensors_by_count(deep_scores_list, num_tokens_per_image, None)

    vision_indices_split = torch.split(vision_indices_global, num_tokens_per_image)
    all_kept_indices_global = []

    # 4. Iteratively process each image for hierarchical selection
    if grid_thw is None:
        grid_thw = [None] * len(shallow_scores_list)
    for shallow_score, deep_score, global_idx_map, grid in zip(
        shallow_scores_list,
        deep_scores_list,
        vision_indices_split,
        grid_thw,
    ):
        shallow_score = shallow_score.squeeze(0)
        deep_score = deep_score.squeeze(0)
        N = shallow_score.shape[0]

        target_k = int(round(N * (1.0 - ratio)))
        if ratio < 1.0 and target_k == 0 and N > 0:
            target_k = 1

        if target_k >= N:
            all_kept_indices_global.append(global_idx_map)
            continue

        shallow_k = int(round((target_k * alpha) / 5.0))
        if alpha > 0 and shallow_k == 0 and target_k >= 5:
            shallow_k = 1

        shallow_indices_final = torch.tensor([], dtype=torch.long, device=device)

        # 4.1 Phase 1: Shallow Spatial Clustering (Anchor + 4-Neighbors)
        if shallow_k > 0:
            _, anchor_indices = torch.topk(shallow_score, k=shallow_k)

            # Infer grid width for connectivity
            if "qwen" in model_type:
                width = int(grid[2].item() // spatial_merge_size)
            elif "llava" in model_type:
                width = int(math.sqrt(N))
            else:
                raise ValueError(
                    f"[TokenCompressor Error] "
                    f"Unsupported architecture for HiPrune: {model_type}"
                )

            # Expand anchors to Cross-Shape neighbors
            neighbors = torch.cat(
                [
                    anchor_indices,
                    anchor_indices - 1,
                    anchor_indices + 1,
                    anchor_indices - width,
                    anchor_indices + width,
                ]
            ).clamp(0, N - 1)

            shallow_indices_final = torch.unique(neighbors)

        # 4.2 Phase 2: Deep Semantic Completion
        current_kept_count = shallow_indices_final.shape[0]
        deep_k = target_k - current_kept_count
        deep_indices_final = torch.tensor([], dtype=torch.long, device=device)

        if deep_k > 0:
            deep_score_masked = deep_score.clone()
            if current_kept_count > 0:
                deep_score_masked[shallow_indices_final] = float("-inf")

            actual_deep_k = min(deep_k, N - current_kept_count)
            if actual_deep_k > 0:
                _, deep_indices_final = torch.topk(deep_score_masked, k=actual_deep_k)

        # Merge local selections and map back to global sequence indices
        final_local = torch.cat([shallow_indices_final, deep_indices_final])
        all_kept_indices_global.append(global_idx_map[final_local])

    # 5. Build final sequence mask
    if not all_kept_indices_global:
        return torch.ones_like(input_ids, dtype=torch.bool)

    kept_indices_tensor = torch.cat(all_kept_indices_global)
    final_indices = torch.cat([non_vision_indices_global.to(device), kept_indices_tensor])

    keep_mask = torch.zeros_like(input_ids.squeeze(0), dtype=torch.bool)
    keep_mask[final_indices] = True

    return keep_mask.unsqueeze(0)
