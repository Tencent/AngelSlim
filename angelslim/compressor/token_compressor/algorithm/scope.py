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
SCOPE Pruning Strategy (Saliency-Coverage Oriented Token Pruning).
"""

import os

import torch

from ..base.context import PruningContext
from .utils.utils import (
    _extract_and_validate_vision_token_info,
    _recompute_attention_maps_for_all_images,
)


def SCOPE(
    visual_feature_vectors,
    num_selected_token,
    cls_attn=None,
    alpha=1.0,
):
    """
    Batched implementation of SCOPE core logic.
    Args:
        visual_feature_vectors: [B, N, D] feature vectors.
        num_selected_token: Target number of tokens per batch.
        cls_attn: [B, N] saliency/attention weights.
        alpha: Scaling factor for saliency.
    """
    norm_vectors = visual_feature_vectors / visual_feature_vectors.norm(dim=-1, keepdim=True)
    cosine_simi = torch.bmm(norm_vectors, norm_vectors.transpose(1, 2))

    B, N = visual_feature_vectors.shape[:2]
    device = visual_feature_vectors.device
    dtype = visual_feature_vectors.dtype

    selected = torch.zeros(B, N, dtype=torch.bool, device=device)
    selected_idx = torch.empty(B, num_selected_token, dtype=torch.long, device=device)
    cur_max = torch.zeros(B, N, dtype=dtype, device=device)

    if cls_attn is not None:
        cls_attn_powered = cls_attn**alpha
    else:
        cls_attn_powered = torch.ones(B, N, dtype=dtype, device=device)

    for i in range(num_selected_token):
        unselected_mask = ~selected
        # Calculate marginal coverage gain
        gains = torch.maximum(
            torch.zeros(1, dtype=dtype, device=device),
            cosine_simi.masked_fill(~unselected_mask.unsqueeze(1), 0) - cur_max.unsqueeze(2),
        ).sum(dim=1)

        # Combine with saliency
        combined_mode = os.environ.get("COMBINED", "multi")
        if combined_mode == "multi":
            gains = gains * cls_attn_powered
        elif combined_mode == "add":
            gains = gains + cls_attn_powered
        else:
            raise NotImplementedError(f"Combined mode {combined_mode} not supported")

        gains = gains.masked_fill(~unselected_mask, float("-inf"))
        best_idx = gains.argmax(dim=1)

        selected[torch.arange(B, device=device), best_idx] = True
        selected_idx[:, i] = best_idx
        cur_max = torch.maximum(
            cur_max,
            cosine_simi[torch.arange(B, device=device), best_idx],
        )

    return selected_idx, cosine_simi


def scope_pruning(context: PruningContext, **kwargs) -> torch.Tensor:
    """
    Framework wrapper for SCOPE pruning strategy.

    Args:
        context (PruningContext): Execution context with feature_map and Vit Q/K.
        **kwargs:
            ratio (float): Pruning ratio.
            layer_idx (int): ViT layer index for saliency computation.
            alpha (float): Scaling factor for attention scores.

    Returns:
        torch.Tensor: Boolean keep_mask of shape [1, seq_len].
    """
    # 1. Parameter parsing
    try:
        ratio = kwargs["ratio"]
        layer_idx = kwargs["layer_idx"]
        alpha = kwargs.get("alpha", 1.0)
    except KeyError as e:
        raise ValueError(f"[TokenCompressor Error] scope_pruning missing required parameter: {e}")

    # 2. Fetch base data via attributes
    input_ids = context.input_ids
    inputs_embeds = context.inputs_embeds

    if input_ids is None or inputs_embeds is None:
        raise ValueError("[TokenCompressor Error] SCOPE requires 'input_ids' and 'inputs_embeds'.")

    device = input_ids.device

    # 3. Extract vision token info
    vision_indices, non_vision_indices, _, _ = _extract_and_validate_vision_token_info(context)
    N_vision = len(vision_indices)

    if N_vision == 0:
        return torch.ones_like(input_ids, dtype=torch.bool)

    # 4. Recompute attention for saliency (using LayerTensorMap)
    q_tensor = context.vit_q[layer_idx]
    k_tensor = context.vit_k[layer_idx]

    if q_tensor is None or k_tensor is None:
        raise ValueError(
            f"[TokenCompressor Error] SCOPE requires ViT layer {layer_idx} Q/K states."
        )

    scores_list, _ = _recompute_attention_maps_for_all_images(q_tensor, k_tensor, context)
    cls_attn = torch.cat(scores_list, dim=1).to(device=device, dtype=inputs_embeds.dtype)

    # 5. Execute SCOPE core
    num_to_keep = int(round(N_vision * (1.0 - ratio)))
    if ratio < 1.0 and num_to_keep == 0 and N_vision > 0:
        num_to_keep = 1

    if num_to_keep >= N_vision:
        return torch.ones_like(input_ids, dtype=torch.bool)

    visual_features = inputs_embeds[:, vision_indices, :]
    selected_idx_local, _ = SCOPE(visual_features, num_to_keep, cls_attn=cls_attn, alpha=alpha)

    # 6. Map indices and build mask
    selected_idx_local = selected_idx_local.squeeze(0)
    kept_vision_global = vision_indices[selected_idx_local]

    final_indices = torch.cat([non_vision_indices.to(device), kept_vision_global.to(device)])
    keep_mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
    if final_indices.numel() > 0:
        keep_mask[final_indices] = True

    return keep_mask.unsqueeze(0)
