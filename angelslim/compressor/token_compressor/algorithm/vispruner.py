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
VisPruner Pruning Strategy.
"""


import torch
import torch.nn.functional as F

from ..base.context import PruningContext
from .utils.utils import (
    _extract_and_validate_vision_token_info,
    _recompute_attention_maps_for_all_images,
    _regroup_tensors_by_count,
)


def vispruner_pruning(context: PruningContext, **kwargs) -> torch.Tensor:
    """
    Executes VisPruner's two-stage pruning algorithm.

    Args:
        context (PruningContext): The execution context with inputs_embeds and Vit Q/K.
        **kwargs:
            ratio (float):
                Pruning ratio for vision tokens.
            layer_idx (int):
                ViT layer index for importance scoring.
            important_ratio_of_kept (float):
                Ratio of the budget allocated to the saliency stage.

    Returns:
        torch.Tensor: Boolean keep_mask of shape [1, sequence_length].
    """
    # 1. Parse parameters
    try:
        pruning_ratio = kwargs["ratio"]
        layer_idx = kwargs["layer_idx"]
        imp_ratio = kwargs.get("important_ratio_of_kept", 0.5)
        keep_ratio = 1.0 - pruning_ratio
    except KeyError as e:
        raise ValueError(
            "[TokenCompressor Error] " f"'vispruner_pruning' missing required parameter: {e}"
        )

    # 2. Access context attributes
    input_ids = context.input_ids
    inputs_embeds = context.inputs_embeds

    if input_ids is None or inputs_embeds is None:
        raise ValueError(
            "[TokenCompressor Error] 'input_ids' or 'inputs_embeds' missing in context."
        )

    device = input_ids.device
    if input_ids.shape[0] != 1:
        raise NotImplementedError(
            "[TokenCompressor Error] VisPruner currently only supports batch_size=1."
        )

    # Locate visual tokens and distribution metadata
    (
        vision_indices_global,
        non_vision_indices_global,
        _,
        num_tokens_per_image,
    ) = _extract_and_validate_vision_token_info(context)

    if len(vision_indices_global) == 0:
        return torch.ones_like(input_ids, dtype=torch.bool)

    # Retrieve ViT Q/K via LayerTensorMap
    q_tensor = context.vit_q[layer_idx]
    k_tensor = context.vit_k[layer_idx]

    if q_tensor is None or k_tensor is None:
        raise ValueError(
            "[TokenCompressor Error] " f"VisPruner requires ViT layer {layer_idx} Q/K states."
        )

    # Recompute importance scores
    final_scores_list, _ = _recompute_attention_maps_for_all_images(q_tensor, k_tensor, context)

    # Align physical scores with logical image segments
    final_scores_list, _ = _regroup_tensors_by_count(final_scores_list, num_tokens_per_image, None)

    # 3. Prepare features and split indices
    all_kept_indices_global = []
    vision_features_total = inputs_embeds[0, vision_indices_global, :]

    vision_features_list = torch.split(vision_features_total, num_tokens_per_image, dim=0)
    vision_indices_split = torch.split(vision_indices_global, num_tokens_per_image, dim=0)

    # 4. Process each image independently
    for scores, features, global_idx_map in zip(
        final_scores_list, vision_features_list, vision_indices_split
    ):
        n_tokens = features.shape[0]
        if n_tokens == 0:
            all_kept_indices_global.append(global_idx_map)
            continue

        scores = scores.squeeze(0).float()

        # Calculate budget allocation
        num_to_keep_total = int(round(n_tokens * keep_ratio))
        if keep_ratio > 0 and num_to_keep_total == 0:
            num_to_keep_total = 1

        if num_to_keep_total >= n_tokens:
            all_kept_indices_global.append(global_idx_map)
            continue

        num_imp = int(round(num_to_keep_total * imp_ratio))
        if num_to_keep_total > 1 and num_imp >= num_to_keep_total:
            num_imp = num_to_keep_total - 1
        num_div = num_to_keep_total - num_imp

        # Phase 1: Saliency selection (Top-K Importance)
        _, imp_indices_local = torch.topk(scores, k=num_imp)

        mask_residual = torch.ones(n_tokens, dtype=torch.bool, device=device)
        mask_residual[imp_indices_local] = False
        residual_indices_local = torch.where(mask_residual)[0]

        # Phase 2: Iterative bipartite matching for diversity
        if num_div > 0 and len(residual_indices_local) > num_div:
            feat_norm = F.normalize(features.float(), p=2, dim=-1)

            while len(residual_indices_local) > num_div:
                r_count = len(residual_indices_local)
                # Select a small batch to prune in each step
                r_batch = min(8, r_count // 2, r_count - num_div)
                if r_batch <= 0:
                    break

                idx_a = residual_indices_local[::2]
                idx_b = residual_indices_local[1::2]

                if len(idx_a) == 0 or len(idx_b) == 0:
                    break

                # Calculate similarity between split sets
                sim_matrix = torch.mm(feat_norm[idx_a], feat_norm[idx_b].t())
                max_sim_in_b, _ = sim_matrix.max(dim=-1)

                # Prune tokens in set A that are most redundant with set B
                sorted_a_rel = max_sim_in_b.argsort(descending=True)
                keep_a_rel = sorted_a_rel[r_batch:]

                residual_indices_local = torch.cat([idx_a[keep_a_rel], idx_b])
                residual_indices_local, _ = residual_indices_local.sort()

        # Merge local selections and map to global space
        final_local = torch.cat([imp_indices_local, residual_indices_local])
        final_local = torch.unique(final_local)

        if len(final_local) > num_to_keep_total:
            final_local = final_local[:num_to_keep_total]

        all_kept_indices_global.append(global_idx_map[final_local])

    # 5. Construct full sequence boolean mask
    if not all_kept_indices_global:
        return torch.ones_like(input_ids, dtype=torch.bool)

    kept_indices_tensor = torch.cat(all_kept_indices_global)
    final_indices = torch.cat([non_vision_indices_global.to(device), kept_indices_tensor])

    keep_mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
    if final_indices.numel() > 0:
        keep_mask[final_indices] = True

    return keep_mask.unsqueeze(0)
