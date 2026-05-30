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
DivPrune Pruning Strategy (Diversity-based Visual Token Pruning).
"""


import torch
import torch.nn.functional as F

from ..base.context import PruningContext
from .utils.utils import _extract_and_validate_vision_token_info


def divprune(context: PruningContext, **kwargs) -> torch.Tensor:
    """
    Executes DivPrune using iterative Farthest Point Sampling (FPS) in feature space.

    Args:
        context (PruningContext): The execution context containing inputs_embeds.
        **kwargs:
            ratio (float): Ratio of vision tokens to prune.

    Returns:
        torch.Tensor: Boolean keep_mask of shape [B, seq_len].
    """
    try:
        ratio = kwargs["ratio"]
    except KeyError:
        raise ValueError("[TokenCompressor Error] 'divprune' requires 'ratio' in params.")

    input_ids = context.input_ids
    inputs_embeds = context.inputs_embeds

    if input_ids is None:
        raise ValueError("[TokenCompressor Error] 'input_ids' not found in context.")
    if inputs_embeds is None:
        raise ValueError("[TokenCompressor Error] DivPrune requires 'inputs_embeds' in context.")

    bsz, seq_len, _ = inputs_embeds.shape
    device = input_ids.device
    if bsz != 1:
        raise NotImplementedError("[TokenCompressor Error] DivPrune only supports batch_size=1.")

    batch_keep_masks = []
    for i in range(bsz):
        input_ids_single = input_ids[i]
        features_single = inputs_embeds[i]

        # 1. Extract vision token indices
        vision_indices, non_vision_indices, _, _ = _extract_and_validate_vision_token_info(context)

        num_vision_tokens = len(vision_indices)
        if num_vision_tokens == 0:
            batch_keep_masks.append(torch.ones_like(input_ids_single, dtype=torch.bool))
            continue

        num_to_keep = int(round(num_vision_tokens * (1 - ratio)))
        if ratio < 1.0 and num_to_keep == 0 and num_vision_tokens > 0:
            num_to_keep = 1
        if num_to_keep >= num_vision_tokens:
            batch_keep_masks.append(torch.ones_like(input_ids_single, dtype=torch.bool))
            continue

        # 2. Compute similarity-based distance matrix (1 - Cosine Similarity)
        visual_features = features_single[vision_indices]
        norm_matrix = F.normalize(visual_features, p=2, dim=1)
        distance_matrix = 1.0 - torch.mm(norm_matrix, norm_matrix.t())
        distance_matrix.fill_diagonal_(float("inf"))

        # 3. Iterative Selection using Farthest Point Sampling logic
        kept_local_indices = torch.empty(num_to_keep, dtype=torch.long, device=device)

        # Maximin Initialization: Pick the point with largest minimum distance
        # to others
        if num_vision_tokens > 1:
            min_dists_to_others, _ = torch.min(distance_matrix, dim=1)
            first_idx = torch.argmax(min_dists_to_others)
        else:
            first_idx = torch.tensor(0, device=device)

        kept_local_indices[0] = first_idx
        num_kept = 1
        min_dists_to_kept = distance_matrix[:, first_idx].clone()

        while num_kept < num_to_keep:
            scores = min_dists_to_kept.clone()
            scores[kept_local_indices[:num_kept]] = -1.0

            new_idx = torch.argmax(scores)
            kept_local_indices[num_kept] = new_idx
            num_kept += 1

            # Update distance vector to the current selected set
            dist_to_new = distance_matrix[:, new_idx]
            min_dists_to_kept = torch.minimum(min_dists_to_kept, dist_to_new)

        # 4. Construct global keep_mask
        kept_vision_indices = vision_indices[kept_local_indices]
        final_indices = torch.cat([non_vision_indices.to(device), kept_vision_indices])

        keep_mask = torch.zeros_like(input_ids_single, dtype=torch.bool)
        if final_indices.numel() > 0:
            keep_mask[final_indices] = True
        batch_keep_masks.append(keep_mask)

    return torch.stack(batch_keep_masks, dim=0)
