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
DART Pruning Strategy (Duplication-Aware Related Token selection).
Reference: "Stop Looking for Important Tokens in Multimodal Language Models:
Duplication Matters More (arXiv:2502.11494)".
"""


import torch
import torch.nn.functional as F

from ..base.context import PruningContext
from .utils.utils import _extract_and_validate_vision_token_info


def dart_pruning(context: PruningContext, **kwargs) -> torch.Tensor:
    """
    Executes DART pruning based on information redundancy rather than just importance.

    Args:
        context (PruningContext):
            The execution context containing feature_map and K tensors.
        **kwargs:
            ratio (float): Ratio of vision tokens to prune.
            pivot_image_token (int|float): Number or ratio of image pivot tokens.
            pivot_text_token (int|float): Number or ratio of text pivot tokens.
            use_post_rope (bool): Whether to use post-RoPE Key tensors.

    Returns:
        torch.Tensor: Boolean keep_mask of shape [B, seq_len].
    """
    # 1. Parse parameters and validate context attributes
    try:
        ratio = kwargs["ratio"]
        pivot_image_token = kwargs["pivot_image_token"]
        pivot_text_token = kwargs["pivot_text_token"]
        layer_idx = kwargs["layer_idx"]
    except KeyError as e:
        raise ValueError(f"[TokenCompressor Error] DART pruning missing required parameter: {e}")

    input_ids = context.input_ids
    feature_map = context.feature_map
    # Using the latest captured LLM Key states via LayerTensorMap
    k_tensor = context.llm_k[layer_idx]

    if input_ids is None:
        raise ValueError("[TokenCompressor Error] 'input_ids' not found in context.")
    if feature_map is None:
        raise ValueError("[TokenCompressor Error] DART requires 'feature_map' in context.")
    if k_tensor is None:
        raise ValueError("[TokenCompressor Error] DART requires Key tensors in context.")

    bsz, seq_len = input_ids.shape
    device = input_ids.device
    if bsz != 1:
        raise NotImplementedError(
            "[TokenCompressor Error] DART pruning only supports batch_size=1."
        )

    batch_keep_masks = []
    for i in range(bsz):
        input_ids_single = input_ids[i]
        features_single = feature_map[i]
        # Reshape Key tensor: [B, H, Seq, Dim] -> [Seq, H * Dim]
        k_single = k_tensor[i].permute(1, 0, 2).reshape(seq_len, -1)

        # 2. Identify vision and non-vision tokens
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

        # 3. Handle pivot quantity calculations
        num_img_pivots_req = pivot_image_token
        num_txt_pivots_req = pivot_text_token
        if isinstance(num_img_pivots_req, float):
            num_img_pivots_req = int(num_img_pivots_req * num_vision_tokens)
        if isinstance(num_txt_pivots_req, float):
            num_txt_pivots_req = int(num_txt_pivots_req * len(non_vision_indices))

        # 4. Phase 1: Pivot Selection (Based on Key L1 Norm)
        text_indices_for_pivot = non_vision_indices.to(device)

        k_visual = k_single[vision_indices]
        k_visual_l1_norm = torch.norm(k_visual, p=1, dim=-1)

        num_img_pivots = min(num_img_pivots_req, num_vision_tokens, num_to_keep)
        img_pivot_indices = torch.tensor([], dtype=torch.long, device=device)
        top_img_pivot_local_indices = torch.tensor([], dtype=torch.long, device=device)
        if num_img_pivots > 0:
            _, top_img_pivot_local_indices = torch.topk(k_visual_l1_norm, k=num_img_pivots)
            img_pivot_indices = vision_indices[top_img_pivot_local_indices]

        txt_pivot_indices = torch.tensor([], dtype=torch.long, device=device)
        if len(text_indices_for_pivot) > 0:
            k_text = k_single[text_indices_for_pivot]
            k_text_l1_norm = torch.norm(k_text, p=1, dim=-1)
            num_txt_pivots = min(num_txt_pivots_req, len(text_indices_for_pivot))
            if num_txt_pivots > 0:
                _, top_txt_pivot_local_indices = torch.topk(k_text_l1_norm, k=num_txt_pivots)
                txt_pivot_indices = text_indices_for_pivot[top_txt_pivot_local_indices]

        pivot_indices = torch.cat([img_pivot_indices, txt_pivot_indices])
        kept_local_indices = set(top_img_pivot_local_indices.cpu().tolist())

        # 5. Phase 2: Collect Related Tokens with Low Duplication
        total_pivots = len(pivot_indices)
        if total_pivots > 0 and len(kept_local_indices) < num_to_keep:
            num_remaining = max(0, num_to_keep - len(kept_local_indices))
            token_topk_per_pivot = (num_remaining + total_pivots - 1) // total_pivots

            visual_features = features_single[vision_indices]
            visual_features_norm = F.normalize(visual_features, p=2, dim=1)

            for pivot_global_idx in pivot_indices:
                if len(kept_local_indices) >= num_to_keep:
                    break

                pivot_feat_norm = F.normalize(
                    features_single[pivot_global_idx].unsqueeze(0),
                    p=2,
                    dim=1,
                )
                # Compute Cosine Similarity
                cos_sim = torch.mm(pivot_feat_norm, visual_features_norm.t()).squeeze(0)

                # Exclude already selected tokens
                if kept_local_indices:
                    mask_indices = torch.tensor(
                        list(kept_local_indices),
                        device=device,
                        dtype=torch.long,
                    )
                    cos_sim.scatter_(0, mask_indices, float("inf"))

                num_to_gather = min(
                    token_topk_per_pivot,
                    num_to_keep - len(kept_local_indices),
                )
                num_available = (cos_sim != float("inf")).sum().item()
                num_to_gather = min(num_to_gather, num_available)

                if num_to_gather > 0:
                    # Select tokens with lowest redundancy (smallest
                    # similarity)
                    _, least_similar_indices = torch.topk(cos_sim, k=num_to_gather, largest=False)
                    kept_local_indices.update(least_similar_indices.cpu().tolist())

        # 6. Build global keep_mask
        final_kept_vision_indices = torch.tensor([], dtype=torch.long, device=device)
        if kept_local_indices:
            valid_indices = torch.tensor(
                list(kept_local_indices),
                device=device,
                dtype=torch.long,
            )
            valid_indices = valid_indices[valid_indices < len(vision_indices)]
            if len(valid_indices) > 0:
                final_kept_vision_indices = vision_indices[valid_indices]

        final_indices = torch.cat([non_vision_indices.to(device), final_kept_vision_indices])
        keep_mask = torch.zeros_like(input_ids_single, dtype=torch.bool)
        if final_indices.numel() > 0:
            keep_mask[final_indices] = True
        batch_keep_masks.append(keep_mask)

    return torch.stack(batch_keep_masks, dim=0)
