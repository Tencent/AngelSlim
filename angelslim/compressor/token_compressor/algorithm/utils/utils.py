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

"""General utility functions for token compression strategies."""

import math
from typing import Any, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from ...base.context import PruningContext

SUPPORTED_MODEL_TYPES = {"llava", "qwen2_5_vl", "llava_ov"}


def _get_config_attr(config: Any, attr_name: str, default: Any = None) -> Any:
    """Helper to access attributes from either a config object or a dictionary."""
    if isinstance(config, dict):
        return config.get(attr_name, default)
    return getattr(config, attr_name, default)


def identify_model_architecture(context: PruningContext) -> str:
    """
    Identifies the model architecture based on the context's model_config.
    """
    config = context.model_config
    if not config:
        raise ValueError("[TokenCompressor Error] 'model_config' not found in context.")

    model_type = str(_get_config_attr(config, "model_type")).lower()

    vision_config = _get_config_attr(config, "vision_config")
    if vision_config:
        vision_model_type = str(_get_config_attr(vision_config, "model_type", "")).lower()
        if "rice" in vision_model_type:
            return "llava_ov"

    if "llavaonevision" in model_type:
        return "llava_ov"
    if "llava" in model_type:
        return "llava"
    if "qwen2_5_vl" in model_type:
        return "qwen2_5_vl"

    raise ValueError(f"[TokenCompressor Error] Unsupported model type: '{model_type}'.")


def get_model_specific_vision_token_ids(
    context: PruningContext,
) -> Set[int]:
    """Retrieves vision-related token IDs specific to the current model architecture."""
    model_type = identify_model_architecture(context)
    config = context.model_config
    token_ids = set()

    if model_type == "llava":
        for key in ["image_token_index", "image_token_id"]:
            val = _get_config_attr(config, key)
            if val is not None:
                token_ids.add(val)
    elif "qwen" in model_type or model_type == "llava_ov":
        for key in [
            "image_token_id",
            "video_token_id",
            "vision_token_id",
        ]:
            val = _get_config_attr(config, key)
            if val is not None:
                token_ids.add(val)

    if not token_ids:
        raise ValueError(
            "[TokenCompressor Error] " f"Failed to extract vision token IDs for {model_type}."
        )
    return token_ids


def _extract_and_validate_vision_token_info(
    context: PruningContext,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Locates vision tokens and validates image block structure against metadata.
    """
    input_ids = context.input_ids
    if input_ids is None:
        raise ValueError("[TokenCompressor Error] 'input_ids' missing in context.")

    input_ids_single = input_ids.squeeze(0)
    model_type = identify_model_architecture(context)

    # 1. Build vision mask
    target_ids = get_model_specific_vision_token_ids(context)
    vision_mask = torch.zeros_like(input_ids_single, dtype=torch.bool)
    for tid in target_ids:
        vision_mask |= input_ids_single == tid

    # 2. Extract indices
    vision_indices = torch.where(vision_mask)[0]
    non_vision_indices = torch.where(~vision_mask)[0]

    # 3. Infer actual block structure
    actual_counts = []
    if len(vision_indices) > 0:
        diffs = vision_indices[1:] - vision_indices[:-1]
        splits = torch.where(diffs > 1)[0] + 1
        actual_counts = [len(b) for b in torch.tensor_split(vision_indices, splits.cpu())]

    # 4. Strict validation for Qwen/LLaVA-OV
    if "qwen" in model_type or model_type == "llava_ov":
        grid = context.image_grid_thw
        merge_size = context.spatial_merge_size
        if grid is not None:
            expected_counts = (
                ((grid[:, 1] // merge_size) * (grid[:, 2] // merge_size)).cpu().tolist()
            )
            if sum(actual_counts) != sum(expected_counts):
                raise ValueError(
                    "[TokenCompressor Error] Strict Check Failed: Token count mismatch."
                )

    return (
        vision_indices,
        non_vision_indices,
        vision_mask,
        actual_counts,
    )


def _recompute_attention_maps_for_all_images(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    context: PruningContext,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Recomputes attention importance scores and metric features per image."""
    final_scores_list, final_keys_list = [], []
    model_type = identify_model_architecture(context)
    head_dim = q_tensor.shape[-1]

    if model_type == "llava":
        for i in range(q_tensor.shape[0]):
            q_cls = q_tensor[i, :, 0:1, :]
            k_patches = k_tensor[i, :, 1:, :]
            weights = torch.softmax(
                torch.matmul(q_cls, k_patches.transpose(-1, -2)) / math.sqrt(head_dim),
                dim=-1,
            )
            final_scores_list.append(weights.mean(dim=0))
            final_keys_list.append(k_patches.mean(dim=0).unsqueeze(0))

    elif "qwen" in model_type or model_type == "llava_ov":
        cu_seqlens = context.cu_seqlens_full
        if cu_seqlens is None:
            raise ValueError("[TokenCompressor Error] Missing 'cu_seqlens_full'.")

        lengths = torch.diff(cu_seqlens).cpu().tolist()
        q_physical = q_tensor.squeeze(0) if q_tensor.dim() == 4 else q_tensor
        k_physical = k_tensor.squeeze(0) if k_tensor.dim() == 4 else k_tensor

        q_splits = torch.split(q_physical, lengths, dim=1)
        k_splits = torch.split(k_physical, lengths, dim=1)

        temp_scores, temp_keys = [], []
        for q_slice, k_slice in zip(q_splits, k_splits):
            if model_type == "llava_ov":
                q_use, k_use = q_slice[:, 0:1, :], k_slice[:, 1:, :]
                weights = torch.softmax(
                    torch.matmul(q_use, k_use.transpose(-1, -2)) / math.sqrt(head_dim),
                    dim=-1,
                )
                score_fine = weights.mean(dim=0).squeeze(0)
            else:
                q_use = q_slice.mean(dim=1, keepdim=True) if q_slice.shape[1] >= 6144 else q_slice
                weights = torch.softmax(
                    torch.matmul(q_use, k_slice.transpose(-1, -2)) / math.sqrt(head_dim),
                    dim=-1,
                )
                score_fine = weights.mean(dim=0).sum(dim=0)
                k_use = k_slice

            merge_unit = context.spatial_merge_size**2
            if merge_unit > 1:
                pad = (merge_unit - (score_fine.shape[0] % merge_unit)) % merge_unit
                s_pad = F.pad(score_fine, (0, pad)).view(-1, merge_unit).sum(-1)
                k_pad = (
                    F.pad(k_use, (0, 0, 0, pad))
                    .view(k_use.shape[0], -1, merge_unit, head_dim)
                    .mean(2)
                )
            else:
                s_pad, k_pad = score_fine, k_use

            temp_scores.append(s_pad)
            temp_keys.append(k_pad)

        if not temp_scores:
            return [], []

        all_scores = torch.cat(temp_scores, dim=0)
        all_keys = torch.cat(temp_keys, dim=1)

        reverse_indices = getattr(context, "reverse_indices", None)
        if reverse_indices is not None:
            if all_scores.shape[0] == reverse_indices.shape[0]:
                all_scores = all_scores[reverse_indices]
                all_keys = all_keys[:, reverse_indices, :]

        all_keys = all_keys.mean(dim=0)

        split_sizes = [t.shape[0] for t in temp_scores]
        s_splits = torch.split(all_scores, split_sizes)
        k_splits = torch.split(all_keys, split_sizes)

        for s, k in zip(s_splits, k_splits):
            final_scores_list.append(s.unsqueeze(0))
            final_keys_list.append(k.unsqueeze(0))

    return final_scores_list, final_keys_list


def get_valid_content_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """Extracts valid text content mask (excluding images/special tokens)."""
    forbidden = {151654, 151655, 151656, 151652, 151653}
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    is_vis = torch.isin(
        input_ids,
        torch.tensor(list(forbidden), device=input_ids.device),
    )
    vis_idx = torch.where(is_vis)[0]
    start = vis_idx[-1].item() + 1 if len(vis_idx) > 0 else 0
    mask[start:] = True
    return mask & (~is_vis)


def _regroup_tensors_by_count(
    source_list: List[torch.Tensor],
    target_counts: List[int],
    grid_list: Optional[torch.Tensor] = None,
):
    """Regroups physical ViT outputs into logical LLM segments."""
    regrouped, regrouped_grids = [], []
    ptr = 0
    for count in target_counts:
        curr, accumulated = [], 0
        if grid_list is not None:
            regrouped_grids.append(grid_list[ptr])
        while accumulated < count:
            t = source_list[ptr]
            curr.append(t)
            accumulated += t.shape[1]
            ptr += 1
        regrouped.append(torch.cat(curr, dim=1))
    return regrouped, (torch.stack(regrouped_grids) if regrouped_grids else None)
