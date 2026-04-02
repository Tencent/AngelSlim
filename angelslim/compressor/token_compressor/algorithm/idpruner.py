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
MMR + Vision Selector Pruning Strategy.
"""

from typing import Callable, Union

import torch

from ..base.context import PruningContext
from .utils.utils import _extract_and_validate_vision_token_info
from .utils.vision_selector_utils import get_universal_selector_scores


def _default_similarity_compute(
    features: torch.Tensor,
) -> torch.Tensor:
    """Default similarity: Cosine Similarity."""
    features_norm = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.matmul(features_norm, features_norm.t())


def _default_importance_compute(scores: torch.Tensor) -> torch.Tensor:
    """Default importance: Linear normalization to [0, 1]."""
    s_min, s_max = scores.min(), scores.max()
    return (scores - s_min) / (s_max - s_min + 1e-6)


def _resolve_callable(func_or_str: Union[Callable, str, None], default_func: Callable) -> Callable:
    """Resolves function parameters as callable objects."""
    if func_or_str is None:
        return default_func
    if callable(func_or_str):
        return func_or_str
    if isinstance(func_or_str, str):
        local_scope = {}
        global_scope = {
            "torch": torch,
            "nn": torch.nn,
            "F": torch.nn.functional,
        }
        exec(func_or_str, global_scope, local_scope)
        return local_scope["compute"]
    return default_func


def idpruner(context: PruningContext, **kwargs) -> torch.Tensor:
    """
    Combined MMR algorithm and Vision Selector for redundant token pruning.

    Args:
        context (PruningContext): Execution context with embeds.
        **kwargs:
            ratio (float): Pruning ratio.
            selector_path (str): Path to the Vision Selector model.
            mmr_lambda (float): Balance coefficient (Importance vs Diversity).
            parallel_k (int): Number of tokens to select per greedy iteration.

    Returns:
        torch.Tensor: Boolean keep_mask of shape [B, seq_len].
    """
    # 1. Parameter parsing
    try:
        ratio = kwargs["ratio"]
        selector_path = kwargs["selector_path"]
        mmr_lambda = kwargs.get("mmr_lambda", 0.5)
        parallel_k = kwargs.get("parallel_k", 1)
        sim_func = _resolve_callable(kwargs.get("similarity_func"), _default_similarity_compute)
        imp_func = _resolve_callable(kwargs.get("importance_func"), _default_importance_compute)
    except KeyError as e:
        raise ValueError(f"[TokenCompressor Error] idpruner missing required parameter: {e}")

    # 2. Fetch context data via attributes
    input_ids = context.input_ids
    inputs_embeds = context.inputs_embeds

    if input_ids is None or inputs_embeds is None:
        raise ValueError("[TokenCompressor Error] Context missing 'input_ids' or 'inputs_embeds'.")

    bsz = inputs_embeds.shape[0]
    device = inputs_embeds.device
    batch_keep_masks = []

    # 3. Process each batch item
    for i in range(bsz):
        ids_single = input_ids[i]
        embeds_single = inputs_embeds[i]

        vision_indices, non_vision_indices, _, _ = _extract_and_validate_vision_token_info(context)
        N = len(vision_indices)

        keep_mask_single = torch.ones_like(ids_single, dtype=torch.bool)

        if N > 0:
            num_to_keep = int(round(N * (1.0 - ratio)))
            if ratio < 1.0 and num_to_keep == 0:
                num_to_keep = 1

            if num_to_keep < N:
                keep_mask_single = torch.zeros_like(ids_single, dtype=torch.bool)
                keep_mask_single[non_vision_indices] = True

                # 4. Feature extraction and importance scoring
                visual_features = embeds_single[vision_indices, :]
                raw_scores = get_universal_selector_scores(
                    selector_path=selector_path,
                    vision_hidden=visual_features.unsqueeze(0),
                    text_hidden=None,
                    mode="raw",
                ).squeeze(0)

                importance = imp_func(raw_scores).float()
                similarity = sim_func(visual_features.float())

                # 5. Parallelized MMR greedy selection
                selected_indices = []
                candidates_mask = torch.ones(N, dtype=torch.bool, device=device)
                max_sim_values = torch.full((N,), -2.0, device=device)

                while len(selected_indices) < num_to_keep:
                    k_step = min(
                        parallel_k,
                        num_to_keep - len(selected_indices),
                    )

                    if len(selected_indices) == 0:
                        mmr_score = importance.clone()
                    else:
                        mmr_score = (mmr_lambda * importance) - ((1 - mmr_lambda) * max_sim_values)

                    mmr_score[~candidates_mask] = -float("inf")
                    _, best_indices = torch.topk(mmr_score, k=k_step)

                    selected_indices.extend(best_indices.tolist())
                    candidates_mask[best_indices] = False

                    # Update max similarity maintenance vector
                    batch_sims = similarity[:, best_indices]
                    batch_max_sim, _ = torch.max(batch_sims, dim=1)
                    max_sim_values = torch.maximum(max_sim_values, batch_max_sim)

                # 6. Construct local keep_mask
                kept_indices = vision_indices[torch.tensor(selected_indices, device=device)]
                keep_mask_single[kept_indices] = True

        batch_keep_masks.append(keep_mask_single)

    return torch.stack(batch_keep_masks, dim=0)
