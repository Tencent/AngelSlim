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

import math

import torch

from ..base.context import PruningContext
from .utils.utils import _extract_and_validate_vision_token_info


def special_token_based_attention_pruning(context: PruningContext, **kwargs) -> torch.Tensor:
    """
    Pruning strategy base on attention scores between specific tokens and vision tokens.
    Supports 'last_text' and 'global_average' strategies.

    Args:
        context (PruningContext): Execution context with model activations.
        **kwargs:
            ratio (float): Pruning ratio for vision tokens.
            query_source (dict): Configuration for selecting Query tokens.
                - strategy (str): 'last_text' or 'global_average'.
                - token_id (int): Required if strategy is 'global_average'.

    Returns:
        torch.Tensor: Boolean keep_mask of shape [B, seq_len].
    """
    # 1. Parse parameters and validate context attributes
    try:
        ratio = kwargs["ratio"]
        query_source = kwargs["query_source"]
        strategy = query_source["strategy"]
        layer_idx = kwargs["layer_idx"]
    except KeyError as e:
        raise ValueError(
            f"[AngelSlim Error] Missing required parameter for attention pruning: {e}"
        )

    input_ids = context.input_ids
    # Using attribute access for LLM states (last captured layer)
    q = context.llm_q[layer_idx]
    k = context.llm_k[layer_idx]

    if input_ids is None or q is None or k is None:
        raise ValueError("[AngelSlim Error] Missing 'input_ids' or Q/K tensors in context.")

    bsz, seq_len = input_ids.shape
    device = input_ids.device
    if bsz != 1:
        raise NotImplementedError(
            "[AngelSlim Error] Attention-based pruning only supports batch_size=1."
        )

    batch_keep_masks = []
    for i in range(bsz):
        ids_single = input_ids[i]
        q_single = q[i]
        k_single = k[i]

        # 2. Identify vision and non-vision tokens via dynamic check
        vision_indices, non_vision_indices, _, _ = _extract_and_validate_vision_token_info(context)

        num_vision_tokens = len(vision_indices)
        if num_vision_tokens == 0:
            batch_keep_masks.append(torch.ones_like(ids_single, dtype=torch.bool))
            continue

        num_to_keep = int(round(num_vision_tokens * (1 - ratio)))
        if ratio < 1.0 and num_to_keep == 0 and num_vision_tokens > 0:
            num_to_keep = 1

        if num_to_keep >= num_vision_tokens:
            batch_keep_masks.append(torch.ones_like(ids_single, dtype=torch.bool))
            continue

        # 3. Locate Query token indices
        query_indices = []
        if strategy == "last_text":
            if len(non_vision_indices) > 0:
                query_indices.append(non_vision_indices[-1])
        elif strategy == "global_average":
            token_id = query_source.get("token_id")
            if token_id is None:
                raise ValueError(
                    "[AngelSlim Error] 'global_average' strategy requires 'token_id'."
                )

            match_indices = torch.where(ids_single == token_id)[0]
            if len(match_indices) > 0:
                query_indices.extend(match_indices.tolist())
        else:
            raise ValueError(f"[AngelSlim Error] Unsupported strategy: '{strategy}'.")

        if not query_indices:
            batch_keep_masks.append(torch.ones_like(ids_single, dtype=torch.bool))
            continue

        # 4. Compute attention scores
        query_idx_tensor = torch.tensor(query_indices, device=device, dtype=torch.long)
        # Average Q vectors across selected query positions
        q_query = q_single[:, query_idx_tensor, :].mean(dim=1, keepdim=True)
        k_visual = k_single[:, vision_indices, :]

        # GQA/MQA broadcast
        num_q_heads, num_kv_heads = (
            q_query.shape[0],
            k_visual.shape[0],
        )
        if num_q_heads != num_kv_heads:
            k_visual = k_visual.repeat_interleave(num_q_heads // num_kv_heads, dim=0)

        attn_logits = torch.einsum("hid,hjd->hij", q_query, k_visual) / math.sqrt(
            q_query.shape[-1]
        )
        scores = torch.softmax(attn_logits, dim=-1).mean(dim=0).squeeze(0)

        # 5. Build keep mask
        _, top_indices = torch.topk(scores, k=min(num_to_keep, num_vision_tokens))
        kept_indices = vision_indices[top_indices]

        final_indices = torch.cat([non_vision_indices.to(device), kept_indices])
        mask = torch.zeros_like(ids_single, dtype=torch.bool)
        mask[final_indices] = True
        batch_keep_masks.append(mask)

    return torch.stack(batch_keep_masks, dim=0)
