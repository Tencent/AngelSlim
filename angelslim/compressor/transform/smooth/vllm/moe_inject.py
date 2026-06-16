# Copyright 2025 Tencent Inc. All Rights Reserved.
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

"""FusedMoE kernel-injection entry points for smooth-stat / alpha-search
collection.

These two functions are imported and called from the patched
``vllm/model_executor/layers/fused_moe/fused_moe.py`` (deployed via
``tools/vllm_patch/install.sh`` to ``vllm/tools/smooth_moe_inject.py``).
The kernel calls them right after computing ``intermediate_cache2``
(= ``silu(gate) * up``) and before the ``w2`` matmul.

The module-level flags ``_MOE_COLLECT_SMOOTH`` /
``_MOE_COLLECT_ALPHA_SEARCH`` are read once at import time from env
vars; ``setup_smooth_hooks`` / ``setup_smooth_alpha_search_hooks``
update them at runtime so the hot-path skips ``os.getenv`` overhead.
"""

import os

import torch

__all__ = [
    "collect_fused_moe_smooth_stats",
    "collect_fused_moe_alpha_search_values",
    "set_moe_collect_smooth",
    "set_moe_collect_alpha_search",
]


# Module-level cache for VLLM_MOE_COLLECT_SMOOTH_STATS flag.
_MOE_COLLECT_SMOOTH = os.getenv("VLLM_MOE_COLLECT_SMOOTH_STATS", "0") == "1"

# Module-level cache for VLLM_MOE_COLLECT_ALPHA_SEARCH flag.
_MOE_COLLECT_ALPHA_SEARCH = os.getenv("VLLM_MOE_COLLECT_ALPHA_SEARCH", "0") == "1"


def set_moe_collect_smooth(enable: bool) -> None:
    """Update the module-level MoE smooth-stats collection flag.

    Called by :func:`setup_smooth_hooks` so that
    :func:`collect_fused_moe_smooth_stats` (hot path) can skip the
    ``os.getenv`` call and read the cached value instead.
    """
    global _MOE_COLLECT_SMOOTH
    _MOE_COLLECT_SMOOTH = bool(enable)


def set_moe_collect_alpha_search(enable: bool) -> None:
    """Update the module-level MoE alpha-search collection flag."""
    global _MOE_COLLECT_ALPHA_SEARCH
    _MOE_COLLECT_ALPHA_SEARCH = bool(enable)


def collect_fused_moe_smooth_stats(
    stage,
    hidden_states,
    topk_ids,
    layer_name=None,
    global_smooth_stats=None,
    ema_momentum=0.9,
):
    """Collect per-expert per-channel absmax & EMA from FusedMoE
    ``down_proj`` input.

    Mirrors :class:`SmoothDownProjInputHook` but operates inside the
    fused kernel where standard PyTorch hooks cannot reach.

    Args:
        stage: must be ``"down_proj"``; other stages are ignored.
        hidden_states: ``[num_tokens * top_k, intermediate_size_shard]``.
        topk_ids: ``[num_tokens, top_k]`` expert assignment matrix.
        layer_name: FusedMoE layer name (passed via
            ``w13_weight._vllm_layer_name_smooth``).
        global_smooth_stats: model-level shared dict (passed via
            ``w13_weight._smooth_stats_of_model``).
        ema_momentum: EMA decay (passed via
            ``w13_weight._smooth_ema_momentum``).
    """
    if global_smooth_stats is None or layer_name is None:
        return
    if stage != "down_proj":
        return  # Only down_proj input is needed for smooth stats

    with torch.no_grad():
        act = hidden_states
        if act.dim() == 1:
            act = act.unsqueeze(0)

        # hidden_states shape: [num_tokens * top_k, intermediate_size]
        # topk_ids shape:      [num_tokens, top_k]
        num_tokens_hs = act.shape[0]
        num_tokens_topk = topk_ids.shape[0]
        top_k = topk_ids.shape[1]

        if num_tokens_hs == num_tokens_topk * top_k:
            flat_expert_ids = topk_ids.reshape(-1)  # [num_tokens * top_k]
        else:
            flat_expert_ids = None

        if flat_expert_ids is not None:
            # --- Per-expert stats ---
            unique_experts = flat_expert_ids.unique()
            for expert_id_tensor in unique_experts:
                expert_id = expert_id_tensor.item()
                if expert_id < 0:
                    continue  # Skip padding tokens (expert_id == -1)
                expert_key = f"{layer_name}.{expert_id}.down_proj"
                if expert_key not in global_smooth_stats:
                    global_smooth_stats[expert_key] = {
                        "absmax": None,
                        "ema": None,
                        "call_count": 0,
                    }
                mask = flat_expert_ids == expert_id_tensor
                expert_hidden = act[mask]
                if expert_hidden.numel() == 0:
                    continue
                cur_absmax = expert_hidden.abs().max(dim=0).values.detach().cpu()
                stats = global_smooth_stats[expert_key]
                if stats["absmax"] is None:
                    stats["absmax"] = cur_absmax
                else:
                    stats["absmax"] = torch.maximum(stats["absmax"], cur_absmax)
                if stats["ema"] is None:
                    stats["ema"] = cur_absmax.clone()
                else:
                    stats["ema"] = ema_momentum * stats["ema"] + (1.0 - ema_momentum) * cur_absmax
                stats["call_count"] = stats.get("call_count", 0) + 1
        else:
            # Fallback: shape mismatch, collect layer-level stats
            valid_act = act
            if valid_act.numel() == 0:
                return
            fallback_key = f"{layer_name}.down_proj"
            if fallback_key not in global_smooth_stats:
                global_smooth_stats[fallback_key] = {
                    "absmax": None,
                    "ema": None,
                    "call_count": 0,
                }
            cur_absmax = valid_act.abs().max(dim=0).values.detach().cpu()
            stats = global_smooth_stats[fallback_key]
            if stats["absmax"] is None:
                stats["absmax"] = cur_absmax
            else:
                stats["absmax"] = torch.maximum(stats["absmax"], cur_absmax)
            if stats["ema"] is None:
                stats["ema"] = cur_absmax.clone()
            else:
                stats["ema"] = ema_momentum * stats["ema"] + (1.0 - ema_momentum) * cur_absmax
            stats["call_count"] = stats.get("call_count", 0) + 1


def collect_fused_moe_alpha_search_values(
    stage,
    hidden_states,
    topk_ids,
    layer_name=None,
    alpha_search_values=None,
    max_tokens_per_expert=512,
):
    """Collect per-expert raw activation tensors for alpha grid search.

    Called from inside the patched vLLM FusedMoE kernel at the same
    injection point as :func:`collect_fused_moe_smooth_stats`.

    Args:
        stage: must be ``"down_proj"``.
        hidden_states: ``[num_tokens * top_k, intermediate_size_shard]``.
        topk_ids: ``[num_tokens, top_k]``.
        layer_name: FusedMoE layer name.
        alpha_search_values: ``model._alpha_search_values`` dict.
        max_tokens_per_expert: cap per expert.
    """
    if alpha_search_values is None or layer_name is None:
        return
    if stage != "down_proj":
        return

    with torch.no_grad():
        act = hidden_states
        if act.dim() == 1:
            act = act.unsqueeze(0)

        num_tokens_hs = act.shape[0]
        num_tokens_topk = topk_ids.shape[0]
        top_k = topk_ids.shape[1]

        if num_tokens_hs == num_tokens_topk * top_k:
            flat_expert_ids = topk_ids.reshape(-1)
        else:
            return  # shape mismatch, skip

        for expert_id_tensor in flat_expert_ids.unique():
            expert_id = expert_id_tensor.item()
            if expert_id < 0:
                continue
            expert_key = f"{layer_name}.{expert_id}.down_proj"
            mask = flat_expert_ids == expert_id_tensor
            expert_act = act[mask].detach().float().cpu()
            if expert_act.numel() == 0:
                continue

            if expert_key not in alpha_search_values:
                alpha_search_values[expert_key] = {"tokens": None}
            stored = alpha_search_values[expert_key]
            if stored["tokens"] is None:
                stored["tokens"] = expert_act
            else:
                stored["tokens"] = torch.cat([stored["tokens"], expert_act], dim=0)
            # Cap
            if stored["tokens"].shape[0] > max_tokens_per_expert:
                indices = torch.linspace(
                    0, stored["tokens"].shape[0] - 1, max_tokens_per_expert
                ).long()
                stored["tokens"] = stored["tokens"][indices]
