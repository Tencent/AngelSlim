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

"""Smooth statistics I/O and per-layer alpha grid search.

Combines:

* The smooth-stat data structure & JSON I/O — the single bridge between
  Phase 1 (vLLM online stat collection) and Phase 2 (offline weight
  conversion).  The JSON file written by ``get_smooth_stats`` is consumed
  by ``apply_qk_smooth`` / ``apply_vo_smooth`` / ``apply_down_proj_smooth``.
* The *pure-tensor* core of :class:`SmoothAlphaSearcher`: it takes
  already-collected ``x``, ``w``, ``act_absmax`` tensors (typically on a
  single TP shard) and returns the best alpha + smooth_weight by
  minimising fake-quant MSE.  No distributed primitives, no module
  discovery, no FusedMoE awareness — those concerns live in
  ``smooth/vllm/searcher_dist.py``.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .tensor_math import (
    smooth_default,
    smooth_per_tensor_act_first,
    smooth_qdq_act,
    smooth_qdq_weight,
)

__all__ = [
    # smooth_stats
    "SmoothStats",
    "load_smooth_stats",
    "save_smooth_stats",
    "load_alpha_search_results",
    "save_alpha_search_results",
    # searcher
    "smooth_alpha_search_layer",
]


# ===========================================================================
# Smooth-stat data structure & JSON I/O
# ===========================================================================


@dataclass
class SmoothStats:
    """A single layer's smooth statistics."""

    absmax: Optional[torch.Tensor]  # [C] per-channel absmax
    ema: Optional[torch.Tensor]  # [C] per-channel EMA
    call_count: int = 0


def load_smooth_stats(json_path: str, use_ema: bool = False) -> dict:
    """Load smooth stats from JSON file and convert lists back to tensors.

    The original ``smooth_stats.json`` schema is::

        {
          "<layer_key>": {
            "absmax": [float, ...],
            "ema":    [float, ...],
            "call_count": int
          },
          ...
        }

    Args:
        json_path: path to smooth_stats.json.
        use_ema: if True, use the EMA field as the per-channel scale;
            otherwise use absmax.

    Returns:
        ``dict[layer_key, {"scale": Tensor[C], "call_count": int}]``.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    field = "ema" if use_ema else "absmax"
    out = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        vals = entry.get(field)
        if vals is None:
            continue
        out[key] = {
            "scale": torch.tensor(vals, dtype=torch.float32),
            "call_count": int(entry.get("call_count", 0)),
        }
    return out


def save_smooth_stats(stats: dict, json_path: str) -> None:
    """Serialise ``model._smooth_stats``-shaped data to JSON.

    Args:
        stats: the dict returned by ``get_smooth_stats(model)`` — already
            JSON-serialisable (lists, ints).
        json_path: target file path.
    """
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)


def load_alpha_search_results(json_path: str) -> dict:
    """Load smooth_alpha_search.json.

    Returns the inner ``"results"`` dict if present, otherwise the whole
    payload (for forwards compat with older / flatter formats)::

        {
          "<layer_key>": {
            "alpha": [float, ...],     # per-rank list under TP
            "smooth_weight": [float],  # full [intermediate_size]
            "loss": float
          },
          ...
        }
    """
    with open(json_path, "r") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "results" in raw:
        return raw["results"]
    return raw


def save_alpha_search_results(payload: dict, json_path: str) -> None:
    """Save alpha search payload (config + results) to JSON."""
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)


# ===========================================================================
# Per-layer alpha grid search
# ===========================================================================


def smooth_alpha_search_layer(
    x: torch.Tensor,
    w: torch.Tensor,
    act_absmax: torch.Tensor,
    config,
) -> Tuple[float, torch.Tensor, float]:
    """Run the per-layer grid search.

    Args:
        x: ``[N, in_features]`` raw activation (float, on the search device).
        w: ``[out_features, in_features]`` weight tensor (float, on device).
        act_absmax: ``[in_features]`` per-channel activation absmax/EMA.
        config: ``SmoothAlphaSearchConfig`` instance.

    Returns:
        Tuple of ``(best_alpha_or_mul, best_smooth_weight, best_loss)``.
        For ``smooth_search_mode="per-tensor-act-first"`` the first
        element is the optimal multiplier; for ``"default"`` it is the
        optimal ``alpha``.
    """
    cfg = config
    device = x.device
    act_absmax = act_absmax.float().to(device).clamp(min=1e-8)
    weight_absmax = w.abs().max(dim=0).values.clamp(min=1e-8)

    # Reference output (computed once)
    y_ref = x @ w.T  # [N, out_feat]

    if cfg.smooth_search_mode == "default":
        alphas = torch.linspace(cfg.alpha_min, cfg.alpha_max, cfg.alpha_steps).tolist()
        losses = torch.empty(len(alphas), device=device)
        for i, alpha in enumerate(alphas):
            smooth = smooth_default(
                act_absmax,
                weight_absmax,
                alpha=alpha,
                smooth_min=cfg.smooth_min,
                smooth_max=cfg.smooth_max,
            )
            x_smoothed = x / smooth.unsqueeze(0)
            w_smoothed = w * smooth.unsqueeze(0)
            x_q = smooth_qdq_act(
                x_smoothed,
                method=cfg.act_quant_method,
                quant_type=cfg.act_quant_type,
                bits=cfg.weight_quant_bits,
            )
            w_q = smooth_qdq_weight(
                w_smoothed,
                method=cfg.weight_quant_method,
                quant_type=cfg.weight_quant_type,
                bits=cfg.weight_quant_bits,
                group_size=cfg.weight_group_size,
                block_size=cfg.block_size,
            )
            y_q = x_q @ w_q.T
            losses[i] = ((y_ref - y_q) ** 2).mean()

        best_idx = int(losses.argmin().item())
        best_alpha = alphas[best_idx]
        best_smooth = smooth_default(
            act_absmax,
            weight_absmax,
            alpha=best_alpha,
            smooth_min=cfg.smooth_min,
            smooth_max=cfg.smooth_max,
        )
        return best_alpha, best_smooth, losses[best_idx].item()

    elif cfg.smooth_search_mode == "per-tensor-act-first":
        muls = torch.linspace(cfg.act_mul_min, cfg.act_mul_max, cfg.alpha_steps).tolist()
        losses = torch.empty(len(muls), device=device)
        for i, mul in enumerate(muls):
            smooth = smooth_per_tensor_act_first(
                act_absmax,
                mul=mul,
                smooth_min=cfg.smooth_min,
                smooth_max=cfg.smooth_max,
            )
            x_smoothed = x / smooth.unsqueeze(0)
            w_smoothed = w * smooth.unsqueeze(0)
            x_q = smooth_qdq_act(
                x_smoothed,
                method=cfg.act_quant_method,
                quant_type=cfg.act_quant_type,
                bits=cfg.weight_quant_bits,
            )
            w_q = smooth_qdq_weight(
                w_smoothed,
                method=cfg.weight_quant_method,
                quant_type=cfg.weight_quant_type,
                bits=cfg.weight_quant_bits,
                group_size=cfg.weight_group_size,
                block_size=cfg.block_size,
            )
            y_q = x_q @ w_q.T
            losses[i] = ((y_ref - y_q) ** 2).mean()

        best_idx = int(losses.argmin().item())
        best_mul = muls[best_idx]
        best_smooth = smooth_per_tensor_act_first(
            act_absmax,
            mul=best_mul,
            smooth_min=cfg.smooth_min,
            smooth_max=cfg.smooth_max,
        )
        return best_mul, best_smooth, losses[best_idx].item()

    else:
        raise ValueError(f"Unknown smooth_search_mode: {cfg.smooth_search_mode}")
