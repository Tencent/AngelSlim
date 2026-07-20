"""Build an optimizer over ONLY the trainable scale parameters (weights frozen).

Separates expert vs dense scale param groups (mcore reduces expert grads over the
expert-DP group), and asserts the framework invariant that no frozen weight leaks in.
"""

from __future__ import annotations

import torch
from torch import nn

from angelslim.compressor.mcore_qad.mcore.quantize import (
    collect_scale_parameters,  # re-exported
)
from angelslim.compressor.mcore_qad.train.config import OptimConfig

__all__ = ["collect_scale_parameters", "build_optimizer"]


def _is_expert(name: str) -> bool:
    return "experts" in name


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named:
        raise RuntimeError("no trainable scale parameters (is the weight spec 'learnable'?)")
    # invariant: every trainable param is a quantizer scale (per-linear scheme, activation
    # quant, or grouped-expert weight_q alpha).
    _scale = (".scheme.", "quantizer", "_mcore_qad_", "weight_q")
    non_scale = [name for name, _ in named if not any(token in name for token in _scale)]
    assert not non_scale, f"non-scale trainable params: {non_scale[:3]}"
    # mcore reduces expert grads over the expert-DP group -> separate param groups.
    dense = [p for n, p in named if not _is_expert(n)]
    expert = [p for n, p in named if _is_expert(n)]
    groups = [g for g in ({"params": dense}, {"params": expert}) if g["params"]]
    return torch.optim.AdamW(
        groups, lr=cfg.lr, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay
    )
