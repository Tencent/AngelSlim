"""Qwen3-MoE HF -> mcore weight conversion."""

from __future__ import annotations

from typing import Dict

from torch import Tensor

from angelslim.compressor.mcore_qad.models.base import (
    attn_to_mcore,
    experts_to_mcore,
    globals_to_mcore,
)


def qwen3_to_mcore(hf: Dict[str, Tensor], cfg, meta) -> Dict[str, Tensor]:
    out: Dict[str, Tensor] = {}
    globals_to_mcore(hf, meta, out)
    for i in range(cfg.num_layers):
        p, m = f"model.layers.{i}", f"decoder.layers.{i}"
        attn_to_mcore(hf, p, m, cfg, out)
        out[f"{m}.mlp.router.weight"] = hf[f"{p}.mlp.gate.weight"]
        experts_to_mcore(hf, p, m, cfg.num_moe_experts, out)
    return out
