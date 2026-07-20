"""hy_v3 (HunYuan-3) HF -> mcore weight conversion.

Handles dense first layers, MoE, shared experts, and expert bias.
"""

from __future__ import annotations

from typing import Dict

from torch import Tensor

from angelslim.compressor.mcore_qad.models.base import (
    attn_to_mcore,
    experts_to_mcore,
    gated_fc1,
    globals_to_mcore,
)


def _first(hf: Dict[str, Tensor], *keys: str) -> str:
    """First present key among aliases (HF naming drifts across transformers 5.x releases:
    the released Hy3 checkpoint uses ``router.gate`` / ``expert_bias`` / ``shared_mlp``,
    while the current modeling emits ``gate`` / ``e_score_correction_bias`` / ``shared_experts``).
    """
    for k in keys:
        if k in hf:
            return k
    return keys[0]


def hy_v3_to_mcore(hf: Dict[str, Tensor], cfg, meta) -> Dict[str, Tensor]:
    out: Dict[str, Tensor] = {}
    globals_to_mcore(hf, meta, out)
    freq = cfg.moe_layer_freq if isinstance(cfg.moe_layer_freq, list) else [1] * cfg.num_layers
    for i in range(cfg.num_layers):
        p, m = f"model.layers.{i}", f"decoder.layers.{i}"
        attn_to_mcore(hf, p, m, cfg, out)
        if freq[i] == 0:  # dense layer
            out[f"{m}.mlp.linear_fc1.weight"] = gated_fc1(
                hf, f"{p}.mlp.gate_proj.weight", f"{p}.mlp.up_proj.weight"
            )
            out[f"{m}.mlp.linear_fc2.weight"] = hf[f"{p}.mlp.down_proj.weight"].contiguous()
        else:  # MoE layer
            out[f"{m}.mlp.router.weight"] = hf[
                _first(hf, f"{p}.mlp.router.gate.weight", f"{p}.mlp.gate.weight")
            ]
            bias_key = _first(hf, f"{p}.mlp.expert_bias", f"{p}.mlp.e_score_correction_bias")
            if bias_key in hf:
                out[f"{m}.mlp.router.expert_bias"] = hf[bias_key]
            experts_to_mcore(hf, p, m, cfg.num_moe_experts, out)
            if cfg.moe_shared_expert_intermediate_size:  # shared expert
                sp = _first(
                    hf,
                    f"{p}.mlp.shared_mlp.gate_proj.weight",
                    f"{p}.mlp.shared_experts.gate_proj.weight",
                ).rsplit(".gate_proj.weight", 1)[0]
                out[f"{m}.mlp.shared_experts.linear_fc1.weight"] = gated_fc1(
                    hf, f"{sp}.gate_proj.weight", f"{sp}.up_proj.weight"
                )
                out[f"{m}.mlp.shared_experts.linear_fc2.weight"] = hf[
                    f"{sp}.down_proj.weight"
                ].contiguous()
    return out
