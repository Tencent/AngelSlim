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

"""Stacked (grouped) weight fake-quant for MoE experts -- one fused Triton call.

The whole local expert stack [E, OUT, IN] is quantized in ONE vectorized call instead of
per-expert (the per-expert path launches ~6000 tiny quant routines/forward -- the measured
bottleneck). Each grouped weight quantizer mirrors its per-linear `schemes/` counterpart,
generalized to a leading expert dim, and uses a fused Triton kernel that writes the
dequantized bf16 output directly + computes d/d_alpha analytically (peak memory ~ output
tensor only). They are pluggable by weight numeric format via GROUPED_WEIGHT_QUANT:

    e2m1 (NVFP4) -> GroupedNVFP4Weight       block-16, E4M3 block scale x per-expert FP32
    int4         -> GroupedInt4GroupWeight   group-128, single-level per-group scale (W4A8)

Add a format = register one small module here; the expert path stays format-agnostic.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from angelslim.compressor.mcore_qad.quant.formats.base import FORMAT_REGISTRY
from angelslim.compressor.mcore_qad.quant.functional import lsq_grad_scale
from angelslim.compressor.mcore_qad.quant.spec import QuantSpec

_E2M1 = FORMAT_REGISTRY.create("e2m1")
_E4M3 = FORMAT_REGISTRY.create("e4m3")
_INT4 = FORMAT_REGISTRY.create("int4")


class GroupedNVFP4Weight(nn.Module):
    """Per-block-learnable NVFP4 quantizer over a stacked expert weight [E, OUT, IN]."""

    def __init__(self, shape3d, group_size: int = 16) -> None:
        super().__init__()
        E, OUT, IN = shape3d
        assert IN % group_size == 0, f"IN {IN} not divisible by block {group_size}"
        self.g = group_size
        nb = IN // group_size
        self.alpha = nn.Parameter(torch.ones(E, OUT, nb))
        self.register_buffer("ref", torch.ones(E, OUT, nb))
        self.register_buffer("S", torch.ones(E, 1, 1))
        self.register_buffer("_init", torch.zeros((), dtype=torch.bool))
        self.lsq = lsq_grad_scale(group_size, _E2M1.max_repr)
        self.enabled = True  # quant_disabled() flips this (QAD teacher)

    @torch.no_grad()
    def _initialize(self, W: Tensor) -> None:
        E, OUT, IN = W.shape
        S = (W.abs().amax(dim=(1, 2), keepdim=True) / (_E2M1.max_repr * _E4M3.max_repr)).clamp_min(
            1e-10
        )
        block_amax = W.reshape(E, OUT, IN // self.g, self.g).abs().amax(dim=-1)
        self.ref.copy_(((block_amax / _E2M1.max_repr) / S).clamp_min(1e-12))
        self.S.copy_(S)
        self.alpha.fill_(1.0)
        self._init.fill_(True)

    def forward(self, W: Tensor) -> Tensor:
        if not self.enabled:  # quant-off teacher pass
            return W
        if not bool(self._init):
            self._initialize(W)
        from angelslim.compressor.mcore_qad.quant.kernels import triton_nvfp4

        return triton_nvfp4(W.contiguous(), self.alpha, self.ref, self.S, self.g, self.lsq)


class GroupedInt4GroupWeight(nn.Module):
    """Per-group-learnable symmetric INT4 quantizer over a stacked expert weight [E, OUT, IN].

    Single-level (plain per-group scale, no E4M3 nesting / no per-expert global), matching
    `schemes/per_group`; the W4 lever in W4A8. group_size defaults to 128 (vLLM W4AFP8).
    """

    def __init__(self, shape3d, group_size: int = 128) -> None:
        super().__init__()
        E, OUT, IN = shape3d
        assert IN % group_size == 0, f"IN {IN} not divisible by group {group_size}"
        self.g = group_size
        nb = IN // group_size
        self.alpha = nn.Parameter(torch.ones(E, OUT, nb))
        self.register_buffer("ref", torch.ones(E, OUT, nb))
        self.register_buffer("_init", torch.zeros((), dtype=torch.bool))
        self.lsq = lsq_grad_scale(group_size, _INT4.max_repr)
        self.enabled = True

    @torch.no_grad()
    def _initialize(self, W: Tensor) -> None:
        E, OUT, IN = W.shape
        group_amax = W.reshape(E, OUT, IN // self.g, self.g).abs().amax(dim=-1)
        self.ref.copy_((group_amax / _INT4.max_repr).clamp_min(1e-12))
        self.alpha.fill_(1.0)
        self._init.fill_(True)

    def forward(self, W: Tensor) -> Tensor:
        if not self.enabled:
            return W
        if not bool(self._init):
            self._initialize(W)
        from angelslim.compressor.mcore_qad.quant.kernels import triton_int4_group

        return triton_int4_group(W.contiguous(), self.alpha, self.ref, self.g, self.lsq)


#: weight numeric format -> grouped weight quantizer class (pluggable).
GROUPED_WEIGHT_QUANT = {
    "e2m1": GroupedNVFP4Weight,
    "int4": GroupedInt4GroupWeight,
}


def build_grouped_weight_quant(weight_spec: QuantSpec, shape3d) -> nn.Module:
    """Build the stacked-expert weight quantizer for a weight QuantSpec (None if identity)."""
    if weight_spec.is_identity():
        return None
    cls = GROUPED_WEIGHT_QUANT.get(weight_spec.fmt)
    if cls is None:
        raise NotImplementedError(
            f"grouped experts have no weight quantizer for fmt {weight_spec.fmt!r}; "
            f"available: {sorted(GROUPED_WEIGHT_QUANT)} (add one in grouped_quant.py)."
        )
    g = weight_spec.group_size or (16 if weight_spec.fmt == "e2m1" else 128)
    return cls(shape3d, group_size=g)
