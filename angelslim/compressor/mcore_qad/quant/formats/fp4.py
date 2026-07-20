"""E2M1 (4-bit float) -- the element format of NVFP4 and MXFP4.

Grid (magnitudes): {0, 0.5, 1, 1.5, 2, 3, 4, 6}; 1 sign bit -> +/- each.
max_repr = 6.0. Non-uniform grid, so we snap to nearest representable level
(not a uniform round), with straight-through gradient.
"""

from __future__ import annotations

import torch
from torch import Tensor

from angelslim.compressor.mcore_qad.quant.formats.base import (
    FORMAT_REGISTRY,
    QuantFormat,
)
from angelslim.compressor.mcore_qad.quant.functional import nearest_level, ste

#: positive E2M1 representable magnitudes (ascending, starting at 0).
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@FORMAT_REGISTRY.register("e2m1")
class E2M1Format(QuantFormat):
    max_repr = 6.0
    symmetric = True

    def __init__(self) -> None:
        self._levels_cpu = torch.tensor(_E2M1_LEVELS, dtype=torch.float32)

    def to_grid(self, x_normalized: Tensor) -> Tensor:
        # 1. clamp into representable range (differentiable -> 0 grad outside).
        x_c = x_normalized.clamp(-self.max_repr, self.max_repr)
        # 2. snap |x| to nearest E2M1 level, reattach sign (no grad here).
        levels = self._levels_cpu.to(device=x_c.device, dtype=x_c.dtype)
        q = torch.sign(x_c) * nearest_level(x_c.abs(), levels)
        # 3. straight-through: forward = q, backward = identity wrt x_c.
        return ste(q, x_c)
