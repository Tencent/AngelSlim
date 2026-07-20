"""Calibrated (frozen) scale loaded from outside -- no gradient."""

from __future__ import annotations

import torch
from torch import Tensor

from angelslim.compressor.mcore_qad.quant.sources.base import (
    SOURCE_REGISTRY,
    ScaleStore,
)


@SOURCE_REGISTRY.register("calibrated")
class CalibratedScale(ScaleStore):
    def __init__(self, shape) -> None:
        super().__init__()
        self.register_buffer("scale", torch.ones(shape))

    def forward(self, ref: Tensor) -> Tensor:
        return self.scale

    def is_learnable(self) -> bool:
        return False

    @torch.no_grad()
    def init_value(self, value: Tensor) -> None:
        self.scale.copy_(value.expand_as(self.scale))
