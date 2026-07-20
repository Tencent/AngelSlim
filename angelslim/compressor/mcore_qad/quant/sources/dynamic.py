"""Dynamic scale: just use the data-derived reference, recomputed every forward.

Used for activation per-(token,block) scales whose layout spans the runtime token
dimension and therefore cannot be a learnable parameter.
"""

from __future__ import annotations

from torch import Tensor

from angelslim.compressor.mcore_qad.quant.sources.base import (
    SOURCE_REGISTRY,
    ScaleStore,
)


@SOURCE_REGISTRY.register("dynamic")
class DynamicScale(ScaleStore):
    def forward(self, ref: Tensor) -> Tensor:
        return ref

    def is_learnable(self) -> bool:
        return False
