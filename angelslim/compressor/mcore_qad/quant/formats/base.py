"""Axis 1 -- Numeric Format: the discrete grid a real value snaps to.

A Format knows ONLY about the representable grid and how to round a *normalized*
(already scale-divided) value onto it, with the right gradient. It knows nothing
about scales, granularity, tensors, or parallelism. This separation is what lets
NVFP4/MXFP4/INT4/INT8 coexist without special-casing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor

from angelslim.compressor.mcore_qad.quant.registry import Registry

FORMAT_REGISTRY: Registry["QuantFormat"] = Registry("format")


class QuantFormat(ABC):
    #: largest representable magnitude on the grid (e.g. E2M1=6.0, INT8=127).
    max_repr: float
    #: whether the grid is symmetric around 0 (affects qmin/qmax).
    symmetric: bool = True

    @abstractmethod
    def to_grid(self, x_normalized: Tensor) -> Tensor:
        """Snap an already scale-normalized tensor onto the grid (with STE/LSQ).

        Input is expected to be roughly in [-max_repr, max_repr]; values outside
        are clamped. Must be differentiable via the chosen estimator.
        """

    def qmin(self) -> float:
        return -self.max_repr if self.symmetric else 0.0

    def qmax(self) -> float:
        return self.max_repr
