"""Uniform integer formats: INT4 / INT8 (and arbitrary bit-widths).

Uniform grid -> simple round(x/1)*1 in normalized space with clamp to [qmin,qmax].
Symmetric only (no zero-point): the framework trains scales only, so asymmetric
quantization is intentionally out of scope.
"""

from __future__ import annotations

from torch import Tensor

from angelslim.compressor.mcore_qad.quant.formats.base import (
    FORMAT_REGISTRY,
    QuantFormat,
)
from angelslim.compressor.mcore_qad.quant.functional import round_ste


class IntFormat(QuantFormat):
    def __init__(self, bits: int, symmetric: bool = True) -> None:
        self.bits = bits
        self.symmetric = symmetric
        self.max_repr = float(2 ** (bits - 1) - 1)  # e.g. 127 for int8

    def to_grid(self, x_normalized: Tensor) -> Tensor:
        """clamp then round-to-nearest on the uniform integer grid, STE gradient.

        Clamp first (differentiable, 0 grad outside range) so saturated elements
        get the LSQ-correct scale gradient; round_ste passes gradient through.
        """
        x_c = x_normalized.clamp(self.qmin(), self.qmax())
        return round_ste(x_c)


@FORMAT_REGISTRY.register("int8")
class Int8Format(IntFormat):
    def __init__(self) -> None:
        super().__init__(bits=8, symmetric=True)


@FORMAT_REGISTRY.register("int4")
class Int4Format(IntFormat):
    def __init__(self) -> None:
        super().__init__(bits=4, symmetric=True)
