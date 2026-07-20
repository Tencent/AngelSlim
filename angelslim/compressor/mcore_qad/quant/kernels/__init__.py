"""Fused Triton fake-quant kernels (fwd dequant + analytic LSQ backward).

One file per numeric format; each exposes a single `triton_*` entry used by the
grouped MoE-expert weight quantizers in `quant.grouped_quant`. Add a format by
adding a file here and registering its module in `GROUPED_WEIGHT_QUANT`.
"""

from angelslim.compressor.mcore_qad.quant.kernels.triton_int4_group import (
    triton_int4_group,
)
from angelslim.compressor.mcore_qad.quant.kernels.triton_nvfp4 import triton_nvfp4

__all__ = ["triton_nvfp4", "triton_int4_group"]
