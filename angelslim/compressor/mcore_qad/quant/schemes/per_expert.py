"""Per-expert scaling for MoE: apply an inner scheme with a leading expert dim.

For a fused expert weight [E, out, in] (or activations [E, tokens, K]), the inner
scheme (e.g. two_level_block / per_channel) already treats all leading dims as
independent groups, so delegating to it yields genuine PER-EXPERT scales (each
expert gets its own block/global/channel scales). Placement is EXPERT: the scale
lives on the EP rank holding the expert; gradients reduce over the expert-DP group.

Note: when experts are separate per-expert modules (mcore SequentialMLP), each
module already owns its own quantizer, so this wrapper is used for FUSED 3D experts.
"""

from __future__ import annotations

from torch import Tensor

from angelslim.compressor.mcore_qad.parallel.parallel_spec import ParallelSpec
from angelslim.compressor.mcore_qad.quant.formats.base import QuantFormat
from angelslim.compressor.mcore_qad.quant.schemes.base import (
    SCHEME_REGISTRY,
    HostInfo,
    ScaleScheme,
)


@SCHEME_REGISTRY.register("per_expert")
class PerExpertScheme(ScaleScheme):
    def __init__(self, inner: ScaleScheme | None = None, **kw) -> None:
        super().__init__()
        if inner is None:
            raise ValueError("PerExpertScheme requires an inner scheme")
        self.inner = inner  # nn.Module submodule -> its scale params are tracked

    def quantize(self, x: Tensor, fmt: QuantFormat) -> Tensor:
        # x has a leading expert dim; the inner scheme treats leading dims as
        # independent groups, so this gives per-expert scales.
        return self.inner.quantize(x, fmt)

    def parallel_spec(self, host: HostInfo) -> ParallelSpec:
        return ParallelSpec.expert()
