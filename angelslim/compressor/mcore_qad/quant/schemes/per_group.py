"""Per-group scaling: one (single-level) scale per contiguous group along the last dim.

Generalizes per-channel (group_size=1). Single-level (unlike two_level_block);
used for INT4 group-128 weight-only etc. Block scale is plain (not E4M3-nested).
"""

from __future__ import annotations

from torch import Tensor

from angelslim.compressor.mcore_qad.parallel.parallel_spec import ParallelSpec
from angelslim.compressor.mcore_qad.quant.formats.base import QuantFormat
from angelslim.compressor.mcore_qad.quant.functional import grad_scale, lsq_grad_scale
from angelslim.compressor.mcore_qad.quant.schemes.base import (
    SCHEME_REGISTRY,
    HostInfo,
    ScaleScheme,
)
from angelslim.compressor.mcore_qad.quant.sources.base import build_store


@SCHEME_REGISTRY.register("per_group")
class PerGroupScheme(ScaleScheme):
    def __init__(
        self,
        group_size: int = 128,
        axis: int = -1,
        source: str = "dynamic",
        block_shape=None,
        **kw,
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.store = build_store(source, block_shape)

    def quantize(self, x: Tensor, fmt: QuantFormat) -> Tensor:
        g = self.group_size
        K = x.shape[-1]
        assert K % g == 0, f"last dim {K} not divisible by group size {g}"
        lead = list(x.shape[:-1])
        xb = x.reshape(*lead, K // g, g)
        ref = (xb.detach().abs().amax(dim=-1) / fmt.qmax()).clamp_min(1e-10)  # [..., n_groups]
        s = self.store(ref).unsqueeze(-1)
        if self.store.is_learnable():
            s = grad_scale(s, lsq_grad_scale(g, fmt.qmax()))
        xq = fmt.to_grid(xb / s) * s
        return xq.reshape_as(x)

    def parallel_spec(self, host: HostInfo) -> ParallelSpec:
        if host.parallel_mode == "row":
            return ParallelSpec.sharded(partition_dim=-1)
        return ParallelSpec.replicated(grad_reduce_groups=["tp"])
