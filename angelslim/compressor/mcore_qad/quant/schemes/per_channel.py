"""Per-channel (per output-row) scaling: one scale per slice along the LAST dim.

Reduces over the last (contraction/in) dim, so the scale shape is x.shape[:-1]:
  * 2D weight [out, in]      -> scale [out]        (per output channel)
  * 3D expert [E, out, in]   -> scale [E, out]     (per expert, per output channel)
This matches the INT8/FP8 per-channel layout vLLM expects, for both dense linears
and fused MoE experts.

Parallel: if the kept rows align with the host TP shard dim -> SHARDED; else
REPLICATED + TP grad all-reduce.
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


@SCHEME_REGISTRY.register("per_channel")
class PerChannelScheme(ScaleScheme):
    def __init__(self, source: str = "dynamic", channel_shape=None, **kw) -> None:
        super().__init__()
        self.store = build_store(source, channel_shape)

    def quantize(self, x: Tensor, fmt: QuantFormat) -> Tensor:
        ref = (x.detach().abs().amax(dim=-1) / fmt.qmax()).clamp_min(1e-10)  # x.shape[:-1]
        s = self.store(ref)
        if self.store.is_learnable():
            s = grad_scale(s, lsq_grad_scale(x.shape[-1], fmt.qmax()))
        s = s.unsqueeze(-1)  # broadcast over last dim
        return fmt.to_grid(x / s) * s

    def parallel_spec(self, host: HostInfo) -> ParallelSpec:
        if host.shard_dim is not None and host.shard_dim == 0:
            return ParallelSpec.sharded(partition_dim=0)
        return ParallelSpec.replicated(grad_reduce_groups=["tp"])
