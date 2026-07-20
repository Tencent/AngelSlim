"""Per-token activation scaling: one (dynamic) scale per token, over the last dim.

The activation analogue of per_channel, but named for its role: for an activation
[*tokens, K] it reduces amax over K -> one scale per token. Dynamic only (the token
axis is runtime-sized, so the scale can never be a stored parameter), which is exactly
what vLLM's W4A8 / FP8 dynamic path expects (per-token, symmetric, runtime). No
trainable parameter -> no parallel sync needed.
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
from angelslim.compressor.mcore_qad.quant.sources.base import build_store


@SCHEME_REGISTRY.register("per_token")
class PerTokenScheme(ScaleScheme):
    def __init__(self, source: str = "dynamic", **kw) -> None:
        super().__init__()
        if source != "dynamic":
            raise ValueError("per_token is dynamic-only (the token axis is runtime-sized)")
        self.store = build_store("dynamic", None)

    def quantize(self, x: Tensor, fmt: QuantFormat) -> Tensor:
        ref = (x.detach().abs().amax(dim=-1, keepdim=True) / fmt.qmax()).clamp_min(1e-10)
        s = self.store(ref)
        return fmt.to_grid(x / s) * s

    def parallel_spec(self, host: HostInfo) -> ParallelSpec:
        return ParallelSpec.replicated(grad_reduce_groups=[])
