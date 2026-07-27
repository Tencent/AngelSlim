# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVFP4's two-level scaling: per-16 block E4M3 scale x per-tensor FP32 global.

Reconstruction for a block b:
    x_hat = (S_global * s_block_e4m3[b]) * q_e2m1
with:
    S_global    = tensor_amax / (qmax_e2m1 * qmax_e4m3)        # positions block
                  scales inside the E4M3 range (=> max block scale ~ qmax_e4m3)
    s_block_raw = (block_amax / qmax_e2m1) / S_global
    s_block_e4m3= E4M3_quantize(s_block_raw)                   # block scale IS E4M3
    eff_scale   = S_global * s_block_e4m3
    q_e2m1      = E2M1_to_grid(x / eff_scale)

Trainability -- `source` controls the per-BLOCK E4M3 scale (the real lever); the
global FP32 scale is always derived from data:
    * weight     : source="learnable" -> per-block E4M3 scale is a learned param
                   (weights frozen -> this is the main lever). It is E4M3-quantized
                   in the forward (faithful) and trained through straight-through.
    * activation : source="dynamic"   -> per-(token,block) scale from live amax.
    * calibrated : per-block buffer loaded externally.

Empirically (downstream/distill-style objective) a learnable per-block scale --
whether kept continuous or passed through E4M3 with STE -- reduces loss ~equally
(~6% in a linear probe), while a learnable GLOBAL scale is a no-op (E4M3 is
floating-point, so S * E4M3(c/S) is ~independent of S). Hence: block is learnable,
global is derived (its only job is to keep block scales inside the E4M3 range).
"""

from __future__ import annotations

from torch import Tensor

from angelslim.compressor.mcore_qad.parallel.parallel_spec import ParallelSpec
from angelslim.compressor.mcore_qad.quant.formats.base import (
    FORMAT_REGISTRY,
    QuantFormat,
)
from angelslim.compressor.mcore_qad.quant.formats.fp8 import E4M3Format
from angelslim.compressor.mcore_qad.quant.functional import grad_scale, lsq_grad_scale
from angelslim.compressor.mcore_qad.quant.schemes.base import (
    SCHEME_REGISTRY,
    HostInfo,
    ScaleScheme,
)
from angelslim.compressor.mcore_qad.quant.sources.base import build_store


@SCHEME_REGISTRY.register("two_level_block")
class TwoLevelBlockScheme(ScaleScheme):
    def __init__(
        self,
        group_size: int = 16,
        axis: int = -1,
        source: str = "dynamic",
        block_scale_fmt: str = "e4m3",
        block_shape=None,
        **kw,
    ) -> None:
        super().__init__()
        assert axis in (-1, None), "v1 quantizes along the last dim (K)"
        self.group_size = group_size
        self.block_fmt = (
            E4M3Format() if block_scale_fmt == "e4m3" else FORMAT_REGISTRY.create(block_scale_fmt)
        )
        # `source` controls the per-block scale (the lever); global is derived.
        self.block_store = build_store(source, block_shape)

    def quantize(self, x: Tensor, fmt: QuantFormat) -> Tensor:
        g = self.group_size
        K = x.shape[-1]
        assert K % g == 0, f"last dim {K} not divisible by block size {g}"
        lead = list(x.shape[:-1])
        xb = x.reshape(*lead, K // g, g)  # [..., nb, g]

        e2, e4 = fmt.max_repr, self.block_fmt.max_repr
        # data statistics are DETACHED: the scale is a constant (or a learnable param);
        # gradient to the input flows via the STE, not by differentiating amax/1/S
        # (differentiating through amax/division is unstable on real activations).
        xd = x.detach()
        S = (xd.abs().amax() / (e2 * e4)).clamp_min(1e-10)  # derived global (>0)

        block_amax = xd.reshape(*lead, K // g, g).abs().amax(dim=-1)  # [..., nb]
        block_ref = (block_amax / e2) / S  # reference for the store
        s_block = self.block_fmt.quantize_scale(self.block_store(block_ref))  # E4M3
        # floor at 1e-10 (not 1e-20): the backward of x/eff has a 1/eff**2 term;
        # 1e-20 -> 1e40 overflows fp32 (-> 0*inf = NaN on zero/underflowed blocks).
        eff = (S * s_block).unsqueeze(-1).clamp_min(1e-10)
        if self.block_store.is_learnable():
            eff = grad_scale(eff, lsq_grad_scale(g, e2))

        xq = fmt.to_grid(xb / eff) * eff
        return xq.reshape_as(x)

    def parallel_spec(self, host: HostInfo) -> ParallelSpec:
        # block scale shape = weight.shape[:-1] + (nb,); it always carries the dim
        # the weight is TP-sharded on, so the scale is SHARDED on both layouts:
        #   column: weight sharded on out (dim0) -> scale sharded on dim0
        #   row:    weight sharded on in  -> scale sharded on its last dim (nb)
        if host.parallel_mode == "column":
            return ParallelSpec.sharded(partition_dim=0)
        if host.parallel_mode == "row":
            return ParallelSpec.sharded(partition_dim=-1)
        return ParallelSpec.replicated(grad_reduce_groups=["tp"])
