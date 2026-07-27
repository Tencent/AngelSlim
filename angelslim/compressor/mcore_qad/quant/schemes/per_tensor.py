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

"""Per-tensor scaling: a single scalar scale for the whole tensor.

Parallel: per-tensor scale is REPLICATED and needs a TP grad all-reduce (each TP rank
holds only a shard / SP sequence-chunk). The DP/CP token dimensions are reduced
uniformly by train/parallel.grad_sync, so they are not listed here.
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


@SCHEME_REGISTRY.register("per_tensor")
class PerTensorScheme(ScaleScheme):
    def __init__(self, source: str = "dynamic", **kw) -> None:
        super().__init__()
        self.store = build_store(source, ())  # scalar

    def quantize(self, x: Tensor, fmt: QuantFormat) -> Tensor:
        ref = (x.detach().abs().amax() / fmt.qmax()).clamp_min(1e-10)
        s = self.store(ref)
        if self.store.is_learnable():
            s = grad_scale(s, lsq_grad_scale(x.numel(), fmt.qmax()))
        return fmt.to_grid(x / s) * s

    def parallel_spec(self, host: HostInfo) -> ParallelSpec:
        return ParallelSpec.replicated(grad_reduce_groups=["tp"])
