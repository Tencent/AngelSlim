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

"""Learnable scale (LSQ-style). The framework's only trainable parameter type.

Parameterized as a LINEAR, bounded MULTIPLICATIVE correction around the data-derived
reference scale:
    scale = ref * clamp(alpha, lo, hi),   alpha learnable, init 1.0
i.e. "learn a multiplicative tweak on top of the min-max scale".

Why linear-multiplicative (not exp/log):
  * gradient is uniform: d scale/d alpha = ref (a constant), NOT amplified by the
    current scale as it would be with scale = exp(alpha) (where d scale/d alpha =
    scale). The exp form can blow up under SGD / large LR; the linear form is stable
    under both SGD and Adam. This matches the dominant LSQ practice (linear-space
    step size) rather than a log parameterization.
  * positivity + boundedness via clamp(alpha, lo, hi) keep the quantizer
    well-conditioned (optimal scales sit within ~2x of min-max).
  * alpha init 1.0 -> starts exactly at the min-max reference.
The LSQ gradient rescale (1/sqrt(N*Qp)) is applied by the scheme on top of this.
"""

from __future__ import annotations

import torch
from torch import Tensor

from angelslim.compressor.mcore_qad.quant.sources.base import (
    SOURCE_REGISTRY,
    ScaleStore,
)


@SOURCE_REGISTRY.register("learnable")
class LearnableScale(ScaleStore):
    def __init__(self, shape, lo: float = 0.25, hi: float = 4.0) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(shape))
        self.register_buffer("ref", torch.ones(shape))
        self.register_buffer("_initialized", torch.zeros((), dtype=torch.bool))
        self.lo, self.hi = lo, hi

    def forward(self, ref: Tensor) -> Tensor:
        if not bool(self._initialized):
            self.init_value(ref)
        return self.ref * self.alpha.clamp(self.lo, self.hi)

    def is_learnable(self) -> bool:
        return True

    @torch.no_grad()
    def init_value(self, value: Tensor) -> None:
        self.ref.copy_(value.clamp_min(1e-12).expand_as(self.ref))
        self.alpha.fill_(1.0)
        self._initialized.fill_(True)
