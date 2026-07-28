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

"""Calibrated (frozen) scale loaded from outside -- no gradient."""

from __future__ import annotations

import torch
from torch import Tensor

from angelslim.compressor.mcore_qad.quant.sources.base import (
    SOURCE_REGISTRY,
    ScaleStore,
)


@SOURCE_REGISTRY.register("calibrated")
class CalibratedScale(ScaleStore):
    def __init__(self, shape) -> None:
        super().__init__()
        self.register_buffer("scale", torch.ones(shape))

    def forward(self, ref: Tensor) -> Tensor:
        return self.scale

    def is_learnable(self) -> bool:
        return False

    @torch.no_grad()
    def init_value(self, value: Tensor) -> None:
        self.scale.copy_(value.expand_as(self.scale))
