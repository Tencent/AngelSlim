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

"""Dynamic scale: just use the data-derived reference, recomputed every forward.

Used for activation per-(token,block) scales whose layout spans the runtime token
dimension and therefore cannot be a learnable parameter.
"""

from __future__ import annotations

from torch import Tensor

from angelslim.compressor.mcore_qad.quant.sources.base import (
    SOURCE_REGISTRY,
    ScaleStore,
)


@SOURCE_REGISTRY.register("dynamic")
class DynamicScale(ScaleStore):
    def forward(self, ref: Tensor) -> Tensor:
        return ref

    def is_learnable(self) -> bool:
        return False
