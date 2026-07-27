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

"""QuantSpec: the declarative, user-facing description of one quant strategy.

A QuantSpec is just data (serializable to YAML). build_quantizer() turns it into
a live Quantizer by looking each axis up in its registry. Example (NVFP4 weight):

    QuantSpec(fmt="e2m1", scheme="two_level_block", group_size=16, axis=-1,
              block_scale_fmt="e4m3", source="learnable")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class QuantSpec:
    fmt: str  # FORMAT_REGISTRY key
    scheme: str  # SCHEME_REGISTRY key
    source: str  # SOURCE_REGISTRY key
    axis: Optional[int] = None  # quantized axis (channel/group/block)
    group_size: Optional[int] = None  # for per_group / block schemes
    block_scale_fmt: Optional[str] = None  # for two_level_block (e.g. "e4m3")
    per_expert: bool = False  # wrap in PerExpertScheme for MoE

    @staticmethod
    def identity() -> "QuantSpec":
        """Sentinel meaning 'keep this tensor in high precision'."""
        return QuantSpec(fmt="__identity__", scheme="__identity__", source="__identity__")

    def is_identity(self) -> bool:
        return self.fmt == "__identity__"
