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

"""Fused Triton fake-quant kernels (fwd dequant + analytic LSQ backward).

One file per numeric format; each exposes a single `triton_*` entry used by the
grouped MoE-expert weight quantizers in `quant.grouped_quant`. Add a format by
adding a file here and registering its module in `GROUPED_WEIGHT_QUANT`.
"""

from angelslim.compressor.mcore_qad.quant.kernels.triton_int4_group import (
    triton_int4_group,
)
from angelslim.compressor.mcore_qad.quant.kernels.triton_nvfp4 import triton_nvfp4

__all__ = ["triton_nvfp4", "triton_int4_group"]
