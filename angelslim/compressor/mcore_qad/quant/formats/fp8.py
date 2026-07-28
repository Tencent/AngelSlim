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

"""FP8 formats: E4M3 and E5M2 (implemented via torch native float8 round-trip).

E4M3 has a dual role:
  * an element format for FP8 W/A quantization, and
  * the *block-scale* dtype inside NVFP4 (the per-16 scale is stored E4M3).
The second use is why faithful NVFP4 simulation must quantize the block scale
THROUGH this format (a common "fake lossless" pitfall is keeping it FP32).
"""

from __future__ import annotations

import torch
from torch import Tensor

from angelslim.compressor.mcore_qad.quant.formats.base import (
    FORMAT_REGISTRY,
    QuantFormat,
)
from angelslim.compressor.mcore_qad.quant.functional import ste


def _fp8_roundtrip(x: Tensor, fp8_dtype: torch.dtype, max_repr: float) -> Tensor:
    """Clamp to range, cast to fp8 and back, straight-through gradient."""
    x_c = x.clamp(-max_repr, max_repr)
    q = x_c.to(fp8_dtype).to(x_c.dtype)
    return ste(q, x_c)


@FORMAT_REGISTRY.register("e4m3")
class E4M3Format(QuantFormat):
    max_repr = 448.0
    symmetric = True

    def to_grid(self, x_normalized: Tensor) -> Tensor:
        return _fp8_roundtrip(x_normalized, torch.float8_e4m3fn, self.max_repr)

    def quantize_scale(self, scale: Tensor) -> Tensor:
        """Quantize a positive scale tensor to the E4M3 grid (NVFP4 block scale).

        Positive-only; straight-through so a learnable global scale upstream still
        receives gradient through this op.
        """
        s = scale.clamp(min=1e-12, max=self.max_repr)
        q = s.to(torch.float8_e4m3fn).to(s.dtype)
        return ste(q, s)


@FORMAT_REGISTRY.register("e5m2")
class E5M2Format(QuantFormat):
    max_repr = 57344.0
    symmetric = True

    def to_grid(self, x_normalized: Tensor) -> Tensor:
        return _fp8_roundtrip(x_normalized, torch.float8_e5m2, self.max_repr)
