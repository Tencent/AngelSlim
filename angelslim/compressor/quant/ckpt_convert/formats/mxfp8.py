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


import torch

from .mxfp4 import _MX

__all__ = ["MXFP8"]

_FP8 = torch.finfo(torch.float8_e4m3fn)


class MXFP8(_MX):
    """W8A8 MXFP8 (compressed-tensors ``mxfp8-quantized``): fp8 weight + uint8 exponent."""

    name = "mxfp8"
    bits = 8
    qmax = float(_FP8.max)
    cs_format = "mxfp8-quantized"
    code_dtype = torch.float8_e4m3fn

    def cs_scheme(self):
        args = self._args()
        return {
            "weights": {**args, "dynamic": False},
            "input_activations": {**args, "dynamic": True},
            "output_activations": None,
        }

    def _scale(self, amax):
        exponent = torch.ceil(torch.log2((amax / _FP8.max).clamp(min=2.0**-127)))
        u8 = (exponent + 127).clamp(0, 254).to(torch.uint8)
        return u8, torch.pow(2.0, u8.float() - 127.0)

    def _encode(self, x):
        return torch.clamp(x, min=_FP8.min, max=_FP8.max).to(torch.float8_e4m3fn)

    def _decode(self, code):
        return code.float()

    def _finish(self, codes):
        return codes.contiguous()

    def tensors(self, base, wkey, q, scale, gscale, act_amax):
        return {f"{base}.weight": q.cpu(), f"{base}.weight_scale": scale.cpu()}
