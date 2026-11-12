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

from ...modules.gptq.ckpt_gptq import gptq_sweep
from .nvfp4 import _cast_fp4, _decode_fp4, _pack

__all__ = ["MXFP4", "_MX"]

GROUP = 32


def _search_exponent(fmt, blocks, amax, alphas):
    """Per-block uint8 exponent over ``ceil(log2(alpha * amax / qmax))`` minimizing the block's
    L2 round-trip error. Only distinct ``log2(alpha)`` offsets matter; ties go to the lowest."""
    alphas = torch.as_tensor(alphas, dtype=torch.float32).clamp(min=2.0**-127)
    offs = torch.unique(torch.log2(alphas)).to(blocks.device)
    chunk = max(1, min(offs.numel(), 32_000_000 // max(1, blocks.numel())))
    bb = blocks.unsqueeze(0)
    safe = torch.where(amax == 0, torch.ones_like(amax), amax)
    base = torch.log2((safe / fmt.qmax).clamp(min=2.0**-127))
    view = [-1] + [1] * base.dim()
    best_err = best_u8 = None
    for start in range(0, offs.numel(), chunk):
        exponent = torch.ceil(base.unsqueeze(0) + offs[start : start + chunk].reshape(view))
        u8 = (exponent + 127).clamp(0, 254).to(torch.uint8)
        scale = torch.pow(2.0, u8.float() - 127.0).unsqueeze(-1)
        err = ((fmt._decode(fmt._encode(bb / scale)) * scale - bb) ** 2).sum(dim=-1)
        cerr, cidx = err.min(dim=0)
        cu8 = u8.gather(0, cidx.unsqueeze(0)).squeeze(0)
        if best_err is None:
            best_err, best_u8 = cerr, cu8
        else:
            better = cerr < best_err
            best_err = torch.where(better, cerr, best_err)
            best_u8 = torch.where(better, cu8, best_u8)
    return best_u8


def _blocks(weight):
    """[.., n, k] -> [.., n, k/32, 32] in fp32, zero-padding k to a multiple of 32."""
    k = weight.shape[-1]
    if k % GROUP:
        weight = torch.nn.functional.pad(weight, (0, GROUP - k % GROUP))
    return weight.float().reshape(*weight.shape[:-1], -1, GROUP)


class _MX:
    """OCP MX: power-of-two scale per 32 inputs, stored as a uint8 exponent (bias 127)."""

    group = GROUP
    needs_act_stats = False
    has_global_scale = False

    def __init__(self, group_size=GROUP):
        if group_size != GROUP:
            raise ValueError(f"{self.name} group_size must be {GROUP}, got {group_size}")

    def _args(self):
        return {"num_bits": self.bits, "type": "float", "strategy": "group",
                "group_size": GROUP, "symmetric": True, "scale_dtype": "torch.uint8"}  # fmt: skip

    def global_scale(self, weights, device=None):
        return None

    def can_stack_rows(self, rows):
        return True

    def _block_scale(self, blocks, alphas):
        """``blocks`` [.., g, 32] -> (uint8 exponent, fp32 scale) per block."""
        amax = blocks.abs().amax(dim=-1)
        if not alphas:
            return self._scale(amax)
        u8 = _search_exponent(self, blocks, amax, alphas)
        u8 = torch.where(amax == 0, self._scale(amax)[0], u8)  # all-zero blocks as in RTN
        return u8, torch.pow(2.0, u8.float() - 127.0)

    def rtn(self, weight, gscale=None, alphas=None):
        k = weight.shape[-1]
        blocks = _blocks(weight)
        u8, scale = self._block_scale(blocks, alphas)
        q = self._encode(blocks / scale.unsqueeze(-1))
        return self._finish(q.reshape(*q.shape[:-2], -1)[..., :k]), u8.contiguous()

    def gptq(self, weight, hinv, gscale=None, alphas=None, static_groups=False):
        def group_scale(block):
            u8, scale = self._block_scale(block.float().unsqueeze(-2), alphas)
            return u8.squeeze(-1), scale  # scale [.., n, 1] broadcasts over the group

        def quantize(w, eff):
            code = self._encode(w / eff)
            return code, self._decode(code) * eff

        codes, scales = gptq_sweep(
            weight, hinv, GROUP, group_scale, quantize, self.code_dtype, static_groups
        )
        return self._finish(codes), torch.stack(scales, dim=-1).contiguous()

    def row_slice(self, q, scale, r0, r1):
        return q[r0:r1], scale[r0:r1]


class MXFP4(_MX):
    """W4A16 MXFP4 (compressed-tensors ``mxfp4-pack-quantized``): packed E2M1 + uint8 exponent."""

    name = "mxfp4"
    bits = 4
    qmax = 6.0
    cs_format = "mxfp4-pack-quantized"
    code_dtype = torch.uint8

    def cs_scheme(self):
        return {"weights": {**self._args(), "dynamic": False}, "input_activations": None,
                "output_activations": None}  # fmt: skip

    def _scale(self, amax):
        scale = torch.pow(2.0, torch.ceil(torch.log2((amax / 6.0).clamp_min(1e-30))))
        scale = torch.where(amax == 0, torch.ones_like(scale), scale)
        return (torch.log2(scale) + 127).to(torch.uint8), scale

    def _encode(self, x):
        return _cast_fp4(x)

    def _decode(self, code):
        return _decode_fp4(code)

    def _finish(self, codes):
        if codes.shape[-1] % 2:
            raise ValueError("mxfp4 needs an even number of input channels")
        return _pack(codes)

    def tensors(self, base, wkey, q, scale, gscale, act_amax):
        return {f"{base}.weight_packed": q.cpu(), f"{base}.weight_scale": scale.cpu()}
