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

__all__ = ["INT4Group"]

QMIN, QMAX = -8.0, 7.0
PACK = 8  # int4 codes per int32 word


def _pack_int32(codes):
    """[.., n, k] int codes in [-8, 7] -> int32 [.., n, k/8]: stored as code + 8, lowest
    input index in the least significant nibble (compressed-tensors ``pack_to_int32``)."""
    nib = (codes.to(torch.int64) + 8) & 0xF
    nib = nib.reshape(*codes.shape[:-1], codes.shape[-1] // PACK, PACK)
    shifts = 4 * torch.arange(PACK, device=codes.device)
    return (nib << shifts).sum(dim=-1).to(torch.uint32).view(torch.int32).contiguous()


class INT4Group:
    """Symmetric INT4 weights, one bf16 scale (amax / 7) per ``group_size`` inputs; activations
    either unquantized (W4A16) or dynamic per-token FP8 (``act: fp8``, W4A8)."""

    name = "int4_group"
    cs_format = "pack-quantized"
    needs_act_stats = False
    has_global_scale = False

    def __init__(self, group_size=128, act="none"):
        if act not in ("none", "fp8"):
            raise ValueError(f"int4_group act must be none or fp8, got {act}")
        self.group = group_size
        self.act = act

    def cs_scheme(self):
        acts = None
        if self.act == "fp8":
            acts = {"num_bits": 8, "type": "float", "strategy": "token", "symmetric": True,
                    "dynamic": True}  # fmt: skip
        return {
            "weights": {"num_bits": 4, "type": "int", "strategy": "group",
                        "group_size": self.group, "symmetric": True,
                        "dynamic": False},  # fmt: skip
            "input_activations": acts,
            "output_activations": None,
        }

    def global_scale(self, weights, device=None):
        return None

    def can_stack_rows(self, rows):
        return True

    def _check(self, weight, alphas):
        if alphas:
            raise ValueError("int4_group does not support alpha_list")
        if weight.shape[-1] % self.group:
            raise ValueError(f"input dim {weight.shape[-1]} is not a multiple of {self.group}")

    @staticmethod
    def _scale(block):
        amax = block.abs().amax(dim=-1).clamp(min=1e-12)
        return (amax / QMAX).to(torch.bfloat16)

    def rtn(self, weight, gscale=None, alphas=None):
        self._check(weight, alphas)
        blocks = weight.float().reshape(*weight.shape[:-1], -1, self.group)
        scale = self._scale(blocks)
        codes = torch.round(blocks / scale.float().unsqueeze(-1)).clamp(QMIN, QMAX)
        return _pack_int32(codes.reshape(weight.shape)), scale.contiguous()

    def gptq(self, weight, hinv, gscale=None, alphas=None, static_groups=False):
        self._check(weight, alphas)

        def group_scale(block):
            scale = self._scale(block)
            return scale, scale.float().clamp(min=1e-12).unsqueeze(-1)

        def quantize(w, eff):
            code = torch.round(w / eff).clamp(QMIN, QMAX)
            return code.to(torch.int8), code * eff

        codes, scales = gptq_sweep(
            weight, hinv, self.group, group_scale, quantize, torch.int8, static_groups
        )
        return _pack_int32(codes), torch.stack(scales, dim=-1).contiguous()

    def row_slice(self, q, scale, r0, r1):
        return q[r0:r1], scale[r0:r1]

    def tensors(self, base, wkey, q, scale, gscale, act_amax):
        shape = torch.tensor([q.shape[-2], q.shape[-1] * PACK], dtype=torch.int64)
        return {f"{base}.weight_packed": q.cpu(), f"{base}.weight_scale": scale.cpu(),
                f"{base}.weight_shape": shape}  # fmt: skip
