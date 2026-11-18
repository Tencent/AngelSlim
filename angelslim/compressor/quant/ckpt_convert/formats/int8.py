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

__all__ = ["INT8Channel"]

QMAX = 127.0


class INT8Channel:
    """W8A8 INT8: one scale per output channel, dynamic per-token activations."""

    name = "int8_channel"
    group = 128  # GPTQ sweep width; the scale itself is per channel
    cs_format = "int-quantized"
    needs_act_stats = False
    has_global_scale = False

    def cs_scheme(self):
        args = {"num_bits": 8, "type": "int", "symmetric": True}
        return {
            "weights": {**args, "strategy": "channel", "dynamic": False},
            "input_activations": {**args, "strategy": "token", "dynamic": True},
            "output_activations": None,
        }

    def global_scale(self, weights, device=None):
        return None

    def can_stack_rows(self, rows):
        return True

    @staticmethod
    def _scale(weight):
        # bf16 arithmetic, as in the original DeepSeek / Meituan channel-int8 cast.
        scale = weight.to(torch.bfloat16).abs().amax(dim=-1, keepdim=True) / QMAX
        return torch.where(scale == 0, torch.ones_like(scale), scale)

    def rtn(self, weight, gscale=None, alphas=None):
        if alphas:
            raise ValueError("int8_channel does not support alpha_list")
        scale = self._scale(weight)
        q = torch.round(weight.to(torch.bfloat16) / scale).clamp(-QMAX, QMAX).to(torch.int8)
        return q.contiguous(), scale.float().contiguous()

    def gptq(self, weight, hinv, gscale=None, alphas=None, static_groups=False):
        if alphas:
            raise ValueError("int8_channel does not support alpha_list")
        scale = self._scale(weight).float().to(hinv.device)

        def group_scale(block):
            return scale, scale

        def quantize(w, eff):
            code = torch.round(w / eff).clamp(-QMAX, QMAX)
            return code.to(torch.int8), code * eff

        codes, _ = gptq_sweep(
            weight, hinv, self.group, group_scale, quantize, torch.int8, static_groups
        )
        return codes.contiguous(), scale.contiguous()

    def row_slice(self, q, scale, r0, r1):
        return q[r0:r1], scale[r0:r1]

    def tensors(self, base, wkey, q, scale, gscale, act_amax):
        return {f"{base}.weight": q.cpu(), f"{base}.weight_scale": scale.cpu()}
