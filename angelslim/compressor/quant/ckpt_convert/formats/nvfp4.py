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

__all__ = ["NVFP4"]

_FP8 = torch.finfo(torch.float8_e4m3fn)
FP4_MAX = 6.0
GROUP = 16
# E2M1 rounding bounds; the odd ones (0.75, 1.75, 3.5) round half up.
_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_E2M1_ROUND_BOUNDS = torch.tensor([0.75, 1.75, 3.5])
_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _cast_fp4(x):
    """Scaled float -> uint8 E2M1 code (sign bit 0b1000 + 3 magnitude bits)."""
    device = x.device
    sign = (x < 0).to(torch.uint8)
    mag = x.abs()
    ord_ = torch.searchsorted(_E2M1_BOUNDS.to(device), mag, out_int32=True).to(torch.uint8)
    round_ = torch.isin(mag, _E2M1_ROUND_BOUNDS.to(device)).to(torch.uint8)
    return (sign * 0b1000 + ord_ + round_).to(torch.uint8)


def _decode_fp4(codes):
    mag = _E2M1_VALUES.to(codes.device)[(codes & 0b0111).long()]
    return torch.where((codes & 0b1000) > 0, -1.0, 1.0) * mag


def _pack(codes):
    return ((codes[..., 1::2] << 4) | codes[..., 0::2]).contiguous()


def _search_block_scale(blocks, amax, ws2, alphas):
    """Continuous per-block scale ``alpha * amax / 6`` minimizing the block's L2 round-trip
    error; alphas are evaluated in chunks of one batched kernel, ties go to the lowest."""
    alphas_t = torch.as_tensor(alphas, dtype=torch.float32, device=blocks.device)
    view = [-1] + [1] * amax.dim()
    chunk = max(1, min(alphas_t.numel(), 32_000_000 // max(1, blocks.numel())))
    bb, amax0 = blocks.unsqueeze(0), amax.unsqueeze(0)
    best_err = best_scale = None
    for start in range(0, alphas_t.numel(), chunk):
        scale = amax0 * alphas_t[start : start + chunk].reshape(view) / FP4_MAX
        q = scale / ws2
        q = torch.where(scale == 0, torch.ones_like(q), q).clamp(min=_FP8.min, max=_FP8.max)
        eff = (q.to(torch.float8_e4m3fn).to(torch.float32) * ws2).clamp(min=1e-12).unsqueeze(-1)
        err = ((_decode_fp4(_cast_fp4(bb / eff)) * eff - bb) ** 2).sum(dim=-1)
        cerr, cidx = err.min(dim=0)
        cscale = scale.gather(0, cidx.unsqueeze(0)).squeeze(0)
        if best_err is None:
            best_err, best_scale = cerr, cscale
        else:
            better = cerr < best_err
            best_err = torch.where(better, cerr, best_err)
            best_scale = torch.where(better, cscale, best_scale)
    return best_scale


def _safe(ws2):
    return torch.where(torch.isfinite(ws2) & (ws2 != 0), ws2, torch.full_like(ws2, 1.0e-6))


class NVFP4:
    """W4A4 NVFP4: E2M1 codes, fp8 scale per 16 inputs, fp32 global scale (compressed-tensors)."""

    name = "nvfp4"
    group = GROUP
    cs_format = "nvfp4-pack-quantized"
    needs_act_stats = True
    has_global_scale = True

    def __init__(self, group_size=GROUP):
        if group_size != GROUP:
            raise ValueError(f"nvfp4 group_size must be {GROUP}, got {group_size}")

    def cs_scheme(self):
        args = {"num_bits": 4, "type": "float", "strategy": "tensor_group",
                "group_size": GROUP, "symmetric": True}  # fmt: skip
        return {
            "weights": {**args, "dynamic": False},
            "input_activations": {**args, "dynamic": "local"},
            "output_activations": None,
        }

    def global_scale(self, weights, device=None):
        """Per-tensor scale ``amax / (6 * 448)`` shared by ``weights`` (fused gate/up)."""
        amax = torch.stack([w.to(device).float().abs().max() for w in weights]).max()
        return _safe((amax / FP4_MAX / _FP8.max).to(torch.float32))

    def can_stack_rows(self, rows):
        return True

    def rtn(self, weight, ws2, alphas=None):
        n, k = weight.shape
        w = weight.detach().to(torch.bfloat16)
        ws2 = _safe(ws2.to(torch.float32).to(w.device))
        blocks = w.view(n, k // GROUP, GROUP)
        amax = blocks.abs().amax(dim=-1).float()
        if alphas:
            scale = _search_block_scale(blocks.float(), amax, ws2, alphas)
        else:
            scale = amax / FP4_MAX
        q = scale / ws2
        q[scale == 0] = 1.0
        wsf = q.clamp(min=_FP8.min, max=_FP8.max).to(torch.float8_e4m3fn)
        eff = (wsf.to(torch.float32) * ws2).unsqueeze(-1)
        return _pack(_cast_fp4((blocks / eff).view(n, k))), wsf.contiguous()

    def gptq(self, weight, hinv, ws2, alphas=None, static_groups=False):
        """``weight`` [n, k] or [B, n, k]; ``ws2`` scalar, per matrix [B] or per row [.., n]."""
        lead, n = tuple(weight.shape[:-2]), weight.shape[-2]
        ws2 = _safe(ws2.to(torch.float32).to(hinv.device))
        if ws2.dim() == 0 or tuple(ws2.shape) == lead + (n,):
            ws2_row = ws2
        elif tuple(ws2.shape) == lead:
            ws2_row = ws2.unsqueeze(-1)
        else:
            raise ValueError(f"global scale {tuple(ws2.shape)} does not fit {tuple(weight.shape)}")
        ws2_srch = ws2_row if ws2_row.dim() == 0 else ws2_row.unsqueeze(-1)

        def group_scale(block):
            amax = block.abs().amax(dim=-1).float()
            if alphas:
                scale = _search_block_scale(
                    block.unsqueeze(-2), amax.unsqueeze(-1), ws2_srch, alphas
                ).squeeze(-1)
            else:
                scale = amax / FP4_MAX
            q = scale / ws2_row
            q[scale == 0] = 1.0
            wsf = q.clamp(min=_FP8.min, max=_FP8.max).to(torch.float8_e4m3fn)
            return wsf, (wsf.to(torch.float32) * ws2_row).clamp(min=1e-12).unsqueeze(-1)

        def quantize(w, eff):
            code = _cast_fp4(w / eff)
            return code, _decode_fp4(code) * eff

        codes, scales = gptq_sweep(
            weight, hinv, GROUP, group_scale, quantize, torch.uint8, static_groups
        )
        wsf = torch.stack([s.view(torch.uint8) for s in scales], dim=-1).view(torch.float8_e4m3fn)
        return _pack(codes), wsf.contiguous()

    def row_slice(self, q, scale, r0, r1):
        return q[r0:r1], scale[r0:r1]

    def tensors(self, base, wkey, q, scale, ws2, act_amax):
        input_scale = torch.tensor(act_amax / (FP4_MAX * _FP8.max), dtype=torch.float32)
        return {
            f"{base}.weight_packed": q.cpu(),
            f"{base}.weight_scale": scale.cpu(),
            f"{base}.weight_global_scale": ws2.detach().float().reciprocal().reshape(1).cpu(),
            f"{base}.input_global_scale": input_scale.reciprocal().reshape(1),
        }
