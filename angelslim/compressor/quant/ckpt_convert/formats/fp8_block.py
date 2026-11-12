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

import math

import torch

from ...modules.gptq.ckpt_gptq import gptq_sweep

__all__ = ["FP8Block"]

_FP8 = torch.finfo(torch.float8_e4m3fn)
FP8_MAX, FP8_MIN = float(_FP8.max), float(_FP8.min)


def _ue8m0(amax):
    """Power-of-two scale ``2^ceil(log2(amax / 448))``."""
    safe = torch.where(amax == 0, torch.ones_like(amax), amax)
    exponent = torch.ceil(torch.log2((safe / FP8_MAX).clamp(min=2.0**-127)))
    return torch.pow(2.0, (exponent + 127.0).clamp(0, 254) - 127.0)


def _search_scale(tiles, amax, alphas, scale_fmt):
    """Per-tile scale over ``alpha * amax / 448`` minimizing the tile's L2 round-trip error.
    ``tiles`` [.., bm, bn], ``amax`` [..]; ties go to the lowest alpha."""
    alphas_t = torch.as_tensor(sorted(alphas), dtype=torch.float32, device=tiles.device)
    view = [-1] + [1] * amax.dim()
    chunk = max(1, min(alphas_t.numel(), 32_000_000 // max(1, tiles.numel())))
    safe = torch.where(amax == 0, torch.ones_like(amax), amax) if scale_fmt == "ue8m0" else amax
    tb, amax0 = tiles.unsqueeze(0), safe.unsqueeze(0)
    best_err = best_scale = None
    for start in range(0, alphas_t.numel(), chunk):
        cand = amax0 * alphas_t[start : start + chunk].reshape(view)
        cand = _ue8m0(cand) if scale_fmt == "ue8m0" else (cand / FP8_MAX)
        cand = torch.where(cand == 0, torch.ones_like(cand), cand)
        eff = cand.unsqueeze(-1).unsqueeze(-1)
        deq = torch.clamp(tb / eff, min=FP8_MIN, max=FP8_MAX).to(torch.float8_e4m3fn).float() * eff
        err = ((deq - tb) ** 2).sum(dim=(-1, -2))
        cerr, cidx = err.min(dim=0)
        cscale = cand.gather(0, cidx.unsqueeze(0)).squeeze(0)
        if best_err is None:
            best_err, best_scale = cerr, cscale
        else:
            better = cerr < best_err
            best_err = torch.where(better, cerr, best_err)
            best_scale = torch.where(better, cscale, best_scale)
    return best_scale


def _tile_scale(tiles, scale_fmt, alphas):
    amax = tiles.abs().amax(dim=(-1, -2))
    if alphas:
        return _search_scale(tiles, amax, alphas, scale_fmt)
    if scale_fmt == "ue8m0":
        return _ue8m0(amax)
    return torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))


class FP8Block:
    """W8A8 FP8 with one scale per ``block_size`` tile and dynamic per-group activations.
    ``legacy`` keeps the ``quant_method: fp8`` layout (``<weight>_scale_inv``, bf16)."""

    name = "fp8_block"
    cs_format = "float-quantized"
    needs_act_stats = False
    has_global_scale = False

    def __init__(self, block_size=(128, 128), scale_fmt="float", legacy=False):
        if scale_fmt not in ("float", "ue8m0"):
            raise ValueError(f"fp8_block scale_fmt must be float or ue8m0, got {scale_fmt}")
        self.block_size = tuple(block_size)
        self.scale_fmt = scale_fmt
        self.legacy = legacy
        self.group = self.block_size[1]

    def cs_scheme(self):
        return {
            "weights": {
                "num_bits": 8,
                "type": "float",
                "strategy": "block",
                "block_structure": list(self.block_size),
                "symmetric": True,
                "dynamic": False,
                "observer": "minmax",
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "strategy": "group",
                "group_size": self.block_size[1],
                "symmetric": True,
                "dynamic": True,
                "observer": None,
            },
            "output_activations": None,
        }

    def global_scale(self, weights, device=None):
        return None

    def can_stack_rows(self, rows):
        return rows % self.block_size[0] == 0

    def rtn(self, weight, gscale=None, alphas=None):
        """Any leading dims; returns (fp8 weight, fp32 scale [.., rows/bm, cols/bn])."""
        bm, bn = self.block_size
        rows, cols = weight.shape[-2:]
        bm = rows if bm == -1 or bm > rows else bm
        bn = cols if bn == -1 or bn > cols else bn
        lead = tuple(weight.shape[:-2])
        param = weight
        if rows % bm:
            pad_shape = [*lead, bm - rows % bm, cols]
            pad = torch.zeros(pad_shape, dtype=param.dtype, device=param.device)
            param = torch.cat([param, pad], dim=-2)
        if cols % bn:
            pad_shape = [*lead, param.shape[-2], bn - cols % bn]
            pad = torch.zeros(pad_shape, dtype=param.dtype, device=param.device)
            param = torch.cat([param, pad], dim=-1)
        padded = param.shape
        gm, gn = math.ceil(rows / bm), math.ceil(cols / bn)
        tiles = param.float().reshape(-1, gm, bm, gn, bn).permute(0, 1, 3, 2, 4)
        if self.legacy and self.scale_fmt == "float" and not alphas:
            # tools/fp8_quant_blockwise.py arithmetic: multiply by 448 / amax.
            amax = tiles.abs().amax(dim=(-1, -2))
            inv = FP8_MAX / torch.where(amax == 0, torch.ones_like(amax), amax)
            q = torch.clamp(tiles * inv[..., None, None], min=FP8_MIN, max=FP8_MAX)
            scale = inv.reciprocal()
        else:
            scale = _tile_scale(tiles, self.scale_fmt, alphas)
            q = torch.clamp(tiles / scale[..., None, None], min=FP8_MIN, max=FP8_MAX)
        q = q.to(torch.float8_e4m3fn).permute(0, 1, 3, 2, 4).reshape(padded)[..., :rows, :cols]
        return q.contiguous(), scale.reshape(*lead, gm, gn).contiguous()

    def gptq(self, weight, hinv, gscale=None, alphas=None, static_groups=False):
        bm, bn = self.block_size
        rows = weight.shape[-2]
        W = weight.to(device=hinv.device, dtype=torch.float32)
        if rows % bm:
            W = torch.nn.functional.pad(W, (0, 0, 0, bm - rows % bm))
        nbm = W.shape[-2] // bm

        def group_scale(block):
            tiles = block.reshape(*block.shape[:-2], nbm, bm, bn)
            scale = _tile_scale(tiles, self.scale_fmt, alphas)
            eff = scale.clamp(min=1e-30).repeat_interleave(bm, dim=-1).unsqueeze(-1)
            return scale, eff

        def quantize(w, eff):
            code = torch.clamp(w / eff, min=FP8_MIN, max=FP8_MAX).to(torch.float8_e4m3fn)
            return code, code.float() * eff

        codes, scales = gptq_sweep(
            W, hinv, bn, group_scale, quantize, torch.float8_e4m3fn, static_groups
        )
        return codes[..., :rows, :].contiguous(), torch.stack(scales, dim=-1).contiguous()

    def row_slice(self, q, scale, r0, r1):
        bm = self.block_size[0]
        return q[r0:r1], scale[r0 // bm : r1 // bm]

    def tensors(self, base, wkey, q, scale, gscale, act_amax):
        if self.legacy:
            tensorwise = self.block_size == (-1, -1)
            suffix = "_scale" if tensorwise else "_scale_inv"
            return {wkey: q.cpu(), wkey + suffix: scale.squeeze().to(torch.bfloat16).cpu()}
        return {f"{base}.weight": q.cpu(), f"{base}.weight_scale": scale.float().cpu()}
