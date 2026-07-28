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

"""Fused Triton NVFP4 weight fake-quant (forward) + analytic LSQ backward.

One block (group of 16 along the quant dim) per program: load 16 weights + the block's
learnable alpha + per-block ref + per-expert global S, produce the dequantized bf16
output directly (no fp32 stack materialization). Backward computes d/d_alpha in closed
form (the same LSQ/STE gradient as the eager `two_level_block`), again per block, with
no autograd graph -- so peak memory is ~the output tensor only.

Math (matches schemes/two_level_block, generalized to a leading expert dim):
    s    = E4M3( clamp(ref*clamp(alpha,LO,HI), 1e-12, 448) )
    eff  = max(S_e * s, 1e-10)
    Wq   = E2M1_snap(W/eff) * eff
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_LO = tl.constexpr(0.25)
_HI = tl.constexpr(4.0)
_E2M1_MAX = tl.constexpr(6.0)
_E4M3_MAX = tl.constexpr(448.0)


@triton.jit
def _snap_e2m1(u):
    """sign(u) * nearest E2M1 level of |clamp(u,6)| (midpoint thresholds)."""
    uc = tl.minimum(tl.maximum(u, -_E2M1_MAX), _E2M1_MAX)
    au = tl.abs(uc)
    q = tl.where(
        au <= 0.25,
        0.0,
        tl.where(
            au < 0.75,
            0.5,
            tl.where(
                au <= 1.25,
                1.0,
                tl.where(
                    au < 1.75,
                    1.5,
                    tl.where(
                        au <= 2.5,
                        2.0,
                        tl.where(au < 3.5, 3.0, tl.where(au <= 5.0, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return tl.where(u < 0, -q, q)


@triton.jit
def _fwd_kernel(W, ALPHA, REF, S, OUT, NPE, n_blocks, G: tl.constexpr):
    p = tl.program_id(0)
    if p >= n_blocks:
        return
    e = p // NPE
    s_e = tl.load(S + e)
    a = tl.load(ALPHA + p)
    r = tl.load(REF + p)
    a = tl.minimum(tl.maximum(a, _LO), _HI)
    s_in = tl.minimum(tl.maximum(r * a, 1e-12), _E4M3_MAX)
    s_e4 = s_in.to(tl.float8e4nv).to(tl.float32)
    eff = tl.maximum(s_e * s_e4, 1e-10)
    off = p * G + tl.arange(0, G)
    w = tl.load(W + off).to(tl.float32)
    out = _snap_e2m1(w / eff) * eff
    tl.store(OUT + off, out.to(OUT.dtype.element_ty))


@triton.jit
def _bwd_kernel(GQ, W, ALPHA, REF, S, GALPHA, NPE, n_blocks, LSQ, G: tl.constexpr):
    p = tl.program_id(0)
    if p >= n_blocks:
        return
    e = p // NPE
    s_e = tl.load(S + e)
    a_raw = tl.load(ALPHA + p)
    r = tl.load(REF + p)
    a = tl.minimum(tl.maximum(a_raw, _LO), _HI)
    sc_raw = r * a  # pre-clamp e4m3 scale input
    s_in = tl.minimum(tl.maximum(sc_raw, 1e-12), _E4M3_MAX)
    s_e4 = s_in.to(tl.float8e4nv).to(tl.float32)
    eff_raw = s_e * s_e4
    eff = tl.maximum(eff_raw, 1e-10)
    off = p * G + tl.arange(0, G)
    w = tl.load(W + off).to(tl.float32)
    gq = tl.load(GQ + off).to(tl.float32)
    u = w / eff
    qhat = _snap_e2m1(u)
    dval = tl.where(tl.abs(u) <= _E2M1_MAX, qhat - u, qhat)  # d(Wq)/d(eff) per element (STE)
    acc = tl.sum(gq * dval, axis=0)
    # Grad survives only where alpha, E4M3 scale, and the effective floor do not clamp.
    mask = (
        (a_raw > _LO) & (a_raw < _HI) & (sc_raw > 1e-12) & (sc_raw < _E4M3_MAX) & (eff_raw > 1e-10)
    )
    d_alpha = tl.where(mask, acc * LSQ * s_e * r, 0.0)
    tl.store(GALPHA + p, d_alpha)


class _TritonNVFP4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, alpha, ref, S, g, lsq, npe):
        W = W.contiguous()
        ctx.save_for_backward(W, alpha, ref, S)
        ctx.g, ctx.lsq, ctx.npe = g, lsq, npe
        out = torch.empty_like(W)
        n_blocks = alpha.numel()
        _fwd_kernel[(n_blocks,)](
            W, alpha.reshape(-1), ref.reshape(-1), S.reshape(-1), out, npe, n_blocks, G=g
        )
        return out

    @staticmethod
    def backward(ctx, g_out):
        W, alpha, ref, S = ctx.saved_tensors
        g_alpha = torch.zeros_like(alpha)
        n_blocks = alpha.numel()
        _bwd_kernel[(n_blocks,)](
            g_out.contiguous(),
            W,
            alpha.reshape(-1),
            ref.reshape(-1),
            S.reshape(-1),
            g_alpha.reshape(-1),
            ctx.npe,
            n_blocks,
            ctx.lsq,
            G=ctx.g,
        )
        return None, g_alpha, None, None, None, None, None


def triton_nvfp4(W, alpha, ref, S, g: int, lsq: float):
    """W [E,OUT,IN] bf16 -> NVFP4 fake-quant bf16. alpha/ref [E,OUT,IN//g], S [E,1,1]."""
    npe = alpha.shape[1] * alpha.shape[2]  # OUT * nb (blocks per expert)
    return _TritonNVFP4.apply(W, alpha, ref, S, g, lsq, npe)
