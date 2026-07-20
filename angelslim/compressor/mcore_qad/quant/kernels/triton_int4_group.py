"""Fused Triton INT4 per-group weight fake-quant (forward) + analytic LSQ backward.

The INT4 sibling of triton_nvfp4: one group (of `g` weights along the quant dim) per
program; load the group + its learnable alpha + per-group ref, write the dequantized
bf16 output directly (no fp32 stack materialized), and compute d/d_alpha in closed form
(the same LSQ/STE gradient as the eager `per_group` scheme). Single-level (plain group
scale -- no E4M3 nesting, no per-tensor global), symmetric INT4 grid (qmax = 7):

    s   = clamp(ref * clamp(alpha, LO, HI), 1e-12)
    Wq  = round(clamp(W/s, -7, 7)) * s          (STE)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_LO = tl.constexpr(0.25)
_HI = tl.constexpr(4.0)
_QMAX = tl.constexpr(7.0)  # int4 symmetric: 2**(4-1) - 1


@triton.jit
def _round_ne(u):
    """Round to nearest, ties away from zero (libdevice-free; matches torch.round off ties)."""
    return tl.where(u >= 0, tl.floor(u + 0.5), tl.ceil(u - 0.5))


@triton.jit
def _fwd_kernel(W, ALPHA, REF, OUT, n_blocks, G: tl.constexpr):
    p = tl.program_id(0)
    if p >= n_blocks:
        return
    a = tl.load(ALPHA + p)
    r = tl.load(REF + p)
    a = tl.minimum(tl.maximum(a, _LO), _HI)
    s = tl.maximum(r * a, 1e-12)
    off = p * G + tl.arange(0, G)
    w = tl.load(W + off).to(tl.float32)
    u = w / s
    q = _round_ne(tl.minimum(tl.maximum(u, -_QMAX), _QMAX))
    tl.store(OUT + off, (q * s).to(OUT.dtype.element_ty))


@triton.jit
def _bwd_kernel(GQ, W, ALPHA, REF, GALPHA, n_blocks, LSQ, G: tl.constexpr):
    p = tl.program_id(0)
    if p >= n_blocks:
        return
    a_raw = tl.load(ALPHA + p)
    r = tl.load(REF + p)
    a = tl.minimum(tl.maximum(a_raw, _LO), _HI)
    s_raw = r * a
    s = tl.maximum(s_raw, 1e-12)
    off = p * G + tl.arange(0, G)
    w = tl.load(W + off).to(tl.float32)
    gq = tl.load(GQ + off).to(tl.float32)
    u = w / s
    qhat = _round_ne(tl.minimum(tl.maximum(u, -_QMAX), _QMAX))
    dval = tl.where(tl.abs(u) <= _QMAX, qhat - u, qhat)  # d(Wq)/d(s) per element (STE)
    acc = tl.sum(gq * dval, axis=0)
    # grad survives only where neither the alpha-clamp nor the scale floor saturates.
    mask = (a_raw > _LO) & (a_raw < _HI) & (s_raw > 1e-12)
    d_alpha = tl.where(mask, acc * LSQ * r, 0.0)
    tl.store(GALPHA + p, d_alpha)


class _TritonInt4Group(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, alpha, ref, g, lsq):
        W = W.contiguous()
        ctx.save_for_backward(W, alpha, ref)
        ctx.g, ctx.lsq = g, lsq
        out = torch.empty_like(W)
        n_blocks = alpha.numel()
        _fwd_kernel[(n_blocks,)](W, alpha.reshape(-1), ref.reshape(-1), out, n_blocks, G=g)
        return out

    @staticmethod
    def backward(ctx, g_out):
        W, alpha, ref = ctx.saved_tensors
        g_alpha = torch.zeros_like(alpha)
        n_blocks = alpha.numel()
        _bwd_kernel[(n_blocks,)](
            g_out.contiguous(),
            W,
            alpha.reshape(-1),
            ref.reshape(-1),
            g_alpha.reshape(-1),
            n_blocks,
            ctx.lsq,
            G=ctx.g,
        )
        return None, g_alpha, None, None, None


def triton_int4_group(W, alpha, ref, g: int, lsq: float):
    """W [..., n_groups*g] bf16 -> INT4 per-group fake-quant bf16. alpha/ref [..., n_groups]."""
    return _TritonInt4Group.apply(W, alpha, ref, g, lsq)
