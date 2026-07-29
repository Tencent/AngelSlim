# Vendored from MInference (https://github.com/microsoft/MInference),
# revision a4eb395, file minference/ops/pit_sparse_flash_attention_v2.py.
#
# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see angelslim/compressor/sparsity/algorithms/minference/kernels/NOTICE].  # noqa: E501
#
# Local modifications (AngelSlim):
#   * Rewrote ``from ..cuda import convert_vertical_slash_indexes`` to load the
#     CUDA index kernel via AngelSlim's JIT loader (``_cuda_ext.get_cuda_ext``).
#   * Dropped the optional sgl_kernel / vllm_flash_attn fast paths
#     (``sparse_attn_func`` / ``convert_vertical_slash_indexes_opt`` /
#     ``vertical_slash_sparse_attention_wo_pad``) — those packages are not a
#     dependency here, and upstream already falls back to the Triton path when
#     they are absent. Numerics of the retained Triton path are unchanged.

"""Vertical-and-slash sparse attention (MInference ``minference`` variant).

Triton mixed block/column-sparse FlashAttention. Consumes pre-estimated
per-head vertical-column (``v_idx``) and slash-diagonal (``s_idx``) indices,
converts them to a block/column layout via the vendored CUDA index kernel, then
runs the Triton kernel.
"""

import math

import torch
import triton
import triton.language as tl

from ._cuda_ext import get_cuda_ext


@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q,
    K,
    V,
    seqlens,
    sm_scale,
    block_count,
    block_offset,
    column_count,
    column_index,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    NUM_ROWS,
    NNZ_S,
    NNZ_V,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen

    for block_index in range(num_blks):
        start_n = tl.load(blks_ptr + block_index)
        cols = start_n + offs_n
        n_mask = cols < seqlen
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    for start_n in range(0, num_cols, BLOCK_N):
        n_mask = start_n + offs_n < num_cols
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & n_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def _triton_mixed_sparse_attention(
    q: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    seqlens: torch.Tensor,  # [BATCH, ]
    block_count: torch.Tensor,
    block_offset: torch.Tensor,
    column_count: torch.Tensor,
    column_index: torch.Tensor,
    sm_scale: float,
    block_size_M: int = 64,
    block_size_N: int = 64,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    _triton_mixed_sparse_attn_fwd_kernel[grid](
        q,
        k,
        v,
        seqlens,
        sm_scale,
        block_count,
        block_offset,
        column_count,
        column_index,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        q.shape[0],
        q.shape[1],
        q.shape[2],
        block_count.shape[-1],
        block_offset.shape[-1],
        column_index.shape[-1],
        BLOCK_M=block_size_M,
        BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4,
        num_stages=2,
    )

    return o


def vertical_slash_sparse_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    s_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    """Vertical-and-slash sparse attention.

    ``v_idx`` / ``s_idx`` are the pre-estimated top-k vertical-column and
    slash-diagonal indices (see the estimation in ``forward.py``).
    """
    cuda_ext = get_cuda_ext()
    if cuda_ext is None:
        raise RuntimeError(
            "MInference CUDA index kernel (convert_vertical_slash_indexes) is "
            "unavailable; cannot run the vertical_and_slash kernel."
        )

    batch_size, num_heads, context_size, head_dim = query.shape
    pad = (block_size_M - context_size) & (block_size_M - 1)
    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])

    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2 ** math.ceil(math.log2(head_dim)) - head_dim
        query = torch.nn.functional.pad(query, [0, target_dim, 0, 0, 0, 0, 0, 0])
        key = torch.nn.functional.pad(key, [0, target_dim, 0, 0, 0, 0, 0, 0])
        value = torch.nn.functional.pad(value, [0, target_dim, 0, 0, 0, 0, 0, 0])

    v_idx = (
        v_idx.to(torch.int32)
        .reshape((batch_size, num_heads, -1))
        .sort(dim=-1, descending=False)[0]
    )
    s_idx = (
        s_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]
    )
    # One seqlen per batch element. The Triton kernel reads ``seqlens[off_hz // H]``
    # for every (batch, head) program, so this MUST be length ``batch_size`` — a
    # length-1 tensor was a device-side out-of-bounds read for batch > 1. All keys
    # are equal length here (the framework rejects padded batches before prefill),
    # so every entry is ``context_size``.
    seqlens = torch.full((batch_size,), context_size, dtype=torch.int32, device=query.device)
    sm_scale = head_dim**-0.5
    block_count, block_offset, column_count, column_index = (
        cuda_ext.convert_vertical_slash_indexes(
            seqlens,
            v_idx,
            s_idx,
            context_size,
            block_size_M,
            block_size_N,
        )
    )

    out = _triton_mixed_sparse_attention(
        query,
        key,
        value,
        seqlens,
        block_count,
        block_offset,
        column_count,
        column_index,
        sm_scale,
        block_size_M,
        block_size_N,
    )

    return out[..., :context_size, :head_dim]
