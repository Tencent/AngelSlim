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

"""CoSA block-selection proxy.

Strided ``Q @ K^T`` -> per-block mass / row-max flags -> keep the smallest
prefix whose cumulative mass covers ``topp`` (plus sink / diagonal /
row-max), emitted as priority-ordered block ids for the hpc kernel.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _flat_group_gemm_fuse_reshape_kernel(
    Q,
    K,
    Out,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_oz,
    stride_oh,
    stride_on,
    chunk_start,
    chunk_end,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_m = tl.program_id(0).to(tl.int64)
    block_n = tl.program_id(1).to(tl.int64)
    batch_id = tl.program_id(2).to(tl.int64) // H
    head_id = tl.program_id(2).to(tl.int64) % H

    if chunk_start + (block_m + 1) * BLOCK_M <= block_n * BLOCK_N:
        return

    Q_ptrs = (
        Q + batch_id * stride_qz + head_id * stride_qh + block_m * BLOCK_M * STRIDE * stride_qn
    )
    K_ptrs = (
        K + batch_id * stride_kz + head_id * stride_kh + block_n * BLOCK_N * STRIDE * stride_kn
    )
    Q_ptrs = (
        Q_ptrs
        + tl.arange(0, BLOCK_M)[:, None] * (stride_qn * STRIDE)
        + tl.arange(0, HEAD_DIM)[None, :]
        + stride_qn * (STRIDE - 1)
    )
    K_ptrs = (
        K_ptrs
        + tl.arange(0, BLOCK_N)[None, :] * (stride_kn * STRIDE)
        + tl.arange(0, HEAD_DIM)[:, None]
    )

    o = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for iter in range(STRIDE):
        q = tl.load(Q_ptrs - iter * stride_qn)
        k = tl.load(K_ptrs + iter * stride_kn)
        o += tl.dot(q, k)

    O_ptrs = (
        Out
        + batch_id * stride_oz
        + head_id * stride_oh
        + block_m * BLOCK_M * stride_on
        + block_n * BLOCK_N
    )
    O_ptrs = O_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_on + tl.arange(0, BLOCK_N)[None, :]
    tl.store(O_ptrs, o.to(Out.type.element_ty))


def flat_group_gemm_fuse_reshape(query_states, key_states, stride, chunk_start, chunk_end):
    batch_size, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[2]

    assert key_states.shape[0] == batch_size
    assert key_states.shape[1] == num_heads
    assert key_states.shape[3] == head_dim

    output = torch.empty(
        (batch_size, num_heads, q_len // stride, kv_len // stride),
        dtype=query_states.dtype,
        device=query_states.device,
    )
    BLOCK_M = 128
    BLOCK_N = 128
    assert q_len % (stride * BLOCK_M) == 0
    assert kv_len % (stride * BLOCK_N) == 0

    grid = (q_len // stride // BLOCK_M, kv_len // stride // BLOCK_N, batch_size * num_heads)
    _flat_group_gemm_fuse_reshape_kernel[grid](
        query_states,
        key_states,
        output,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        key_states.stride(0),
        key_states.stride(1),
        key_states.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        chunk_start,
        chunk_end,
        num_heads,
        stride,
        head_dim,
        BLOCK_M,
        BLOCK_N,
    )
    return output


@triton.jit
def _fused_cosa_proxy_kernel(
    In,
    Out_score,
    Out_rowmax,
    scale,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    score_stride_0,
    score_stride_1,
    score_stride_2,
    rowmax_stride_0,
    rowmax_stride_1,
    rowmax_stride_2,
    real_q_len,
    k_len,
    chunk_start,
    chunk_end,
    segment_size: tl.constexpr,
    block_size: tl.constexpr,
):
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    offs_q = tl.arange(0, block_size) + chunk_start + block_id * block_size
    offs_k = tl.arange(0, segment_size)

    num_iters = k_len // segment_size
    num_iters_before_causal = (chunk_start + (block_id + 1) * block_size - 1) // segment_size

    m_i = tl.zeros([block_size], dtype=tl.float32) - float("inf")

    input_ptr = (
        In
        + batch_id * input_stride_0
        + head_id * input_stride_1
        + block_id * block_size * input_stride_2
    )
    input_ptr = (
        input_ptr + tl.arange(0, segment_size) + tl.arange(0, block_size)[:, None] * input_stride_2
    )

    score_ptr = (
        Out_score
        + batch_id * score_stride_0
        + head_id * score_stride_1
        + block_id * score_stride_2
    )
    score_ptr = score_ptr + tl.arange(0, segment_size // block_size)

    rowmax_ptr = (
        Out_rowmax
        + batch_id * rowmax_stride_0
        + head_id * rowmax_stride_1
        + block_id * rowmax_stride_2
    )
    rowmax_ptr = rowmax_ptr + tl.arange(0, segment_size // block_size)

    for iter in range(0, num_iters_before_causal):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        m_i = tl.maximum(m_i, tl.max(X, 1))

    for iter in range(num_iters_before_causal, num_iters_before_causal + 1):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        mask = offs_q[:, None] >= (offs_k[None, :] + iter * segment_size)
        X = tl.where(mask, X, -1.0e6)
        m_i = tl.maximum(m_i, tl.max(X, 1))

    sum_mask = offs_q[:, None] < real_q_len

    for iter in range(0, num_iters_before_causal):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        blockscore = tl.max(X, 2)
        E = tl.exp2(blockscore - m_i[:, None])
        E = tl.where(sum_mask, E, 0)
        score = tl.sum(E, 0)
        hr = tl.max(tl.where(E >= (1.0 - 1e-3), 1.0, 0.0), 0)
        tl.store(
            score_ptr + iter * segment_size // block_size, score.to(Out_score.type.element_ty)
        )
        tl.store(rowmax_ptr + iter * segment_size // block_size, hr.to(Out_rowmax.type.element_ty))

    for iter in range(num_iters_before_causal, num_iters_before_causal + 1):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        mask = offs_q[:, None] >= (offs_k[None, :] + iter * segment_size)
        X = tl.where(mask, X, -1.0e6)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        blockscore = tl.max(X, 2)
        E = tl.exp2(blockscore - m_i[:, None])
        E = tl.where(sum_mask, E, 0)
        score = tl.sum(E, 0)
        hr = tl.max(tl.where(E >= (1.0 - 1e-3), 1.0, 0.0), 0)
        tl.store(
            score_ptr + iter * segment_size // block_size, score.to(Out_score.type.element_ty)
        )
        tl.store(rowmax_ptr + iter * segment_size // block_size, hr.to(Out_rowmax.type.element_ty))

    for iter in range(num_iters_before_causal + 1, num_iters):
        zeros = tl.zeros([segment_size // block_size], dtype=tl.float32)
        tl.store(
            score_ptr + iter * segment_size // block_size, zeros.to(Out_score.type.element_ty)
        )
        tl.store(
            rowmax_ptr + iter * segment_size // block_size, zeros.to(Out_rowmax.type.element_ty)
        )


def fused_cosa_proxy(
    attn_weights_slice,
    reshaped_block_size,
    segment_size,
    chunk_start,
    chunk_end,
    real_q_len,
    scale,
):
    batch_size, num_heads, q_len, k_len = attn_weights_slice.shape
    assert q_len % reshaped_block_size == 0
    assert k_len % segment_size == 0
    assert segment_size % reshaped_block_size == 0
    assert attn_weights_slice.stride(-1) == 1

    attn_score = torch.empty(
        (batch_size, num_heads, q_len // reshaped_block_size, k_len // reshaped_block_size),
        dtype=attn_weights_slice.dtype,
        device=attn_weights_slice.device,
    )
    has_rowmax = torch.empty(
        (batch_size, num_heads, q_len // reshaped_block_size, k_len // reshaped_block_size),
        dtype=torch.int8,
        device=attn_weights_slice.device,
    )
    grid = (q_len // reshaped_block_size, num_heads, batch_size)
    _fused_cosa_proxy_kernel[grid](
        attn_weights_slice,
        attn_score,
        has_rowmax,
        scale,
        attn_weights_slice.stride(0),
        attn_weights_slice.stride(1),
        attn_weights_slice.stride(2),
        attn_score.stride(0),
        attn_score.stride(1),
        attn_score.stride(2),
        has_rowmax.stride(0),
        has_rowmax.stride(1),
        has_rowmax.stride(2),
        real_q_len,
        k_len,
        chunk_start,
        chunk_end,
        segment_size,
        reshaped_block_size,
    )
    return attn_score, has_rowmax


def select_and_order_blocks_row_max(
    mass: torch.Tensor,
    has_rowmax: torch.Tensor,
    q_block_num: int,
    k_block_num: int,
    threshold: float,
) -> torch.Tensor:
    batch_size, head_num, chunk_num, block_num = mass.shape
    device = mass.device
    mass = mass.to(torch.float32)
    has_rowmax = has_rowmax.to(torch.bool)

    threshold = float(threshold)
    total_sum = mass.sum(dim=-1, keepdim=True)
    if threshold == 1.0:
        # inf budget: total_sum * 1.0 can drop the last block to float rounding.
        required_sum = torch.full_like(total_sum, float("inf"))
    else:
        required_sum = total_sum * threshold

    causal = torch.tril(
        torch.ones(chunk_num, block_num, dtype=torch.bool, device=device),
        diagonal=block_num - chunk_num,
    ).view(1, 1, chunk_num, block_num)

    mandatory = torch.zeros((1, 1, chunk_num, block_num), dtype=torch.bool, device=device)
    mandatory[:, :, :, 0] = True
    rows = torch.arange(chunk_num, device=device)
    diag_cols = rows + (block_num - chunk_num)
    mandatory[0, 0, rows, diag_cols] = True
    mandatory = mandatory & causal

    C = total_sum.amax() + 1.0
    priority = mass + C * has_rowmax.to(mass.dtype) + (2.0 * C) * mandatory.to(mass.dtype)
    neg_inf = torch.full_like(priority, float("-inf"))
    priority = torch.where(causal.expand_as(priority), priority, neg_inf)

    sort_perm = priority.argsort(dim=-1, descending=True)
    mass_in_order = torch.gather(mass, dim=-1, index=sort_perm)
    cum_before = torch.cat(
        [
            torch.zeros((batch_size, head_num, chunk_num, 1), device=device),
            mass_in_order[:, :, :, :-1],
        ],
        dim=-1,
    ).cumsum(dim=-1)
    keep_in_order = cum_before < required_sum

    mand_in_order = torch.gather(mandatory.expand_as(mass).to(torch.bool), dim=-1, index=sort_perm)
    causal_in_order = torch.gather(causal.expand_as(mass).to(torch.bool), dim=-1, index=sort_perm)
    kept = (keep_in_order | mand_in_order) & causal_in_order

    row_blockmask = sort_perm.to(torch.int32)
    row_blockmask = torch.where(kept, row_blockmask, torch.full_like(row_blockmask, -1))
    assert not (row_blockmask == -1).all(dim=-1).any(), "row_blockmask contains a row with all -1"
    return row_blockmask.contiguous()


def _select_chunk(
    attn_sum_chunk,
    has_rowmax_chunk,
    chunk_idx,
    npc,
    offset_u,
    q_block_num_u,
    k_block_num_u,
    topp,
):
    rows_lo = chunk_idx * npc
    rows_hi = min((chunk_idx + 1) * npc, q_block_num_u)
    n_rows = rows_hi - rows_lo
    kv_u = rows_hi + offset_u

    mass_chunk = attn_sum_chunk[:, :, :n_rows, :kv_u].contiguous()
    hr_chunk = has_rowmax_chunk[:, :, :n_rows, :kv_u].contiguous()
    sel = select_and_order_blocks_row_max(mass_chunk, hr_chunk, n_rows, kv_u, topp)
    if kv_u < k_block_num_u:
        sel = F.pad(sel, (0, k_block_num_u - kv_u), value=-1)
    return sel


def _validate_topp(topp) -> float:
    if isinstance(topp, bool) or not isinstance(topp, (int, float)):
        raise TypeError(f"topp must be a scalar float, got {type(topp).__name__}.")
    return float(topp)


def cosa_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size: int,
    stride: int,
    topp: float = 0.8,
    chunk_size: int = 16384,
):
    topp = _validate_topp(topp)

    _, _, k_len, head_dim = key_states.shape
    _, _, q_len, _ = query_states.shape

    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_block_num = (q_len + q_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    assert k_chunk_num >= q_chunk_num

    # Keep F.pad on the input device (do not force "cuda:0" under device_map).
    if k_num_to_pad > 0:
        pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0)
    else:
        pad_key_states = key_states
    if q_num_to_pad > 0:
        pad_query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0)
    else:
        pad_query_states = query_states

    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size

    q_block_num_u = (q_len + block_size - 1) // block_size
    k_block_num_u = (k_len + block_size - 1) // block_size
    offset_u = k_block_num_u - q_block_num_u
    npc = num_blocks_per_chunk

    scale = 1.4426950408889634 / math.sqrt(head_dim) / stride
    segment_size = min(4096, reshaped_block_size)
    real_q_len = k_reshaped_seq_len - k_reshaped_num_to_pad

    attn_sum_list = []
    row_blockmask_list = []

    for chunk_idx in range(q_chunk_num):
        chunk_start = (
            k_block_num - q_block_num
        ) * reshaped_block_size + chunk_idx * reshaped_chunk_size
        chunk_end = chunk_start + reshaped_chunk_size
        attn_weights_slice = flat_group_gemm_fuse_reshape(
            pad_query_states[
                :,
                :,
                (chunk_idx * reshaped_chunk_size)
                * stride : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size)
                * stride,
                :,
            ],
            pad_key_states,
            stride,
            chunk_start,
            chunk_end,
        )
        attn_sum, has_rowmax = fused_cosa_proxy(
            attn_weights_slice,
            reshaped_block_size,
            segment_size,
            chunk_start,
            chunk_end,
            real_q_len,
            scale,
        )
        del attn_weights_slice

        sel = _select_chunk(
            attn_sum,
            has_rowmax,
            chunk_idx,
            npc,
            offset_u,
            q_block_num_u,
            k_block_num_u,
            topp,
        )
        attn_sum_list.append(attn_sum)
        row_blockmask_list.append(sel)

    attn_sums = torch.cat(attn_sum_list, dim=-2)
    second = torch.cat(row_blockmask_list, dim=-2)
    return attn_sums, second
