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


"""Triton kernel: strided group GEMM with fused reshape for block-logit scoring.

This kernel computes a strided dot product between query and key blocks,
producing a downsampled attention-score matrix used to estimate per-block
importance in the Stem scoring pipeline.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def flat_group_gemm_fuse_reshape_kernel(
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
    is_causal: tl.constexpr,
):
    """Triton kernel: one tile of the strided Q·K^T group GEMM.

    Args:
        Q: Query tensor pointer, shape ``(B, H, L_q, D)``.
        K: Key tensor pointer, shape ``(B, H, L_kv, D)``.
        Out: Output tensor pointer, shape ``(B, H, L_q // stride, L_kv // stride)``.
        stride_qz: Stride of Q along the batch dimension.
        stride_qh: Stride of Q along the head dimension.
        stride_qn: Stride of Q along the sequence dimension.
        stride_kz: Stride of K along the batch dimension.
        stride_kh: Stride of K along the head dimension.
        stride_kn: Stride of K along the sequence dimension.
        stride_oz: Stride of Out along the batch dimension.
        stride_oh: Stride of Out along the head dimension.
        stride_on: Stride of Out along the row (downsampled query) dimension.
        chunk_start: Logical chunk start boundary (downsampled coords) for causal masking.
        chunk_end: Logical chunk end boundary (downsampled coords).
        H: Number of attention heads (compile-time constant).
        STRIDE: Striding factor for downsampling (compile-time constant).
        HEAD_DIM: Per-head hidden dimension (compile-time constant).
        BLOCK_M: Tile size along the query (M) axis (compile-time constant).
        BLOCK_N: Tile size along the key (N) axis (compile-time constant).
        is_causal: Whether to apply causal masking (compile-time constant).
    """
    block_m = tl.program_id(0).to(tl.int64)
    block_n = tl.program_id(1).to(tl.int64)
    batch_id = tl.program_id(2).to(tl.int64) // H
    head_id = tl.program_id(2).to(tl.int64) % H

    # Early exit for causal tiles that are entirely above the diagonal.
    if is_causal and chunk_start + (block_m + 1) * BLOCK_M <= block_n * BLOCK_N:
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

    # Accumulate strided dot products.
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for _iter in range(STRIDE):
        q = tl.load(Q_ptrs - _iter * stride_qn)
        k = tl.load(K_ptrs + _iter * stride_kn)
        acc += tl.dot(q, k)

    O_ptrs = (
        Out
        + batch_id * stride_oz
        + head_id * stride_oh
        + block_m * BLOCK_M * stride_on
        + block_n * BLOCK_N
    )
    O_ptrs = O_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_on + tl.arange(0, BLOCK_N)[None, :]
    tl.store(O_ptrs, acc.to(Out.type.element_ty))


def flat_group_gemm_fuse_reshape(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    stride: int,
    chunk_start: int,
    chunk_end: int,
    is_causal: bool = True,
) -> torch.Tensor:
    """Launch the strided group-GEMM Triton kernel.

    Args:
        query_states: Query tensor of shape ``(B, H, L_q, D)``.
        key_states: Key tensor of shape ``(B, H, L_kv, D)``.
        stride: Striding factor.
        chunk_start: Logical chunk start boundary (in downsampled coordinates)
            used for the causal early-exit check inside the kernel.
        chunk_end: Logical chunk end boundary (in downsampled coordinates).
        is_causal: Whether to apply causal masking.

    Returns:
        Downsampled score matrix of shape ``(B, H, L_q // stride, L_kv // stride)``.
    """
    B, H, Lq, D = query_states.shape
    Lkv = key_states.shape[2]

    assert key_states.shape[0] == B
    assert key_states.shape[1] == H
    assert key_states.shape[3] == D

    BLOCK_M = 128
    BLOCK_N = 128
    assert (
        Lq % (stride * BLOCK_M) == 0
    ), f"q_len ({Lq}) must be divisible by stride*BLOCK_M ({stride * BLOCK_M})"
    assert (
        Lkv % (stride * BLOCK_N) == 0
    ), f"kv_len ({Lkv}) must be divisible by stride*BLOCK_N ({stride * BLOCK_N})"

    output = torch.empty(
        (B, H, Lq // stride, Lkv // stride),
        dtype=query_states.dtype,
        device=query_states.device,
    )

    grid = (Lq // stride // BLOCK_M, Lkv // stride // BLOCK_N, B * H)
    flat_group_gemm_fuse_reshape_kernel[grid](
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
        H,
        stride,
        D,
        BLOCK_M,
        BLOCK_N,
        is_causal,
    )

    return output
