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
import os
import pickle

import torch
import triton
import triton.language as tl

try:
    from eval.check_env import get_env_name
except ImportError:

    def get_env_name():
        """Fallback: assume 'vlm' environment when eval module is unavailable."""
        return "vlm"


# Set to True to enable Triton autotuning (benchmarks kernel configs on your hardware).
# When False (default), pre-computed configs are loaded from the cache file.
# To run autotuning: set this to True in the source (or module attribute) before the first import.
USE_TRITON_AUTOTUNE: bool = False
BEST_EFF_CONFIGS_CACHE = {}
env_prefix = get_env_name()

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(
    _KERNEL_DIR, "cache", f"{env_prefix}_vecattention_kernels_best_eff_configs.pkl"
)


def load_best_eff_configs(acc_config, file_path=CONFIG_PATH):
    """Load the best efficiency kernel config for a given accuracy config tuple.

    Caches results in memory to avoid repeated disk reads.

    Args:
        acc_config: Tuple of (q_pooling_size, k_local_size, group_k_block, head_dim, causal).
        file_path: Path to the pickled config cache file.

    Returns:
        dict of kernel hyperparameters (BLOCK_SIZE_Q, num_warps, num_stages).
    """
    global BEST_EFF_CONFIGS_CACHE
    if acc_config in BEST_EFF_CONFIGS_CACHE:
        return BEST_EFF_CONFIGS_CACHE[acc_config]

    try:
        with open(file_path, "rb") as f:
            all_configs = pickle.load(f)
    except FileNotFoundError:
        all_configs = {}

    eff_config = None
    for k, v in all_configs.items():
        if acc_config == k[: len(acc_config)]:
            eff_config = v
            break
    if eff_config is None:
        # No cached config: fall back to safe defaults
        eff_config = {"BLOCK_SIZE_Q": 64, "num_warps": 4, "num_stages": 3}

    BEST_EFF_CONFIGS_CACHE[acc_config] = eff_config
    return eff_config


def fuse_qk_softmax_minp_wo_causal(
    avg_q_chunk,
    k,
    avg_q_chunk_offset,
    gap,
    causal=True,
    q_pooling_size=128,
    k_local_size=128,
    wo_initial=False,
    group_k_block=1,
):
    """Fused Triton kernel for Q·K^T scoring with MinP column selection.

    Computes block-level attention scores between pooled query vectors and
    full key vectors, then selects key columns whose score exceeds a
    per-row running maximum minus ``gap`` (i.e., MinP-style thresholding).

    Args:
        avg_q_chunk: Pooled query blocks, shape (B, H, q_num_blocks, head_dim).
        k: Full key tensor, shape (B, H, k_len, head_dim).
        avg_q_chunk_offset: Starting Q-block index in the full sequence (for chunked prefill).
        gap: Logit gap threshold (= -log(min_p)). Float or (B, H) Tensor for per-head thresholds.
        causal: Whether to apply causal masking.
        q_pooling_size: Number of tokens per query block.
        k_local_size: Key block size for column selection (BLOCK_SIZE_K in the kernel).
        wo_initial: If True, skip the initial (sink) k-block to avoid double-counting.
        group_k_block: Number of consecutive k-blocks merged into one Triton program.

    Returns:
        column_count: (B, H, q_num_blocks) int32 — number of selected columns per Q-block.
        column_index: (B, H, q_num_blocks, max_count) int32 — indices of selected key tokens.
    """
    batch_size, num_heads, q_num_blocks, head_dim = avg_q_chunk.shape
    assert avg_q_chunk.shape[1] == k.shape[1]
    k_len = k.shape[-2]
    sm_scale = 1.0 / math.sqrt(head_dim)
    BLOCK_SIZE_K = k_local_size

    if USE_TRITON_AUTOTUNE:
        BLOCK_SIZE_Q = 128
        kernel_autotune_args = {}
    else:
        kernel_autotune_args = load_best_eff_configs(
            (q_pooling_size, k_local_size, group_k_block, head_dim, causal)
        )
        BLOCK_SIZE_Q = kernel_autotune_args.get("BLOCK_SIZE_Q", 64)

    padded_q_num_blocks = math.ceil(q_num_blocks / BLOCK_SIZE_Q) * BLOCK_SIZE_Q
    avg_q_chunk = torch.cat(
        [
            avg_q_chunk,
            torch.zeros(
                (batch_size, num_heads, padded_q_num_blocks - q_num_blocks, head_dim),
                dtype=avg_q_chunk.dtype,
                device=avg_q_chunk.device,
            ),
        ],
        dim=2,
    )
    avg_q_chunk_real_length = q_num_blocks

    padded_k_len = math.ceil(k_len / BLOCK_SIZE_K) * BLOCK_SIZE_K
    k = torch.cat(
        [
            k,
            torch.full(
                (batch_size, num_heads, padded_k_len - k_len, head_dim),
                torch.nan,
                dtype=k.dtype,
                device=k.device,
            ),
        ],
        dim=2,
    )

    assert not causal or q_pooling_size % BLOCK_SIZE_K == 0

    column_count = torch.zeros(
        (batch_size, num_heads, padded_q_num_blocks), dtype=torch.int32, device=avg_q_chunk.device
    )
    column_index = torch.zeros(
        (batch_size, num_heads, padded_q_num_blocks, padded_k_len),
        dtype=torch.int32,
        device=avg_q_chunk.device,
    )

    def grid(META):
        return (
            padded_q_num_blocks // META["BLOCK_SIZE_Q"],
            math.ceil(padded_k_len / (k_local_size * group_k_block)),
            batch_size * num_heads,
        )

    if isinstance(gap, float):
        kernel_args = [
            avg_q_chunk,
            k,
            sm_scale,
            column_count,
            column_index,
            avg_q_chunk.stride(0),
            avg_q_chunk.stride(1),
            avg_q_chunk.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            column_count.stride(0),
            column_count.stride(1),
            column_index.stride(0),
            column_index.stride(1),
            column_index.stride(2),
            avg_q_chunk_offset,
            avg_q_chunk_real_length,
            k_len,
            num_heads,
            head_dim,
            gap,
            causal,
            q_pooling_size,
            k_local_size,
            group_k_block,
            wo_initial,
        ]
        _causal_fuse_qk_cutoff_wo_causal_kernel[grid](*kernel_args, **kernel_autotune_args)
    elif isinstance(gap, torch.Tensor):
        kernel_args = [
            avg_q_chunk,
            k,
            sm_scale,
            gap,
            column_count,
            column_index,
            avg_q_chunk.stride(0),
            avg_q_chunk.stride(1),
            avg_q_chunk.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            column_count.stride(0),
            column_count.stride(1),
            column_index.stride(0),
            column_index.stride(1),
            column_index.stride(2),
            avg_q_chunk_offset,
            avg_q_chunk_real_length,
            k_len,
            num_heads,
            head_dim,
            causal,
            q_pooling_size,
            k_local_size,
            group_k_block,
            wo_initial,
        ]
        _causal_fuse_qk_cutoff_wo_causal_perHead_kernel[grid](*kernel_args, **kernel_autotune_args)

    column_count = column_count[:, :, :q_num_blocks].contiguous()
    max_column_count = torch.max(column_count)
    column_index = column_index[
        :, :, :q_num_blocks, : max(min(k_len, max_column_count), 1)
    ].contiguous()
    return column_count, column_index


def clean_count(nargs):
    nargs["column_count"].fill_(0)


configs = [
    triton.Config({"BLOCK_SIZE_Q": BSQ}, num_warps=w, num_stages=s, pre_hook=clean_count)
    for BSQ in [16, 32, 64, 128]
    for s in [1, 2, 3]
    for w in [4, 8]
]

if USE_TRITON_AUTOTUNE:
    decorator = triton.autotune(
        configs=configs,
        key=["q_pooling_size", "k_local_size", "group_k_block", "HEAD_DIM", "CAUSAL"],
        cache_results=True,
    )
else:

    def decorator(func):
        return func


@decorator
@triton.jit
def _causal_fuse_qk_cutoff_wo_causal_kernel(
    q,
    k,
    sm_scale,
    column_count,
    column_index,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_ccb,
    stride_cch,
    stride_cib,
    stride_cih,
    stride_cin,
    q_chunk_offset,
    q_length,
    k_length,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    gap: tl.constexpr,
    CAUSAL: tl.constexpr,
    q_pooling_size: tl.constexpr,
    k_local_size: tl.constexpr,
    group_k_block: tl.constexpr,
    wo_init: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
):
    BLOCK_SIZE_K: tl.constexpr = k_local_size

    qblock_id = tl.program_id(0).to(tl.int64)
    kblock_id = tl.program_id(1).to(tl.int64) * group_k_block
    batch_id = tl.program_id(2).to(tl.int64) // NUM_HEADS
    head_id = tl.program_id(2).to(tl.int64) % NUM_HEADS

    k_offset_start = (
        kblock_id * BLOCK_SIZE_K if not wo_init else max(q_pooling_size, kblock_id * BLOCK_SIZE_K)
    )
    max_valid_offset = (
        k_length
        if not CAUSAL
        else min(
            k_length,
            (q_chunk_offset + (qblock_id + 1) * BLOCK_SIZE_Q) * q_pooling_size - q_pooling_size,
        )
    )
    k_offset_end = min(max_valid_offset, kblock_id * BLOCK_SIZE_K + k_local_size * group_k_block)

    if k_offset_end <= k_offset_start:
        return

    offset_at_causal = q_chunk_offset * q_pooling_size + qblock_id * BLOCK_SIZE_Q * q_pooling_size
    k_ptrs = k + batch_id * stride_kb + head_id * stride_kh + k_offset_start * stride_kn
    k_ptrs = (
        k_ptrs + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_kn + tl.arange(0, HEAD_DIM)[:, None]
    )

    q_ptrs = q + batch_id * stride_qb + head_id * stride_qh + qblock_id * BLOCK_SIZE_Q * stride_qn
    q_ptrs = (
        q_ptrs + tl.arange(0, BLOCK_SIZE_Q)[:, None] * stride_qn + tl.arange(0, HEAD_DIM)[None, :]
    )
    qblock = tl.load(q_ptrs)
    qblock = (qblock * sm_scale).to(qblock.type.element_ty)

    column_count_ptr = (
        column_count + batch_id * stride_ccb + head_id * stride_cch + qblock_id * BLOCK_SIZE_Q
    )
    column_count_ptr = column_count_ptr + tl.arange(0, BLOCK_SIZE_Q)[:, None]
    column_index_ptr = (
        column_index
        + batch_id * stride_cib.to(tl.int64)
        + head_id * stride_cih
        + qblock_id * BLOCK_SIZE_Q * stride_cin
    )
    column_index_ptr = (
        column_index_ptr
        + tl.arange(0, BLOCK_SIZE_Q)[:, None] * stride_cin
        + tl.arange(0, BLOCK_SIZE_K)[None, :]
    )

    qk_max = tl.zeros((BLOCK_SIZE_Q, 1), dtype=tl.float32) - float("inf")

    for k_offset in tl.range(k_offset_start, k_offset_end, BLOCK_SIZE_K):
        kblock = tl.load(k_ptrs)
        qk = tl.dot(qblock, kblock)
        qk_max = tl.maximum(qk_max, tl.max(qk, axis=1, keep_dims=True))
        qk_mask = (qk + gap) >= qk_max
        if CAUSAL:
            qk_mask = qk_mask & (
                tl.arange(0, BLOCK_SIZE_Q)[:, None]
                > ((k_offset - offset_at_causal) // q_pooling_size)
            )

        row_counts = tl.sum(qk_mask.to(tl.int32), axis=1, keep_dims=True)
        idx = tl.arange(0, BLOCK_SIZE_K)[None, :] + k_offset
        idx = tl.where(qk_mask, idx, k_length)
        idx = tl.sort(idx, dim=1, descending=False)

        # Requires triton > 3.4.0 for correct atomic_add broadcast behaviour.
        # See: https://github.com/triton-lang/triton/issues/7402
        column_index_offset = tl.atomic_add(column_count_ptr, row_counts, sem="relaxed")
        tl.store(column_index_ptr + column_index_offset, idx, mask=idx < k_length)

        k_ptrs = k_ptrs + BLOCK_SIZE_K * stride_kn


@decorator
@triton.jit
def _causal_fuse_qk_cutoff_wo_causal_perHead_kernel(
    q,
    k,
    sm_scale,
    gaps,
    column_count,
    column_index,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_ccb,
    stride_cch,
    stride_cib,
    stride_cih,
    stride_cin,
    q_chunk_offset,
    q_length,
    k_length,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    q_pooling_size: tl.constexpr,
    k_local_size: tl.constexpr,
    group_k_block: tl.constexpr,
    wo_init: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
):
    """Per-head variant: each head reads its own gap value from the ``gaps`` tensor."""
    BLOCK_SIZE_K: tl.constexpr = k_local_size

    qblock_id = tl.program_id(0).to(tl.int64)
    kblock_id = tl.program_id(1).to(tl.int64) * group_k_block
    batch_id = tl.program_id(2).to(tl.int64) // NUM_HEADS
    head_id = tl.program_id(2).to(tl.int64) % NUM_HEADS

    gap_ptr = gaps + batch_id * NUM_HEADS + head_id
    gap = tl.load(gap_ptr)

    k_offset_start = (
        kblock_id * BLOCK_SIZE_K if not wo_init else max(q_pooling_size, kblock_id * BLOCK_SIZE_K)
    )
    max_valid_offset = (
        k_length
        if not CAUSAL
        else min(
            k_length,
            (q_chunk_offset + (qblock_id + 1) * BLOCK_SIZE_Q) * q_pooling_size - q_pooling_size,
        )
    )
    k_offset_end = min(max_valid_offset, kblock_id * BLOCK_SIZE_K + k_local_size * group_k_block)

    if k_offset_end <= k_offset_start:
        return

    offset_at_causal = q_chunk_offset * q_pooling_size + qblock_id * BLOCK_SIZE_Q * q_pooling_size
    k_ptrs = k + batch_id * stride_kb + head_id * stride_kh + k_offset_start * stride_kn
    k_ptrs = (
        k_ptrs + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_kn + tl.arange(0, HEAD_DIM)[:, None]
    )

    q_ptrs = q + batch_id * stride_qb + head_id * stride_qh + qblock_id * BLOCK_SIZE_Q * stride_qn
    q_ptrs = (
        q_ptrs + tl.arange(0, BLOCK_SIZE_Q)[:, None] * stride_qn + tl.arange(0, HEAD_DIM)[None, :]
    )
    qblock = tl.load(q_ptrs)
    qblock = (qblock * sm_scale).to(qblock.type.element_ty)

    column_count_ptr = (
        column_count + batch_id * stride_ccb + head_id * stride_cch + qblock_id * BLOCK_SIZE_Q
    )
    column_count_ptr = column_count_ptr + tl.arange(0, BLOCK_SIZE_Q)[:, None]
    column_index_ptr = (
        column_index
        + batch_id * stride_cib.to(tl.int64)
        + head_id * stride_cih
        + qblock_id * BLOCK_SIZE_Q * stride_cin
    )
    column_index_ptr = (
        column_index_ptr
        + tl.arange(0, BLOCK_SIZE_Q)[:, None] * stride_cin
        + tl.arange(0, BLOCK_SIZE_K)[None, :]
    )

    qk_max = tl.zeros((BLOCK_SIZE_Q, 1), dtype=tl.float32) - float("inf")

    for k_offset in tl.range(k_offset_start, k_offset_end, BLOCK_SIZE_K):
        kblock = tl.load(k_ptrs)
        qk = tl.dot(qblock, kblock)
        qk_max = tl.maximum(qk_max, tl.max(qk, axis=1, keep_dims=True))
        qk_mask = (qk + gap) >= qk_max
        if CAUSAL:
            qk_mask = qk_mask & (
                tl.arange(0, BLOCK_SIZE_Q)[:, None]
                > ((k_offset - offset_at_causal) // q_pooling_size)
            )

        row_counts = tl.sum(qk_mask.to(tl.int32), axis=1, keep_dims=True)
        idx = tl.arange(0, BLOCK_SIZE_K)[None, :] + k_offset
        idx = tl.where(qk_mask, idx, k_length)
        idx = tl.sort(idx, dim=1, descending=False)

        # Requires triton > 3.4.0 for correct atomic_add broadcast behaviour.
        # See: https://github.com/triton-lang/triton/issues/7402
        column_index_offset = tl.atomic_add(column_count_ptr, row_counts, sem="relaxed")
        tl.store(column_index_ptr + column_index_offset, idx, mask=idx < k_length)

        k_ptrs = k_ptrs + BLOCK_SIZE_K * stride_kn


# ====================== Autotuning utilities =========================


def find_best_eff_config(
    q_pooling_size: int,
    k_local_size: int,
    group_k_block: int,
    seq_len: int = 64 * 1024,
    head_dim: int = 128,
    causal: bool = True,
):
    """Run Triton autotune for one (q_pooling_size, k_local_size, group_k_block) combination.

    Must be called with USE_TRITON_AUTOTUNE=1.

    Returns:
        The best Triton Config object, or None if not found in cache.
    """
    device = "cuda"
    dtype = torch.float16
    batch_size, num_heads = 1, 32
    gap = 1e6

    k_len = ((seq_len + q_pooling_size - 1) // q_pooling_size) * q_pooling_size
    q_num_blocks = k_len // q_pooling_size

    avg_q = torch.randn(batch_size, num_heads, q_num_blocks, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, k_len, head_dim, dtype=dtype, device=device)
    fuse_qk_softmax_minp_wo_causal(
        avg_q, k, 0, gap, causal, q_pooling_size, k_local_size, causal, group_k_block
    )
    torch.cuda.synchronize()
    key = (q_pooling_size, k_local_size, group_k_block, head_dim, causal)
    for k, v in _causal_fuse_qk_cutoff_wo_causal_kernel.cache.items():
        if key == k[: len(key)]:
            return v
    return None


def find_all_config(
    q_pooling_size_list=None,
    k_local_size_list=None,
    group_k_block_list=None,
    seq_len=64 * 1024,
    head_dim=128,
    causal=True,
    save=False,
    save_path=CONFIG_PATH,
    load_existing_results=True,
):
    """Sweep all (q_pooling_size, k_local_size, group_k_block) combinations.

    Find the best kernel config.

    Args:
        q_pooling_size_list: List of Q pooling sizes to sweep (default [64, 128]).
        k_local_size_list: List of K local sizes to sweep (default [16, 32, 64, 128]).
        group_k_block_list: List of group_k_block values to sweep (default [1, 2, 4, 8]).
        seq_len: Sequence length used for benchmarking.
        head_dim: Head dimension.
        causal: Whether to benchmark causal attention.
        save: If True, pickle results to ``save_path``.
        save_path: Output path for the pickled config cache.
        load_existing_results: If True, skip configurations already in the saved file.

    Returns:
        dict mapping config tuples to their best kernel hyperparameters.
    """
    if q_pooling_size_list is None:
        q_pooling_size_list = [64, 128]
    if k_local_size_list is None:
        k_local_size_list = [16, 32, 64, 128]
    if group_k_block_list is None:
        group_k_block_list = [1, 2, 4, 8]

    if load_existing_results and os.path.exists(save_path):
        with open(save_path, "rb") as f:
            best_configs = pickle.load(f)
        print(f"Loaded existing best configs from {save_path}")
    else:
        best_configs = {}

    for q_pooling_size in q_pooling_size_list:
        for k_local_size in k_local_size_list:
            if k_local_size >= q_pooling_size:
                # k_local_size == q_pooling_size causes register spilling
                continue
            for group_k_block in group_k_block_list:
                key = (q_pooling_size, k_local_size, group_k_block, head_dim, causal)
                if key in best_configs:
                    continue
                print(f"Testing config: {key}")
                best_eff_config = find_best_eff_config(
                    q_pooling_size, k_local_size, group_k_block, seq_len, head_dim, causal
                )
                if best_eff_config is None:
                    raise ValueError(f"No best config found for: {key}")
                best_eff_config = best_eff_config.__dict__
                best_configs[key] = {
                    **best_eff_config["kwargs"],
                    **{
                        k: v
                        for k, v in best_eff_config.items()
                        if k not in ["kwargs", "pre_hook", "ir_override"]
                    },
                }
                print(f"Best config for {key}: {best_configs[key]}")

    if save:
        with open(save_path, "wb") as f:
            pickle.dump(best_configs, f)
        print(f"Saved best configs to {save_path}")

    return best_configs


def autotune_main():
    """Entry point for generating the kernel efficiency config cache.

    Sets ``USE_TRITON_AUTOTUNE = True`` automatically so that the Triton autotune
    decorator is active for the benchmarking sweep.
    """
    global USE_TRITON_AUTOTUNE
    USE_TRITON_AUTOTUNE = True
    q_pooling_size_list = [64, 128]
    k_local_size_list = [16, 32, 64, 128]
    group_k_block_list = [1, 2, 4, 8, 16, 32, 64, 128]

    find_all_config(
        q_pooling_size_list,
        k_local_size_list,
        group_k_block_list,
        seq_len=64 * 1024,
        save=True,
        save_path=CONFIG_PATH,
        load_existing_results=False,
        causal=True if env_prefix == "vlm" else False,
    )


if __name__ == "__main__":
    autotune_main()


# ---------------------------------------------------------------------------
# Utility functions: pooling kernel and helpers
# ---------------------------------------------------------------------------


@triton.jit
def bnhd_pool_kernel(
    x_ptr,
    y_ptr,
    pool_type: tl.constexpr,
    batch_size,
    seq_len,
    num_heads,
    head_dim: tl.constexpr,
    stride_xb,
    stride_xn,
    stride_xh,
    stride_xd,
    stride_yb,
    stride_yn,
    stride_yh,
    stride_yd,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_h = tl.program_id(2)

    x_ptr = (
        x_ptr
        + pid_b * stride_xb
        + pid_n * BLOCK_SIZE_N * stride_xn
        + pid_h * BLOCK_SIZE_H * stride_xh
    )

    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)

    cur_block_size_n = min(seq_len - pid_n * BLOCK_SIZE_N, BLOCK_SIZE_N)

    x_mask = (
        (off_n < seq_len - pid_n * BLOCK_SIZE_N)[:, None, None]
        & (off_h < num_heads - pid_h * BLOCK_SIZE_H)[None, :, None]
        & (off_d < head_dim)[None, None, :]
    )
    x = tl.load(
        x_ptr
        + off_n[:, None, None] * stride_xn
        + off_h[None, :, None] * stride_xh
        + off_d[None, None, :] * stride_xd,
        mask=x_mask,
        other=0,
    )
    if pool_type == 0:
        y = tl.sum(x, axis=0) / cur_block_size_n
    elif pool_type == 1:
        y = tl.max(x, axis=0)
    elif pool_type == 2:
        y = tl.min(x, axis=0)
    elif pool_type == 3:
        y = tl.max(tl.abs(x), axis=0)
    elif pool_type == 4:
        y = tl.sum(x, axis=0)
    else:
        y = tl.sum(x, axis=0) / cur_block_size_n

    y_ptr = y_ptr + pid_b * stride_yb + pid_n * stride_yn + pid_h * BLOCK_SIZE_H * stride_yh
    y_mask = (off_h < num_heads - pid_h * BLOCK_SIZE_H)[:, None] & (off_d < head_dim)[None, :]
    tl.store(y_ptr + off_h[:, None] * stride_yh + off_d[None, :] * stride_yd, y, mask=y_mask)


def triton_bnhd_pool(x: torch.Tensor, kernel_size: int, pool_type: str = "avg"):
    b, n, h, d = x.shape
    assert d in {16, 32, 64, 128}
    assert kernel_size in {1, 16, 32, 64, 128, 256, 512}
    if kernel_size == 1:
        return x
    m = triton.cdiv(n, kernel_size)
    y = torch.zeros(b, m, h, d, device=x.device, dtype=x.dtype)

    if pool_type == "last":
        if n % kernel_size == 0:
            return x[:, kernel_size - 1 :: kernel_size, ...]
        else:
            return torch.cat((x[:, kernel_size - 1 :: kernel_size, ...], x[:, -1:, ...]), dim=1)

    block_size_h = triton.next_power_of_2(h)
    while kernel_size * block_size_h * d > 128 * 128 * 128:
        block_size_h = block_size_h // 2
    assert block_size_h != 0

    block_size_d = triton.next_power_of_2(d)
    pool_str_to_type = {"avg": 0, "max": 1, "min": 2, "maxabs": 3, "sum": 4}
    pool_type = pool_str_to_type[pool_type]

    def grid(META):
        return (
            b,
            triton.cdiv(n, META["BLOCK_SIZE_N"]),
            triton.cdiv(h, META["BLOCK_SIZE_H"]),
        )

    bnhd_pool_kernel[grid](
        x,
        y,
        pool_type,
        b,
        n,
        h,
        d,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        BLOCK_SIZE_N=kernel_size,
        BLOCK_SIZE_H=block_size_h,
        BLOCK_SIZE_D=block_size_d,
    )
    return y


def causal_mask_in_uneqal_block(seqlen, block_size_q, block_size_k, device):
    num_q_blocks = math.ceil(seqlen / block_size_q)
    num_k_blocks = math.ceil(seqlen / block_size_k)
    q_start = torch.arange(num_q_blocks, device=device) * block_size_q
    q_end = q_start + block_size_q - 1
    k_start = torch.arange(num_k_blocks, device=device) * block_size_k
    k_end = k_start + block_size_k - 1
    overlap = (q_start.unsqueeze(1) <= k_end.unsqueeze(0)) & (
        k_start.unsqueeze(0) <= q_end.unsqueeze(1)
    )
    assert overlap.any(dim=-1).all()
    last_true_idx = overlap.float().cumsum(dim=-1).argmax(dim=-1)
    col_idx = torch.arange(num_k_blocks, device=overlap.device).unsqueeze(0)
    mask = col_idx <= last_true_idx.unsqueeze(1)
    return overlap, mask


def average_vector(q, block_size, use_triton=True):
    """Average pool query vectors: (B, H, L, D) -> (B, H, num_blocks, D)."""
    batch_size, num_heads, seq_len, head_dim = q.shape
    dtype = q.dtype
    q = q.float()
    num_blocks = math.ceil(seq_len / block_size)
    if use_triton:
        q = q.transpose(1, 2)
        return triton_bnhd_pool(q, block_size).transpose(1, 2).to(dtype)
    else:
        pad_q = num_blocks * block_size - seq_len
        avg_q = (
            torch.nn.functional.pad(q, (0, 0, 0, pad_q), value=0)
            .view(batch_size, num_heads, num_blocks, block_size, head_dim)
            .mean(-2)
        )
        avg_q[:, :, -1, :] = avg_q[:, :, -1, :] * block_size / (block_size - pad_q)
        return avg_q.to(dtype)
