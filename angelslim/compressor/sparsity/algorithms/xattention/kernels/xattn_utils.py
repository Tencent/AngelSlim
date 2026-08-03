# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see NOTICE for details]
#
# Vendored into AngelSlim from MInference (https://github.com/microsoft/MInference)
# at revision a4eb395: minference/ops/op_utils/xattn_utils.py. MIT, (c) Microsoft.
# Vendored whole, then reformatted to this repo's style (black/isort, line
# length 99) with logic unchanged, so the two Triton helpers used by
# `xattention.py` (`flat_group_gemm_fuse_reshape`, `softmax_fuse_block_sum`)
# keep their exact upstream behavior. The distributed (`torch.distributed`)
# helpers further down are not on the `Xattention_prefill` path and are retained
# only for provenance fidelity.

import torch
import torch.distributed as dist
import triton
import triton.language as tl

LN2 = 1 / 1.4426950408889634


def create_causal_mask(batch_size, head_num, block_size, block_num, divide_block_num):
    """
    Creates a causal attention mask used in transformer-based models.

    Parameters:
    - batch_size (int): The number of sequences in the batch.
    - head_num (int): The number of attention heads.
    - block_size (int): The size of each block in the sequence.
    - block_num (int): The total number of blocks in the sequence.
    - divide_block_num (int): The block index at which causality is applied.

    Returns:
    - torch.Tensor: A mask tensor of shape (batch_size, head_num, block_size, total_size)
    where total_size = block_size * block_num. The mask enforces causal attention by
    setting certain positions to `-inf` to prevent information leakage from future tokens.
    """
    divide_block_num += 1
    if divide_block_num < 1 or divide_block_num > block_num:
        raise ValueError(
            f"divide_block_num ({divide_block_num}) must be between 1 and block_num ({block_num})."
        )

    total_size = block_size * block_num
    device = "cuda"
    mask = torch.zeros(block_size, total_size, device=device)
    if divide_block_num < block_num:
        mask[:, divide_block_num * block_size :] = float("-inf")

    if divide_block_num - 1 < block_num:
        start_col = (divide_block_num - 1) * block_size
        end_col = start_col + block_size
        upper_tri_mask = torch.triu(
            torch.full((block_size, block_size), float("-inf"), device=device),
            diagonal=1,
        )
        mask[:, start_col:end_col] = upper_tri_mask

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.expand(batch_size, head_num, block_size, total_size)
    return mask


def find_blocks_chunked(
    input_tensor: torch.Tensor,  # (batch_size, num_heads, num_block_q, num_block_k)
    current_index,  #
    threshold,
    num_to_choose,
    decoding: bool,
    mode: str = "both",
    causal=True,
):
    """
    Finds and selects relevant blocks of attention for transformer-based models based on a
    threshold or a predefined number of blocks.

    Parameters:
    - input_tensor (torch.Tensor): The input tensor of shape (batch_size, head_num, num_block_q, num_block_k).  # noqa: E501
    - current_index (int): The current index in the sequence processing.
    - threshold (float or None): A threshold value used to determine the minimum attention weight sum.  # noqa: E501
    - num_to_choose (int or None): The number of blocks to be selected, ensuring sufficient information retrieval.  # noqa: E501
    - decoding (bool): If True, operates in decoding mode; otherwise, it's in encoding mode.
    - mode (str): Defines the processing mode, either 'both', 'prefill', or 'decode'.
    - causal (bool): If True, applies causal masking to prevent future information leakage.

    Returns:
    - torch.Tensor: A boolean mask of shape (batch_size, head_num, num_block_q, num_block_k),
    indicating which blocks should be attended to.
    """
    assert threshold is None or num_to_choose is None
    batch_size, head_num, num_block_q, num_block_k = input_tensor.shape
    input_tensor = input_tensor.to(float)

    total_sum = input_tensor.sum(dim=-1, keepdim=True)
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(float)
        required_sum = total_sum * threshold.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
            (batch_size, head_num, num_block_q, 1)
        ).to(input_tensor.device)
    else:
        required_sum = total_sum * threshold

    mask = torch.zeros_like(input_tensor, dtype=torch.bool)
    mask[:, :, :, 0] = 1
    mask[:, :, :, current_index : current_index + num_block_q] = (
        torch.eye(num_block_q, device=mask.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(1, head_num, num_block_q, num_block_q)
    )
    # Note that other_values only contains the values of the current block
    # (the sink blocks and diagonal are filled with 0)
    other_values = input_tensor.masked_fill(mask, 0)

    # Get sorted values
    sorted_values, _ = torch.sort(other_values, dim=-1, descending=True)
    sorted_values = sorted_values.to(input_tensor.device)
    sorted_values = torch.cat(
        [
            torch.zeros((batch_size, head_num, num_block_q, 1), device=input_tensor.device),
            torch.where(mask, input_tensor, 0).sum(
                dim=-1, keepdim=True
            ),  # shape: (batch_size, head_num, num_block_q, 1)
            sorted_values[
                :, :, :, :-2
            ],  # :-2 excludes the first and diagonal (which are marked 0 in other_values)
        ],
        dim=-1,
    )

    # Get sorted indices
    # index will select the already-masked (sink and diagonal) at the beginning
    _, index = torch.sort(
        torch.where(mask, 100000 * (1 + input_tensor), input_tensor),
        dim=-1,
        descending=True,
    )

    # [batch_size, head_num, num_block_q, num_block_k]
    cumulative_sum_without_self = torch.cat(
        [
            torch.zeros((batch_size, head_num, num_block_q, 1), device=input_tensor.device),
            sorted_values[:, :, :, 0:-1],
        ],
        dim=-1,
    ).cumsum(dim=-1)

    # Mask for indices where cumulative sum is below the required threshold.
    index_mask = cumulative_sum_without_self < required_sum
    index = torch.where(index_mask, index, 0)

    mask = mask.view(batch_size, head_num * num_block_q, num_block_k)
    index = index.view(batch_size, head_num * num_block_q, num_block_k)
    mask[:, torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1), index] = True
    mask = mask.view(batch_size, head_num, num_block_q, num_block_k)

    assert bool(
        (torch.where(mask, input_tensor, 0).sum(dim=-1, keepdim=True) >= required_sum * 0.99).all()
    ), f"mask sum {torch.where(mask,input_tensor,0).sum(dim=-1, keepdim=True)} < required_sum {required_sum}"  # noqa: E231, E501

    try:
        if causal:
            assert (~mask[:, :, :, current_index + num_block_q :]).all()
    except:  # noqa: B001, E722
        mask[:, :, :, current_index + num_block_q :] = False

    if causal:
        if decoding:
            assert mask[:, :, :, 0].all() and mask[:, :, :, -1].all()
        else:
            lambda_mask = torch.zeros_like(input_tensor, dtype=bool, device=input_tensor.device)
            lambda_mask[:, :, :, 0] = 1
            lambda_mask[:, :, :, current_index : current_index + num_block_q] = (
                torch.eye(num_block_q, device=lambda_mask.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, head_num, num_block_q, num_block_q)
            )
            assert torch.where(lambda_mask, mask, True).all()

    return mask


def shuffle_zigzag_masks(
    block_masks: torch.Tensor,  # [batch_size, num_qo_heads, num_blocks_local, num_blocks]
    process_group: dist.ProcessGroup = None,
):
    dim = len(block_masks.shape) - 1
    if not block_masks.is_contiguous():
        block_masks = block_masks.contiguous()

    # We must use outplace, otherwise it will raise error at backward due to inplace operations.
    # We can not change to_send directly and create a new tensor to store the result.
    to_send_f = torch.zeros_like(block_masks)

    # assume the input sequence length is 8, and computation runs on 4 GPUs
    # the seq is represented as [0 1 2 3 4 5 6 7], world size is 4
    # the input status before `shuffle_zigzag_input` is
    # - gpu A: [0 1]
    # - gpu B: [2 3]
    # - gpu C: [4 5]
    # - gpu D: [6 7]
    # the value of `to_send_slice` is
    # - gpu A: [1]
    # - gpu B: [3]
    # - gpu C: [5]
    # - gpu D: [7]
    block_seq_len = block_masks.shape[dim] // 2
    left_slicer = [slice(None)] * dim + [slice(None, block_seq_len)]
    right_slicer = [slice(None)] * dim + [slice(block_seq_len, None)]
    to_send_slice = block_masks[right_slicer].contiguous()

    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)

    res = torch.zeros_like(to_send_slice)

    _ops = []
    offset = (dist.get_rank() // world_size) * world_size
    # rank  src_rank
    # 0     3
    # 1     2
    # 2     1
    # 3     0
    src_rank = (world_size - rank - 1) % world_size + offset
    send_op = dist.P2POp(dist.isend, to_send_slice, src_rank, group=process_group)
    recv_op = dist.P2POp(dist.irecv, res, src_rank, group=process_group)

    _ops.append(send_op)
    _ops.append(recv_op)

    response = dist.batch_isend_irecv(_ops)
    for resp in response:
        resp.wait()

    if rank >= world_size // 2:  # D: 6 7, -> 1 6
        to_send_f[right_slicer] = block_masks[left_slicer]
        to_send_f[left_slicer] = res
    else:  # A: 0 1, -> 0 7
        to_send_f[left_slicer] = block_masks[left_slicer]
        to_send_f[right_slicer] = res
    # after shuffle, the status of `to_send_f`
    # GPU A: [0 7]
    # GPU B: [2 5]
    # GPU C: [3 4]
    # GPU D: [1 6]

    return to_send_f


@triton.jit
def softmax_fuse_block_sum_kernel_causal(
    In,
    Out,
    scale,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    real_q_len,
    k_len,  # we assume k_len is divisible by chunk size
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
    l_i = tl.zeros([block_size], dtype=tl.float32) + 1.0

    input_ptr = (
        In
        + batch_id * input_stride_0
        + head_id * input_stride_1
        + block_id * block_size * input_stride_2
    )
    input_ptr = (
        input_ptr + tl.arange(0, segment_size) + tl.arange(0, block_size)[:, None] * input_stride_2
    )

    output_ptr = (
        Out + batch_id * output_stride_0 + head_id * output_stride_1 + block_id * output_stride_2
    )
    output_ptr = output_ptr + tl.arange(0, segment_size // block_size)

    for iter in range(0, num_iters_before_causal):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)

        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local

        m_i = m_new

    for iter in range(num_iters_before_causal, num_iters_before_causal + 1):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        mask = offs_q[:, None] >= (offs_k[None, :] + iter * segment_size)
        X = tl.where(mask, X, -1.0e6)
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)

        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local

        m_i = m_new

    l_i_inv = 1.0 / l_i

    sum_mask = offs_q[:, None] < real_q_len

    for iter in range(0, num_iters_before_causal):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty))

    for iter in range(num_iters_before_causal, num_iters_before_causal + 1):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        mask = offs_q[:, None] >= (offs_k[None, :] + iter * segment_size)
        X = tl.where(mask, X, -1.0e6)
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty))

    for iter in range(num_iters_before_causal + 1, num_iters):
        X = tl.zeros([segment_size // block_size], dtype=tl.float32)
        tl.store(output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty))


@triton.jit
def softmax_fuse_block_sum_kernel_non_causal(
    In,
    Out,
    scale,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    real_q_len,
    k_len,  # we assume k_len is divisible by chunk size
    chunk_start,
    chunk_end,
    segment_size: tl.constexpr,
    block_size: tl.constexpr,
):
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    offs_q = tl.arange(0, block_size) + chunk_start + block_id * block_size
    offs_k = tl.arange(0, segment_size)  # noqa: F841

    num_iters = k_len // segment_size

    m_i = tl.zeros([block_size], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_size], dtype=tl.float32) + 1.0

    input_ptr = (
        In
        + batch_id * input_stride_0
        + head_id * input_stride_1
        + block_id * block_size * input_stride_2
    )
    input_ptr = (
        input_ptr + tl.arange(0, segment_size) + tl.arange(0, block_size)[:, None] * input_stride_2
    )

    output_ptr = (
        Out + batch_id * output_stride_0 + head_id * output_stride_1 + block_id * output_stride_2
    )
    output_ptr = output_ptr + tl.arange(0, segment_size // block_size)

    for iter in range(0, num_iters):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)

        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local

        m_i = m_new

    l_i_inv = 1.0 / l_i

    sum_mask = offs_q[:, None] < real_q_len

    for iter in range(0, num_iters):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty))


@triton.jit
def flat_group_gemm_kernel(
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
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_m = tl.program_id(0).to(tl.int64)
    block_n = tl.program_id(1).to(tl.int64)
    batch_id = tl.program_id(2).to(tl.int64) // H
    head_id = tl.program_id(2).to(tl.int64) % H

    if chunk_start + (block_m + 1) * BLOCK_M <= block_n * BLOCK_N:
        return

    Q_ptrs = Q + batch_id * stride_qz + head_id * stride_qh + block_m * BLOCK_M * stride_qn
    K_ptrs = K + batch_id * stride_kz + head_id * stride_kh + block_n * BLOCK_N * stride_kn

    Q_ptrs = Q_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_qn + tl.arange(0, BLOCK_K)[None, :]
    K_ptrs = K_ptrs + tl.arange(0, BLOCK_N)[None, :] * stride_kn + tl.arange(0, BLOCK_K)[:, None]

    num_iters = HEAD_DIM // BLOCK_K
    o = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for iter in range(num_iters):
        q = tl.load(Q_ptrs + iter * BLOCK_K)
        k = tl.load(K_ptrs + iter * BLOCK_K)
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
    is_caual: tl.constexpr,
):
    block_m = tl.program_id(0).to(tl.int64)
    block_n = tl.program_id(1).to(tl.int64)
    batch_id = tl.program_id(2).to(tl.int64) // H
    head_id = tl.program_id(2).to(tl.int64) % H

    if is_caual:
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


def softmax_fuse_block_sum(
    attn_weights_slice,
    reshaped_block_size,
    segment_size,
    chunk_start,
    chunk_end,
    real_q_len,
    scale,
    is_causal=True,
):
    batch_size, num_heads, q_len, k_len = attn_weights_slice.shape
    assert q_len % reshaped_block_size == 0
    try:
        assert k_len % segment_size == 0
    except:  # noqa: B001, E722
        raise AssertionError(f"xAttention error, k_len: {k_len}, segment size: {segment_size}")
    assert segment_size % reshaped_block_size == 0
    assert attn_weights_slice.stride(-1) == 1

    output = torch.empty(
        (batch_size, num_heads, q_len // reshaped_block_size, k_len // reshaped_block_size),
        dtype=attn_weights_slice.dtype,
        device=attn_weights_slice.device,
    )

    grid = (q_len // reshaped_block_size, num_heads, batch_size)

    if is_causal:
        softmax_fuse_block_sum_kernel_causal[grid](
            attn_weights_slice,
            output,
            scale,
            attn_weights_slice.stride(0),
            attn_weights_slice.stride(1),
            attn_weights_slice.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            real_q_len,
            k_len,
            chunk_start,
            chunk_end,
            segment_size,
            reshaped_block_size,
        )
    else:
        softmax_fuse_block_sum_kernel_non_causal[grid](
            attn_weights_slice,
            output,
            scale,
            attn_weights_slice.stride(0),
            attn_weights_slice.stride(1),
            attn_weights_slice.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            real_q_len,
            k_len,
            chunk_start,
            chunk_end,
            segment_size,
            reshaped_block_size,
        )

    return output


def flat_group_gemm(query_states, key_states, chunk_start, chunk_end):
    batch_size, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[2]

    output = torch.empty(
        (batch_size, num_heads, q_len, kv_len),
        dtype=query_states.dtype,
        device=query_states.device,
    )
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (q_len // BLOCK_M, kv_len // BLOCK_N, batch_size * num_heads)
    flat_group_gemm_kernel[grid](
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
        head_dim,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

    return output


def flat_group_gemm_fuse_reshape(
    query_states, key_states, stride, chunk_start, chunk_end, is_causal=True
):
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
    assert q_len % (stride * BLOCK_M) == 0, f"q_len={q_len}, stride={stride}, BLOCK_M={BLOCK_M}"
    assert kv_len % (stride * BLOCK_N) == 0, f"kv_len={kv_len}, stride={stride}, BLOCK_N={BLOCK_N}"

    grid = (q_len // stride // BLOCK_M, kv_len // stride // BLOCK_N, batch_size * num_heads)
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
        num_heads,
        stride,
        head_dim,
        BLOCK_M,
        BLOCK_N,
        is_causal,
    )

    return output
