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


"""Pure-PyTorch implementation of the Stem sparse prefill.

This module provides two main entry points:

* :func:`_compute_triton_block_logits` — compute block-level importance
  scores using a Triton-accelerated strided group GEMM.
* :func:`stem_forward_torch` — full Stem prefill: score → schedule →
  top-k mask → block-sparse (or pseudo-sparse) attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    from block_sparse_attn import block_sparse_attn_func

    HAS_BLOCK_SPARSE_KERNEL = True
except ImportError:
    print(
        "⚠️ [Stem] 'block_sparse_attn' not found. " "Falling back to pseudo-sparse implementation."
    )
    HAS_BLOCK_SPARSE_KERNEL = False


# ---------------------------------------------------------------------------
# Per-layer sparsity schedule
# ---------------------------------------------------------------------------

# Default per-layer keep-ratio: first 2 layers keep 100%, remaining layers 20%.
_DEFAULT_LAYER_KEEP_RATIOS: list[float] = [1.0, 1.0] + [0.2] * 36

# Short-sequence thresholds — below these lengths the sparsity schedule is
# relaxed to avoid losing too much context.
_SHORT_SEQ_THRESHOLD_FULL = 56  # keep-all threshold
_SHORT_SEQ_THRESHOLD_LINEAR = 160  # linear blend threshold
_SHORT_SEQ_LINEAR_RATE = 0.2
_SHORT_SEQ_LINEAR_BIAS = 30
_LONG_SEQ_RATE = 0.1
_LONG_SEQ_BIAS = 30


def generate_exact_k_schedule(
    num_blocks: int,
    alpha: float,
    layer_idx: int,
    device: torch.device,
    num_heads: int | None = None,
) -> torch.Tensor:
    """Generate the per-block top-k budget schedule.

    For each query-block position, the schedule tells how many key-blocks
    should be kept (before the alpha decay).

    Parameters
    ----------
    num_blocks : int
        Number of query blocks (``Qb``).
    alpha : float
        Decay factor applied beyond the initial keep region.
    layer_idx : int
        Transformer layer index (controls the base keep ratio).
    device : torch.device
        Target device for the returned tensor.
    num_heads : int | None
        If given, return a ``(num_heads, num_blocks)`` tensor with
        per-head schedules; otherwise return ``(num_blocks,)``.

    Returns
    -------
    torch.Tensor
        Budget schedule — ``(num_blocks,)`` or ``(num_heads, num_blocks)``.
    """
    keep_ratio = _DEFAULT_LAYER_KEEP_RATIOS[layer_idx]
    per_head_ratios = [keep_ratio] * (num_heads or 1)

    def _build_single_schedule(k_val: int, n: int) -> torch.Tensor:
        # Relax budget for short sequences.
        if k_val != n:
            if n < _SHORT_SEQ_THRESHOLD_FULL:
                k_val = n
            elif n < _SHORT_SEQ_THRESHOLD_LINEAR:
                k_val = int(n * _SHORT_SEQ_LINEAR_RATE + _SHORT_SEQ_LINEAR_BIAS)
            else:
                k_val = int(n * _LONG_SEQ_RATE) + _LONG_SEQ_BIAS

        schedule = torch.full((n,), k_val, dtype=torch.long, device=device)
        if n > k_val:
            decay_len = n - k_val
            k_end = k_val * alpha
            ideal_vals = torch.linspace(
                float(k_val),
                float(k_end),
                steps=decay_len,
                dtype=torch.float64,
                device=device,
            )
            schedule[k_val:] = torch.clamp(torch.floor(ideal_vals).long(), min=1, max=k_val)
        return schedule

    schedules = [
        _build_single_schedule(int(ratio * num_blocks), num_blocks) for ratio in per_head_ratios
    ]
    stacked = torch.stack(schedules, dim=0)
    return stacked.squeeze(0) if stacked.shape[0] == 1 else stacked


# ---------------------------------------------------------------------------
# Block-level importance scoring
# ---------------------------------------------------------------------------


def _block_downsample(
    x: torch.Tensor,
    seq_len: int,
    num_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Downsample *x* along the sequence dimension via per-block max-pooling.

    Args:
        x: Input tensor of shape ``(..., seq_len, D)``.
        seq_len: Original sequence length.
        num_blocks: Target block count.
        block_size: Block size.

    Returns:
        Tensor of shape ``(..., num_blocks, D)`` — the maximum value in
        each block.
    """
    padded_len = num_blocks * block_size
    if seq_len % block_size != 0:
        pad = torch.zeros(
            x.shape[:-2] + (padded_len - seq_len, x.shape[-1]),
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat([x, pad], dim=-2)
    return x.view(x.shape[:-2] + (num_blocks, block_size, x.shape[-1])).max(dim=-2).values


def _compute_triton_block_logits(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_size: int,
    stride: int,
    chunk_size: int,
    norm: float,
    causal: bool,
) -> torch.Tensor:
    """Compute block-level importance logits using the Triton strided GEMM.

    The scoring process:

    1. Pad Q / K / V to a multiple of ``chunk_size``.
    2. Compute strided Q·K^T via the Triton kernel (chunk by chunk).
    3. Add a value-norm bonus term (log-normalised, ReLU-gated).
    4. Reduce to ``(Qb, Kb)`` block logits by averaging over sub-blocks.
    5. Apply a causal block mask.

    Parameters
    ----------
    query_states : torch.Tensor   — ``(B, H, L_q, D)``
    key_states   : torch.Tensor   — ``(B, H, L_kv, D)``
    value_states : torch.Tensor   — ``(B, H, L_kv, D)``
    block_size   : int             — size of each block (typically 128)
    stride       : int             — striding factor for the GEMM
    chunk_size   : int             — chunk width for iterating over Q
    norm         : float           — additional scaling denominator
    causal       : bool            — whether to apply causal masking

    Returns
    -------
    torch.Tensor
        Block logits of shape ``(B, H, Qb, Kb)``.
    """
    from ..ops.stem_kernel import flat_group_gemm_fuse_reshape

    B, H, k_len, head_dim = key_states.shape
    _, _, q_len, _ = query_states.shape
    assert H == query_states.shape[1], "Q and K must have the same number of heads."
    dtype = query_states.dtype
    device = query_states.device

    # --- Step 1: pad to a multiple of chunk_size --------------------------
    target_seq_len = max(k_len, q_len)
    target_seq_len = ((target_seq_len + chunk_size - 1) // chunk_size) * chunk_size
    k_pad = target_seq_len - k_len
    q_pad = target_seq_len - q_len

    if k_pad > 0:
        pad_key_states = F.pad(key_states, (0, 0, 0, k_pad), value=0).to("cuda")
        value_norm = torch.norm(value_states, p=2, dim=-1, keepdim=True).to(device)
        pad_value_states = F.pad(value_norm, (0, 0, 0, k_pad), value=0).to("cuda")
    else:
        pad_key_states = key_states
        value_norm = torch.norm(value_states, p=2, dim=-1, keepdim=True).to(device)
        pad_value_states = value_norm

    if q_pad > 0:
        pad_query_states = F.pad(query_states, (0, 0, 0, q_pad), value=0).to("cuda")
    else:
        pad_query_states = query_states

    # --- Derived dimensions -----------------------------------------------
    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride

    pad_q_len = pad_query_states.shape[2]
    pad_k_len = pad_key_states.shape[2]
    assert pad_q_len == pad_k_len == target_seq_len

    q_down_len = pad_q_len // stride
    k_down_len = pad_k_len // stride
    pad_Qb = pad_q_len // block_size
    pad_Kb = pad_k_len // block_size
    chunk_base = (pad_Kb - pad_Qb) * reshaped_block_size

    # --- Step 2: value-norm bonus term ------------------------------------
    v_down = _block_downsample(pad_value_states, pad_k_len, k_down_len, stride).squeeze(-1)
    v_log_norm = torch.log(v_down + 1e-6)

    LAMBDA_MAG = 0.2  # magnitude of the value-norm bonus

    valid_len_down = k_len // stride
    if valid_len_down > 0:
        valid_v = v_log_norm[:, :, :valid_len_down]
        v_mean = valid_v.mean(dim=-1, keepdim=True)
        v_std = valid_v.std(dim=-1, keepdim=True)
    else:
        v_mean = v_log_norm.mean(dim=-1, keepdim=True)
        v_std = v_log_norm.std(dim=-1, keepdim=True)

    v_log_norm = (v_log_norm - v_mean) / (v_std + 1e-6)
    v_log_norm = F.relu(v_log_norm)
    v_bonus = LAMBDA_MAG * v_log_norm

    # --- Step 3: chunked strided Q·K^T via Triton -------------------------
    scores = torch.zeros((B, H, q_down_len, k_down_len), dtype=dtype, device=device)
    scale = query_states.new_tensor(1.0 / (math.sqrt(head_dim) * stride * norm), dtype=dtype)
    q_chunk_num = pad_q_len // chunk_size

    for chunk_idx in range(q_chunk_num):
        chunk_q_start = chunk_idx * reshaped_chunk_size
        chunk_q_end = chunk_q_start + reshaped_chunk_size
        q_slice_start = chunk_q_start * stride
        q_slice_end = chunk_q_end * stride

        attn_weights_slice = flat_group_gemm_fuse_reshape(
            pad_query_states[:, :, q_slice_start:q_slice_end, :].contiguous(),
            pad_key_states.contiguous(),
            stride,
            chunk_base + chunk_q_start,
            chunk_base + chunk_q_end,
            is_causal=causal,
        )

        attn_scores = attn_weights_slice * scale + v_bonus.unsqueeze(-2)
        scores[:, :, chunk_q_start:chunk_q_end, :] = attn_scores
        del attn_weights_slice

    # --- Step 4: reduce to block logits -----------------------------------
    scores = scores.view(B, H, pad_Qb, reshaped_block_size, pad_Kb, reshaped_block_size)
    block_logits = scores.mean(dim=3).mean(dim=4)

    # Trim to the actual (unpadded) block counts.
    Qb = (q_len + block_size - 1) // block_size
    Kb = (k_len + block_size - 1) // block_size
    block_logits = block_logits[:, :, :Qb, :Kb]

    # --- Step 5: causal block mask ----------------------------------------
    if causal:
        qb_idx = torch.arange(Qb, device=device).view(1, 1, -1, 1)
        kb_idx = torch.arange(Kb, device=device).view(1, 1, 1, -1)
        block_logits = block_logits.masked_fill(kb_idx > qb_idx, float("-inf"))

    return block_logits.to(dtype)


# ---------------------------------------------------------------------------
# Stem prefill — torch backend
# ---------------------------------------------------------------------------


def stem_forward_torch(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    prefill_kwargs: dict,
) -> torch.Tensor:
    """Stem sparse-prefill implementation using pure PyTorch + optional Triton.

    Workflow:
      1. Compute block-level importance scores via :func:`_compute_triton_block_logits`.
      2. Build a per-layer top-k schedule (:func:`generate_exact_k_schedule`).
      3. Select the top-k blocks, add initial-blocks and sliding-window blocks.
      4. Run block-sparse attention (real kernel if available, else pseudo-sparse).

    Parameters
    ----------
    query_states  : ``(B, H, L_q, D)``
    key_states    : ``(B, H, L_kv, D)``
    value_states  : ``(B, H, L_kv, D)``
    prefill_kwargs : dict
        Contains ``"attn_forward_config"`` and ``"layer_idx"``.

    Returns
    -------
    torch.Tensor
        Attention output — ``(B, H, L_q, D)``.
    """
    config = prefill_kwargs["attn_forward_config"]
    layer_idx = prefill_kwargs["layer_idx"]

    block_size: int = config.get("block_size", 128)
    config_alpha = config.get("stem_alpha", 1.0)

    B, H, Lq, head_dim = query_states.shape
    _, _, Tk, _ = key_states.shape

    scaling = head_dim**-0.5
    Qb = (Lq + block_size - 1) // block_size
    Kb = (Tk + block_size - 1) // block_size

    stride: int = config.get("stride", 8)
    chunk_size: int = config.get("chunk_size", 2048)
    norm: float = config.get("norm", 1.0)

    # --- 1. Block-level scoring -------------------------------------------
    block_logits = _compute_triton_block_logits(
        query_states,
        key_states,
        value_states,
        block_size=block_size,
        stride=stride,
        chunk_size=chunk_size,
        norm=norm,
        causal=True,
    )

    # --- 2. Per-layer sparsity schedule -----------------------------------
    mask_block = torch.zeros((B, H, Qb, Kb), device=query_states.device, dtype=torch.bool)

    alpha = config_alpha[layer_idx] if isinstance(config_alpha, (list, tuple)) else config_alpha
    sched = generate_exact_k_schedule(Qb, alpha, layer_idx, query_states.device, num_heads=H)
    growth = max(1.0, alpha)

    if sched.dim() == 1:
        head_needed = torch.clamp(torch.ceil(sched[0].to(torch.float32) * growth), max=Kb)
        needed_k = int(head_needed.item())
        budget = sched.view(1, 1, -1, 1)
    else:
        if sched.shape[0] != H:
            raise ValueError("Per-head k_start configuration does not match number of heads.")
        head_start = sched[:, 0].to(torch.float32)
        head_needed = torch.clamp(torch.ceil(head_start * growth), max=Kb)
        needed_k = int(head_needed.max().item())
        budget = sched.view(1, H, -1, 1)

    needed_k = max(0, min(needed_k, Kb))

    # --- 3. Top-k block selection -----------------------------------------
    if needed_k > 0:
        topk_vals, topk_idx = torch.topk(block_logits, needed_k, dim=-1)
        rank = (
            torch.arange(needed_k, device=query_states.device)
            .view(1, 1, 1, -1)
            .expand(B, H, Qb, needed_k)
        )
        keep = (rank < budget) & torch.isfinite(topk_vals)
        if keep.any():
            mask_block.scatter_(3, topk_idx, keep)

    # Causal block mask.
    q_range = torch.arange(Qb, device=query_states.device).view(1, 1, Qb, 1)
    k_range = torch.arange(Kb, device=query_states.device).view(1, 1, 1, Kb)
    causal_block_mask = k_range <= q_range

    # Always keep the first few blocks (sink tokens).
    initial_blocks: int = int(config.get("initial_blocks", 4))
    if initial_blocks > 0:
        mask_block |= (k_range < min(initial_blocks, Kb)) & causal_block_mask

    # Sliding-window blocks.
    window_size: int = int(config.get("window_size", 4))
    if window_size > 0:
        recent_start = torch.clamp(q_range - (window_size - 1), min=0)
        mask_block |= (k_range >= recent_start) & causal_block_mask

    mask_block &= causal_block_mask

    # --- 4. Block-sparse attention ----------------------------------------
    if HAS_BLOCK_SPARSE_KERNEL:
        q_kernel = query_states.transpose(1, 2).reshape(B * Lq, H, head_dim)
        k_kernel = key_states.transpose(1, 2).reshape(B * Tk, H, head_dim)
        v_kernel = value_states.transpose(1, 2).reshape(B * Tk, H, head_dim)

        q_cu = torch.arange(
            0,
            (B + 1) * Lq,
            step=Lq,
            dtype=torch.int32,
            device=query_states.device,
        )
        k_cu = torch.arange(
            0,
            (B + 1) * Tk,
            step=Tk,
            dtype=torch.int32,
            device=query_states.device,
        )
        head_mask_type = torch.ones(H, dtype=torch.int32, device=query_states.device)

        torch.cuda.synchronize()
        attn_output = block_sparse_attn_func(
            q_kernel,
            k_kernel,
            v_kernel,
            q_cu,
            k_cu,
            head_mask_type,
            None,
            mask_block.contiguous(),
            Lq,
            Tk,
            p_dropout=0.0,
            deterministic=True,
            is_causal=True,
        )
        torch.cuda.synchronize()
        return attn_output.view(B, Lq, H, head_dim).transpose(1, 2)

    # Fallback: pseudo-sparse attention (expand block mask to full mask).
    mask_full = (
        mask_block.unsqueeze(-1)
        .unsqueeze(-3)
        .expand(-1, -1, -1, block_size, -1, block_size)
        .reshape(B, H, Qb * block_size, Kb * block_size)
    )
    mask_full = mask_full[..., :Lq, :Tk]

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if Lq > 1:
        causal_bool = torch.ones((Lq, Tk), device=query_states.device, dtype=torch.bool).triu(1)
        attn_weights = attn_weights.masked_fill(causal_bool[None, None, :, :], float("-inf"))

    attn_weights = attn_weights.masked_fill(~mask_full, float("-inf"))
    probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return torch.matmul(probs, value_states)
