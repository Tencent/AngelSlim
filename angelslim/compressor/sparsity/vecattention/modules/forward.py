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


"""VecAttention-patched attention forward pass for Vision-Language Models.

This module provides the replacement ``forward`` method that is bound to each
attention layer by the patch function. During **prefill** (``q_len > 1``) it
delegates to the VecAttention sparse backend; during **decode** (``q_len == 1``)
it falls back to the model's original attention implementation.

VecAttention applies sparse attention only to the vision token region;
text tokens before/after use standard full attention.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..ops.vecattention_kernel import average_vector, fuse_qk_softmax_minp_wo_causal

# Ensure VecAttention's custom vllm_flash_attn is importable.
# The vllm-flash-attention source lives as a standalone submodule under ops/.
_OPS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ops")
_VLLM_FA_DIR = os.path.join(_OPS_DIR, "vllm-flash-attention")
if os.path.isdir(_VLLM_FA_DIR) and _VLLM_FA_DIR not in sys.path:
    sys.path.insert(0, _VLLM_FA_DIR)

try:
    from vllm_flash_attn import sparse_attn_func
except ImportError:
    raise ImportError(
        "vllm_flash_attn with sparse_attn_func not found. "
        "Please init the submodule and build:\n"
        "  git submodule update --init --recursive\n"
        "  cd angelslim/compressor/sparsity/vecattention/ops/vllm-flash-attention\n"
        "  pip install -e . --no-build-isolation"
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (GQA support)."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager scaled dot-product attention for decode fallback."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def _full_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Standard scaled dot-product attention (no sparsity).

    Used for text token segments where VecAttention is not applied.
    """
    scaling = query.shape[-1] ** -0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    if causal:
        Lq, Lk = query.shape[2], key.shape[2]
        causal_mask = torch.ones(Lq, Lk, dtype=torch.bool, device=query.device).triu(
            diagonal=Lk - Lq + 1
        )
        attn_weights = attn_weights.masked_fill(causal_mask[None, None, :, :], float("-inf"))

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(attn_weights, value)


# ---------------------------------------------------------------------------
# VecAttention prefill core
# ---------------------------------------------------------------------------


def vecattention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    threshold: Union[float, torch.Tensor] = None,
    q_pooling_size: int = 128,
    k_local_size: int = 128,
    group_k_block: int = 1,
    causal: bool = True,
    chunk_size: int = 16 * 1024,
) -> torch.Tensor:
    """Vectorized sparse attention prefill (VecAttention).

    Selects important key-value blocks per query using a fused Triton kernel
    that applies a per-head MinP threshold on averaged Q*K^T scores, then
    runs sparse attention via vllm_flash_attn.sparse_attn_func.

    Args:
        query_states: (batch, num_heads, seq_len, head_dim)
        key_states:   (batch, num_kv_heads, seq_len, head_dim)
        value_states: (batch, num_kv_heads, seq_len, head_dim)
        threshold: MinP threshold.
        q_pooling_size: Query block size for vector pooling. Must be 64 or 128.
        k_local_size: Key local block size for column selection.
        group_k_block: Number of k-blocks processed together.
        causal: Whether to apply causal masking.
        chunk_size: Prefill chunk size (tokens).

    Returns:
        attn_output: (batch, num_heads, seq_len, head_dim)
    """
    assert chunk_size % q_pooling_size == 0, "chunk_size must be a multiple of q_pooling_size"
    assert q_pooling_size in [64, 128], "q_pooling_size must be 64 or 128"
    SPATTN_BLOCK_SIZE_K = 64

    if isinstance(threshold, torch.Tensor):
        gap = -torch.log(threshold + 1e-9)
    else:
        gap = -math.log(threshold + 1e-9, math.e)

    batch_size, num_heads, seq_len, head_dim = query_states.shape
    num_q_blocks = math.ceil(seq_len / q_pooling_size)

    attn_output = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        dtype=query_states.dtype,
        device=query_states.device,
    )

    if causal:
        n = q_pooling_size // SPATTN_BLOCK_SIZE_K
        blk_count = torch.full(
            (batch_size, num_heads, num_q_blocks),
            2 * n,
            dtype=torch.int32,
            device=query_states.device,
        )
        blk_count[..., 0] = math.ceil(min(seq_len, q_pooling_size) / SPATTN_BLOCK_SIZE_K)
        if seq_len > q_pooling_size and seq_len % q_pooling_size != 0:
            blk_count[..., -1] = n + math.ceil(
                (seq_len - math.floor(seq_len / q_pooling_size) * q_pooling_size)
                / SPATTN_BLOCK_SIZE_K
            )
        blk_idx = torch.zeros(
            batch_size,
            num_heads,
            num_q_blocks,
            2 * n,
            dtype=torch.int32,
            device=query_states.device,
        )
        offsets = (
            torch.arange(n, device=query_states.device, dtype=torch.int32) * SPATTN_BLOCK_SIZE_K
        )
        blk_idx[..., :n] = offsets
        base = (
            torch.arange(0, num_q_blocks, device=query_states.device, dtype=torch.int32)
            * q_pooling_size
        ).unsqueeze(-1)
        blk_idx[..., n:] = base + offsets
    else:
        blk_count = torch.zeros(
            batch_size, num_heads, num_q_blocks, dtype=torch.int32, device=query_states.device
        )
        blk_idx = torch.zeros(
            batch_size, num_heads, num_q_blocks, 1, dtype=torch.int32, device=query_states.device
        )

    for i in range(0, seq_len, chunk_size):
        q_chunk = query_states[:, :, i : i + chunk_size, :]

        avg_q_chunk = average_vector(q_chunk, q_pooling_size, use_triton=True)
        col_count, col_idx = fuse_qk_softmax_minp_wo_causal(
            avg_q_chunk,
            key_states,
            i // q_pooling_size,
            gap,
            causal,
            q_pooling_size,
            k_local_size,
            wo_initial=causal,
            group_k_block=group_k_block,
        )

        blk_count_chunk = blk_count[:, :, i // q_pooling_size : (i + chunk_size) // q_pooling_size]
        blk_idx_chunk = blk_idx[:, :, i // q_pooling_size : (i + chunk_size) // q_pooling_size, :]

        k_chunk = key_states[:, :, : i + chunk_size, :] if causal else key_states
        v_chunk = value_states[:, :, : i + chunk_size, :] if causal else value_states

        attn_output[:, :, i : i + chunk_size, :] = sparse_attn_func(
            q_chunk.transpose(1, 2).contiguous(),
            k_chunk.transpose(1, 2).contiguous(),
            v_chunk.transpose(1, 2).contiguous(),
            q_pooling_size,
            blk_count_chunk.contiguous(),
            blk_idx_chunk.contiguous(),
            col_count,
            col_idx,
            return_softmax_lse=False,
            causal=causal,
        ).transpose(1, 2)

    return attn_output


# ---------------------------------------------------------------------------
# Patched forward for Qwen2.5-VL models
# ---------------------------------------------------------------------------


def qwen_vl_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """VecAttention-patched attention forward for Qwen2.5-VL.

    During prefill, applies VecAttention sparse attention to the vision token
    region and full attention to surrounding text tokens.
    During decode, falls back to the model's original attention.
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb,
    )

    rope_scaling = getattr(self, "rope_scaling", None) or self.config.rope_scaling
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    config = self.vecattention_config
    vision_start = config.attn_kwargs.get("vision_start_position", None)
    vision_end = config.attn_kwargs.get("vision_end_position", None)

    if q_len > 1 and q_len >= config.block_size_q:
        if vision_start is None and vision_end is None:
            attn_output = vecattention_forward(
                query_states,
                key_states,
                value_states,
                threshold=config.threshold,
                q_pooling_size=config.block_size_q,
                k_local_size=config.block_size_k,
                group_k_block=config.group_k_block,
                causal=True,
                chunk_size=config.chunk_size,
            )
        else:
            if vision_start is not None and vision_start > 0:
                attn_output_before = _full_attention(
                    query_states[:, :, :vision_start, :],
                    key_states[:, :, :vision_start, :],
                    value_states[:, :, :vision_start, :],
                    causal=True,
                )

            ve = vision_end if vision_end is not None else q_len
            attn_output_va = vecattention_forward(
                query_states[:, :, :ve, :],
                key_states[:, :, :ve, :],
                value_states[:, :, :ve, :],
                threshold=config.threshold,
                q_pooling_size=config.block_size_q,
                k_local_size=config.block_size_k,
                group_k_block=config.group_k_block,
                causal=True,
                chunk_size=config.chunk_size,
            )

            if vision_start is not None and vision_start > 0:
                attn_output_va[:, :, :vision_start, :] = attn_output_before

            if vision_end is not None and vision_end < q_len:
                attn_output_after = _full_attention(
                    query_states[:, :, vision_end:, :],
                    key_states,
                    value_states,
                    causal=True,
                )
                attn_output = torch.cat([attn_output_va, attn_output_after], dim=2)
            else:
                attn_output = attn_output_va

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_weights = None
    else:
        attention_interface = ALL_ATTENTION_FUNCTIONS.get(
            getattr(self.config, "_attn_implementation", None),
            eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else getattr(self, "attention_dropout", 0.0),
            scaling=self.head_dim**-0.5,
            **kwargs,
        )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
