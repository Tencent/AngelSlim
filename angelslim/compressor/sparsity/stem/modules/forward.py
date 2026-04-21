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


"""Stem-patched attention forward pass.

This module provides the replacement ``forward`` method that is bound to each
attention layer by :func:`stem.patch.stem_patch`.  During **prefill**
(``q_len > 1``) it delegates to the Stem sparse backend; during **decode**
(``q_len == 1``) it falls back to the model's original attention implementation
(eager, FlashAttention-2, SDPA, etc.).

The code mirrors the structure of
``transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward``
(Transformers >= 5.2) and should be kept in sync with upstream changes.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack

from ..backends import stem_forward

# ---------------------------------------------------------------------------
# Helper functions (identical to upstream Qwen3)
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by splitting and concatenating halves."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding (RoPE) to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (GQA support).

    ``(B, num_kv_heads, L, D)`` -> ``(B, num_attention_heads, L, D)``
    """
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Fallback eager attention (used in decode phase, mirrors upstream)
# ---------------------------------------------------------------------------


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
    """Eager (non-sparse) scaled dot-product attention.

    Used as the **decode** fallback when ``q_len == 1`` and no specialised
    attention implementation (e.g. FlashAttention-2) is configured.
    Matches the upstream ``eager_attention_forward`` in Transformers >= 5.2.
    """
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _assert_no_padding_mask_for_stem(attention_mask: torch.Tensor, k_len: int) -> None:
    """Verify that the attention mask has no padding (required by Stem prefill).

    Raises
    ------
    ValueError
        If the mask is not 4-D or if the last query row contains ``-inf``
        entries (indicating padding tokens).
    """
    if attention_mask.ndim != 4:
        raise ValueError(f"attention_mask must be 4-D, got shape={tuple(attention_mask.shape)}")
    last_row = attention_mask[:, :, -1, :k_len]
    if not torch.isfinite(last_row).all():
        raise ValueError("Stem prefill requires no padding mask (last query row has -inf).")


# ---------------------------------------------------------------------------
# Patched attention forward
# ---------------------------------------------------------------------------


def attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Stem-patched attention forward — drop-in replacement for
    ``Qwen3Attention.forward`` (Transformers >= 5.2).

    * **Prefill** (``q_len > 1``): delegates to :func:`stem_forward` which
      computes block-sparse attention according to the configured backend.
    * **Decode** (``q_len == 1``): uses the model's original attention
      implementation (eager / FlashAttention-2 / SDPA / flex).
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # --- QKV projection & RoPE (identical to upstream) --------------------
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # --- KV cache update (Transformers >= 5.2 style) ----------------------
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    q_len = query_states.shape[2]
    k_len = key_states.shape[2]

    # --- Prefill (Stem sparse attention) ----------------------------------
    if q_len > 1:
        if attention_mask is not None:
            _assert_no_padding_mask_for_stem(attention_mask, k_len)

        prefill_kwargs = {
            "layer_idx": self.layer_idx,
            "attn_forward_config": self.attn_forward_config,
        }
        backend = self.attn_forward_config.get("backend", "torch")

        # HPC kernels (both bf16 and fp8) handle GQA internally;
        # only the pure-torch path needs explicit KV head repeat.
        if backend == "hpc":
            stem_key_states = key_states
            stem_value_states = value_states
        else:
            stem_key_states = repeat_kv(key_states, self.num_key_value_groups)
            stem_value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = stem_forward(
            query_states, stem_key_states, stem_value_states, prefill_kwargs
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_weights = None

    # --- Decode (standard attention, mirrors upstream) ---------------------
    else:
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
