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
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack

# Shared per-architecture helpers (single source of truth). Stem previously
# carried its own copies of these (rotate_half / apply_rotary_pos_emb /
# repeat_kv / eager_attention_forward / a padding check), but they had drifted:
# the local RoPE was full-rotary only (broke Qwen3.5 partial rotary) and the
# local padding check only inspected the LAST query row of a 4D mask (missing 2D
# masks and padding on earlier rows). The shared versions are partial-rotary-safe
# and dtype-aware, so Stem reuses them rather than re-implementing.
from ..._forward_templates._common import (
    apply_rotary_pos_emb,
    assert_no_padding_mask,
    eager_attention_forward,
    reject_output_attentions,
    repeat_kv,
)
from ..backends import stem_forward

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
        # Thread cache_position (+ sin/cos) into the cache update,
        # mirroring the qwen3/qwen3_5 forward templates. DynamicCache ignores
        # these, but StaticCache / quantized / sliding-window caches need
        # cache_position to write the correct slots — dropping it writes wrong
        # slots when InferEngine overrides use_cache=True with such a cache.
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": kwargs.get("cache_position"),
        }
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    q_len = query_states.shape[2]
    k_len = key_states.shape[2]

    # --- Prefill (Stem sparse attention) ----------------------------------
    # ONLY on a true first-fill: q_len>1 AND no cache prefix (k_len==q_len). A
    # cached 2nd turn or chunked/prefix-cache prefill has k_len>q_len; stem's
    # block scoring assumes query i aligns to key i, so k_len>q_len mis-anchors
    # the block-causal mask. Route that case to the configured attention.
    if q_len > 1 and k_len == q_len:
        reject_output_attentions(kwargs)
        # Shared dtype-aware padding guard (handles None / 2D / 4D-float /
        # 4D-bool); replaces stem's weaker last-row-only 4D-only check.
        assert_no_padding_mask(attention_mask, k_len)

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
            # Not every attention module sets ``sliding_window`` (Qwen3 does;
            # Hunyuan-V3's HYV3Attention does not) — read it defensively so the
            # decode path works across every compatible arch.
            sliding_window=getattr(self, "sliding_window", None),
            **kwargs,
        )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
