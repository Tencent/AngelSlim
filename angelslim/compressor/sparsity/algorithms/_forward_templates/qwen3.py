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

"""Qwen3 / Qwen3-MoE sparse-attention forward template (plain attention).

Mirrors ``transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward``
(transformers >= 5.2): plain ``q_proj`` (no gate), ``q_norm`` / ``k_norm`` on the
head dim. Prefill delegates to the algorithm's ``prefill_fn``; decode (q_len==1)
falls back to the model's configured attention implementation.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ._common import (
    apply_rotary_pos_emb,
    assert_no_padding_mask,
    eager_attention_forward,
    reject_output_attentions,
)


def build_qwen3_forward(prefill_fn, supports_padding_mask: bool = False):
    """Return a ``forward`` that runs ``prefill_fn`` on the prefill path.

    ``supports_padding_mask``: when False (the default for the sparse algorithms,
    whose key selection never sees the mask), a padded prefill batch is rejected
    rather than silently attending to padding tokens.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # Thread cache_position (+ sin/cos) into the cache update.
            # DynamicCache ignores them, but StaticCache / quantized / sliding
            # caches need cache_position to write the correct slots — dropping
            # it silently corrupts those cache types.
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": kwargs.get("cache_position"),
            }
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Sparse prefill ONLY on a true first-fill: q_len>1 AND no cache prefix
        # (k_len==q_len). A cached 2nd turn or chunked/prefix-cache prefill has
        # k_len>q_len; the sparse algorithms select keys from raw Q/K assuming
        # query i aligns to key i (no bottom-right offset), so feeding them
        # k_len>q_len mis-anchors the causal/diagonal math. Route that case to
        # the model's configured attention (dense but correct over the full
        # cache) instead.
        q_len = query_states.shape[2]
        k_len = key_states.shape[2]
        if q_len > 1 and k_len == q_len:
            reject_output_attentions(kwargs)
            if not supports_padding_mask:
                assert_no_padding_mask(attention_mask, key_states.shape[2])
            attn_output = prefill_fn(self, query_states, key_states, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_weights = None
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
                sliding_window=getattr(self, "sliding_window", None),
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return forward
