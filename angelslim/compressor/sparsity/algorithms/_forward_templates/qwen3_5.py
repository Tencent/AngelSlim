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

"""Qwen3.5 / Qwen3.5-MoE sparse-attention forward template (GATED attention).

Qwen3.5 attention is *gated*:
``q_proj`` outputs ``num_attention_heads * head_dim * 2``; the tensor is chunked
into (query, gate); after attention the output is multiplied by
``sigmoid(gate)`` before ``o_proj``. A plain-Qwen3 template would slice the gate
half as query heads, doubling the head count and running sparse on garbage.

This template mirrors the real
``transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Attention.forward``
(transformers 5.9) exactly, inserting the sparse prefill in place of the dense
attention call and preserving the gate application (the GateApplyStep).

Only ``full_attention`` layers carry ``self_attn`` and reach this template; the
``linear_attention`` (gated delta-net) layers are filtered out by
``resolve_sparsable_layers``, so the linear tower is untouched.
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


def build_qwen3_5_forward(prefill_fn, supports_padding_mask: bool = False):
    """Return a gated ``forward`` that runs ``prefill_fn`` on the prefill path.

    ``supports_padding_mask``: when False (sparse default), a padded prefill batch
    is rejected — the algorithm's key selection never sees the mask.
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

        # Gated attention: q_proj -> (query, gate). Mirrors upstream exactly.
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # Thread cache_position (+ sin/cos) into the cache update (see
            # qwen3.py template). Dropping it corrupts StaticCache / quantized /
            # sliding cache slot placement.
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
        # k_len>q_len; the sparse algorithms assume query i aligns to key i, so
        # feeding them k_len>q_len mis-anchors the causal/diagonal math. Route
        # that case to the configured attention (dense but correct).
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
                # Qwen3.5 attention may not set sliding_window; read
                # defensively so the first decode token doesn't AttributeError.
                sliding_window=getattr(self, "sliding_window", None),
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # GateApplyStep — restore the gate before o_proj.
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return forward
