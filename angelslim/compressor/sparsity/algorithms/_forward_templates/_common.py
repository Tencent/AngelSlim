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

"""Shared helpers for per-architecture sparse-attention forward templates.

Sparse algorithms differ only in the *prefill* step; the QKV
projection / q_norm / RoPE / KV-cache / GQA-repeat / decode-fallback / o_proj
preamble is a property of the model **architecture**, not the algorithm. With
two architectures (Qwen3, Qwen3.5-gated) and several algorithms, repeating the
preamble per (algorithm × architecture) is the duplication this package removes.

A *template* is ``build(attn_module, prefill_fn)`` -> bound ``forward``. The
``prefill_fn`` is the algorithm's contribution; it has the signature::

    prefill_fn(attn_module, query_states, key_states, value_states) -> attn_output

with ``query/key/value`` in ``(B, H, L, D)`` **pre-repeat** layout — i.e. K/V are
at ``num_key_value_heads`` granularity (GQA), exactly as they come out of the
KV cache. **The template does NOT call ``repeat_kv`` on the prefill path** (an
earlier version of this docstring claimed it did — it does not; the only
``repeat_kv`` in this module is inside ``eager_attention_forward``, the DECODE
fallback). The **algorithm** is responsible for repeating K/V if its kernel
needs ``num_attention_heads``-granularity heads (the minference family does this
in ``MInference.build_attn_forward``, mirroring upstream MInference
which calls ``repeat_kv`` before kernel dispatch). ``attn_output`` is returned in
``(B, H, L, D)`` and the template handles the transpose / reshape / gate / o_proj
tail.
"""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    """Partial-rotary-safe RoPE.

    Rotates only the first ``rotary_dim = cos.shape[-1]`` dims and passes the
    rest through, matching both plain Qwen3 (rotary_dim == head_dim, nothing
    passes through) and Qwen3.5 partial rotary (rotary_dim < head_dim, e.g.
    head_dim 256 with partial_rotary_factor 0.25 -> rotary_dim 64).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """``(B, num_kv_heads, L, D)`` -> ``(B, num_attention_heads, L, D)`` (GQA)."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def assert_no_padding_mask(attention_mask, k_len: int) -> None:
    """Hard-fail ONLY on a genuine padding mask.

    Sparse prefill algorithms whose ``traits.supports_padding_mask`` is False
    estimate which keys to keep from the raw Q/K/V and never see the mask — so a
    PADDED batch would let them attend to padding tokens as if real. We make that
    a LOUD precondition. But we must NOT reject a *legitimate no-padding* mask
    (e.g. a 2D all-ones mask, which transformers/fa2 commonly pass for a single
    unpadded sequence). Mask conventions handled:

      * ``None``                       → no mask, OK.
      * 2D ``(B, S)`` bool/int         → HF padding mask; 1/True = keep, 0/False
                                         = PAD. Any 0 over the first ``k_len``
                                         keys → hard-fail; all-ones → OK.
      * 4D ``(B, H, Sq, Sk)`` → masked positions depend on the mask DTYPE, which
                                         transformers picks by attention backend:
                                         a FLOAT additive mask marks masked keys
                                         with ``finfo.min`` (a finite large-
                                         negative, NOT -inf), a BOOL keep-mask
                                         marks them ``False``. A right-padded
                                         batch shows masked keys in the LAST query
                                         row; none masked there → OK.

    Anything else (unexpected ndim) is left to the kernel rather than guessed at.
    """
    if attention_mask is None:
        return
    nd = attention_mask.ndim
    if nd == 2:
        # boolean/int keep-mask: a 0/False anywhere in the kept-key span is PAD.
        m = attention_mask[..., :k_len]
        if m.dtype == torch.bool:
            has_pad = (~m).any().item()
        else:
            has_pad = (m == 0).any().item()
        if has_pad:
            raise ValueError(_PAD_MSG)
        return
    if nd == 4:
        # A right-padded batch shows masked keys in the LAST query row over valid
        # keys. But a 4D mask can ALSO carry a non-padding custom bias, which the
        # sparse algorithm (selecting keys from raw Q/K/V) cannot honor. We must
        # refuse a custom mask without false-positiving on a plain CAUSAL mask
        # (whose masked strict upper triangle is expected). So check the
        # CAUSAL-ALLOWED region only (key j <= query i, aligned to the bottom-right
        # for q_len <= k_len): any masked key there is padding or a custom bias on
        # attendable keys → refuse. Masked keys in the future (upper-tri) are
        # normal causal masking and are ignored.
        #
        # "masked" is DTYPE-dependent and must NOT be tested with isfinite():
        # transformers emits a FLOAT additive mask whose masked positions are
        # ``finfo.min`` (finite, ~-3.4e38 for bf16 — isfinite() is True), or a
        # BOOL keep-mask (no inf at all). isfinite() flags neither, so we branch
        # on dtype: float → "very negative" (< finfo.min/2 catches finfo.min and
        # -inf alike); bool → not-kept (~mask).
        am = attention_mask[:, :, :, :k_len]
        q_len = am.shape[2]
        offset = k_len - q_len  # bottom-right causal alignment
        qi = torch.arange(q_len, device=am.device).view(q_len, 1)
        kj = torch.arange(k_len, device=am.device).view(1, k_len)
        causal_allowed = kj <= (qi + offset)  # (q_len, k_len) bool
        if am.dtype == torch.bool:
            masked = ~am
        else:
            masked = am < (torch.finfo(am.dtype).min / 2)
        masked_in_allowed = masked & causal_allowed[None, None]
        if masked_in_allowed.any():
            raise ValueError(_PAD_MSG)
        return
    # Unknown rank: do not guess — let the downstream kernel handle/raise.
    return


_PAD_MSG = (
    "sparse prefill requires an unpadded batch: the attention_mask carries "
    "padding. This algorithm does not support a padding mask "
    "(traits.supports_padding_mask=False) — its key selection would treat "
    "padding tokens as real. Use batch size 1, or unpadded / left-pad-free "
    "inputs, for sparse prefill."
)


def reject_output_attentions(kwargs) -> None:
    """Hard-fail when a sparse prefill is asked to return attention weights.

    The sparse prefill path computes ``attn_output`` from a block-sparse (or
    kernel-fused) kernel that never materializes the full ``(B, H, Sq, Sk)``
    probability matrix, so it returns ``attn_weights=None``. If the caller set
    ``output_attentions=True`` we would silently hand back ``None`` instead of
    the requested tensor — a quiet HF-contract violation. Make it explicit: the
    user must disable ``output_attentions`` or use dense attention.
    """
    if kwargs.get("output_attentions"):
        raise ValueError(
            "sparse prefill does not support output_attentions=True: the "
            "sparse kernels never form the full attention matrix, so the "
            "weights cannot be returned. Disable output_attentions, or run "
            "dense attention for layers where you need the weights."
        )


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Eager SDPA decode fallback (mirrors upstream Qwen3 / Qwen3.5)."""
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
