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

"""Decoder-layer location helpers.

Locate the decoder layers regardless of LLM vs
multimodal nesting, and filter out layers that carry no ``self_attn`` (Qwen3.5
interleaves ``linear_attention`` and ``full_attention`` layers via
``config.layer_types``; only ``full_attention`` layers expose ``self_attn``).
"""

from __future__ import annotations

from ._base import IncompatibleConfigError


def resolve_layers(model):
    """Return the decoder ``ModuleList`` (LLM body or VLM language tower).

    Order of preference:
      1. ``model.model.language_model.layers`` — multimodal language tower
         (Qwen3.5 / Qwen3.5-MoE / Qwen2.5-VL).
      2. ``model.model.layers`` — plain decoder-only LLM (Qwen3 / Qwen3-MoE).
    """
    inner = getattr(model, "model", None)
    if inner is None:
        # Raise the subsystem's own error type (was a bare AttributeError),
        # so callers' `except IncompatibleConfigError` catches it consistently.
        raise IncompatibleConfigError(
            f"{type(model).__name__} has no `.model`; cannot locate decoder layers."
        )
    lm = getattr(inner, "language_model", None)
    if lm is not None and hasattr(lm, "layers"):
        return lm.layers
    if hasattr(inner, "layers"):
        return inner.layers
    raise IncompatibleConfigError(
        f"Cannot locate decoder layers on {type(model).__name__}. Sparsity "
        f"requires either model.model.layers or "
        f"model.model.language_model.layers."
    )


def resolve_sparsable_layers(model):
    """Return only the decoder layers that carry a ``self_attn`` module.

    Qwen3.5 / Qwen3.5-MoE set ``layer_types[i] = "linear_attention"`` for most
    layers; those carry ``self.linear_attn`` (a gated delta-net), NOT
    ``self.self_attn``. Sparse prefill only applies to ``full_attention``
    layers. For plain Qwen3 every layer has ``self_attn`` so this is a no-op
    filter.
    """
    return [layer for layer in resolve_layers(model) if hasattr(layer, "self_attn")]
