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

"""Model-patching logic: replace the standard attention forward with VecAttention's."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .modules.forward import qwen_vl_attn_forward

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .vecattention_configuration import VecAttentionConfig


def vecattention_patch(
    model: "PreTrainedModel", config: "VecAttentionConfig"
) -> "PreTrainedModel":
    """Replace each attention layer's ``forward`` with VecAttention sparse prefill.

    Supports Qwen2.5-VL and Qwen2-VL models.

    Args:
        model: A HuggingFace VLM model (e.g. Qwen2.5-VL).
        config: VecAttention runtime configuration.

    Returns:
        The same *model* object, mutated in-place with VecAttention.

    Raises:
        ValueError: If the model's ``model_type`` is not supported.
    """
    model_type = model.config.model_type.lower()

    if "qwen2_5_vl" in model_type or "qwen2_vl" in model_type:
        _patch_qwen_vl(model, config)
    else:
        raise ValueError(
            f"VecAttention does not support model_type={model_type!r}. "
            f"Supported: qwen2_5_vl, qwen2_vl."
        )

    return model


def _patch_qwen_vl(model: "PreTrainedModel", config: "VecAttentionConfig") -> None:
    """Apply VecAttention patch to Qwen2.5-VL / Qwen2-VL models."""
    if hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    elif hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise AttributeError(
            "Cannot find attention layers in Qwen VL model. "
            "Expected model.model.language_model.layers or model.model.layers."
        )

    AttentionClass = layers[0].self_attn.__class__

    for i, layer in enumerate(layers):
        attn = layer.self_attn
        attn.layer_idx = i
        attn.vecattention_config = config
        attn.forward = qwen_vl_attn_forward.__get__(attn, AttentionClass)
