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


"""Model-patching logic: replace the standard attention forward with Stem's."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .modules.forward import attn_forward

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .stem_configuration import StemConfig


def stem_patch(model: "PreTrainedModel", config: "StemConfig") -> "PreTrainedModel":
    """Replace each attention layer's ``forward`` with :func:`attn_forward`.

    Currently only **Qwen3** models are supported.

    Args:
        model: A HuggingFace causal-LM model (e.g. ``Qwen3ForCausalLM``).
        config: Stem runtime configuration.

    Returns:
        The same *model* object, mutated in-place with Stem attention.

    Raises:
        ValueError: If the model's ``model_type`` does not contain ``"qwen3"``.
    """
    model_type = model.config.model_type.lower()
    if "qwen3" not in model_type:
        raise ValueError(f"Only Qwen3 is supported, got model_type={model_type!r}")

    AttentionClass = model.model.layers[0].self_attn.__class__

    # Ensure every layer carries its own index (used by schedule functions).
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = i

    def _apply_stem_forward(module: object) -> None:
        """Bind the Stem ``attn_forward`` and config to each attention module."""
        if isinstance(module, AttentionClass):
            module.attn_forward_config = config.attn_kwargs
            module.forward = attn_forward.__get__(module, AttentionClass)

    model.apply(_apply_stem_forward)
    return model
