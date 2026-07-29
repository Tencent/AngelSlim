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

"""Modality resolution from HF ``config.model_type``.

``SlimModelFactory.get_series_by_models`` keys on AngelSlim
*registered class names* (``"Qwen3_5"``), NOT HF ``model_type`` strings
(``"qwen3_5"``). The sparse patcher decides modality from the HF model_type, so
it needs its own table. Algorithms declare ``model_modal="any"`` (they patch
the language tower regardless of outer modality); this table is used to label
the model and to keep the door open for future modality-specific guards.
"""

from __future__ import annotations

# HF config.model_type -> coarse modality. Language-tower sparse prefill works
# for both "llm" and the language tower of "vlm" models.
_HF_TYPE_TO_MODAL = {
    "qwen3": "llm",
    "qwen3_moe": "llm",
    "qwen3_next": "llm",
    "qwen3_5": "vlm",  # multimodal wrapper; sparse covers the language tower
    "qwen3_5_text": "llm",  # the text tower itself (Qwen3_5ForCausalLM.config)
    "qwen3_5_moe": "vlm",
    "qwen3_5_moe_text": "llm",
    "qwen3_vl": "vlm",
    "qwen3_vl_moe": "vlm",
    "qwen3_omni_moe": "omni",
    # Qwen2.5-VL / Qwen2-VL: VecAttention's original target family. The sparse
    # patcher covers the language tower (model.model.language_model.layers).
    "qwen2_5_vl": "vlm",
    "qwen2_vl": "vlm",
}


def resolve_modal(model) -> str:
    """Return the coarse modality string for ``model`` from its HF model_type."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return _HF_TYPE_TO_MODAL.get(model_type, "llm")
