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

"""Per-architecture sparse-attention forward templates.

A template owns the architecture-specific QKV / RoPE / cache / GQA / gate /
decode / o_proj preamble; the algorithm supplies only a ``prefill_fn``. This
keeps each algorithm architecture-agnostic and each architecture's quirks
(Qwen3.5 gating) in exactly one place.

  ``get_forward_template(model_type)(prefill_fn)`` -> bound ``forward`` callable.
"""

from __future__ import annotations

from ..._base import IncompatibleConfigError
from .hy3 import build_hy3_forward
from .qwen3 import build_qwen3_forward
from .qwen3_5 import build_qwen3_5_forward

TEMPLATE_REGISTRY = {
    "qwen3": build_qwen3_forward,
    "qwen3_moe": build_qwen3_forward,
    # Qwen3.5 gated attention. A causal LM reports the text-tower model_type
    # ("qwen3_5_text"); the multimodal wrapper reports "qwen3_5". Map both.
    "qwen3_5": build_qwen3_5_forward,
    "qwen3_5_text": build_qwen3_5_forward,
    "qwen3_5_moe": build_qwen3_5_forward,
    "qwen3_5_moe_text": build_qwen3_5_forward,
    # Hunyuan-V3: plain (ungated) attention, MoE only in the MLP tower. Uses the
    # same forward as Qwen3.
    "hy_v3": build_hy3_forward,
}


def get_forward_template(model_type: str):
    """Return the template builder for ``model_type`` (raises if unsupported)."""
    if model_type not in TEMPLATE_REGISTRY:
        raise IncompatibleConfigError(
            f"No sparse forward template registered for model_type="
            f"{model_type!r}. Supported model types: {sorted(TEMPLATE_REGISTRY)}."
        )
    return TEMPLATE_REGISTRY[model_type]
