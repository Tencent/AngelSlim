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

"""High-level entry point for applying VecAttention to a VLM model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .patch import vecattention_patch
from .vecattention_configuration import VecAttentionConfig

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class VecAttentionInference:
    """Callable object that patches a model to use VecAttention sparse prefill.

    Only the prefill phase (q_len > 1) uses VecAttention; decode falls back
    to the model's original attention implementation.

    Usage::

        vec = VecAttentionInference(attn_kwargs={"threshold": 0.9})
        model = vec(model)

    Args:
        attn_kwargs: Forwarded to ``VecAttentionConfig``. See its docstring
            for valid keys (threshold, block_size_q, block_size_k,
            group_k_block, chunk_size).
    """

    def __init__(self, attn_kwargs: dict | None = None) -> None:
        self.config = VecAttentionConfig(attn_kwargs=attn_kwargs)

    def __call__(self, model: "PreTrainedModel") -> "PreTrainedModel":
        """Apply the VecAttention patch and return the modified model."""
        return vecattention_patch(model, self.config)

    def __repr__(self) -> str:
        return f"VecAttentionInference(config={self.config!r})"
