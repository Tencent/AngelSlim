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


"""High-level entry point for applying the Stem patch to a HuggingFace model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .patch import stem_patch
from .stem_configuration import StemConfig

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class StemInference:
    """Callable object that patches a model to use Stem sparse attention.

    Usage::

        stem = StemInference(attn_kwargs={"backend": "hpc", "hpc_dtype": "fp8"})
        model = stem(model)

    Args:
        attn_kwargs: Forwarded to ``StemConfig``. See its docstring for valid keys.
    """

    def __init__(self, attn_kwargs: dict | None = None) -> None:
        self.config = StemConfig(attn_kwargs=attn_kwargs)

    def __call__(self, model: "PreTrainedModel") -> "PreTrainedModel":
        """Apply the Stem attention patch and return the modified model."""
        return stem_patch(model, self.config)

    def __repr__(self) -> str:
        return f"StemInference(config={self.config!r})"
