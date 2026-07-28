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

"""Core abstractions for the sparse-attention subsystem.

The Protocol suite collapses to a single ``SparsityAlgorithm``
ABC. After retiring KV-compression, calibration, and
search, the framework has exactly one capability axis (prefill), so a Protocol-of-one is
just ceremony. The KV-compression Protocol can return later alongside its first
concrete subclass (rule of three).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field  # noqa: F401
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch  # noqa: F401
    from transformers import PreTrainedModel


class IncompatibleConfigError(RuntimeError):
    """Raised when a sparse configuration cannot be applied to a model/runtime.

    Used for all hard-fails: TP/PP/multi-node, FP8-attn collision,
    unsupported model_type, CUDA-Graph / vLLM.
    The message must be actionable — tell the user exactly what to change.
    """


@dataclass(frozen=True)
class AlgorithmTraits:
    """Declarative properties of a sparse algorithm, consumed by the patcher.

    ``requires_unrepeated_kv`` is False for every current algorithm —
    the per-architecture forward templates run ``repeat_kv`` before the kernel,
    mirroring MInference upstream.

    ``model_modal`` is "any" for all current algorithms because the
    patcher restricts traversal to the language tower regardless of whether the
    surrounding model is multimodal. Modality is a property of the *model*
    (resolved via ``_modal.resolve_modal``), not the algorithm.
    """

    requires_unrepeated_kv: bool = False
    needs_calibration: bool = False
    supports_padding_mask: bool = False
    # "any" => the patcher may apply this to the language tower of any model
    # whose model_type is in ``compatible_model_types``.
    model_modal: str = "any"
    # HF ``config.model_type`` strings this algorithm supports. None => all.
    compatible_model_types: Optional[frozenset] = None


class SparsityAlgorithm(ABC):
    """Base class for all sparse-prefill algorithms.

    Subclasses live under ``angelslim/compressor/sparsity/algorithms/<name>/algorithm.py`` and are
    registered with :class:`SparsityAlgorithmRegistry`. The package path is the
    namespace, so class names carry no ``Sparsity`` suffix:
    ``Stem``, ``MInference``, ``XAttention``, ``FlexPrefill``, ``FlashPrefill``.

    Lifecycle:
      1. ``__init__(attn_kwargs)`` — store config; no model yet.
      2. ``setup(model)`` — per-instance derivations that need the model
         (e.g. Stem's per-layer keep-ratio schedule). Called once by
         the patcher before any forward is bound.
      3. ``build_attn_forward(attn_module, model)`` — return the bound forward
         callable that replaces ``attn_module.forward``.
    """

    #: registry key + YAML ``compression.sparsity.name`` value
    name: str = ""

    def __init__(self, attn_kwargs: Optional[dict] = None):
        self.attn_kwargs: dict = dict(attn_kwargs or {})
        # Whether the missing-kernel path may fall back to pseudo-sparse.
        # Default False => hard-fail. Overridden from YAML.
        self.allow_pseudo_sparse: bool = bool(self.attn_kwargs.pop("allow_pseudo_sparse", False))
        self._per_instance_ready = False

    @property
    @abstractmethod
    def traits(self) -> AlgorithmTraits:
        """Declarative properties consumed by the patcher."""

    def setup(self, model: "PreTrainedModel") -> None:
        """Per-instance derivations that need the model. Override as needed.

        Default is a no-op. Stem overrides this to derive the per-layer
        keep-ratio schedule against ``model.config.num_hidden_layers``.
        """
        self._per_instance_ready = True

    @abstractmethod
    def build_attn_forward(self, attn_module, model: "PreTrainedModel"):
        """Return the callable to bind as ``attn_module.forward``.

        The returned function is bound by the patcher via ``push_attn_forward``.
        It MUST only touch submodule parameters through ``__call__``
        (e.g. ``self.q_proj(x)``), never via ``.weight`` directly — this is the
        ZeRO-3 contract.
        """
