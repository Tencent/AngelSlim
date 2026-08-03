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

"""Stem sparse-prefill algorithm under the SparsityAlgorithm framework.

Stem is an in-repo (Tencent) block-sparse prefill method. This adapter wraps
the existing Stem forward (``compressor/sparsity/stem/modules/forward.py``) and
its torch/HPC backends, exposing them through the unified framework:

  * per-instance ``layer_keep_ratios`` derived against the real
    ``num_hidden_layers`` (the legacy hardcoded 38-entry table only fit
    Qwen3-8B and IndexError'd / silently mis-scheduled other sizes).
  * ``allow_pseudo_sparse`` gates the missing-kernel
    fallback; default is hard-fail.

Decode falls back to the model's configured attention implementation (q3:
default fa2) via ``ALL_ATTENTION_FUNCTIONS`` inside the Stem forward.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Optional

from ..._base import AlgorithmTraits, IncompatibleConfigError, SparsityAlgorithm
from ...registry import SparsityAlgorithmRegistry

if TYPE_CHECKING:
    from transformers import PreTrainedModel

# The Stem keep-ratio schedule was tuned on Qwen3-8B (36 hidden layers in the
# released checkpoint; the legacy table was 38 entries = 2 warmup + 36 steady).
# The shape is: a short warmup region kept at 1.0, then a steady keep ratio.
_WARMUP_LAYERS = 2
_WARMUP_RATIO = 1.0
_STEADY_RATIO = 0.2


def interpolate_default_schedule(num_layers: int) -> List[float]:
    """Project the Qwen3-tuned warmup/steady schedule onto ``num_layers``.

    First ``min(_WARMUP_LAYERS, num_layers)`` layers keep 100%; the rest keep
    the steady ratio. This is a heuristic; production runs should pass an
    explicit ``layer_keep_ratios`` list.
    """
    if num_layers <= 0:
        raise IncompatibleConfigError(f"num_hidden_layers must be positive, got {num_layers}")
    warmup = min(_WARMUP_LAYERS, num_layers)
    return [_WARMUP_RATIO] * warmup + [_STEADY_RATIO] * (num_layers - warmup)


@SparsityAlgorithmRegistry.register("stem")
class Stem(SparsityAlgorithm):
    """Stem block-sparse prefill (Qwen3 / Qwen3-MoE language tower)."""

    name = "stem"

    @property
    def traits(self) -> AlgorithmTraits:
        return AlgorithmTraits(
            requires_unrepeated_kv=False,
            needs_calibration=False,
            supports_padding_mask=False,
            model_modal="any",
            # Stem's forward template matches Qwen3-style attention (q_norm,
            # k_norm, plain q_proj). Qwen3.5 gated attention needs its own
            # template, tracked separately; Stem stays LLM-first.
            compatible_model_types=frozenset({"qwen3", "qwen3_moe", "hy_v3"}),
        )

    def setup(self, model: "PreTrainedModel") -> None:
        """Derive per-instance config (layer_keep_ratios) against the model."""
        n = int(model.config.num_hidden_layers)

        user_ratios: Optional[list] = self.attn_kwargs.get("layer_keep_ratios")
        if user_ratios is not None:
            if len(user_ratios) != n:
                raise IncompatibleConfigError(
                    f"layer_keep_ratios has {len(user_ratios)} entries, model "
                    f"has {n} hidden layers; they must match."
                )
            ratios = list(user_ratios)
        else:
            ratios = interpolate_default_schedule(n)
            warnings.warn(
                f"[sparse:stem] no layer_keep_ratios in config; interpolating "
                f"Qwen3-tuned warmup/steady schedule onto {n} layers. Provide "
                f"explicit ratios for production use.",
                UserWarning,
                stacklevel=2,
            )

        # Build the attn_forward_config dict consumed by the Stem backend.
        self._attn_forward_config = dict(self.attn_kwargs)
        self._attn_forward_config["layer_keep_ratios"] = ratios
        self._attn_forward_config["allow_pseudo_sparse"] = self.allow_pseudo_sparse
        self._per_instance_ready = True

    def build_attn_forward(self, attn_module, model: "PreTrainedModel"):
        """Bind the Stem forward to ``attn_module`` and return it.

        Reuses the validated Stem forward from
        ``compressor/sparsity/algorithms/stem/modules/forward.py``. We attach the
        derived per-instance config onto the module (the forward reads
        ``self.attn_forward_config``) and return the bound method.
        """
        if not self._per_instance_ready:
            raise IncompatibleConfigError(
                "Stem.setup(model) must be called before build_attn_forward."
            )
        from .modules.forward import attn_forward

        attn_module.attn_forward_config = self._attn_forward_config
        return attn_forward.__get__(attn_module, type(attn_module))
