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

"""MInference sparse-prefill algorithm under the SparsityAlgorithm framework.

The three user-facing variant names
(``minference`` / ``a_shape`` / ``tri_shape``) all map to this single
``MInference`` class via the registry (one ``name`` field, no ``attn_type``).
The package path is the namespace, so the class carries no ``Sparsity`` suffix.

``requires_unrepeated_kv = False`` — all three variants consume
post-``repeat_kv`` K/V at ``num_attention_heads`` granularity, matching MInference
upstream (``minference_forward.py`` calls ``repeat_kv`` before kernel dispatch).
The forward template does the repeat; the algorithm only declares the trait.

The production fused Triton/CUDA kernels are vendored at
kernel-build time. Until then this binds the pure-torch reference
(``reference.py``) — the *pseudo-sparse* path. When real kernels
land, ``build_attn_forward`` swaps the dispatch without touching the framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..._base import AlgorithmTraits, IncompatibleConfigError, SparsityAlgorithm
from ...registry import SparsityAlgorithmRegistry

if TYPE_CHECKING:
    from transformers import PreTrainedModel

# The three user-facing variant names that map to this one class.
_VARIANTS = ("minference", "a_shape", "tri_shape")


class MInference(SparsityAlgorithm):
    """MInference vertical-slash / streaming sparse prefill (Qwen3 / Qwen3-MoE).

    ``variant`` selects the prefill pattern:
      * ``a_shape``   — sink + sliding window (StreamingLLM).
      * ``tri_shape`` — a_shape + full-causal tail.
      * ``minference`` — per-head vertical-and-slash (needs a pattern JSON for
        searched budgets; falls back to dense ``(1000, 6096)`` per head).
    """

    def __init__(self, attn_kwargs: Optional[dict] = None, *, variant: str = "minference"):
        if variant not in _VARIANTS:
            raise ValueError(f"unknown minference variant {variant!r}; one of {_VARIANTS}")
        super().__init__(attn_kwargs)
        self.variant = variant
        # name reflects the user-facing variant (mirrors the registry key).
        self.name = variant

    @property
    def traits(self) -> AlgorithmTraits:
        return AlgorithmTraits(
            requires_unrepeated_kv=False,  # all 3 run on repeated K/V
            needs_calibration=False,
            supports_padding_mask=False,
            model_modal="any",
            # Per-architecture forward templates cover both plain Qwen3
            # and gated Qwen3.5 attention, so MInference supports both families.
            # Only full_attention layers are patched; Qwen3.5's linear_attention
            # (gated delta-net) layers are filtered out by resolve_sparsable_layers.
            # NB: a Qwen3.5 *causal LM* reports model_type "qwen3_5_text" (the text
            # tower); the multimodal wrapper reports "qwen3_5". Both are accepted.
            compatible_model_types=frozenset(
                {
                    "qwen3",
                    "qwen3_moe",
                    "qwen3_5",
                    "qwen3_5_text",
                    "qwen3_5_moe",
                    "qwen3_5_moe_text",
                    "hy_v3",
                }
            ),
        )

    def setup(self, model: "PreTrainedModel") -> None:
        """Resolve the per-variant config against the model.

        * a_shape / tri_shape: streaming kwargs (n_init / n_local / n_last).
        * minference: load + fingerprint-validate the pattern JSON (or None for
          the dense per-head fallback).
        """
        cfg = dict(self.attn_kwargs)

        # Streaming kwargs (a_shape / tri_shape). Defaults match upstream.
        streaming_kwargs = {
            "n_init": int(cfg.get("n_init", 128)),
            "n_local": int(cfg.get("n_local", 3968)),
            "n_last": int(cfg.get("n_last", 100)),
        }

        best_pattern = None
        if self.variant == "minference":
            from .patterns._loader import load_pattern

            pattern_path = cfg.get("pattern_path")  # already abs-resolved upstream
            best_pattern = load_pattern(pattern_path, model.config)

        self._attn_forward_config = {
            "variant": self.variant,
            "streaming_kwargs": streaming_kwargs,
            "best_pattern": best_pattern,
            "allow_pseudo_sparse": self.allow_pseudo_sparse,
        }
        self._per_instance_ready = True

    def build_attn_forward(self, attn_module, model: "PreTrainedModel"):
        if not self._per_instance_ready:
            raise IncompatibleConfigError(
                "MInference.setup(model) must be called before build_attn_forward."
            )
        from .._forward_templates import get_forward_template
        from .._forward_templates._common import repeat_kv
        from .prefill import minference_prefill

        cfg = self._attn_forward_config

        def _prefill_fn(attn, query_states, key_states, value_states):
            # Kernels consume post-repeat K/V at num_attention_heads.
            k_rep = repeat_kv(key_states, attn.num_key_value_groups)
            v_rep = repeat_kv(value_states, attn.num_key_value_groups)
            return minference_prefill(
                cfg["variant"],
                query_states,
                k_rep,
                v_rep,
                streaming_kwargs=cfg.get("streaming_kwargs") or {},
                best_pattern=cfg.get("best_pattern"),
                layer_idx=attn.layer_idx,
                head_dim=attn.head_dim,
                allow_pseudo_sparse=bool(cfg.get("allow_pseudo_sparse", False)),
            )

        model_type = getattr(model.config, "model_type", None)
        template = get_forward_template(model_type)
        attn_module.attn_forward_config = cfg
        # Forward the trait so the template rejects a padded prefill batch
        # (minference's per-head key selection never sees the mask).
        forward = template(_prefill_fn, supports_padding_mask=self.traits.supports_padding_mask)
        return forward.__get__(attn_module, type(attn_module))


def _make_factory(variant: str):
    """Return a registry factory that binds ``variant`` and forwards kwargs."""

    def _factory(attn_kwargs=None):
        return MInference(attn_kwargs=attn_kwargs, variant=variant)

    return _factory


# Register all three user-facing names onto the one class.
for _v in _VARIANTS:
    SparsityAlgorithmRegistry.register_factory(_v, _make_factory(_v))
