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

"""FlexPrefill sparse-prefill algorithm.

FlexPrefill (MInference family, MIT) adaptively chooses, per head, the smallest
set of key-blocks whose cumulative attention mass covers ``gamma`` (default
0.9), estimated from the last query block; ``tau`` decides when a head degrades
to block-sparse. Unlike minference's vertical_slash (which needs a CUDA index
ext), the real path is pure Triton — only ``triton`` + ``flash_attn`` are
required, and head_dim must be in {16,32,64,128}.

Reuses the framework Stem/MInference proved: ABC / registry / patcher / guards /
forward templates (qwen3 + qwen3.5 gated). The algorithm supplies only a
``prefill_fn``; head_dim 256 (Qwen3.5) routes to the torch reference,
exactly like minference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional  # noqa: F401

from ..._base import AlgorithmTraits, IncompatibleConfigError, SparsityAlgorithm
from ...registry import SparsityAlgorithmRegistry

if TYPE_CHECKING:
    from transformers import PreTrainedModel


@SparsityAlgorithmRegistry.register("flexprefill")
class FlexPrefill(SparsityAlgorithm):
    """FlexPrefill adaptive vertical-slash sparse prefill (Qwen3 / Qwen3.5)."""

    name = "flexprefill"

    @property
    def traits(self) -> AlgorithmTraits:
        return AlgorithmTraits(
            requires_unrepeated_kv=False,  # kernel sees post-repeat K/V
            needs_calibration=False,
            supports_padding_mask=False,  # per-head selection never sees the mask
            model_modal="any",
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
        cfg = dict(self.attn_kwargs)
        self._cfg = {
            "gamma": float(cfg.get("gamma", 0.9)),
            "tau": float(cfg.get("tau", 0.1)),
            "min_budget": cfg.get("min_budget", None),
            "max_budget": cfg.get("max_budget", None),
            "block_size": int(cfg.get("block_size", 128)),
            "allow_pseudo_sparse": self.allow_pseudo_sparse,
        }
        self._per_instance_ready = True

    def build_attn_forward(self, attn_module, model: "PreTrainedModel"):
        if not self._per_instance_ready:
            raise IncompatibleConfigError(
                "FlexPrefill.setup(model) must be called before build_attn_forward."
            )
        from .._forward_templates import get_forward_template
        from .._forward_templates._common import repeat_kv
        from .prefill import flexprefill_prefill

        cfg = self._cfg

        def _prefill_fn(attn, query_states, key_states, value_states):
            # The kernel consumes post-repeat K/V at num_attention_heads.
            k_rep = repeat_kv(key_states, attn.num_key_value_groups)
            v_rep = repeat_kv(value_states, attn.num_key_value_groups)
            return flexprefill_prefill(
                query_states,
                k_rep,
                v_rep,
                gamma=cfg["gamma"],
                tau=cfg["tau"],
                min_budget=cfg["min_budget"],
                max_budget=cfg["max_budget"],
                block_size=cfg["block_size"],
                head_dim=attn.head_dim,
                allow_pseudo_sparse=bool(cfg["allow_pseudo_sparse"]),
            )

        model_type = getattr(model.config, "model_type", None)
        template = get_forward_template(model_type)
        attn_module.attn_forward_config = cfg
        forward = template(_prefill_fn, supports_padding_mask=self.traits.supports_padding_mask)
        return forward.__get__(attn_module, type(attn_module))
