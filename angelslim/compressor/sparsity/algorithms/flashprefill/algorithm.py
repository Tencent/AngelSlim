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

"""FlashPrefill sparse-prefill algorithm.

FlashPrefill (Fan et al., arXiv:2603.06199) keeps, per query-block, the smallest
set of key-blocks whose block-approximate energy clears a **Max-based Dynamic
Threshold** ``thresh_I = alpha * max_J s[I,J]`` (keep ``s[I,J] >= thresh_I``) —
a single-pass max-reduction with NO Top-k sort and NO Top-p cumsum, which is what
distinguishes it from xattention (top-p) and flexprefill (gamma-coverage). Plus
unconditional sink (first 256 tokens) + local window (last 512). ``alpha`` is the
sole hyperparameter: alpha=0 keeps every causal block (== dense), larger is
sparser.

CLEAN-ROOM IMPLEMENTATION. The upstream repo (qhfan/FlashPrefill) ships with no
license, so the license fail-open policy forbids vendoring its code. Instead
the *method* (block-pooled discovery + max-based thresholding) is implemented
from the paper in ``blockmask.py`` / ``reference.py`` (original AngelSlim work),
and the selected-block attention runs on ``block_sparse_attn`` — the SAME
BSD-3-Clause-licensed kernel Stem / xattention use. No upstream FlashPrefill bytes are
present; see the package NOTICE. head_dim 256 (Qwen3.5) routes to the torch
reference, exactly like minference / flexprefill / xattention.

Reuses the framework the prior algorithms proved: ABC / registry / patcher /
guards / forward templates (qwen3 + qwen3.5 gated). The algorithm supplies only a
``prefill_fn``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._base import AlgorithmTraits, IncompatibleConfigError, SparsityAlgorithm
from ...registry import SparsityAlgorithmRegistry

if TYPE_CHECKING:
    from transformers import PreTrainedModel


@SparsityAlgorithmRegistry.register("flashprefill")
class FlashPrefill(SparsityAlgorithm):
    """FlashPrefill max-based dynamic-threshold block-sparse prefill (Qwen3/3.5)."""

    name = "flashprefill"

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
            "alpha": float(cfg.get("alpha", 0.2)),
            "block_size": int(cfg.get("block_size", 128)),
            "sink": int(cfg.get("sink", 256)),
            "window": int(cfg.get("window", 512)),
            "last_n_block_full": int(cfg.get("last_n_block_full", 2)),
            "allow_pseudo_sparse": self.allow_pseudo_sparse,
        }
        self._per_instance_ready = True

    def build_attn_forward(self, attn_module, model: "PreTrainedModel"):
        if not self._per_instance_ready:
            raise IncompatibleConfigError(
                "FlashPrefill.setup(model) must be called before build_attn_forward."
            )
        from .._forward_templates import get_forward_template
        from .._forward_templates._common import repeat_kv
        from .prefill import flashprefill_prefill

        cfg = self._cfg

        def _prefill_fn(attn, query_states, key_states, value_states):
            # The kernel consumes post-repeat K/V at num_attention_heads.
            k_rep = repeat_kv(key_states, attn.num_key_value_groups)
            v_rep = repeat_kv(value_states, attn.num_key_value_groups)
            return flashprefill_prefill(
                query_states,
                k_rep,
                v_rep,
                alpha=cfg["alpha"],
                block_size=cfg["block_size"],
                sink=cfg["sink"],
                window=cfg["window"],
                last_n_block_full=cfg["last_n_block_full"],
                head_dim=attn.head_dim,
                allow_pseudo_sparse=bool(cfg["allow_pseudo_sparse"]),
            )

        model_type = getattr(model.config, "model_type", None)
        template = get_forward_template(model_type)
        attn_module.attn_forward_config = cfg
        forward = template(_prefill_fn, supports_padding_mask=self.traits.supports_padding_mask)
        return forward.__get__(attn_module, type(attn_module))
