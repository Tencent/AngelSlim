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

"""VecAttention sparse-prefill algorithm — framework migration.

VecAttention keeps, per query-block, the key columns whose mean-query logit
clears a per-head **MinP threshold** (``gap = -log(threshold)``: keep key ``j``
iff ``qk[j] + gap >= max_j qk``), plus an attention-sink (initial) block and a
local band of ``block_size_k`` blocks. ``threshold`` is the sole sparsity knob:
``threshold -> 0`` keeps everything (== dense), larger is sparser. This is a
DISTINCT selection rule from the other six — MinP-on-logits, not top-p coverage
(xattention), gamma-coverage (flexprefill), or alpha-of-max (flashprefill).

This is the migration of the standalone ``compressor/sparsity/vecattention/``
subsystem onto the unified framework (the last sparse algorithm outside it).
The original
targeted Qwen2.5-VL via a bound ``forward`` + its own patcher; this migration
makes it a first-class registered algorithm on the SAME ABC / registry / patcher
/ forward-template path as the others, targeting Qwen3 / Qwen3.5 (the family we
have real weights for). The selected-column attention runs on the external
``vllm_flash_attn.sparse_attn_func`` kernel (an optional ``pip install``-ed
vLLM-flash-attention fork, not vendored in-tree); when it is not installed — or
head_dim is 256 (Qwen3.5) — it routes to the torch reference,
exactly like minference / flexprefill / xattention / flashprefill at head_dim 256.

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


@SparsityAlgorithmRegistry.register("vecattention")
class VecAttention(SparsityAlgorithm):
    """VecAttention per-head MinP column-sparse prefill (Qwen3 / Qwen3.5)."""

    name = "vecattention"

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
        block_size_q = int(cfg.get("block_size_q", 64))
        chunk_size = int(cfg.get("chunk_size", 64 * 1024))
        # Mirror the legacy VecAttentionConfig validation (kernel hard constraints).
        if block_size_q not in (64, 128):
            raise IncompatibleConfigError(
                f"vecattention block_size_q must be 64 or 128, got {block_size_q}"
            )
        if chunk_size % block_size_q != 0:
            raise IncompatibleConfigError(
                f"vecattention chunk_size ({chunk_size}) must be a multiple of "
                f"block_size_q ({block_size_q})"
            )
        self._cfg = {
            "threshold": float(cfg.get("threshold", 0.1)),
            "block_size_q": block_size_q,
            "block_size_k": int(cfg.get("block_size_k", 16)),
            "group_k_block": int(cfg.get("group_k_block", 16)),
            "chunk_size": chunk_size,
            "allow_pseudo_sparse": self.allow_pseudo_sparse,
        }
        self._per_instance_ready = True

    def build_attn_forward(self, attn_module, model: "PreTrainedModel"):
        if not self._per_instance_ready:
            raise IncompatibleConfigError(
                "VecAttention.setup(model) must be called before build_attn_forward."
            )
        from .._forward_templates import get_forward_template
        from .._forward_templates._common import repeat_kv
        from .prefill import vecattention_prefill

        cfg = self._cfg

        def _prefill_fn(attn, query_states, key_states, value_states):
            # The kernel consumes post-repeat K/V at num_attention_heads.
            k_rep = repeat_kv(key_states, attn.num_key_value_groups)
            v_rep = repeat_kv(value_states, attn.num_key_value_groups)
            return vecattention_prefill(
                query_states,
                k_rep,
                v_rep,
                threshold=cfg["threshold"],
                block_size_q=cfg["block_size_q"],
                block_size_k=cfg["block_size_k"],
                group_k_block=cfg["group_k_block"],
                chunk_size=cfg["chunk_size"],
                head_dim=attn.head_dim,
                allow_pseudo_sparse=bool(cfg["allow_pseudo_sparse"]),
            )

        model_type = getattr(model.config, "model_type", None)
        template = get_forward_template(model_type)
        attn_module.attn_forward_config = cfg
        forward = template(_prefill_fn, supports_padding_mask=self.traits.supports_padding_mask)
        return forward.__get__(attn_module, type(attn_module))
