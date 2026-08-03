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

"""XAttention sparse-prefill algorithm.

XAttention (MInference family, MIT; method from mit-han-lab/x-attention) keeps,
per head, the smallest set of key-blocks whose cumulative (antidiagonal-strided)
attention mass covers ``threshold`` (default 0.9), then runs the selected-block
attention through ``block_sparse_attn`` — the SAME kernel Stem uses. The estimate
is strided (``stride``, default 8): it sub-samples the query/key sequence to
approximate block importance cheaply, so unlike flexprefill there is no exact
"keep-all == dense" knob; on real (peaked) attention it is faithful (rel ~0.0005
at threshold 0.98 on a Qwen3-8B layer), and at short sequences (<= 2 blocks) it
returns dense exactly.

Upstream hard constraints honored here: ``block_size == 128`` and ``batch == 1``.
head_dim must be in {16,32,64,128} — ``block_sparse_attn`` miscomputes at
head_dim 256, so Qwen3.5 (head_dim 256) routes to the torch reference,
exactly like minference / flexprefill.

Reuses the framework Stem/MInference/FlexPrefill proved: ABC / registry /
patcher / guards / forward templates (qwen3 + qwen3.5 gated). The algorithm
supplies only a ``prefill_fn``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._base import AlgorithmTraits, IncompatibleConfigError, SparsityAlgorithm
from ...registry import SparsityAlgorithmRegistry

if TYPE_CHECKING:
    from transformers import PreTrainedModel


@SparsityAlgorithmRegistry.register("xattention")
class XAttention(SparsityAlgorithm):
    """XAttention antidiagonal-strided block-sparse prefill (Qwen3 / Qwen3.5)."""

    name = "xattention"

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
            "stride": int(cfg.get("stride", 8)),
            "norm": int(cfg.get("norm", 1)),
            "threshold": float(cfg.get("threshold", 0.9)),
            "block_size": int(cfg.get("block_size", 128)),
            "chunk_size": int(cfg.get("chunk_size", 2048)),
            "allow_pseudo_sparse": self.allow_pseudo_sparse,
        }
        self._per_instance_ready = True

    def build_attn_forward(self, attn_module, model: "PreTrainedModel"):
        if not self._per_instance_ready:
            raise IncompatibleConfigError(
                "XAttention.setup(model) must be called before build_attn_forward."
            )
        from .._forward_templates import get_forward_template
        from .._forward_templates._common import repeat_kv
        from .prefill import xattention_prefill

        cfg = self._cfg

        def _prefill_fn(attn, query_states, key_states, value_states):
            # The kernel consumes post-repeat K/V at num_attention_heads.
            k_rep = repeat_kv(key_states, attn.num_key_value_groups)
            v_rep = repeat_kv(value_states, attn.num_key_value_groups)
            return xattention_prefill(
                query_states,
                k_rep,
                v_rep,
                stride=cfg["stride"],
                norm=cfg["norm"],
                threshold=cfg["threshold"],
                block_size=cfg["block_size"],
                chunk_size=cfg["chunk_size"],
                head_dim=attn.head_dim,
                allow_pseudo_sparse=bool(cfg["allow_pseudo_sparse"]),
            )

        model_type = getattr(model.config, "model_type", None)
        template = get_forward_template(model_type)
        attn_module.attn_forward_config = cfg
        forward = template(_prefill_fn, supports_padding_mask=self.traits.supports_padding_mask)
        return forward.__get__(attn_module, type(attn_module))
