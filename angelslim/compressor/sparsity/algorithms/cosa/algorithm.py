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

"""CoSA sparse-prefill algorithm.

CoSA keeps, per query-block, the smallest set of causal key-blocks whose
cumulative attention mass covers ``topp``. Mass is estimated from a
``stride``-subsampled proxy of ``Q @ K^T`` (Triton, ``ops/proxy.py``). The
selection is emitted as block ids **ordered by descending priority**; the
``hpc`` kernel walks that order and skips the tail below ``skipsoftmax_threshold``.

Hard constraints: ``triton`` + internal ``hpc``, bf16, head_dim 128,
``block_size == 128``, ``batch == 1``. No pseudo-sparse fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._base import AlgorithmTraits, IncompatibleConfigError, SparsityAlgorithm
from ...registry import SparsityAlgorithmRegistry

if TYPE_CHECKING:
    from transformers import PreTrainedModel

_DEFAULT_STRIDE = 8
_DEFAULT_TOPP = 0.8
_DEFAULT_BLOCK_SIZE = 128
# hpc anyorderskip: ``< 0`` disables threshold skip; ``>= 0`` enables.
_DEFAULT_SKIPSOFTMAX_THRESHOLD = 1000


@SparsityAlgorithmRegistry.register("cosa")
class CoSA(SparsityAlgorithm):
    """CoSA ordered block-sparse prefill (Qwen3 / Qwen3-MoE / Hunyuan-V3)."""

    name = "cosa"

    @property
    def traits(self) -> AlgorithmTraits:
        return AlgorithmTraits(
            requires_unrepeated_kv=False,  # proxy and kernel both see post-repeat K/V
            needs_calibration=False,
            supports_padding_mask=False,  # block selection never sees the mask
            model_modal="any",
            compatible_model_types=frozenset({"qwen3", "qwen3_moe", "hy_v3"}),
        )

    def setup(self, model: "PreTrainedModel") -> None:
        if self.allow_pseudo_sparse:
            raise IncompatibleConfigError(
                "CoSA has no pseudo-sparse reference path, so "
                "allow_pseudo_sparse=true cannot be honoured. Remove it; the "
                "proxy requires `triton` and the attention requires the `hpc` "
                "extension unconditionally."
            )

        cfg = dict(self.attn_kwargs)
        stride = int(cfg.get("stride", _DEFAULT_STRIDE))
        if stride < 1:
            raise IncompatibleConfigError(f"stride must be >= 1, got {stride}.")

        chunk_size = cfg.get("chunk_size", None)
        self._cfg = {
            "stride": stride,
            "topp": self._resolve_topp(cfg.get("topp", _DEFAULT_TOPP)),
            "block_size": int(cfg.get("block_size", _DEFAULT_BLOCK_SIZE)),
            "chunk_size": None if chunk_size is None else int(chunk_size),
            "skipsoftmax_threshold": float(
                cfg.get("skipsoftmax_threshold", _DEFAULT_SKIPSOFTMAX_THRESHOLD)
            ),
        }
        self._per_instance_ready = True

    @staticmethod
    def _resolve_topp(topp) -> float:
        """Validate ``topp`` as a single scalar in ``(0, 1]``."""
        if isinstance(topp, bool) or not isinstance(topp, (int, float)):
            raise IncompatibleConfigError(
                f"topp must be a scalar float, got {type(topp).__name__}. "
                f"Per-head topp tables are not supported."
            )
        topp = float(topp)
        if not 0.0 < topp <= 1.0:
            raise IncompatibleConfigError(
                f"topp must be in (0, 1], got {topp}. It is a cumulative "
                f"attention-mass coverage target; 1.0 keeps every causal block."
            )
        return topp

    def build_attn_forward(self, attn_module, model: "PreTrainedModel"):
        if not self._per_instance_ready:
            raise IncompatibleConfigError(
                "CoSA.setup(model) must be called before build_attn_forward."
            )
        from .._forward_templates import get_forward_template
        from .._forward_templates._common import repeat_kv
        from .prefill import cosa_prefill

        cfg = self._cfg

        def _prefill_fn(attn, query_states, key_states, value_states):
            # The kernel consumes post-repeat K/V at num_attention_heads.
            return cosa_prefill(
                query_states,
                repeat_kv(key_states, attn.num_key_value_groups),
                repeat_kv(value_states, attn.num_key_value_groups),
                stride=cfg["stride"],
                topp=cfg["topp"],
                block_size=cfg["block_size"],
                chunk_size=cfg["chunk_size"],
                skipsoftmax_threshold=cfg["skipsoftmax_threshold"],
            )

        model_type = getattr(model.config, "model_type", None)
        template = get_forward_template(model_type)
        attn_module.attn_forward_config = cfg
        forward = template(_prefill_fn, supports_padding_mask=self.traits.supports_padding_mask)
        return forward.__get__(attn_module, type(attn_module))
