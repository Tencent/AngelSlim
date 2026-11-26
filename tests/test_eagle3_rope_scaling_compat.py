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

"""transformers >= 5 RoPE config compatibility for Eagle3 inference.

transformers 5 renames ``rope_scaling["type"]`` to ``rope_type`` and may put
additional keys (``rope_theta``, ``mrope_section``, ...) in the same dict.
The Eagle3 drafter and target attention used to index ``["type"]`` and
``["factor"]`` directly, so a config saved by the current transformers raised
``KeyError: 'type'`` during benchmark/inference (issue #330).

These tests are CPU-only and do not load model weights.
"""

from types import SimpleNamespace

import pytest

from angelslim.compressor.speculative.inference.models.eagle3.configuration_eagle3_model import (
    Eagle3Config,
)
from angelslim.compressor.speculative.inference.models.eagle3.draft.llama3_eagle3 import (
    LlamaAttention as DrafterAttention,
)
from angelslim.compressor.speculative.inference.models.eagle3.draft.llama3_eagle3 import (
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaRotaryEmbedding,
)
from angelslim.compressor.speculative.inference.models.eagle3.target.modeling_llama_kv import (
    LlamaAttention as TargetAttention,
)
from angelslim.compressor.speculative.inference.models.eagle3.target.modeling_llama_kv import (
    LlamaLinearScalingRotaryEmbedding as TargetLinearScalingRotaryEmbedding,
)
from angelslim.compressor.speculative.inference.models.eagle3.target.modeling_llama_kv import (
    LlamaRotaryEmbedding as TargetRotaryEmbedding,
)

ROPE_THETA = 12345.0


def _drafter_config(rope_scaling):
    return SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        head_dim=8,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_scaling=rope_scaling,
        rope_theta=ROPE_THETA,
    )


def _target_config(rope_scaling):
    return SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        pretraining_tp=1,
        max_position_embeddings=64,
        rope_scaling=rope_scaling,
        rope_theta=ROPE_THETA,
    )


def test_eagle3_config_accepts_transformers5_rope_scaling_dict():
    """``rope_type`` plus extra keys must not fail Eagle3Config validation."""
    config = Eagle3Config(
        rope_scaling={"rope_type": "linear", "factor": 2.0, "rope_theta": ROPE_THETA}
    )

    assert config.rope_scaling["rope_type"] == "linear"


def test_eagle3_config_accepts_default_rope_scaling():
    """transformers 5 may materialize even default RoPE as a dict."""
    config = Eagle3Config(rope_scaling={"rope_type": "default", "rope_theta": ROPE_THETA})

    assert config.rope_scaling["rope_type"] == "default"


def test_eagle3_config_keeps_legacy_type_key_working():
    config = Eagle3Config(rope_scaling={"type": "linear", "factor": 2.0})

    assert config.rope_scaling["type"] == "linear"


def test_eagle3_config_rejects_unknown_rope_scaling_type():
    with pytest.raises(ValueError, match="name field"):
        Eagle3Config(rope_scaling={"rope_type": "yarn", "factor": 2.0})


def test_drafter_attention_reads_transformers5_linear_scaling():
    attention = DrafterAttention(_drafter_config({"rope_type": "linear", "factor": 2.0}))

    assert isinstance(attention.rotary_emb, LlamaLinearScalingRotaryEmbedding)
    assert attention.rotary_emb.scaling_factor == 2.0
    assert attention.rotary_emb.base == ROPE_THETA


def test_drafter_attention_reads_transformers5_default_scaling():
    attention = DrafterAttention(_drafter_config({"rope_type": "default"}))

    assert isinstance(attention.rotary_emb, LlamaRotaryEmbedding)
    assert attention.rotary_emb.base == ROPE_THETA


def test_drafter_attention_reads_legacy_dynamic_scaling():
    attention = DrafterAttention(_drafter_config({"type": "dynamic", "factor": 2.0}))

    assert isinstance(attention.rotary_emb, LlamaDynamicNTKScalingRotaryEmbedding)
    assert attention.rotary_emb.scaling_factor == 2.0
    assert attention.rotary_emb.base == ROPE_THETA


def test_target_attention_reads_transformers5_default_scaling():
    """A default RoPE dict must select classic RoPE, not the LLaMA-31 fallback."""
    attention = TargetAttention(_target_config({"rope_type": "default"}))

    assert isinstance(attention.rotary_emb, TargetRotaryEmbedding)
    assert attention.rotary_emb.base == ROPE_THETA


def test_target_attention_reads_transformers5_linear_scaling():
    attention = TargetAttention(_target_config({"rope_type": "linear", "factor": 2.0}))

    assert isinstance(attention.rotary_emb, TargetLinearScalingRotaryEmbedding)
    assert attention.rotary_emb.scaling_factor == 2.0
    assert attention.rotary_emb.base == ROPE_THETA
