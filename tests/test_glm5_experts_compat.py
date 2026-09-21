# Copyright 2025 Tencent Inc.  All rights reserved.
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

"""GLM5 adapter must import and wrap the fused experts module on every
transformers release in the supported range: the class was renamed from
``GlmMoeDsaNaiveMoe`` to ``GlmMoeDsaExperts`` in transformers 5.13."""

import importlib

import pytest
import torch
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa

from angelslim.models.llm import glm5_1


def _fused_experts_class():
    return getattr(
        modeling_glm_moe_dsa,
        "GlmMoeDsaNaiveMoe",
        getattr(modeling_glm_moe_dsa, "GlmMoeDsaExperts", None),
    )


def test_glm5_module_imports_and_resolves_fused_experts_class():
    assert importlib.import_module("angelslim.models.llm") is not None
    assert glm5_1.GlmMoeDsaNaiveMoe is _fused_experts_class()


def test_experts_with_linear_matches_fused_experts_forward():
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )

    torch.manual_seed(0)
    config = GlmMoeDsaConfig(
        hidden_size=8, moe_intermediate_size=6, num_local_experts=4, hidden_act="silu"
    )
    fused = _fused_experts_class()(config)
    with torch.no_grad():
        fused.gate_up_proj.normal_()
        fused.down_proj.normal_()
    assert glm5_1._is_glm_naive_moe(fused)

    linearized = glm5_1.GlmExpertsWithLinear(fused)

    tokens, top_k = 5, 2
    hidden_states = torch.randn(tokens, config.hidden_size)
    top_k_index = torch.stack(
        [torch.randperm(config.num_local_experts)[:top_k] for _ in range(tokens)]
    )
    top_k_weights = torch.rand(tokens, top_k)
    with torch.no_grad():
        expected = fused(hidden_states, top_k_index, top_k_weights)
        actual = linearized(hidden_states, top_k_index, top_k_weights)
    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    pytest.main([__file__])
