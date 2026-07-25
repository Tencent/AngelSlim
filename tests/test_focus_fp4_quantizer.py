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

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from angelslim.compressor.qat.modules.quantizer import Quantizer, QuantLinear
from angelslim.compressor.qat.plugins import learnable_scale


def _quant_info(group_size=16):
    return SimpleNamespace(
        quant_algo="w4a8_fp8",
        quant_algo_info={
            "w": "int4_per-group",
            "a": "fp8_per-token-dynamic",
            "w_group_size": group_size,
        },
    )


def _nvfp4_config(group_size=16, *, use_subgroup=False, num_sub=2):
    return {
        "weight": {
            "qtype": "nvfp4",
            "granularity": "per-group",
            "group_size": group_size,
            "is_sym": True,
            "dynamic": False,
            "use_subgroup_scale": use_subgroup,
            "num_sub": num_sub,
        },
        "activation": {
            "qtype": "nvfp4",
            "granularity": "per-group",
            "group_size": group_size,
            "is_sym": True,
            "dynamic": True,
        },
    }


def test_nvfp4_weight_scale2_initialization_uses_gathered_weight(monkeypatch):
    quant_linear = QuantLinear(
        torch.nn.Linear(16, 2, bias=False),
        _nvfp4_config(),
        _quant_info(),
        use_weight_quant=True,
        use_act_quant=False,
    )
    local_weight = torch.full_like(quant_linear.weight, 0.25)
    gathered_weight = torch.linspace(-12.0, 9.0, quant_linear.weight.numel()).reshape_as(
        quant_linear.weight
    )
    with torch.no_grad():
        quant_linear.weight.copy_(local_weight)
        quant_linear.weight_quantizer.scale_2.fill_(1.0)
    quant_linear.weight_quantizer.init = False
    events = []

    @contextmanager
    def fake_gather(param, modifier_rank=None):
        original = param.detach().clone()
        events.append(modifier_rank)
        with torch.no_grad():
            param.copy_(gathered_weight)
        try:
            yield param
        finally:
            with torch.no_grad():
                param.copy_(original)

    monkeypatch.setattr(learnable_scale, "gathered_param_if_zero3", fake_gather)
    learnable_scale.initialize_nvfp4_weight_scale_2(torch.nn.Sequential(quant_linear))

    expected = gathered_weight.abs().amax() / (6.0 * 448.0)
    assert events == [None]
    assert quant_linear.weight_quantizer.init
    torch.testing.assert_close(
        quant_linear.weight_quantizer.scale_2,
        expected.reshape(1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(quant_linear.weight, local_weight)


def test_nvfp4_weight_scale2_initialization_rejects_nonfinite_weight():
    quant_linear = QuantLinear(
        torch.nn.Linear(16, 2, bias=False),
        _nvfp4_config(),
        _quant_info(),
        use_weight_quant=True,
        use_act_quant=False,
    )
    with torch.no_grad():
        quant_linear.weight[0, 0] = torch.nan
    quant_linear.weight_quantizer.init = False

    with pytest.raises(ValueError, match="Non-finite NVFP4 weight amax"):
        learnable_scale.initialize_nvfp4_weight_scale_2(torch.nn.Sequential(quant_linear))


def test_nvfp4_subgroup_requires_even_parent_partition():
    weight = torch.ones(2, 16)
    with pytest.raises(ValueError, match="must be divisible by num_sub"):
        Quantizer(
            _nvfp4_config(use_subgroup=True, num_sub=3),
            _quant_info(),
            x=weight,
            weight_shape=weight.shape,
        )
