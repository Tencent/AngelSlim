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

"""Unit tests for vllm_calib activation / KV / MoE hooks, with vLLM replaced by stubs."""

import sys
import types

import pytest
import torch

from angelslim.compressor.quant.vllm_calib import hooks, kvcache


class _FakeAttention(torch.nn.Module):
    def forward(self, x):
        return x


class _FakeLinearBase(torch.nn.Linear):
    pass


class _FakeFusedMoE(torch.nn.Module):
    def __init__(self, num_experts=2):
        super().__init__()
        self.global_num_experts = num_experts
        self.w13_weight = types.SimpleNamespace(dtype=torch.float16, shape=(num_experts, 4, 2))
        self.w2_weight = types.SimpleNamespace(dtype=torch.float16)

    def forward(self, x):
        return x


class _FakeTritonExperts:
    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
              global_num_experts, *args):  # fmt: skip
        return triton_moe_stub.moe_kernel_quantize_input(hidden_states * 2)[0]


triton_moe_stub = types.ModuleType("vllm.model_executor.layers.fused_moe.experts.triton_moe")
triton_moe_stub.TritonExperts = _FakeTritonExperts
triton_moe_stub.moe_kernel_quantize_input = lambda a, *args, **kwargs: (a, None)


class _ToyModel(torch.nn.Module):
    def __init__(self, *, with_attention=False, with_moe=True):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        if with_attention:
            self.attn = _FakeAttention()
        if with_moe:
            self.moe = _FakeFusedMoE(num_experts=2)


@pytest.fixture(autouse=True)
def vllm_stubs(monkeypatch):
    def stub(name, **attrs):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)

    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers"):
        stub(name)
    stub("vllm.model_executor.layers.fused_moe")
    stub("vllm.model_executor.layers.fused_moe.layer", FusedMoE=_FakeFusedMoE)
    stub("vllm.model_executor.layers.linear", LinearBase=_FakeLinearBase)
    stub("vllm.model_executor.layers.attention", Attention=_FakeAttention)
    stub("vllm.model_executor.layers.fused_moe.experts", triton_moe=triton_moe_stub)
    monkeypatch.setitem(sys.modules, triton_moe_stub.__name__, triton_moe_stub)
    monkeypatch.setattr(hooks, "_is_fp8_model", lambda *_: False)
    monkeypatch.setattr(hooks, "patch_image_embeds", lambda _m: "embeds off")


def test_setup_activation_hooks_registers_linear_and_moe_stats():
    model = _ToyModel(with_attention=False, with_moe=True)
    hooks.setup_activation_hooks(model, kv_granularity="none")

    assert model._angelslim_fp8_scale_only is False
    assert len(model._activation_hooks) == 1
    assert "linear" in model._activation_stats
    assert "moe.gate_up_proj" in model._moe_activation_stats
    assert "moe.down_proj" in model._moe_activation_stats
    assert model.moe.w13_weight._vllm_layer_name == "moe"
    assert model.moe.w13_weight._moe_activation_stats_of_model is model._moe_activation_stats
    assert model.moe.w13_weight._moe_scale_only is False


def test_setup_activation_hooks_per_tensor_kv():
    model = _ToyModel(with_attention=True, with_moe=False)
    hooks.setup_activation_hooks(model, kv_granularity="per-tensor")

    assert len(model._activation_hooks) == 1
    assert len(model._kvcache_hooks) == 1
    for stats in (model._kcache_stats["attn"], model._vcache_stats["attn"]):
        assert stats["min"].item() == float("inf")
        assert stats["max"].item() == float("-inf")

    hooks.setup_activation_hooks(model, kv_granularity="per-tensor")
    assert len(model._kvcache_hooks) == 1


def test_setup_activation_hooks_per_head_kv(monkeypatch):
    monkeypatch.setattr(kvcache, "infer_num_kv_heads_total", lambda _m: 8)
    model = _ToyModel(with_attention=True, with_moe=False)
    hooks.setup_activation_hooks(model, kv_granularity="per-head")

    assert model._kvcache_num_kv_heads_total == 8
    assert len(model._kvcache_perhead_hooks) == 1
    for side in ("k", "v"):
        stats = model._kvcache_perhead_stats[side]["attn"]
        assert stats["min"].item() == float("inf")
        assert stats["max"].item() == float("-inf")

    hooks.setup_activation_hooks(model, kv_granularity="per-head")
    assert len(model._kvcache_perhead_hooks) == 1


def test_get_activation_stats_reduces_and_converts(monkeypatch):
    calls = []
    monkeypatch.setattr(hooks, "all_reduce_minmax", lambda stats, label: calls.append(label))
    model = types.SimpleNamespace(
        _activation_stats={"linear": {"min": torch.tensor(-1.25), "max": torch.tensor(2.5)}},
        _kcache_stats={"attn": {"min": torch.tensor(-0.5), "max": torch.tensor(0.7)}},
        _vcache_stats={"attn": {"min": torch.tensor(-0.3), "max": torch.tensor(0.9)}},
    )

    got = hooks.get_activation_stats(model)

    assert got["linear"] == {"min": pytest.approx(-1.25), "max": pytest.approx(2.5)}
    assert got["attn.k_cache"] == {"min": pytest.approx(-0.5), "max": pytest.approx(0.7)}
    assert got["attn.v_cache"] == {"min": pytest.approx(-0.3), "max": pytest.approx(0.9)}
    assert len(calls) == 3


def test_get_moe_stats_reduces_and_returns_scalars(monkeypatch):
    calls = []
    monkeypatch.setattr(hooks, "all_reduce_minmax", lambda stats, label: calls.append(label))
    model = types.SimpleNamespace(
        _moe_activation_stats={
            "l.gate_up_proj": {"min": torch.tensor(-3.0), "max": torch.tensor(8.0)}
        }
    )

    got = hooks.get_moe_stats(model)

    assert calls == ["MoE"]
    assert got["l.gate_up_proj"] == {"min": pytest.approx(-3.0), "max": pytest.approx(8.0)}


def test_setup_mtp_activation_hooks_registers_linear_kv_and_moe():
    draft_model = _ToyModel(with_attention=True, with_moe=True)
    hooks.setup_mtp_activation_hooks(draft_model)

    assert len(draft_model._activation_hooks) == 1
    assert len(draft_model._kvcache_hooks) == 1
    assert "attn" in draft_model._kcache_stats
    assert "moe.gate_up_proj" in draft_model._moe_activation_stats
    assert "moe.down_proj" in draft_model._moe_activation_stats


def test_cli_worker_callables_are_not_defined_in_the_entry_point():
    """Hooks shipped via collective_rpc are pickled by qualified name. Defined in the module
    that `-m` runs, they resolve to __main__ and the worker cannot unpickle them, so the CLI
    must import them from a library module."""
    import ast
    import os

    from angelslim.compressor.quant import vllm_calib

    pkg = os.path.dirname(vllm_calib.__file__)
    for fname, names in (
        ("hessian_collect.py",
         ("setup_hessian_hooks", "setup_hessian_hooks_on_draft", "dump_hessian_stats")),
        ("calibrate.py",
         ("setup_activation_hooks", "get_activation_stats", "get_moe_stats")),
    ):  # fmt: skip
        with open(os.path.join(pkg, fname), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in names:
            assert name not in defined, f"{name} must be imported by {fname}, not defined in it"


def test_moe_hook_feeds_both_stages_to_the_stats(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_COLLECT_STATS", "1")
    model = torch.nn.Module()
    model.experts = _FakeFusedMoE()  # collectors only record layers named "*experts*"
    hooks.setup_activation_hooks(model, kv_granularity="none")

    w1 = model.experts.w13_weight
    x = torch.tensor([[1.0, -2.0], [3.0, 0.5]])
    topk = torch.tensor([[0], [1]])
    _FakeTritonExperts().apply(None, x, w1, None, None, topk, None, -1)

    stats = model._moe_activation_stats
    assert stats["experts.gate_up_proj"]["max"].item() == 3.0
    assert stats["experts.down_proj"]["min"].item() == -4.0
    assert hooks._ctx.call is None


def test_moe_hook_fp8_scale_only_records_scale_amax(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_COLLECT_STATS", "1")
    monkeypatch.setattr(hooks, "_is_fp8_model", lambda *_: True)
    scale = torch.tensor([0.5])
    quant = lambda a, *_, **__: (a, scale)  # noqa: E731
    monkeypatch.setattr(triton_moe_stub, "moe_kernel_quantize_input", quant)
    hooks._installed = False
    model = torch.nn.Module()
    model.experts = _FakeFusedMoE()
    hooks.setup_activation_hooks(model, kv_granularity="none")

    w1 = model.experts.w13_weight
    x = torch.tensor([[1.0, -2.0], [3.0, 0.5]])
    _FakeTritonExperts().apply(None, x, w1, None, None, torch.tensor([[0], [1]]), None, -1,
                               None, torch.tensor([0.25]))  # fmt: skip

    stats = model._moe_activation_stats
    assert "experts.gate_up_proj" not in stats  # no BF16 keys in scale-only mode
    assert stats["experts.gate_up_proj.scale_amax"]["bf16_amax_recovered"].item() == 0.25 * 448
    assert stats["experts.down_proj.scale_amax"]["bf16_amax_recovered"].item() == 0.5 * 448


def test_smooth_moe_observer_collects_down_proj(monkeypatch):
    from angelslim.compressor.transform.smooth.vllm import moe_inject

    monkeypatch.setattr(moe_inject, "_MOE_COLLECT_SMOOTH", True)
    stats = {}
    w1 = types.SimpleNamespace(
        shape=(2, 4, 2), _vllm_layer_name_smooth="experts", _smooth_stats_of_model=stats,
        _smooth_ema_momentum=0.9,
    )  # fmt: skip
    monkeypatch.setattr(hooks, "_observers", [])
    hooks.install()
    hooks.register_observer(moe_inject.smooth_moe_observer)
    x = torch.tensor([[1.0, -2.0], [3.0, 0.5]])
    _FakeTritonExperts().apply(None, x, w1, None, None, torch.tensor([[0], [1]]), None, -1)

    assert torch.equal(stats["experts.0.down_proj"]["absmax"], torch.tensor([2.0, 4.0]))
    assert torch.equal(stats["experts.1.down_proj"]["absmax"], torch.tensor([6.0, 1.0]))
