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

import importlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Seq2SeqTrainer

from angelslim.compressor.qat.modules.quantizer import QuantLinear
from angelslim.compressor.qat.plugins.learnable_scale import LearnableScalePlugin
from angelslim.compressor.qat.trainers.end2end_trainer import (
    End2EndTrainer,
    QATSeq2SeqTrainer,
)
from angelslim.utils.config_parser import SlimConfigParser

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _focus_quant_info():
    return SimpleNamespace(
        quant_algo="w4a8_fp8",
        quant_algo_info={
            "w": "int4_per-group",
            "a": "fp8_per-token-dynamic",
            "w_group_size": 4,
        },
    )


def _focus_quant_config():
    return {
        "use_weight_quant": True,
        "use_activation_quant": True,
        "learnable": {
            "act_scale": False,
            "weight_scale": True,
            "quant_max_scale_lr": 0.02,
        },
        "weight": {
            "qtype": "mxfp4",
            "granularity": "per-group",
            "group_size": 4,
            "dynamic": False,
            "use_subgroup_scale": True,
            "num_sub": 2,
        },
        "activation": {
            "qtype": "mxfp4",
            "granularity": "per-group",
            "group_size": 4,
            "dynamic": True,
        },
    }


def _focus_eval_quant_config(qtype):
    return {
        "use_weight_quant": True,
        "use_activation_quant": True,
        "learnable": {
            "act_scale": False,
            "weight_scale": True,
        },
        "weight": {
            "qtype": qtype,
            "granularity": "per-group",
            "group_size": 4,
            "is_sym": True,
            "dynamic": False,
        },
        "activation": {
            "qtype": qtype,
            "granularity": "per-group",
            "group_size": 4,
            "is_sym": True,
            "dynamic": True,
        },
    }


@pytest.mark.parametrize("weight_qtype", ["mxfp4", "nvfp4"])
def test_focus_zero3_does_not_require_ptq_warmstart(weight_qtype):
    plugin = LearnableScalePlugin.__new__(LearnableScalePlugin)
    plugin.config = {
        "weight": {"qtype": weight_qtype},
        "activation": {"dynamic": True},
    }
    plugin.use_weight_quant = True
    plugin.use_activation_quant = True

    assert plugin._zero3_needs_no_warmstart()


def test_static_activation_still_requires_ptq_warmstart():
    plugin = LearnableScalePlugin.__new__(LearnableScalePlugin)
    plugin.config = {
        "weight": {"qtype": "mxfp4"},
        "activation": {"dynamic": False},
    }
    plugin.use_weight_quant = True
    plugin.use_activation_quant = True

    assert not plugin._zero3_needs_no_warmstart()


def test_focus_learn_strategy_freezes_weight_and_enables_only_weight_scales():
    linear = torch.nn.Linear(8, 2, bias=False)
    quant_linear = QuantLinear(
        linear,
        _focus_quant_config(),
        _focus_quant_info(),
        use_weight_quant=True,
        use_act_quant=True,
    )
    model = torch.nn.Sequential(quant_linear)

    plugin = LearnableScalePlugin.__new__(LearnableScalePlugin)
    plugin.quant_model = SimpleNamespace(model=model)
    plugin.learn_act_scale = False
    plugin.learn_weight_scale = True
    plugin.learn_kv_scale = False
    plugin.learn_lwc = False
    plugin.learn_norm = False
    plugin._apply_learn_strategy()

    assert not quant_linear.weight.requires_grad
    assert quant_linear.weight_quantizer.max_scale.requires_grad
    assert quant_linear.weight_quantizer.quant_max_scale.requires_grad
    assert not quant_linear.act_quantizer.max_scale.requires_grad


class _ScaleOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_scale = torch.nn.Parameter(torch.ones(2))
        self.quant_max_scale = torch.nn.Parameter(torch.ones(3))
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)


def test_focus_optimizer_keeps_quant_max_scale_learning_rate():
    qat_config = SimpleNamespace(
        hf_args={"learning_rate": 0.01, "weight_decay": 0.1},
        plugin_config={
            "quant_config": {
                "learnable": {"quant_max_scale_lr": 0.02},
                "lwc": {"enable_lwc": False},
            }
        },
    )
    trainer = End2EndTrainer.__new__(End2EndTrainer)
    trainer.quant_model = SimpleNamespace(model=_ScaleOnlyModel())
    trainer.config = {"compress_config": SimpleNamespace(QAT=qat_config)}
    trainer._init_optimizer()

    assert [group["lr"] for group in trainer.optimizer.param_groups] == [0.01, 0.02]
    assert len(trainer.optimizer.param_groups[0]["params"]) == 1
    assert trainer.optimizer.param_groups[0]["params"][0] is trainer.quant_model.model.max_scale
    assert len(trainer.optimizer.param_groups[1]["params"]) == 1
    assert (
        trainer.optimizer.param_groups[1]["params"][0] is trainer.quant_model.model.quant_max_scale
    )


def test_qat_trainer_reuses_prebuilt_optimizer():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    trainer = QATSeq2SeqTrainer.__new__(QATSeq2SeqTrainer)
    trainer._qat_prebuilt_optimizer = optimizer

    assert QATSeq2SeqTrainer.create_optimizer(trainer) is optimizer
    assert trainer.optimizer is optimizer


def test_qat_trainer_avoids_double_gradient_accumulation_scaling(monkeypatch):
    def fake_init(self, *args, **kwargs):
        self.optimizer = None

    monkeypatch.setattr(Seq2SeqTrainer, "__init__", fake_init)
    trainer = QATSeq2SeqTrainer()
    assert trainer.model_accepts_loss_kwargs is True


@pytest.mark.parametrize(
    ("config_name", "expected_qtype", "expected_save_format"),
    [
        ("qwen3-4b_focus_mxfp4_w4a4_zero3.yaml", "mxfp4_rceil", "fake"),
        ("qwen3-4b_focus_mxfp4_w4a4_real_zero3.yaml", "mxfp4_rceil", "real"),
        ("qwen3-4b_focus_nvfp4_w4a4_zero3.yaml", "nvfp4", "fake"),
        ("qwen3-4b_focus_nvfp4_w4a4_real_zero3.yaml", "nvfp4", "real"),
    ],
)
def test_focus_qwen3_config_parses(config_name, expected_qtype, expected_save_format):
    config = SlimConfigParser().parse(str(_REPO_ROOT / "configs/qwen3/fp4" / config_name))
    qat = config.compression_config.QAT
    quant_config = qat.plugin_config["quant_config"]

    assert config.model_config.model_path == "Qwen/Qwen3-4B"
    assert config.model_config.device_map == "None"
    assert qat.from_ptq_ckpt is None
    assert qat.loss_topk == 1000
    assert quant_config["weight"]["qtype"] == expected_qtype
    assert quant_config["activation"]["dynamic"] is True
    assert quant_config["learnable"]["act_scale"] is False
    expected_num_sub = 4 if "mxfp4" in expected_qtype else 2
    expected_quant_lr = 5.0e-2 if "mxfp4" in expected_qtype else 1.0e-3
    assert quant_config["weight"]["use_subgroup_scale"] is True
    assert quant_config["weight"]["num_sub"] == expected_num_sub
    assert quant_config["learnable"]["quant_max_scale_lr"] == expected_quant_lr
    assert qat.save_format == expected_save_format
    assert qat.hf_args["deepspeed"] == "configs/qwen3/fp4/ds_config_zero3_focus.json"


def test_focus_zero3_disables_mixed_dtype_parameter_persistence():
    config_path = _REPO_ROOT / "configs/qwen3/fp4/ds_config_zero3_focus.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["zero_optimization"]["stage3_param_persistence_threshold"] == 0


def test_zero3_shard_loader_resolves_hub_model_id(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    zero3_io = importlib.import_module("angelslim.utils.zero3_io")
    snapshot_dir = tmp_path / "hub-snapshot"
    snapshot_dir.mkdir()
    shard_path = snapshot_dir / "model.safetensors"
    save_file({"model.weight": torch.ones(2, 2)}, shard_path)

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot_dir)

    hub = importlib.import_module("huggingface_hub")
    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)

    shards = list(zero3_io.iter_safetensors_shards("Qwen/Qwen3-4B"))

    assert calls == [
        {
            "repo_id": "Qwen/Qwen3-4B",
            "allow_patterns": ["*.safetensors", "*.safetensors.index.json"],
        }
    ]
    assert shards == [(str(shard_path), ["model.weight"])]


def test_quant_inplace_commits_modified_zero3_weight(monkeypatch):
    plugin_module = importlib.import_module("angelslim.compressor.qat.plugins.learnable_scale")
    events = []

    @contextmanager
    def fake_gathered_params(params, modifier_rank=None):
        params = list(params)
        snapshots = [param.detach().clone() for param in params]
        events.append(("quantizer", modifier_rank))
        yield params
        if modifier_rank is None:
            for param, snapshot in zip(params, snapshots):
                param.copy_(snapshot)

    @contextmanager
    def fake_gathered_weight(param, modifier_rank=None):
        snapshot = param.detach().clone()
        events.append(("weight", modifier_rank))
        yield param
        if modifier_rank is None:
            param.copy_(snapshot)

    monkeypatch.setattr(plugin_module, "gathered_params_if_zero3", fake_gathered_params)
    monkeypatch.setattr(plugin_module, "gathered_param_if_zero3", fake_gathered_weight)

    linear = torch.nn.Linear(8, 2, bias=False)
    quant_linear = QuantLinear(
        linear,
        _focus_quant_config(),
        _focus_quant_info(),
        use_weight_quant=True,
        use_act_quant=False,
    )
    values = torch.tensor(
        [
            [-5.5, -2.75, -1.25, -0.3, 0.2, 0.9, 2.4, 5.1],
            [-4.2, -1.7, -0.6, -0.1, 0.4, 1.3, 3.2, 4.7],
        ]
    )
    with torch.no_grad():
        quant_linear.weight.copy_(values)
        expected = quant_linear.weight_quantizer(quant_linear.weight).clone()

    plugin_module.quant_inplace(torch.nn.Sequential(quant_linear))

    assert events == [("quantizer", None), ("weight", 0)]
    assert not torch.equal(expected, values)
    torch.testing.assert_close(quant_linear.weight, expected)


@pytest.mark.parametrize("qtype", ["mxfp4", "nvfp4"])
def test_eval_only_fake_resume_preserves_baked_weights_and_logits(qtype):
    torch.manual_seed(7)
    inputs = torch.randn(2, 8)

    source = QuantLinear(
        torch.nn.Linear(8, 2, bias=False),
        _focus_eval_quant_config(qtype),
        _focus_quant_info(),
        use_weight_quant=True,
        use_act_quant=True,
        resume=True,
    )
    # Match a trained model: NVFP4 dynamic activation scale has already been
    # initialized by at least one forward before the fake checkpoint is saved.
    source(inputs)
    source_plugin = LearnableScalePlugin.__new__(LearnableScalePlugin)
    source_plugin.quant_model = SimpleNamespace(model=source)
    source_plugin.use_weight_quant = True
    source_plugin.use_activation_quant = True
    source_plugin.after_train()

    assert not source.use_weight_quant
    assert source.use_act_quant
    expected_weight = source.weight.detach().clone()
    expected_logits = source(inputs).detach().clone()
    state_dict = {name: value.detach().clone() for name, value in source.state_dict().items()}

    restored = QuantLinear(
        torch.nn.Linear(8, 2, bias=False),
        _focus_eval_quant_config(qtype),
        _focus_quant_info(),
        use_weight_quant=True,
        use_act_quant=True,
        resume=True,
    )
    restored.load_state_dict(state_dict)
    restored_plugin = LearnableScalePlugin.__new__(LearnableScalePlugin)
    restored_plugin.quant_model = SimpleNamespace(model=restored)
    restored_plugin.use_weight_quant = True
    restored_plugin.use_activation_quant = True

    scale_2_before = None
    if qtype == "nvfp4":
        assert not restored.act_quantizer.init
        scale_2_before = restored.act_quantizer.scale_2.detach().clone()

    restored_plugin.after_train(skip_weight_bake=True)

    assert not restored.use_weight_quant
    assert restored.use_act_quant
    torch.testing.assert_close(restored.weight, expected_weight)
    if qtype == "nvfp4":
        assert restored.weight_quantizer.init
        assert restored.act_quantizer.init

    actual_logits = restored(inputs)
    torch.testing.assert_close(actual_logits, expected_logits)
    if scale_2_before is not None:
        torch.testing.assert_close(restored.act_quantizer.scale_2, scale_2_before)


def test_focus_eval_rejects_missing_activation_quantizer():
    quant_linear = QuantLinear(
        torch.nn.Linear(8, 2, bias=False),
        _focus_eval_quant_config("mxfp4"),
        _focus_quant_info(),
        use_weight_quant=True,
        use_act_quant=False,
        resume=True,
    )
    plugin = LearnableScalePlugin.__new__(LearnableScalePlugin)
    plugin.quant_model = SimpleNamespace(model=quant_linear)
    plugin.use_weight_quant = True
    plugin.use_activation_quant = True

    with pytest.raises(RuntimeError, match="Activation quantization must remain enabled"):
        plugin.after_train(skip_weight_bake=True)


def test_fake_save_uses_consolidated_state_for_zero3(tmp_path, monkeypatch):
    qat_module = importlib.import_module("angelslim.compressor.qat.qat")
    model = torch.nn.Linear(2, 2)
    qat = qat_module.QAT.__new__(qat_module.QAT)
    qat.save_fmt = "fake"
    qat.quant_model = SimpleNamespace(model=model)
    qat.trainer = SimpleNamespace(external_trainer=SimpleNamespace(model=model))

    expected = {"weight": torch.full((2, 2), 7.0)}
    monkeypatch.setattr(qat_module, "model_has_zero3_params", lambda _: True)
    monkeypatch.setattr(qat_module, "consolidated_state_dict", lambda _: expected)

    qat.save(str(tmp_path / "focus"))
    saved = torch.load(tmp_path / "focus_fake_quant_model.pt", map_location="cpu")
    torch.testing.assert_close(saved["weight"], expected["weight"])


def test_resume_allows_missing_tied_lm_head(tmp_path):
    checkpoint = tmp_path / "focus.pt"
    torch.save({"model.embed_tokens.weight": torch.ones(1)}, checkpoint)

    class _TiedModel:
        def __init__(self):
            self.retied = False

        def load_state_dict(self, state_dict, strict):
            assert not strict
            assert "model.embed_tokens.weight" in state_dict
            return SimpleNamespace(missing_keys=["lm_head.weight"], unexpected_keys=[])

        def tie_weights(self):
            self.retied = True

    class _PluginManager:
        def __init__(self):
            self.after_kwargs = None

        def call_before_train(self, **kwargs):
            pass

        def call_after_train(self, **kwargs):
            self.after_kwargs = kwargs

    model = _TiedModel()
    trainer = End2EndTrainer.__new__(End2EndTrainer)
    trainer.quant_model = SimpleNamespace(model=model)
    plugin_manager = _PluginManager()
    trainer.plugin_manager = plugin_manager
    trainer.resume_ckpt_dir = str(checkpoint)
    trainer.do_train = False
    trainer.prepare_dataset = lambda _: setattr(trainer, "train_dataset", [])
    trainer.prepare_trainer = lambda: None

    trainer.run(dataloader=None)
    assert model.retied
    assert plugin_manager.after_kwargs == {"skip_weight_bake": True}
