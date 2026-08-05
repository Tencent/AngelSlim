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

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from angelslim.compressor.qat.export import export_focus_mxfp4_checkpoint
from angelslim.compressor.qat.modules.quantizer import Quantizer, QuantLinear
from angelslim.compressor.qat.qat import QAT
from angelslim.compressor.qat.trainers.end2end_trainer import End2EndTrainer
from angelslim.compressor.quant.modules.mxfp4 import (
    decode_e8m0_scale,
    encode_e8m0_scale,
    mxfp4_cast_to_e2m1,
    mxfp4_quantize_pack,
    mxfp4_unpack_dequantize,
)


def _quant_info():
    return SimpleNamespace(
        quant_algo="w4a8_fp8",
        quant_algo_info={
            "w": "int4_per-group",
            "a": "fp8_per-token-dynamic",
            "w_group_size": 32,
        },
    )


def _mxfp4_config():
    return {
        "weight": {
            "qtype": "mxfp4_rceil",
            "granularity": "per-group",
            "group_size": 32,
            "is_sym": True,
            "dynamic": False,
            "mantissa_rounding": "even",
        }
    }


def test_e2m1_ties_use_round_half_to_even():
    values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    assert mxfp4_cast_to_e2m1(values).tolist() == [0, 2, 2, 4, 4, 6, 6]


def test_e8m0_bit_pattern_round_trip():
    scale = torch.tensor([2.0**-127, 0.5, 1.0, 2.0, 2.0**127])
    encoded = encode_e8m0_scale(scale)
    assert encoded.tolist() == [0, 126, 127, 128, 254]
    torch.testing.assert_close(decode_e8m0_scale(encoded), scale, rtol=0, atol=0)


def test_packed_dequant_matches_focus_fake_quantizer():
    generator = torch.Generator().manual_seed(20260722)
    weight = torch.randn(3, 64, generator=generator, dtype=torch.float32) * 1.7
    quantizer = Quantizer(
        _mxfp4_config(),
        _quant_info(),
        x=weight,
        weight_shape=weight.shape,
    )
    learned_scale = torch.tensor(
        [[0.75, 1.25], [1.0, 0.625], [1.5, 0.875]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        quantizer.max_scale.copy_(learned_scale)

    expected = quantizer(weight)
    packed, encoded_scale = mxfp4_quantize_pack(weight, quantizer.max_scale)
    restored = mxfp4_unpack_dequantize(packed, encoded_scale)

    assert packed.shape == (3, 32)
    assert packed.dtype == torch.uint8
    assert encoded_scale.shape == (3, 2)
    assert encoded_scale.dtype == torch.uint8
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def test_subgroup_packed_dequant_matches_focus_fake_quantizer():
    generator = torch.Generator().manual_seed(20260723)
    weight = torch.randn(3, 64, generator=generator, dtype=torch.float32) * 1.7
    config = _mxfp4_config()
    config["weight"]["use_subgroup_scale"] = True
    config["weight"]["num_sub"] = 4
    quantizer = Quantizer(
        config,
        _quant_info(),
        x=weight,
        weight_shape=weight.shape,
    )
    with torch.no_grad():
        quantizer.max_scale.copy_(torch.tensor([[0.75, 1.25], [1.0, 0.625], [1.5, 0.875]]))
        quantizer.quant_max_scale.copy_(
            torch.linspace(-1.5, 2.0, quantizer.quant_max_scale.numel()).reshape_as(
                quantizer.quant_max_scale
            )
        )

    expected = quantizer(weight)
    packed, encoded_scale = mxfp4_quantize_pack(
        weight,
        quantizer.max_scale,
        quant_max_scale=quantizer.quant_max_scale,
        num_sub=4,
    )
    restored = mxfp4_unpack_dequantize(packed, encoded_scale)

    torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def test_real_export_rejects_non_deployable_shapes_and_scales():
    with pytest.raises(ValueError, match="divisible by 32"):
        mxfp4_quantize_pack(torch.ones(2, 33), torch.ones(2, 2))
    with pytest.raises(ValueError, match="finite and positive"):
        mxfp4_quantize_pack(torch.ones(2, 32), torch.tensor([[1.0], [-1.0]]))


def test_fake_checkpoint_exports_compressed_tensors_schema(tmp_path):
    model_path = tmp_path / "base"
    output_path = tmp_path / "exported"
    model_path.mkdir()
    base_weight = torch.linspace(-3.0, 3.0, 64).reshape(2, 32).to(torch.bfloat16)
    save_file(
        {
            "model.layers.0.mlp.down_proj.weight": base_weight,
            "model.embed_tokens.weight": torch.ones(4, 8, dtype=torch.bfloat16),
        },
        model_path / "model.safetensors",
    )
    (model_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )
    (model_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    checkpoint_path = tmp_path / "focus_fake.pt"
    learned_scale = torch.tensor([[0.75], [1.25]])
    subgroup_scale = torch.tensor([[-1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, -1.0]])
    torch.save(
        {
            # Deliberately wrong baked weight: export must use the frozen base weight.
            "model.layers.0.mlp.down_proj.weight": torch.zeros_like(base_weight),
            "model.layers.0.mlp.down_proj.weight_quantizer.max_scale": learned_scale,
            "model.layers.0.mlp.down_proj.weight_quantizer.quant_max_scale": subgroup_scale,
        },
        checkpoint_path,
    )

    summary = export_focus_mxfp4_checkpoint(
        checkpoint_path,
        model_path,
        output_path,
        max_shard_size="1GB",
    )
    assert summary["exported_layer_count"] == 1
    assert summary["subgroup_layer_count"] == 1

    with safe_open(output_path / "model.safetensors", framework="pt", device="cpu") as reader:
        assert set(reader.keys()) == {
            "model.embed_tokens.weight",
            "model.layers.0.mlp.down_proj.weight_packed",
            "model.layers.0.mlp.down_proj.weight_scale",
        }
        packed = reader.get_tensor("model.layers.0.mlp.down_proj.weight_packed")
        encoded_scale = reader.get_tensor("model.layers.0.mlp.down_proj.weight_scale")

    expected_packed, expected_scale = mxfp4_quantize_pack(
        base_weight,
        learned_scale,
        quant_max_scale=subgroup_scale,
    )
    torch.testing.assert_close(packed, expected_packed, rtol=0, atol=0)
    torch.testing.assert_close(encoded_scale, expected_scale, rtol=0, atol=0)

    config = json.loads((output_path / "config.json").read_text(encoding="utf-8"))
    quant_config = config["quantization_config"]
    assert quant_config["quant_method"] == "compressed-tensors"
    assert quant_config["format"] == "mxfp4-pack-quantized"
    assert quant_config["config_groups"]["group_0"]["input_activations"]["dynamic"] is True


def test_qat_real_convert_builds_packed_state_dict_from_original_weight():
    linear = torch.nn.Linear(64, 3, bias=False)
    quant_linear = QuantLinear(
        linear,
        _mxfp4_config(),
        _quant_info(),
        use_weight_quant=True,
        use_act_quant=False,
    )
    with torch.no_grad():
        quant_linear.weight_quantizer.max_scale.copy_(
            torch.tensor([[0.75, 1.25], [1.0, 0.625], [1.5, 0.875]])
        )
    model = torch.nn.Sequential(quant_linear)

    qat = QAT.__new__(QAT)
    qat.save_fmt = "real"
    qat.plugin_config = {"quant_config": {"weight": {"qtype": "mxfp4_rceil"}}}
    qat.quant_model = SimpleNamespace(model=model)
    qat._rank0_state_dict = None
    qat.convert()

    expected_packed, expected_scale = mxfp4_quantize_pack(
        quant_linear.weight,
        quant_linear.weight_quantizer.max_scale,
    )
    assert set(qat._rank0_state_dict) == {"0.weight_packed", "0.weight_scale"}
    torch.testing.assert_close(
        qat._rank0_state_dict["0.weight_packed"], expected_packed, rtol=0, atol=0
    )
    torch.testing.assert_close(
        qat._rank0_state_dict["0.weight_scale"], expected_scale, rtol=0, atol=0
    )


def test_qat_real_convert_supports_offline_only_subgroup_coefficients():
    config = _mxfp4_config()
    config["weight"]["use_subgroup_scale"] = True
    config["weight"]["num_sub"] = 4
    quant_linear = QuantLinear(
        torch.nn.Linear(64, 3, bias=False),
        config,
        _quant_info(),
        use_weight_quant=True,
        use_act_quant=False,
    )
    qat = QAT.__new__(QAT)
    qat.save_fmt = "real"
    qat.plugin_config = {"quant_config": {"weight": {"qtype": "mxfp4_rceil"}}}
    qat.quant_model = SimpleNamespace(model=torch.nn.Sequential(quant_linear))
    qat._rank0_state_dict = None
    qat.convert()

    expected_packed, expected_scale = mxfp4_quantize_pack(
        quant_linear.weight,
        quant_linear.weight_quantizer.max_scale,
        quant_max_scale=quant_linear.weight_quantizer.quant_max_scale,
        num_sub=4,
    )
    torch.testing.assert_close(
        qat._rank0_state_dict["0.weight_packed"], expected_packed, rtol=0, atol=0
    )
    torch.testing.assert_close(
        qat._rank0_state_dict["0.weight_scale"], expected_scale, rtol=0, atol=0
    )
    assert all("quant_max_scale" not in key for key in qat._rank0_state_dict)


@pytest.mark.parametrize("qtype", ["mxfp4_rceil", "nvfp4"])
def test_focus_real_export_rejects_baked_resume_checkpoint(qtype):
    qat = QAT.__new__(QAT)
    qat.save_fmt = "real"
    qat.plugin_config = {"quant_config": {"weight": {"qtype": qtype}}}
    qat.config = {
        "compress_config": SimpleNamespace(
            QAT=SimpleNamespace(resume_ckpt_dir="/tmp/focus_fake_quant_model.pt")
        )
    }
    qat.quant_model = SimpleNamespace(model=torch.nn.Identity())

    with pytest.raises(RuntimeError, match="quantizing the weights a second time"):
        qat.convert()


def test_qat_real_convert_rejects_non_positive_learned_scale():
    quant_linear = QuantLinear(
        torch.nn.Linear(64, 3, bias=False),
        _mxfp4_config(),
        _quant_info(),
        use_weight_quant=True,
        use_act_quant=False,
    )
    with torch.no_grad():
        quant_linear.weight_quantizer.max_scale[0, 0] = -1.0
    qat = QAT.__new__(QAT)
    qat.save_fmt = "real"
    qat.plugin_config = {"quant_config": {"weight": {"qtype": "mxfp4_rceil"}}}
    qat.quant_model = SimpleNamespace(model=torch.nn.Sequential(quant_linear))
    qat._rank0_state_dict = None

    with pytest.raises(ValueError, match="max_scale.*positive"):
        qat.convert()


@pytest.mark.parametrize("qtype", ["mxfp4_rceil", "nvfp4"])
def test_real_focus_fp4_training_skips_fake_weight_bake(qtype):
    class PluginManager:
        after_kwargs = None

        def call_before_train(self, **kwargs):
            pass

        def call_after_train(self, **kwargs):
            self.after_kwargs = kwargs

    qat_config = SimpleNamespace(
        save_format="real",
        plugin_config={"quant_config": {"weight": {"qtype": qtype}}},
    )
    trainer = End2EndTrainer.__new__(End2EndTrainer)
    trainer.config = {"compress_config": SimpleNamespace(QAT=qat_config)}
    trainer.quant_model = SimpleNamespace(model=torch.nn.Identity())
    trainer.plugin_manager = PluginManager()
    trainer.resume_ckpt_dir = None
    trainer.do_train = False
    trainer.prepare_dataset = lambda _: setattr(trainer, "train_dataset", [])
    trainer.prepare_trainer = lambda: None

    trainer.run(dataloader=None)
    assert trainer.plugin_manager.after_kwargs == {"skip_weight_bake": True}


@pytest.mark.parametrize(
    ("do_train", "save_format", "message"),
    [
        (True, "fake", "cannot be used to continue QAT"),
        (False, "real", "cannot be converted through direct"),
    ],
)
def test_focus_resume_mode_rejects_double_quantization(do_train, save_format, message):
    qat_config = SimpleNamespace(
        save_format=save_format,
        plugin_config={"quant_config": {"weight": {"qtype": "mxfp4_rceil"}}},
    )
    trainer = End2EndTrainer.__new__(End2EndTrainer)
    trainer.config = {"compress_config": SimpleNamespace(QAT=qat_config)}
    trainer.resume_ckpt_dir = "/tmp/focus_fake_quant_model.pt"
    trainer.do_train = do_train

    with pytest.raises(RuntimeError, match=message):
        trainer._validate_focus_resume_mode()
