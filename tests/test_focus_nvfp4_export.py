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

from angelslim.compressor.qat.export import export_focus_nvfp4_checkpoint
from angelslim.compressor.qat.modules.quantizer import Quantizer, QuantLinear
from angelslim.compressor.qat.qat import QAT
from angelslim.compressor.quant.modules.nvfp4 import (
    harmonize_nvfp4_fused_scales,
    nvfp4_quantize_pack,
    nvfp4_unpack_dequantize,
)


def _quant_info():
    return SimpleNamespace(
        quant_algo="w4a8_fp8",
        quant_algo_info={
            "w": "int4_per-group",
            "a": "fp8_per-token-dynamic",
            "w_group_size": 16,
        },
    )


def _nvfp4_config(use_subgroup=False):
    return {
        "weight": {
            "qtype": "nvfp4",
            "granularity": "per-group",
            "group_size": 16,
            "is_sym": True,
            "dynamic": False,
            "mantissa_rounding": "even",
            "use_subgroup_scale": use_subgroup,
            "num_sub": 2,
        },
        "activation": {
            "qtype": "nvfp4",
            "granularity": "per-group",
            "group_size": 16,
            "is_sym": True,
            "dynamic": True,
        },
    }


def _fused_scale_state(input_global_scale=500.0):
    tensor_scales = {
        "model.layers.0.self_attn.q_proj": 0.30,
        "model.layers.0.self_attn.k_proj": 0.55,
        "model.layers.0.self_attn.v_proj": 1.00,
        "model.layers.0.mlp.gate_proj": 0.45,
        "model.layers.0.mlp.up_proj": 0.90,
    }
    local_scale = torch.tensor([[0.5, 1.25], [2.5, 4.0]]).to(torch.float8_e4m3fn)
    state = {}
    for index, (prefix, tensor_scale) in enumerate(tensor_scales.items()):
        state[f"{prefix}.weight_packed"] = torch.full((2, 16), index, dtype=torch.uint8)
        state[f"{prefix}.weight_scale"] = local_scale.clone()
        state[f"{prefix}.weight_global_scale"] = torch.tensor([1.0 / tensor_scale])
        state[f"{prefix}.input_global_scale"] = torch.tensor([input_global_scale])
    return state


def test_harmonize_nvfp4_fused_scales_preserves_codes_and_effective_scales():
    state = _fused_scale_state()
    packed_before = {
        key: value.clone() for key, value in state.items() if key.endswith(".weight_packed")
    }
    effective_before = {
        prefix: state[f"{prefix}.weight_scale"].float() / state[f"{prefix}.weight_global_scale"]
        for prefix in (
            key[: -len(".weight_packed")] for key in state if key.endswith(".weight_packed")
        )
    }

    summary = harmonize_nvfp4_fused_scales(state)

    assert summary["fused_group_count"] == 2
    assert summary["fused_layer_count"] == 5
    assert summary["max_global_scale_ratio"] > 3.0
    for key, packed in packed_before.items():
        torch.testing.assert_close(state[key], packed, rtol=0, atol=0)

    for members in (
        ("q_proj", "k_proj", "v_proj"),
        ("gate_proj", "up_proj"),
    ):
        kind = "self_attn" if len(members) == 3 else "mlp"
        prefixes = [f"model.layers.0.{kind}.{member}" for member in members]
        globals_after = [state[f"{prefix}.weight_global_scale"] for prefix in prefixes]
        assert all(torch.equal(globals_after[0], value) for value in globals_after[1:])
        for prefix in prefixes:
            effective_after = (
                state[f"{prefix}.weight_scale"].float() / state[f"{prefix}.weight_global_scale"]
            )
            torch.testing.assert_close(
                effective_after,
                effective_before[prefix],
                rtol=0.063,
                atol=0,
            )


def test_harmonize_nvfp4_fused_scales_rejects_mismatched_input_scale():
    state = _fused_scale_state()
    state["model.layers.0.self_attn.k_proj.input_global_scale"] = torch.tensor([400.0])

    with pytest.raises(ValueError, match="mismatched input_global_scale"):
        harmonize_nvfp4_fused_scales(state)


def test_harmonize_nvfp4_fused_scales_rejects_block_scale_underflow():
    state = _fused_scale_state()
    key = "model.layers.0.self_attn.q_proj.weight_scale"
    state[key] = torch.full_like(state[key], torch.finfo(torch.float8_e4m3fn).tiny)

    with pytest.raises(ValueError, match="exceed the supported E4M3 range"):
        harmonize_nvfp4_fused_scales(state)


def test_nvfp4_packed_dequant_matches_focus_fake_quantizer():
    weight = torch.randn(3, 64, generator=torch.Generator().manual_seed(20260724))
    quantizer = Quantizer(
        _nvfp4_config(),
        _quant_info(),
        x=weight,
        weight_shape=weight.shape,
    )
    with torch.no_grad():
        quantizer.max_scale.copy_(
            torch.tensor([[0.75, 1.25, 0.8, 1.1], [1.0, 0.625, 1.3, 0.9], [1.5, 0.875, 1.0, 0.7]])
        )

    expected = quantizer(weight)
    packed, local_scale, global_scale = nvfp4_quantize_pack(
        weight, quantizer.max_scale, quantizer.scale_2
    )
    restored = nvfp4_unpack_dequantize(packed, local_scale, global_scale)

    assert packed.shape == (3, 32)
    assert packed.dtype == torch.uint8
    assert local_scale.shape == (3, 4)
    assert local_scale.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def test_nvfp4_subgroup_packing_matches_focus_fake_quantizer():
    weight = torch.randn(3, 64, generator=torch.Generator().manual_seed(20260725))
    quantizer = Quantizer(
        _nvfp4_config(use_subgroup=True),
        _quant_info(),
        x=weight,
        weight_shape=weight.shape,
    )
    with torch.no_grad():
        quantizer.quant_max_scale.copy_(
            torch.linspace(-1.5, 2.0, quantizer.quant_max_scale.numel()).reshape_as(
                quantizer.quant_max_scale
            )
        )

    expected = quantizer(weight)
    packed, local_scale, global_scale = nvfp4_quantize_pack(
        weight,
        quantizer.max_scale,
        quantizer.scale_2,
        quant_max_scale=quantizer.quant_max_scale,
        num_sub=2,
    )
    restored = nvfp4_unpack_dequantize(packed, local_scale, global_scale)
    # compressed-tensors stores the reciprocal global scale, so dequantization
    # can differ by one FP32 rounding step while preserving identical FP4 codes.
    torch.testing.assert_close(restored, expected, rtol=1e-7, atol=0)


def test_nvfp4_fake_checkpoint_exports_compressed_tensors_schema(tmp_path):
    model_path = tmp_path / "base"
    output_path = tmp_path / "exported"
    model_path.mkdir()
    base_weight = torch.linspace(-3.0, 3.0, 32).reshape(2, 16)
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

    checkpoint_path = tmp_path / "focus_fake.pt"
    prefix = "model.layers.0.mlp.down_proj"
    learned_scale = torch.tensor([[0.75], [1.25]])
    weight_scale_2 = base_weight.abs().amax().reshape(1) / (6.0 * 448.0)
    input_scale_2 = torch.tensor([0.002])
    torch.save(
        {
            f"{prefix}.weight": torch.zeros_like(base_weight),
            f"{prefix}.weight_quantizer.max_scale": learned_scale,
            f"{prefix}.weight_quantizer.scale_2": weight_scale_2,
            f"{prefix}.act_quantizer.scale_2": input_scale_2,
        },
        checkpoint_path,
    )

    summary = export_focus_nvfp4_checkpoint(
        checkpoint_path, model_path, output_path, max_shard_size="1GB"
    )
    assert summary["exported_layer_count"] == 1

    with safe_open(output_path / "model.safetensors", framework="pt", device="cpu") as reader:
        assert set(reader.keys()) == {
            "model.embed_tokens.weight",
            f"{prefix}.input_global_scale",
            f"{prefix}.weight_global_scale",
            f"{prefix}.weight_packed",
            f"{prefix}.weight_scale",
        }
        packed = reader.get_tensor(f"{prefix}.weight_packed")
        local_scale = reader.get_tensor(f"{prefix}.weight_scale")
        weight_global_scale = reader.get_tensor(f"{prefix}.weight_global_scale")
        input_global_scale = reader.get_tensor(f"{prefix}.input_global_scale")

    expected_packed, expected_scale, expected_global = nvfp4_quantize_pack(
        base_weight, learned_scale, weight_scale_2
    )
    torch.testing.assert_close(packed, expected_packed, rtol=0, atol=0)
    torch.testing.assert_close(local_scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(weight_global_scale, expected_global, rtol=0, atol=0)
    torch.testing.assert_close(input_global_scale, 1.0 / input_scale_2, rtol=0, atol=0)

    config = json.loads((output_path / "config.json").read_text(encoding="utf-8"))
    quant_config = config["quantization_config"]
    assert quant_config["format"] == "nvfp4-pack-quantized"
    group = quant_config["config_groups"]["group_0"]
    assert group["weights"]["strategy"] == "tensor_group"
    assert group["input_activations"]["dynamic"] == "local"


def test_nvfp4_fake_checkpoint_harmonizes_qkv_fused_scales(tmp_path):
    model_path = tmp_path / "base"
    output_path = tmp_path / "exported"
    model_path.mkdir()
    prefixes = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    base_weights = {
        f"{prefix}.weight": (torch.linspace(-1.0, 1.0, 32).reshape(2, 16) * (index + 1))
        for index, prefix in enumerate(prefixes)
    }
    save_file(
        {
            **base_weights,
            "model.embed_tokens.weight": torch.ones(4, 8, dtype=torch.bfloat16),
        },
        model_path / "model.safetensors",
    )
    (model_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )

    checkpoint_path = tmp_path / "focus_fake.pt"
    checkpoint = {}
    input_scale_2 = torch.tensor([0.002])
    expected_state = {}
    raw_packed = {}
    for index, prefix in enumerate(prefixes):
        weight = base_weights[f"{prefix}.weight"]
        max_scale = torch.tensor([[0.75 + index * 0.1], [1.25 - index * 0.1]])
        scale_2 = weight.abs().amax().reshape(1) / (6.0 * 448.0)
        checkpoint.update(
            {
                f"{prefix}.weight": torch.zeros_like(weight),
                f"{prefix}.weight_quantizer.max_scale": max_scale,
                f"{prefix}.weight_quantizer.scale_2": scale_2,
                f"{prefix}.act_quantizer.scale_2": input_scale_2,
            }
        )
        packed, local_scale, weight_global = nvfp4_quantize_pack(weight, max_scale, scale_2)
        expected_state[f"{prefix}.weight_packed"] = packed
        expected_state[f"{prefix}.weight_scale"] = local_scale
        expected_state[f"{prefix}.weight_global_scale"] = weight_global
        expected_state[f"{prefix}.input_global_scale"] = 1.0 / input_scale_2
        raw_packed[prefix] = packed.clone()
    torch.save(checkpoint, checkpoint_path)
    expected_summary = harmonize_nvfp4_fused_scales(expected_state)

    summary = export_focus_nvfp4_checkpoint(
        checkpoint_path, model_path, output_path, max_shard_size="1GB"
    )

    assert summary["fusion_scale_harmonization"] == expected_summary
    assert summary["fusion_scale_harmonization"]["fused_group_count"] == 1
    with safe_open(output_path / "model.safetensors", framework="pt", device="cpu") as reader:
        for prefix in prefixes:
            torch.testing.assert_close(
                reader.get_tensor(f"{prefix}.weight_packed"),
                raw_packed[prefix],
                rtol=0,
                atol=0,
            )
            for suffix in (
                "weight_scale",
                "weight_global_scale",
                "input_global_scale",
            ):
                torch.testing.assert_close(
                    reader.get_tensor(f"{prefix}.{suffix}"),
                    expected_state[f"{prefix}.{suffix}"],
                    rtol=0,
                    atol=0,
                )


def test_qat_nvfp4_real_convert_builds_deployment_state_dict():
    quant_linear = QuantLinear(
        torch.nn.Linear(64, 3, bias=False),
        _nvfp4_config(use_subgroup=True),
        _quant_info(),
        use_weight_quant=True,
        use_act_quant=True,
    )
    quant_linear(torch.randn(2, 64))

    qat = QAT.__new__(QAT)
    qat.save_fmt = "real"
    qat.plugin_config = {"quant_config": {"weight": {"qtype": "nvfp4"}}}
    qat.quant_model = SimpleNamespace(model=torch.nn.Sequential(quant_linear))
    qat._rank0_state_dict = None
    qat.convert()

    assert set(qat._rank0_state_dict) == {
        "0.input_global_scale",
        "0.weight_global_scale",
        "0.weight_packed",
        "0.weight_scale",
    }
    assert all("quant_max_scale" not in key for key in qat._rank0_state_dict)


def test_qat_nvfp4_real_convert_matches_offline_fusion_harmonization():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.mlp = torch.nn.Module()
            generator = torch.Generator().manual_seed(20260726)
            for parent, members in (
                (self.self_attn, ("q_proj", "k_proj", "v_proj")),
                (self.mlp, ("gate_proj", "up_proj")),
            ):
                for index, name in enumerate(members):
                    linear = torch.nn.Linear(64, 3 + index, bias=False)
                    with torch.no_grad():
                        linear.weight.copy_(
                            torch.randn(
                                linear.weight.shape,
                                generator=generator,
                                dtype=linear.weight.dtype,
                            )
                            * (index + 1)
                        )
                    quant_linear = QuantLinear(
                        linear,
                        _nvfp4_config(),
                        _quant_info(),
                        use_weight_quant=True,
                        use_act_quant=True,
                    )
                    quant_linear.act_quantizer.scale_2.copy_(torch.tensor([0.002]))
                    quant_linear.act_quantizer.init = True
                    setattr(parent, name, quant_linear)

    model = ToyModel()
    expected_state = {}
    for prefix, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue
        packed, local_scale, weight_global = nvfp4_quantize_pack(
            module.weight,
            module.weight_quantizer.max_scale,
            module.weight_quantizer.scale_2,
        )
        expected_state[f"{prefix}.weight_packed"] = packed
        expected_state[f"{prefix}.weight_scale"] = local_scale
        expected_state[f"{prefix}.weight_global_scale"] = weight_global
        expected_state[f"{prefix}.input_global_scale"] = (
            1.0 / module.act_quantizer.scale_2.float()
        ).reshape(1)
    expected_summary = harmonize_nvfp4_fused_scales(expected_state)

    qat = QAT.__new__(QAT)
    qat.save_fmt = "real"
    qat.plugin_config = {"quant_config": {"weight": {"qtype": "nvfp4"}}}
    qat.quant_model = SimpleNamespace(model=model)
    qat._rank0_state_dict = None
    qat.convert()

    assert qat._focus_nvfp4_fusion_summary == expected_summary
    assert qat._focus_nvfp4_fusion_summary["fused_group_count"] == 2
    for key, expected in expected_state.items():
        torch.testing.assert_close(qat._rank0_state_dict[key], expected, rtol=0, atol=0)
