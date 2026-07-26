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

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from angelslim.compressor.qat.export import (
    export_focus_mxfp4_checkpoint,
    export_focus_nvfp4_checkpoint,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PREFIX = "model.layers.0.mlp.down_proj"
_WEIGHT_KEY = f"{_PREFIX}.weight"


def _load_validator(qtype):
    validator_path = _REPO_ROOT / "tools" / f"validate_focus_{qtype}_export.py"
    spec = importlib.util.spec_from_file_location(f"focus_{qtype}_validator", validator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_safetensors(path):
    with safe_open(path, framework="pt", device="cpu") as reader:
        return {key: reader.get_tensor(key) for key in reader.keys()}


def _write_export_fixture(tmp_path, qtype):
    model_path = tmp_path / f"{qtype}-base"
    export_path = tmp_path / f"{qtype}-export"
    model_path.mkdir()

    base_weight = torch.linspace(-3.0, 3.0, 64).reshape(2, 32).to(torch.bfloat16)
    save_file(
        {
            _WEIGHT_KEY: base_weight,
            "model.embed_tokens.weight": torch.ones(4, 8, dtype=torch.bfloat16),
        },
        model_path / "model.safetensors",
    )
    (model_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )

    checkpoint_path = tmp_path / f"{qtype}_fake_quant_model.pt"
    if qtype == "mxfp4":
        state_dict = {
            _WEIGHT_KEY: torch.zeros_like(base_weight),
            f"{_PREFIX}.weight_quantizer.max_scale": torch.tensor([[0.75], [1.25]]),
            f"{_PREFIX}.weight_quantizer.quant_max_scale": torch.tensor(
                [[-1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, -1.0]]
            ),
        }
        exporter = export_focus_mxfp4_checkpoint
    else:
        state_dict = {
            _WEIGHT_KEY: torch.zeros_like(base_weight),
            f"{_PREFIX}.weight_quantizer.max_scale": torch.tensor([[0.75, 1.0], [1.25, 0.875]]),
            f"{_PREFIX}.weight_quantizer.quant_max_scale": torch.tensor(
                [[-1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, -1.0]]
            ),
            f"{_PREFIX}.weight_quantizer.scale_2": (
                base_weight.float().abs().amax().reshape(1) / (6.0 * 448.0)
            ),
            f"{_PREFIX}.act_quantizer.scale_2": torch.tensor([0.002]),
        }
        exporter = export_focus_nvfp4_checkpoint

    torch.save(state_dict, checkpoint_path)
    exporter(
        checkpoint_path,
        model_path,
        export_path,
        max_shard_size="1GB",
    )
    return checkpoint_path, model_path, export_path


@pytest.mark.parametrize("qtype", ["mxfp4", "nvfp4"])
def test_focus_export_validator_supports_schema_only_mode(tmp_path, qtype):
    validator = _load_validator(qtype)
    _, model_path, export_path = _write_export_fixture(tmp_path, qtype)

    summary = validator.validate_export(
        export_path=str(export_path),
        model_path=str(model_path),
    )

    assert summary["status"] == "PASS"
    assert summary["validation_mode"] == "schema_only"
    assert summary["validated_layer_count"] == 1
    assert summary["samples"] == []


@pytest.mark.parametrize("qtype", ["mxfp4", "nvfp4"])
def test_focus_export_validator_preserves_bit_exact_mode(tmp_path, qtype):
    validator = _load_validator(qtype)
    checkpoint_path, model_path, export_path = _write_export_fixture(tmp_path, qtype)

    summary = validator.validate_export(
        export_path=str(export_path),
        checkpoint_path=str(checkpoint_path),
        model_path=str(model_path),
        max_weights=1,
    )

    assert summary["status"] == "PASS"
    assert summary["validation_mode"] == "bit_exact"
    assert summary["validated_layer_count"] == 1
    assert len(summary["samples"]) == 1


@pytest.mark.parametrize(
    ("qtype", "missing_suffix"),
    [
        ("mxfp4", ".weight_scale"),
        ("nvfp4", ".input_global_scale"),
    ],
)
def test_focus_schema_validator_rejects_missing_required_tensor(tmp_path, qtype, missing_suffix):
    validator = _load_validator(qtype)
    _, model_path, export_path = _write_export_fixture(tmp_path, qtype)
    tensor_path = export_path / "model.safetensors"
    tensors = _read_safetensors(tensor_path)
    del tensors[f"{_PREFIX}{missing_suffix}"]
    save_file(tensors, tensor_path)

    with pytest.raises(KeyError, match="Missing packed tensors"):
        validator.validate_export(
            export_path=str(export_path),
            model_path=str(model_path),
        )


@pytest.mark.parametrize("qtype", ["mxfp4", "nvfp4"])
def test_focus_schema_validator_rejects_invalid_packed_shape(tmp_path, qtype):
    validator = _load_validator(qtype)
    _, model_path, export_path = _write_export_fixture(tmp_path, qtype)
    tensor_path = export_path / "model.safetensors"
    tensors = _read_safetensors(tensor_path)
    tensors[f"{_PREFIX}.weight_packed"] = torch.zeros(2, 15, dtype=torch.uint8)
    save_file(tensors, tensor_path)

    with pytest.raises(ValueError, match="Invalid packed weight layout"):
        validator.validate_export(
            export_path=str(export_path),
            model_path=str(model_path),
        )


def test_focus_schema_validator_rejects_retained_unpacked_weight(tmp_path):
    validator = _load_validator("mxfp4")
    _, model_path, export_path = _write_export_fixture(tmp_path, "mxfp4")
    tensor_path = export_path / "model.safetensors"
    tensors = _read_safetensors(tensor_path)
    tensors[_WEIGHT_KEY] = torch.ones(2, 32, dtype=torch.bfloat16)
    save_file(tensors, tensor_path)

    with pytest.raises(ValueError, match="retained unpacked quantized weight"):
        validator.validate_export(
            export_path=str(export_path),
            model_path=str(model_path),
        )
