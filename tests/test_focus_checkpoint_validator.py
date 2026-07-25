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
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VALIDATOR_PATH = _REPO_ROOT / "tools" / "validate_focus_fp4_checkpoint.py"
_WEIGHT_KEY = "model.layers.0.self_attn.q_proj.weight"
_PREFIX = _WEIGHT_KEY[: -len(".weight")]


def _load_validator():
    spec = importlib.util.spec_from_file_location("focus_checkpoint_validator", _VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fixture(tmp_path, qtype, change_weight=True):
    base_weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    model_dir = tmp_path / "base-model"
    model_dir.mkdir()
    save_file({_WEIGHT_KEY: base_weight}, model_dir / "model.safetensors")

    fake_weight = base_weight + 0.25 if change_weight else base_weight.clone()
    state_dict = {
        _WEIGHT_KEY: fake_weight,
        f"{_PREFIX}.weight_quantizer.max_scale": torch.ones(2, 1),
    }
    if qtype == "nvfp4":
        state_dict[f"{_PREFIX}.weight_quantizer.scale_2"] = torch.ones(1)
        state_dict[f"{_PREFIX}.act_quantizer.scale_2"] = torch.ones(1)
    else:
        state_dict[f"{_PREFIX}.act_quantizer.max_scale"] = torch.ones(1)

    checkpoint = tmp_path / f"{qtype}_fake_quant_model.pt"
    torch.save(state_dict, checkpoint)
    return checkpoint, model_dir


@pytest.mark.parametrize(("qtype", "group_size"), [("mxfp4", 4), ("nvfp4", 4)])
def test_focus_checkpoint_validator_accepts_changed_fake_weights(tmp_path, qtype, group_size):
    validator = _load_validator()
    checkpoint, model_dir = _write_fixture(tmp_path, qtype, change_weight=True)

    summary = validator.validate_checkpoint(
        checkpoint_path=str(checkpoint),
        model_path=str(model_dir),
        qtype=qtype,
        group_size=group_size,
        max_weights=1,
    )

    assert summary["status"] == "PASS"
    assert summary["max_scale_count"] == 1
    assert summary["samples"][0]["changed_fraction"] == 1.0


def test_focus_checkpoint_validator_rejects_unmodified_base_weight(tmp_path):
    validator = _load_validator()
    checkpoint, model_dir = _write_fixture(tmp_path, "mxfp4", change_weight=False)

    with pytest.raises(RuntimeError, match="identical to the frozen base model"):
        validator.validate_checkpoint(
            checkpoint_path=str(checkpoint),
            model_path=str(model_dir),
            qtype="mxfp4",
            group_size=4,
            max_weights=1,
        )


def test_focus_checkpoint_validator_checks_all_scale_shapes(tmp_path):
    validator = _load_validator()
    checkpoint, model_dir = _write_fixture(tmp_path, "mxfp4", change_weight=True)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict[f"{_PREFIX}.weight_quantizer.max_scale"] = torch.ones(2, 2)
    torch.save(state_dict, checkpoint)

    with pytest.raises(RuntimeError, match="Scale shape mismatch"):
        validator.validate_checkpoint(
            checkpoint_path=str(checkpoint),
            model_path=str(model_dir),
            qtype="mxfp4",
            group_size=4,
            max_weights=1,
        )


def test_focus_checkpoint_validator_checks_every_base_weight_shape(tmp_path):
    validator = _load_validator()
    checkpoint, model_dir = _write_fixture(tmp_path, "mxfp4", change_weight=True)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict[_WEIGHT_KEY] = torch.ones(2, 8)
    state_dict[f"{_PREFIX}.weight_quantizer.max_scale"] = torch.ones(2, 2)
    torch.save(state_dict, checkpoint)

    with pytest.raises(RuntimeError, match="Shape mismatch"):
        validator.validate_checkpoint(
            checkpoint_path=str(checkpoint),
            model_path=str(model_dir),
            qtype="mxfp4",
            group_size=4,
            max_weights=1,
        )


def test_focus_checkpoint_validator_rejects_nonfinite_weight(tmp_path):
    validator = _load_validator()
    checkpoint, model_dir = _write_fixture(tmp_path, "mxfp4", change_weight=True)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict[_WEIGHT_KEY][0, 0] = torch.nan
    torch.save(state_dict, checkpoint)

    with pytest.raises(RuntimeError, match="Non-finite fake weight"):
        validator.validate_checkpoint(
            checkpoint_path=str(checkpoint),
            model_path=str(model_dir),
            qtype="mxfp4",
            group_size=4,
            max_weights=1,
        )


@pytest.mark.parametrize(
    ("qtype", "scale_key"),
    [
        ("mxfp4", f"{_PREFIX}.act_quantizer.max_scale"),
        ("nvfp4", f"{_PREFIX}.weight_quantizer.scale_2"),
    ],
)
def test_focus_checkpoint_validator_requires_exact_scalar_scale_shape(tmp_path, qtype, scale_key):
    validator = _load_validator()
    checkpoint, model_dir = _write_fixture(tmp_path, qtype, change_weight=True)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict[scale_key] = torch.ones(1, 1)
    torch.save(state_dict, checkpoint)

    with pytest.raises(RuntimeError, match="Invalid"):
        validator.validate_checkpoint(
            checkpoint_path=str(checkpoint),
            model_path=str(model_dir),
            qtype=qtype,
            group_size=4,
            max_weights=1,
        )


def test_focus_checkpoint_validator_requires_nvfp4_scales(tmp_path):
    validator = _load_validator()
    checkpoint, model_dir = _write_fixture(tmp_path, "nvfp4", change_weight=True)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    del state_dict[f"{_PREFIX}.act_quantizer.scale_2"]
    torch.save(state_dict, checkpoint)

    with pytest.raises(RuntimeError, match="Missing activation scale"):
        validator.validate_checkpoint(
            checkpoint_path=str(checkpoint),
            model_path=str(model_dir),
            qtype="nvfp4",
            group_size=4,
            max_weights=1,
        )
