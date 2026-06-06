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

"""Unit tests for JSON configuration parsing in ``angelslim.utils.config_parser``.

These tests are CPU-only and require neither a GPU nor model weights: they
exercise the pure configuration-parsing logic that ``Engine.prepare_model``
relies on when loading a previously compressed model from
``angelslim_config.json``.
"""

import json

import pytest

from angelslim.utils.config_parser import parse_json_full_config


def _write_json(tmp_path, payload):
    config_path = tmp_path / "angelslim_config.json"
    config_path.write_text(json.dumps(payload))
    return str(config_path)


def test_json_roundtrip_preserves_compression_config(tmp_path):
    """The compression section must survive a JSON load round-trip.

    ``Engine.prepare_model`` forwards ``slim_config.compression_config`` to
    ``from_pretrained``; if it is dropped during parsing the compressed model is
    reloaded without any compression metadata.
    """
    payload = {
        "model_config": {"name": "Qwen", "model_path": "Base Model Path"},
        "compression_config": {
            "name": "PTQ",
            "quantization": {"name": "fp8_dynamic", "bits": 8},
        },
        "global_config": {"save_path": "Save Model Path"},
    }

    full_config = parse_json_full_config(_write_json(tmp_path, payload))

    assert full_config.compression_config is not None
    assert full_config.compression_config.name == ["PTQ"]
    assert full_config.compression_config.quantization is not None
    assert full_config.compression_config.quantization.name == "fp8_dynamic"


def test_json_missing_required_section_raises_value_error(tmp_path):
    """A missing required section should raise a descriptive ValueError."""
    payload = {"model_config": {"name": "Qwen", "model_path": "Base Model Path"}}

    with pytest.raises(ValueError, match="compression_config"):
        parse_json_full_config(_write_json(tmp_path, payload))
