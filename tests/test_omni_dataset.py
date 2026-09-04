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

"""Unit tests for ``OmniDataset._load_file_based_dataset``.

These tests are CPU-only and require neither a GPU nor model weights.
Heavy dependencies are stubbed so the conversation-construction logic
can be exercised in isolation.

Regression: assistant messages were silently dropped from the
conversation list, producing incomplete calibration data.
"""

import importlib.util
import json
import os
import sys
import types

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OMNI_DATASET_PATH = os.path.join(_REPO_ROOT, "angelslim", "data", "omni_dataset.py")


def _install_stubs():
    """Register lightweight stand-ins so ``omni_dataset.py`` imports cleanly."""

    def _module(name):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    if "torch" not in sys.modules:
        _module("torch")
        _module("torch.utils")
        torch_utils_data = _module("torch.utils.data")
        torch_utils_data.Dataset = type("Dataset", (), {})
    if "transformers" not in sys.modules:
        transformers = _module("transformers")
        transformers.ProcessorMixin = type("ProcessorMixin", (), {})

    for pkg in ("angelslim", "angelslim.utils", "angelslim.data"):
        if pkg not in sys.modules:
            mod = _module(pkg)
            mod.__path__ = []

    if "angelslim.utils.lazy_imports" not in sys.modules:
        lazy = _module("angelslim.utils.lazy_imports")
        lazy.qwen_omni_utils = types.ModuleType("qwen_omni_utils")

    if "angelslim.data.base_dataset" not in sys.modules:
        base_mod = _module("angelslim.data.base_dataset")
        base_mod.BaseDataset = type("BaseDataset", (), {"__init__": lambda *a, **k: None})


def _load_omni_dataset_cls():
    _install_stubs()
    spec = importlib.util.spec_from_file_location(
        "angelslim.data.omni_dataset", _OMNI_DATASET_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OmniDataset


@pytest.fixture(scope="module")
def omni_cls():
    return _load_omni_dataset_cls()


def _make_instance(cls):
    """Create an uninitialized OmniDataset for testing."""
    inst = cls.__new__(cls)
    inst.data = []
    inst.processor = None
    inst.use_audio_in_video = False
    return inst


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_assistant_messages_are_preserved(tmp_path, omni_cls):
    """Assistant turns must appear in the conversation passed downstream.

    Regression: the conversation loop only handled ``system`` and ``user``
    roles, silently dropping ``assistant`` turns from calibration data.
    """
    data_file = tmp_path / "data.jsonl"
    _write_jsonl(
        data_file,
        [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                    {"role": "user", "content": "And 3+3?"},
                    {"role": "assistant", "content": "6"},
                ]
            }
        ],
    )

    inst = _make_instance(omni_cls)
    captured = []
    inst._process_and_append = lambda msgs: captured.append(msgs)

    inst._load_file_based_dataset(str(data_file), num_samples=-1)

    assert len(captured) == 1
    roles = [m["role"] for m in captured[0]]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_system_message_is_preserved(tmp_path, omni_cls):
    """System messages must still appear in the conversation."""
    data_file = tmp_path / "data.jsonl"
    _write_jsonl(
        data_file,
        [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ]
            }
        ],
    )

    inst = _make_instance(omni_cls)
    captured = []
    inst._process_and_append = lambda msgs: captured.append(msgs)

    inst._load_file_based_dataset(str(data_file), num_samples=-1)

    roles = [m["role"] for m in captured[0]]
    assert roles == ["system", "user", "assistant"]


def test_assistant_content_is_wrapped_as_text(tmp_path, omni_cls):
    """Assistant content should be wrapped in the ``[{type: text, text: ...}]`` format."""
    data_file = tmp_path / "data.jsonl"
    _write_jsonl(
        data_file,
        [
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ]
            }
        ],
    )

    inst = _make_instance(omni_cls)
    captured = []
    inst._process_and_append = lambda msgs: captured.append(msgs)

    inst._load_file_based_dataset(str(data_file), num_samples=-1)

    assistant_msg = captured[0][1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == [{"type": "text", "text": "Hello!"}]


def test_num_samples_limits_loaded_records(tmp_path, omni_cls):
    """When num_samples > 0, only that many records are processed."""
    data_file = tmp_path / "data.jsonl"
    _write_jsonl(
        data_file,
        [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(5)],
    )

    inst = _make_instance(omni_cls)
    captured = []
    inst._process_and_append = lambda msgs: captured.append(msgs)

    inst._load_file_based_dataset(str(data_file), num_samples=2)
    assert len(captured) == 2
