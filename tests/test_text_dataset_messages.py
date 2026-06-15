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

"""Unit tests for ``TextDataset._prepare_messages``.

These tests are CPU-only and require neither a GPU, model weights, nor the
heavy ``torch``/``transformers``/``datasets``/``pyarrow`` stack: those imports
pulled in by ``angelslim.data.text_dataset`` are stubbed so the pure-dict
message-preparation logic can be exercised in isolation. ``_prepare_messages``
does not touch ``self``, so it is invoked on an uninitialized instance.
"""

import importlib.util
import os
import sys
import types

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEXT_DATASET_PATH = os.path.join(_REPO_ROOT, "angelslim", "data", "text_dataset.py")


def _install_stubs():
    """Register lightweight stand-ins so ``text_dataset.py`` imports cleanly."""

    def _module(name):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    if "torch" not in sys.modules:
        _module("torch")
    if "pyarrow" not in sys.modules:
        pyarrow = _module("pyarrow")
        pyarrow.parquet = _module("pyarrow.parquet")
    if "datasets" not in sys.modules:
        datasets = _module("datasets")
        datasets.load_dataset = lambda *a, **k: None
    if "transformers" not in sys.modules:
        transformers = _module("transformers")
        transformers.ProcessorMixin = object

    # angelslim.data.base_dataset.BaseDataset (the only intra-package import)
    for pkg in ("angelslim", "angelslim.data"):
        if pkg not in sys.modules:
            mod = _module(pkg)
            mod.__path__ = []  # mark as package
    name = "angelslim.data.base_dataset"
    if name not in sys.modules:
        mod = _module(name)
        mod.BaseDataset = type("BaseDataset", (), {})


def _load_text_dataset_cls():
    _install_stubs()
    spec = importlib.util.spec_from_file_location(
        "angelslim.data.text_dataset", _TEXT_DATASET_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TextDataset


@pytest.fixture(scope="module")
def prepare():
    cls = _load_text_dataset_cls()
    instance = cls.__new__(cls)  # skip __init__ (which loads data / a processor)
    return instance._prepare_messages


def test_conversations_with_system_field_does_not_crash(prepare):
    """A ShareGPT record carrying a ``system`` field must not raise.

    Regression: the branch guarded on ``data["system"]`` but read
    ``data["system_prompt"]``, raising ``KeyError: 'system_prompt'``.
    """
    data = {
        "system": "You are a helpful assistant.",
        "conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello!"},
        ],
    }
    messages = prepare(data)
    assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_conversations_keeps_all_turns(prepare):
    """Every turn of a multi-turn conversation is preserved and role-mapped.

    Regression: only the first two turns were kept (positional user/assistant),
    dropping the rest of a multi-turn ShareGPT record.
    """
    data = {
        "conversations": [
            {"from": "human", "value": "q1"},
            {"from": "gpt", "value": "a1"},
            {"from": "human", "value": "q2"},
            {"from": "gpt", "value": "a2"},
        ]
    }
    messages = prepare(data)
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in messages] == ["q1", "a1", "q2", "a2"]


def test_conversations_system_turn_is_not_duplicated(prepare):
    """A leading system turn must not be doubled by the system-field prepend."""
    data = {
        "system": "S",
        "conversations": [
            {"from": "system", "value": "S"},
            {"from": "human", "value": "q"},
            {"from": "gpt", "value": "a"},
        ],
    }
    messages = prepare(data)
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]


def test_conversations_skips_malformed_turn(prepare):
    """Turns without a text payload are skipped instead of crashing later."""
    data = {
        "conversations": [
            {"from": "human", "value": "q"},
            {"from": "gpt"},  # no value/content
            {"from": "gpt", "value": "a"},
        ]
    }
    messages = prepare(data)
    assert [m["content"] for m in messages] == ["q", "a"]


def test_messages_branch_unchanged(prepare):
    """The ``messages`` branch keeps its existing behavior (no regression)."""
    data = {
        "system_prompt": "sys",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ],
    }
    messages = prepare(data)
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_input_output_branch_unchanged(prepare):
    """The plain input/output branch keeps its existing behavior."""
    data = {"input": "q", "output": "a"}
    messages = prepare(data)
    assert messages == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
