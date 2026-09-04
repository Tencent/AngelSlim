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

"""Unit tests for ``MultiModalDataset`` helpers.

These tests are CPU-only and require neither a GPU nor model weights.
Heavy dependencies are stubbed so the data-preparation logic can be
exercised in isolation.

Regression: ``_process_and_append`` crashed with ``TypeError`` when
``quant_algo`` was ``None`` (sparsity-only or distill-only pipelines).
Regression: ``_extract_vision_info`` only caught ``ValueError`` from
``Image.open()``, but missing files raise ``FileNotFoundError``.
"""

import importlib.util
import os
import sys
import types

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MM_DATASET_PATH = os.path.join(_REPO_ROOT, "angelslim", "data", "multimodal_dataset.py")


def _install_stubs():
    """Register lightweight stand-ins so ``multimodal_dataset.py`` imports cleanly."""

    def _module(name):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    if "torch" not in sys.modules:
        torch = _module("torch")
        torch.Tensor = type("Tensor", (), {})
        _module("torch.utils")
        torch_utils_data = _module("torch.utils.data")
        torch_utils_data.Dataset = type("Dataset", (), {})
    if "datasets" not in sys.modules:
        datasets = _module("datasets")
        datasets.load_dataset = lambda *a, **k: None
    if "tqdm" not in sys.modules:
        tqdm_mod = _module("tqdm")
        tqdm_mod.tqdm = lambda x, **kw: x
    if "transformers" not in sys.modules:
        transformers = _module("transformers")
        transformers.ProcessorMixin = type("ProcessorMixin", (), {})

    for pkg in ("angelslim", "angelslim.utils", "angelslim.data"):
        if pkg not in sys.modules:
            mod = _module(pkg)
            mod.__path__ = []

    if "angelslim.utils.lazy_imports" not in sys.modules:
        lazy = _module("angelslim.utils.lazy_imports")
        lazy.qwen_vl_utils = types.ModuleType("qwen_vl_utils")

    if "angelslim.data.base_dataset" not in sys.modules:
        base_mod = _module("angelslim.data.base_dataset")
        base_mod.BaseDataset = type("BaseDataset", (), {"__init__": lambda *a, **k: None})


def _load_mm_dataset_cls():
    _install_stubs()
    spec = importlib.util.spec_from_file_location(
        "angelslim.data.multimodal_dataset", _MM_DATASET_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MultiModalDataset


@pytest.fixture(scope="module")
def mm_cls():
    return _load_mm_dataset_cls()


def _make_instance(cls, quant_algo=None, model_name="Qwen3VL"):
    """Create an uninitialized MultiModalDataset with specific attributes."""
    inst = cls.__new__(cls)
    inst.quant_algo = quant_algo
    inst.model_name = model_name
    inst.data = []
    inst.processor = None
    inst.max_length = 512
    return inst


def test_process_and_append_quant_algo_none_does_not_crash(mm_cls):
    """When quant_algo is None (sparsity/distill pipeline), padding defaults to True.

    Regression: ``"int4_" in None`` raised ``TypeError``.
    """
    inst = _make_instance(mm_cls, quant_algo=None, model_name="Qwen3VL")

    call_args = {}

    def fake_apply_chat_template(*args, **kwargs):
        call_args["padding"] = kwargs.get("padding")
        # Return a dict mimicking processor output with input_ids
        dummy_tensor = type("T", (), {"roll": lambda s, **k: s, "__setitem__": lambda *a: None})()
        return {"input_ids": dummy_tensor}

    inst.processor = type("P", (), {"apply_chat_template": fake_apply_chat_template})()

    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]
    inst._process_and_append(messages)
    assert call_args["padding"] is True


def test_process_and_append_int4_uses_max_length_padding(mm_cls):
    """When quant_algo contains 'int4_', padding should be 'max_length'."""
    inst = _make_instance(mm_cls, quant_algo="int4_gptq", model_name="Qwen3VL")

    call_args = {}

    def fake_apply_chat_template(*args, **kwargs):
        call_args["padding"] = kwargs.get("padding")
        dummy_tensor = type("T", (), {"roll": lambda s, **k: s, "__setitem__": lambda *a: None})()
        return {"input_ids": dummy_tensor}

    inst.processor = type("P", (), {"apply_chat_template": fake_apply_chat_template})()

    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]
    inst._process_and_append(messages)
    assert call_args["padding"] == "max_length"


def test_extract_vision_info_missing_file_raises_value_error(mm_cls):
    """Missing image path should raise ValueError with a helpful message.

    Regression: ``Image.open()`` raised ``FileNotFoundError`` which was
    not caught, producing an unhelpful traceback.
    """
    inst = _make_instance(mm_cls)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "/nonexistent/path/img.jpg"}],
        }
    ]
    with pytest.raises(ValueError, match="Could not open image file"):
        inst._extract_vision_info(messages)


def test_extract_vision_info_skips_non_list_content(mm_cls):
    """Messages with string content (e.g. HunyuanVL adapted) are skipped."""
    inst = _make_instance(mm_cls)
    messages = [
        {"role": "assistant", "content": "plain text response"},
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    ]
    images, videos = inst._extract_vision_info(messages)
    assert images == []
    assert videos == []
