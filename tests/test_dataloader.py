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

"""Unit tests for ``DataLoaderFactory`` data-type resolution.

These tests are CPU-only and require neither a GPU, model weights, nor the
heavy ``torch``/``transformers`` stack: the dataset classes and third-party
imports pulled in by ``angelslim.data.dataloader`` are stubbed so that the
pure ``_resolve_data_type`` string logic can be exercised in isolation.
"""

import importlib.util
import os
import sys
import types

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATALOADER_PATH = os.path.join(_REPO_ROOT, "angelslim", "data", "dataloader.py")


def _install_stubs():
    """Register lightweight stand-ins so ``dataloader.py`` imports cleanly."""

    def _module(name):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    # torch.utils.data.DataLoader
    if "torch" not in sys.modules:
        torch = _module("torch")
        torch.utils = _module("torch.utils")
        torch_data = _module("torch.utils.data")
        torch_data.DataLoader = object
        torch.utils.data = torch_data

    # transformers.ProcessorMixin
    if "transformers" not in sys.modules:
        transformers = _module("transformers")
        transformers.ProcessorMixin = object

    # angelslim.data.<dataset> siblings imported by dataloader.py
    for pkg in ("angelslim", "angelslim.data"):
        if pkg not in sys.modules:
            mod = _module(pkg)
            mod.__path__ = []  # mark as package
    for leaf, cls in (
        ("audio_dataset", "AudioDataset"),
        ("base_dataset", "BaseDataset"),
        ("multimodal_dataset", "MultiModalDataset"),
        ("omni_dataset", "OmniDataset"),
        ("text2image_dataset", "Text2ImageDataset"),
        ("text_dataset", "TextDataset"),
    ):
        name = f"angelslim.data.{leaf}"
        if name not in sys.modules:
            mod = _module(name)
            setattr(mod, cls, type(cls, (), {}))


def _load_dataloader_module():
    _install_stubs()
    spec = importlib.util.spec_from_file_location("angelslim.data.dataloader", _DATALOADER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def factory():
    return _load_dataloader_module().DataLoaderFactory


@pytest.mark.parametrize(
    "source,expected",
    [
        ("calib.json", "TextDataset"),
        ("CALIB.JSON", "TextDataset"),
        ("data.parquet", "TextDataset"),
        ("images/", "MultiModalDataset"),
        ({"split": "train"}, "MultiModalDataset"),
    ],
)
def test_auto_resolves_to_a_dispatchable_id(factory, source, expected):
    """``"auto"`` must resolve to a canonical id the dispatch can match.

    Regression: auto-detection previously produced ``"text"``/``"multimodal"``
    which no dispatch branch matched, so every ``data_type="auto"`` call raised
    ``ValueError: Unsupported data type``.
    """
    assert factory._resolve_data_type("auto", source) == expected


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("text", "TextDataset"),
        ("multimodal", "MultiModalDataset"),
        ("text2image", "Text2ImageDataset"),
        ("omni", "OmniDataset"),
        ("audio", "AudioDataset"),
    ],
)
def test_documented_short_aliases_resolve(factory, alias, expected):
    """The short aliases named in the docstring must be accepted."""
    assert factory._resolve_data_type(alias, "calib.json") == expected


@pytest.mark.parametrize(
    "canonical",
    [
        "TextDataset",
        "MultiModalDataset",
        "Text2ImageDataset",
        "OmniDataset",
        "AudioDataset",
    ],
)
def test_canonical_ids_pass_through(factory, canonical):
    """Existing config-supplied class names must remain valid (no regression)."""
    assert factory._resolve_data_type(canonical, "calib.json") == canonical


def test_unknown_data_type_raises_value_error(factory):
    """An unrecognized value raises a descriptive error, not a silent miss."""
    with pytest.raises(ValueError, match="Unsupported data type"):
        factory._resolve_data_type("nonexistent", "calib.json")
