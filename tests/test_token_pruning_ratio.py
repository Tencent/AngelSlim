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

"""Unit tests for ``resolve_num_tokens_to_keep`` ratio validation.

These tests are CPU-only and require neither a GPU nor model weights: the
``torch`` imports and the ``PruningContext`` dependency pulled in by
``algorithm/utils/utils.py`` are stubbed so that the pure keep-count resolution
logic shared by every pruning strategy can be exercised in isolation.

Regression target: every strategy converts a drop ``ratio`` into an absolute
keep count via ``round(num_vision_tokens * (1 - ratio))``. An out-of-range
``ratio`` (e.g. a config typo such as ``ratio: 5``) used to flow straight into
that expression, yielding a negative ``num_to_keep`` that crashed
``torch.empty`` / ``torch.topk`` or silently corrupted the kept-token set.
"""

import importlib.util
import math
import os
import sys
import types

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UTILS_PATH = os.path.join(
    _REPO_ROOT,
    "angelslim",
    "compressor",
    "token_compressor",
    "algorithm",
    "utils",
    "utils.py",
)
_MODULE_NAME = "angelslim.compressor.token_compressor.algorithm.utils.utils"


def _install_stubs():
    """Register lightweight stand-ins so ``utils.py`` imports cleanly."""

    def _module(name):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    class _AnyAttrModule(types.ModuleType):
        """A module whose every attribute access yields a harmless placeholder.

        ``utils.py`` references ``torch.Tensor`` (and friends) inside function type
        annotations, which are evaluated at import time; returning ``object`` for any
        attribute lets the module load without the real ``torch`` installed.
        """

        def __getattr__(self, _name):
            return object

    # torch + torch.nn.functional (only referenced in annotations, never called here)
    if "torch" not in sys.modules:
        torch = _AnyAttrModule("torch")
        sys.modules["torch"] = torch
        torch_nn = _AnyAttrModule("torch.nn")
        sys.modules["torch.nn"] = torch_nn
        torch_functional = _AnyAttrModule("torch.nn.functional")
        sys.modules["torch.nn.functional"] = torch_functional
        torch_nn.functional = torch_functional
        torch.nn = torch_nn

    # Parent packages of the target module, marked as packages.
    for pkg in (
        "angelslim",
        "angelslim.compressor",
        "angelslim.compressor.token_compressor",
        "angelslim.compressor.token_compressor.base",
        "angelslim.compressor.token_compressor.algorithm",
        "angelslim.compressor.token_compressor.algorithm.utils",
    ):
        if pkg not in sys.modules:
            mod = _module(pkg)
            mod.__path__ = []  # mark as package

    # angelslim...base.context.PruningContext (resolved via the relative import)
    ctx_name = "angelslim.compressor.token_compressor.base.context"
    if ctx_name not in sys.modules:
        ctx = _module(ctx_name)
        ctx.PruningContext = type("PruningContext", (), {})


def _load_utils_module():
    _install_stubs()
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def resolve():
    return _load_utils_module().resolve_num_tokens_to_keep


@pytest.mark.parametrize(
    "ratio,num_tokens,expected",
    [
        (0.0, 100, 100),  # drop nothing -> keep all
        (0.25, 100, 75),
        (0.5, 100, 50),
        (0.9, 100, 10),
        (1.0, 100, 0),  # drop everything -> keep none
        (0.0, 0, 0),  # no vision tokens at all
    ],
)
def test_in_range_ratio_resolves_expected_keep_count(resolve, ratio, num_tokens, expected):
    assert resolve(ratio, num_tokens) == expected


def test_rounding_to_zero_retains_one_token(resolve):
    """A benign rounding-to-zero must never drop the whole image (ratio < 1.0)."""
    # round(3 * (1 - 0.99)) == round(0.03) == 0 -> clamped up to 1
    assert resolve(0.99, 3) == 1


def test_ratio_one_keeps_zero_not_one(resolve):
    """ratio == 1.0 is an explicit "keep none"; the retain-one guard must not fire."""
    assert resolve(1.0, 3) == 0


@pytest.mark.parametrize("ratio", [1.0001, 5, 100.0])
def test_ratio_above_one_raises(resolve, ratio):
    """ratio > 1.0 previously produced a negative keep count (torch crash)."""
    with pytest.raises(ValueError, match=r"\[TokenCompressor Error\] 'ratio'"):
        resolve(ratio, 100)


@pytest.mark.parametrize("ratio", [-0.1, -1, -100.0])
def test_negative_ratio_raises(resolve, ratio):
    with pytest.raises(ValueError, match=r"\[TokenCompressor Error\] 'ratio'"):
        resolve(ratio, 100)


def test_nan_ratio_raises(resolve):
    with pytest.raises(ValueError, match=r"\[TokenCompressor Error\] 'ratio'"):
        resolve(math.nan, 100)


@pytest.mark.parametrize("ratio", [None, "0.5", True])
def test_non_numeric_ratio_raises(resolve, ratio):
    """``bool`` is rejected explicitly so ``True``/``False`` never act as 1/0."""
    with pytest.raises(ValueError, match=r"\[TokenCompressor Error\] 'ratio'"):
        resolve(ratio, 100)
