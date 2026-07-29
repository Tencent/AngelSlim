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

"""Sparse-attention subsystem.

Imports are LAZY. Eagerly importing an algorithm whose optional
kernel is unbuilt used to crash the whole subsystem (and anything that touched
it). PEP 562 module-level ``__getattr__`` defers each import until the symbol is
actually accessed, so ``import angelslim.compressor.sparsity`` stays cheap and
never depends on an optional kernel being present.

The legacy standalone entry points (``StemInference`` / ``VecAttentionInference``
+ their ``*_patch`` / config classes) were removed once every algorithm became a
first-class registered ``SparsityAlgorithm``. Construct algorithms through
``angelslim.compressor.sparsity.registry.SparsityAlgorithmRegistry`` instead.
"""

from __future__ import annotations

__all__ = ["Sparsity"]

# Map public symbol -> (submodule, attribute). Imported on first access only.
_LAZY = {
    "Sparsity": (".sparsity", "Sparsity"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    submodule, attr = target
    mod = importlib.import_module(submodule, __name__)
    return getattr(mod, attr)


def __dir__():
    return sorted(__all__)
