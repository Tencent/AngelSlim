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

"""Registry mapping user-facing sparse algorithm names to implementations.

A single ``name`` field carries the user-facing variant, mirroring PTQ
(``compression.quantization.name = fp8_static``). The MInference family maps
three names (``minference`` / ``a_shape`` / ``tri_shape``) onto one class.

The KV-compression family is not implemented in this release. The forward-compat
guard is a simple name allow-list — any of the deferred KV names raises
``NotImplementedError``.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Type  # noqa: F401

from ._base import SparsityAlgorithm

# Algorithm names for the deferred KV-compression family. Selecting one raises
# an actionable NotImplementedError rather than silently doing nothing.
_KV_COMPRESSION = frozenset({"snapkv", "streamingllm", "pyramidkv", "quest"})


class SparsityAlgorithmRegistry:
    """Name -> SparsityAlgorithm factory registry.

    Algorithms register a factory (usually the class itself, or a partial that
    binds a fixed ``variant``). ``create`` instantiates with ``attn_kwargs``.
    """

    _factories: Dict[str, Callable[..., SparsityAlgorithm]] = {}
    _algorithms_loaded = False

    @classmethod
    def _ensure_algorithms_registered(cls) -> None:
        """Import the algorithms package so its classes register themselves.

        Registration is an import side-effect of
        ``angelslim.compressor.sparsity.algorithms``. Now that the compressor
        package no longer eager-imports that subpackage (to keep a bare
        ``import ...sparsity.registry`` from pulling the quant stack), the
        registry triggers it lazily on first use — so ``create`` / ``available``
        are correct regardless of whether the caller imported ``algorithms``
        first. Idempotent; the recursion guard makes the algorithms' own
        ``register`` calls (which re-enter nothing here) safe.
        """
        if cls._algorithms_loaded:
            return
        cls._algorithms_loaded = True  # set BEFORE importing to avoid re-entry
        try:
            import importlib

            importlib.import_module("angelslim.compressor.sparsity.algorithms")
        except Exception:  # noqa: BLE001
            # A missing optional kernel must not break registry lookups; the
            # algorithms package itself guards per-algorithm imports. Leave the
            # flag set so we don't retry a doomed import on every call.
            pass

    @classmethod
    def register(cls, name: str):
        """Decorator: register an algorithm class under ``name``."""

        def _wrap(factory: Callable[..., SparsityAlgorithm]):
            # Last registration wins (mirrors CompressorFactory): the assignment
            # below already overwrites any prior entry.
            cls._factories[name] = factory
            return factory

        return _wrap

    @classmethod
    def register_factory(cls, name: str, factory: Callable[..., SparsityAlgorithm]):
        """Imperative variant of :meth:`register` (for family aliases)."""
        cls._factories[name] = factory

    @classmethod
    def create(cls, name: str, attn_kwargs: Optional[dict] = None) -> SparsityAlgorithm:
        cls._ensure_algorithms_registered()
        if name in _KV_COMPRESSION:
            raise NotImplementedError(
                f"Sparse algorithm {name!r} is a KV-compression method, which "
                f"is out of scope for this release (sparse-prefill only). The "
                f"available prefill algorithms are: {sorted(cls._factories)}."
            )
        if name not in cls._factories:
            available = sorted(cls._factories)
            raise ValueError(
                f"Sparse algorithm {name!r} not registered. "
                f"Available: {available}. "
                f"(KV-compression names {sorted(_KV_COMPRESSION)} are not "
                f"implemented.)"
            )
        return cls._factories[name](attn_kwargs=attn_kwargs)

    @classmethod
    def available(cls) -> list:
        cls._ensure_algorithms_registered()
        return sorted(cls._factories)
