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

"""A tiny string->class registry for the pluggable quant axes.

Each axis (formats, schemes, sources) owns one ``Registry``. Adding a strategy means
writing one class and decorating it with ``@some_registry.register("name")`` -- no core
code changes. This keeps the combinatorial explosion of quant strategies out of
if/else branches. (Model adapters use their own tuple-based registry in models/base.py.)
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._table: Dict[str, Type[T]] = {}

    def register(self, key: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator: register a class under ``key`` (raises on duplicate keys)."""

        def _decorator(cls: Type[T]) -> Type[T]:
            if key in self._table:
                raise KeyError(f"[{self.name}] key already registered: {key!r}")
            self._table[key] = cls
            return cls

        return _decorator

    def get(self, key: str) -> Type[T]:
        if key not in self._table:
            raise KeyError(f"[{self.name}] unknown key {key!r}; available: {sorted(self._table)}")
        return self._table[key]

    def create(self, key: str, *args, **kwargs) -> T:
        return self.get(key)(*args, **kwargs)

    def keys(self):
        return tuple(sorted(self._table))
