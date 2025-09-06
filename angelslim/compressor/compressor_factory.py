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

from typing import Any, Callable, Dict, Optional, Type, Union

from ..utils import print_info


class CompressorFactory:
    """
    Factory class for model compression methods with flexible registration.
    Supports both explicit name registration and direct class name registration.
    """

    _compress_methods: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, name: Optional[Union[str, Callable]] = None) -> Callable:
        """Decorator to register compression methods. Supports two usage patterns:
        1. @CompressorFactory.register("explicit_name")
        2. @CompressorFactory.register (uses class name as key)
        """

        # Handler for direct class registration (@CompressorFactory.register)
        def register_class(compress_cls: Type[Any]) -> Type[Any]:
            """Register a class using its own name as the key"""
            key = compress_cls.__name__
            if key in cls._compress_methods:
                print_info(
                    f"Compression method '{key}' already exists, will be overwritten."
                )
            cls._compress_methods[key] = compress_cls
            return compress_cls

        # Handler for named registration (@CompressorFactory.register("name"))
        def register_with_name(key: str) -> Callable[[Type[Any]], Type[Any]]:
            """Decorator that registers a class with a custom key"""

            def decorator(compress_cls: Type[Any]) -> Type[Any]:
                if key in cls._compress_methods:
                    print_info(f"register '{key}' already exists, will be overwritten.")
                cls._compress_methods[key] = compress_cls
                return compress_cls

            return decorator

        # Determine registration type based on input
        if name is None:
            # Case 1: Direct class registration (@CompressorFactory.register)
            return register_class
        elif isinstance(name, str):
            # Case 2: Explicit name registration (@CompressorFactory.register("name"))
            return register_with_name(name)
        elif callable(name):
            # Case 3: Direct class registration (called without parentheses)
            return register_class(name)
        else:
            raise TypeError("Invalid argument type for registration")

    @classmethod
    def create(cls, names: list, model: Any, slim_config: Any) -> Any:
        """Create compressor instance"""
        compressor = []
        for name in names:
            if name not in cls._compress_methods:
                available = list(cls._compress_methods.keys())
                raise ValueError(
                    f"Compress method '{name}' not registered. Available: {available}"
                )
            compressor.append(cls._compress_methods[name](model, slim_config))
        return compressor

    @classmethod
    def get_available_compressor(cls) -> list:
        return list(cls._compress_methods.keys())
