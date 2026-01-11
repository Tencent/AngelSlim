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

from typing import Any, Callable, Dict, Type

from .base_dataset_builder import DatasetBuilder


class DatasetBuilderFactory:
    """
    Factory class for dataset builders with flexible registration.
    """

    _builders: Dict[str, Type[DatasetBuilder]] = {}

    @classmethod
    def register(
        cls,
        training_mode: str = "online",
        modal_type: str = "LLM",
        target_model_type: str = None,
    ) -> Callable[[Type[DatasetBuilder]], Type[DatasetBuilder]]:
        """Decorator to register dataset builders."""

        def decorator(builder_cls: Type[DatasetBuilder]) -> Type[DatasetBuilder]:
            if (training_mode, modal_type, target_model_type) in cls._builders:
                print(
                    f"DatasetBuilder for training_mode '{training_mode}'"
                    f" modal_type '{modal_type}' already exists."
                    f" target_model_type '{target_model_type}' already exists."
                )
            cls._builders[(training_mode, modal_type, target_model_type)] = builder_cls
            return builder_cls

        return decorator

    @classmethod
    def create(
        cls,
        training_mode: str = "online",
        modal_type: str = "LLM",
        target_model_type: str = None,
        **kwargs: Any,
    ) -> DatasetBuilder:
        """Create a dataset builder instance based on training_mode and modal_type."""
        if (training_mode, modal_type, target_model_type) not in cls._builders:
            available = list(cls._builders.keys())
            raise ValueError(
                f"Unknown training_mode '{training_mode}' "
                f"modal_type '{modal_type}' "
                f"target_model_type '{target_model_type}'. "
                f"Available: {available}"
            )

        builder_class = cls._builders[(training_mode, modal_type, target_model_type)]
        return builder_class(**kwargs)

    @classmethod
    def get_available_builders(cls) -> list:
        """Get list of all registered modal types."""
        return list(cls._builders.keys())
