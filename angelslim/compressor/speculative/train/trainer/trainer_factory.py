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

from .eagle3_trainer import Eagle3Trainer


class Eagle3TrainerFactory:
    """
    Factory class for dataset builders with flexible registration.
    """

    _builders: Dict[tuple[str, str], Type[Eagle3Trainer]] = {}

    @classmethod
    def register(
        cls, training_mode: str = "online", modal_type: str = "LLM"
    ) -> Callable[[Type[Eagle3Trainer]], Type[Eagle3Trainer]]:
        """Decorator to register dataset builders with a modal_type."""

        def decorator(builder_cls: Type[Eagle3Trainer]) -> Type[Eagle3Trainer]:
            if (training_mode, modal_type) in cls._builders:
                print(
                    f"Eagle3Trainer for training_mode '{training_mode}' "
                    f"and modal_type '{modal_type}' already exists."
                )
            cls._builders[(training_mode, modal_type)] = builder_cls
            return builder_cls

        return decorator

    @classmethod
    def create(
        cls, training_mode: str = "online", modal_type: str = "LLM", **kwargs: Any
    ) -> Eagle3Trainer:
        """Create a dataset builder instance based on modal_type."""
        if (training_mode, modal_type) not in cls._builders:
            available = list(cls._builders.keys())
            raise ValueError(
                f"Unknown modal_type training_mode '{training_mode}'"
                f" and '{modal_type}'. Available: {available}"
            )

        builder_class = cls._builders[(training_mode, modal_type)]
        return builder_class(**kwargs)

    @classmethod
    def get_available_builders(cls) -> list:
        """Get list of all registered modal types."""
        return list(cls._builders.keys())
