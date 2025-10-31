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

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

import torch
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel


class DraftModelFactory:
    """
    Factory class for draft models with flexible registration.
    Supports both explicit name registration and direct class name registration.
    """

    _draft_models: Dict[str, Type[PreTrainedModel]] = {}

    @classmethod
    def register(cls, name: Optional[Union[str, Callable]] = None) -> Callable:
        """Decorator to register draft models. Supports two usage patterns:
        1. @DraftModelFactory.register("explicit_name")
        2. @DraftModelFactory.register (uses class name as key)
        """

        # Handler for direct class registration (@DraftModelFactory.register)
        def register_class(model_cls: Type[PreTrainedModel]) -> Type[PreTrainedModel]:
            """Register a class using its own name as the key"""
            key = model_cls.__name__
            if key in cls._draft_models:
                print(f"Draft model '{key}' already exists, will be overwritten.")
            cls._draft_models[key] = model_cls
            return model_cls

        # Handler for named registration (@DraftModelFactory.register("name"))
        def register_with_name(
            key: str,
        ) -> Callable[[Type[PreTrainedModel]], Type[PreTrainedModel]]:
            """Decorator that registers a class with a custom key"""

            def decorator(model_cls: Type[PreTrainedModel]) -> Type[PreTrainedModel]:
                if key in cls._draft_models:
                    print(f"Draft model '{key}' already exists, will be overwritten.")
                cls._draft_models[key] = model_cls
                return model_cls

            return decorator

        # Determine registration type based on input
        if name is None:
            # Case 1: Direct class registration (@DraftModelFactory.register)
            return register_class
        elif isinstance(name, str):
            # Case 2: Explicit name registration (@DraftModelFactory.register("name"))
            return register_with_name(name)
        elif callable(name):
            # Case 3: Direct class registration (called without parentheses)
            return register_class(name)
        else:
            raise TypeError("Invalid argument type for registration")

    @classmethod
    def _get_model_class(cls, config: PretrainedConfig) -> Type[PreTrainedModel]:
        """Get the appropriate model class based on config."""
        architectures = getattr(config, "architectures", [])

        if len(architectures) != 1:
            raise ValueError("Exactly one architecture expected in config")

        arch = architectures[0]

        if arch not in cls._draft_models:
            available = list(cls._draft_models.keys())
            raise ValueError(f"Unknown architecture: {arch}. Available: {available}")

        return cls._draft_models[arch]

    @classmethod
    def from_config(
        cls,
        config: Union[PretrainedConfig, Dict[str, Any], str],
    ) -> PreTrainedModel:
        """Create model from config."""
        # Get model class
        model_class = cls._get_model_class(config)
        model = model_class(config=config)

        dtype_mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        model_dtype = dtype_mapping.get(config.torch_dtype, torch.bfloat16)
        model = model.to(dtype=model_dtype)

        return model

    @classmethod
    def get_available_models(cls) -> list:
        """Get list of all registered draft models."""
        return list(cls._draft_models.keys())


class DraftModelConfig:
    """Helper class for loading draft model configurations."""

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> PretrainedConfig:
        """Create config from file."""
        # Check if it's a file or directory
        if isinstance(config_path, str):
            config_path = Path(config_path)

        if config_path.is_file():
            # It's a config file, load it directly
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            # Get architectures to determine model class
            architectures = config_dict.get("architectures", [])
            if not architectures:
                raise ValueError("Config file must contain 'architectures' field")

            arch = architectures[0]
            if arch not in DraftModelFactory._draft_models:
                available = DraftModelFactory.get_available_models()
                raise ValueError(
                    f"Unknown architecture: {arch}. Available: {available}"
                )

            # Get the model class and its config class
            model_class = DraftModelFactory._draft_models[arch]
            config_class = model_class.config_class

            # Create config instance from dict
            config = config_class(**config_dict)
        else:
            # It's a directory or model name, use AutoConfig
            config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        return config


def create_draft_model(
    config: Union[PretrainedConfig],
    **kwargs: Any,
) -> PreTrainedModel:
    """Convenience function to load draft model."""
    return DraftModelFactory.from_config(config, **kwargs)
