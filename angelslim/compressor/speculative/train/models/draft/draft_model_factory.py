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
from typing import Any, Dict, Union

import torch
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from .llama_eagle3 import Eagle3LLamaforCausalLM


class DraftModelFactory:
    """
    Factory for creating draft models based on architecture field.

    This approach keeps model_type="llama" but uses the architectures field
    to determine which model class to instantiate.
    """

    _ARCHITECTURE_MAPPING = {
        "Eagle3LLamaforCausalLM": Eagle3LLamaforCausalLM,
    }

    @classmethod
    def _get_model_class(cls, config: PretrainedConfig):
        """Get the appropriate model class based on config."""
        architectures = getattr(config, "architectures", [])

        if len(architectures) != 1:
            raise ValueError("Exactly one architecture expected in config")

        arch = architectures[0]

        if cls._ARCHITECTURE_MAPPING.get(arch, None) is None:
            raise ValueError(f"Unknown architecture: {arch}")

        return cls._ARCHITECTURE_MAPPING[arch]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        **kwargs: Any,
    ) -> PreTrainedModel:
        """
        Load a pretrained model with architecture-based selection.

        Args:
            model_name_or_path: Path to pretrained model
            load_emb: Whether to load embeddings (for Eagle3 models)
            **kwargs: Additional arguments

        Returns:
            Loaded model instance

        Example:
            >>> # Load Eagle3 model (architectures=["Eagle3LLamaforCausalLM"])
            >>> model = DraftModelFactory.from_pretrained("/path/to/eagle3/model")
        """
        # Load config
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        # Get model class
        model_class = cls._get_model_class(config)

        model = model_class.from_pretrained(model_name_or_path, **kwargs)
        return model

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


class DraftModelConfig:
    _ARCHITECTURE_MAPPING = {
        "Eagle3LLamaforCausalLM": Eagle3LLamaforCausalLM,
    }

    @classmethod
    def from_file(cls, config_path: Union[str, Path]):
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
            if arch not in cls._ARCHITECTURE_MAPPING:
                raise ValueError(f"Unknown architecture: {arch}")

            # Get the model class and its config class
            model_class = cls._ARCHITECTURE_MAPPING[arch]
            config_class = model_class.config_class

            # Create config instance from dict
            config = config_class(**config_dict)
        else:
            # It's a directory or model name, use AutoConfig
            config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        return config


def create_draft_model(
    config_path: Union[str, Path],
    **kwargs: Any,
) -> PreTrainedModel:
    """Convenience function to load draft model."""
    return DraftModelFactory.from_config(config_path, **kwargs)
