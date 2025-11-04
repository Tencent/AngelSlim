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

from typing import Any, Dict, Optional, Type


class SlimModelFactory:
    """Factory for model creation using class decorator registration"""

    # Registry to store mapping of class names to model classes
    registry: Dict[str, Type] = {}
    series_registry: Dict[str, str] = {}

    ALLOWED_SERIES = ("LLM", "VLM", "Diffusion", "Omni")

    @classmethod
    def register(cls, model_class: Type) -> Type:
        """Class decorator for automatic registration"""
        class_name = model_class.__name__
        if class_name in cls.registry:
            raise ValueError(f"Model class '{class_name}' is already registered")
        cls.registry[class_name] = model_class

        module_path = model_class.__module__.lower()
        if "llm" in module_path:
            series = "LLM"
        elif "vlm" in module_path:
            series = "VLM"
        elif "diffusion" in module_path:
            series = "Diffusion"
        elif "omni" in module_path:
            series = "Omni"
        else:
            raise ValueError(
                f"model_class '{class_name}' is not in a valid series: {cls.ALLOWED_SERIES}"  # noqa: E501
            )

        cls.series_registry[class_name] = series

        return model_class

    @classmethod
    def create(
        cls,
        model_name: str,
        model: Optional[Any] = None,
        deploy_backend: str = "vllm",
        **kwargs,
    ) -> Any:
        """Create an instance of the specified model"""
        if model_name not in cls.registry:
            available = ", ".join(cls.registry.keys())
            raise ValueError(
                f"Unknown model: '{model_name}'. Available models: {available}"
            )

        return cls.registry[model_name](
            model=model,
            deploy_backend=deploy_backend,
            **kwargs,
        )

    @classmethod
    def get_model_class(cls, model_name: str) -> Type:
        """Get the class for a registered model"""
        if model_name not in cls.registry:
            available = ", ".join(cls.registry.keys())
            raise ValueError(
                f"Unknown model: '{model_name}'. Available models: {available}"
            )
        return cls.registry[model_name]

    @classmethod
    def get_registered_models(cls) -> Dict[str, Type]:
        """Get all registered model classes"""
        return cls.registry.copy()

    @classmethod
    def get_series_by_models(cls, model_name: str) -> str:
        """Get all model classes for a specific series"""
        if model_name not in cls.registry:
            available = ", ".join(cls.registry.keys())
            raise ValueError(
                f"Unknown model: '{model_name}'. Available models: {available}"
            )
        return cls.series_registry.get(model_name, [])
