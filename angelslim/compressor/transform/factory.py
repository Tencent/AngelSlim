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

from .base import TransformBase

__all__ = ["TransformFactory"]


class _NoOpTransform(TransformBase):
    """No-op transform returned when slim_config has no transform_config."""

    def __init__(self, quant_model, slim_config=None):
        # slim_config may be a dict (PTQ path), skip TransformBase.__init__ attribute assignment
        self.quant_model = quant_model
        self.config = slim_config

    def run(self):
        pass


class TransformFactory:
    """Factory for creating TransformBase instances from config.

    Usage
    -----
        transform = TransformFactory.create(slim_model, slim_config)
        transform.run()

    The transform name is read from ``slim_config.transform_config["name"]``,
    which corresponds to the ``transform.name`` field in the YAML config:

        transform:
          name: SpinQuant
          spin_config: ...

    Registering a new transform
    ---------------------------
        @TransformFactory.register("MyTransform")
        class MyTransform(TransformBase):
            ...
    """

    _registry: dict[str, type[TransformBase]] = {}

    @classmethod
    def create(cls, quant_model, slim_config) -> TransformBase:
        """Instantiate a transform from slim_config.

        Args:
            quant_model: The wrapped slim model.
            slim_config: Config object with a ``transform_config`` dict containing ``"name"``.

        Returns:
            An unrun TransformBase instance. Call ``.run()`` to apply the transform.

        Raises:
            ValueError: If transform name is missing or not registered.
        """
        # slim_config may be a dict (PTQ path) or an object with attributes (transform path)
        if isinstance(slim_config, dict):
            transform_config = slim_config.get("transform_config")
        else:
            transform_config = getattr(slim_config, "transform_config", None)

        if not transform_config:
            return _NoOpTransform(quant_model, slim_config)

        name = (
            transform_config.get("name")
            if isinstance(transform_config, dict)
            else getattr(transform_config, "name", None)
        )
        if not name:
            return _NoOpTransform(quant_model, slim_config)

        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown transform '{name}'. Available: {available}")

        return cls._registry[name](quant_model, slim_config)

    @classmethod
    def register(cls, name: str):
        """Decorator to register a TransformBase subclass under the given name.

        Args:
            name: The string key used in YAML ``transform.name``.

        Example:
            @TransformFactory.register("MyTransform")
            class MyTransform(TransformBase):
                ...
        """

        def decorator(cls_):
            if not issubclass(cls_, TransformBase):
                raise TypeError(f"{cls_.__name__} must be a subclass of TransformBase")
            cls._registry[name] = cls_
            return cls_

        return decorator

    @classmethod
    def list_transforms(cls) -> list[str]:
        """Return names of all registered transforms."""
        return list(cls._registry.keys())
