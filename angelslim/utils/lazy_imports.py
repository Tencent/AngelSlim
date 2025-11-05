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

import importlib
from typing import Any

"""
Central lazy import module for AngelSlim toolkit.
This module provides lazy loading functionality for optional dependencies,
delaying actual imports until the packages are first used.
"""


class LazyModule:
    """
    A proxy class for lazy module loading.

    This class delays the actual import of a module until its attributes are
    first accessed, which helps reduce startup time and memory usage when
    dealing with optional dependencies that may not be used in every execution.

    Attributes:
        _module_name (str): The full name of the module to import
        _extra_group (str): The extra dependency group required for this module
        _module (ModuleType): The actual imported module (None until first access)
        _submodules (dict): Cache for LazyModule instances of submodules

    Example:
        >>> ray = LazyModule('ray', 'speculative')
        >>> # The actual import happens here on first attribute access
        >>> ray.init()
    """

    def __init__(self, module_name: str, extra_group: str = None):
        """
        Initialize a lazy module wrapper.

        Args:
            module_name: Full name of the module to import (e.g., 'diffusers')
            extra_group: Name of the extra dependency group required for this module
        """
        self._module_name = module_name
        self._extra_group = extra_group
        self._module = None
        self._submodules = {}

    def _import_module(self):
        """
        Import the target module if not already imported.

        Raises:
            ImportError: If the module cannot be imported
        """
        if self._module is None:
            try:
                self._module = importlib.import_module(self._module_name)
            except ImportError as e:
                if self._extra_group:
                    raise ImportError(
                        f"Module '{self._module_name}' requires "
                        f"additional dependencies. Please install: "
                        f"pip install 'angelslim[{self._extra_group}]'"
                    ) from e
                raise

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the actual module.

        On first access, this method imports the target module and then
        delegates the attribute lookup to the actual module. For submodules,
        it returns a LazyModule instance to support multi-level access.

        Args:
            name: Name of the attribute to access

        Returns:
            The requested attribute from the target module, or a LazyModule
            instance for submodules

        Raises:
            ImportError: If the module cannot be imported and an
                extra_group is specified, provides installation instructions
        """
        # Return cached submodule if exists
        if name in self._submodules:
            return self._submodules[name]

        self._import_module()

        # Try to get the attribute from the imported module
        try:
            attr = getattr(self._module, name)
            # If it's a module, wrap it in LazyModule for consistent behavior
            if isinstance(attr, type(self._module)):
                submodule_name = f"{self._module_name}.{name}"
                lazy_submodule = LazyModule(submodule_name, self._extra_group)
                lazy_submodule._module = attr  # Cache the already imported module
                self._submodules[name] = lazy_submodule
                return lazy_submodule
            return attr
        except AttributeError:
            # If attribute not found, try importing as a submodule
            submodule_name = f"{self._module_name}.{name}"
            try:
                # Create a LazyModule for the submodule
                lazy_submodule = LazyModule(submodule_name, self._extra_group)
                # Trigger import to verify it exists
                lazy_submodule._import_module()
                # Cache it
                self._submodules[name] = lazy_submodule
                # Also cache in parent module for consistency
                setattr(self._module, name, lazy_submodule._module)
                return lazy_submodule
            except ImportError:
                # If submodule import fails, re-raise the original AttributeError
                raise AttributeError(
                    f"module '{self._module_name}' has no attribute '{name}'"
                )


class LazyAttribute:
    """
    A proxy class for lazy loading of specific module attributes.

    This class delays the import of a module and retrieval of a specific attribute
    until the attribute is first accessed. Useful for optimizing imports of
    large modules when only specific components are needed.

    Attributes:
        _module_name (str): The name of the module containing the target attribute
        _attribute_name (str): The name of the specific attribute to load
        _extra_group (str): The extra dependency group required for this attribute
        _attribute (Any): The actual attribute value (None until first access)
    """

    def __init__(self, module_name: str, attribute_name: str, extra_group: str = None):
        """
        Initialize a lazy attribute wrapper.

        Args:
            module_name: Name of the module containing the target attribute
            attribute_name: Name of the specific attribute to load lazily
            extra_group: Name of the extra dependency group required
        """
        self._module_name = module_name
        self._attribute_name = attribute_name
        self._extra_group = extra_group
        self._attribute = None

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the target attribute.

        On first access, this method imports the module and retrieves the
        target attribute, then delegates subsequent attribute access to it.

        Args:
            name: Name of the attribute to access

        Returns:
            The requested attribute from the target attribute

        Raises:
            ImportError: If the module cannot be imported and an extra_group
                is specified, provides installation instructions
        """
        if self._attribute is None:
            try:
                module = importlib.import_module(self._module_name)
                self._attribute = getattr(module, self._attribute_name)
            except ImportError as e:
                if self._extra_group:
                    raise ImportError(
                        f"Attribute '{self._attribute_name}' requires "
                        f"additional dependencies. Please install: "
                        f"pip install 'angelslim[{self._extra_group}]'"
                    ) from e
                raise
        return getattr(self._attribute, name)


# Create global lazy loading objects for optional dependencies

# --- Speculative decoding related lazy imports ---
ray = LazyModule("ray", "speculative")
fastchat = LazyModule("fastchat", "speculative")
openai = LazyModule("openai", "speculative")
anthropic = LazyModule("anthropic", "speculative")
jsonschema_specifications = LazyModule("jsonschema_specifications", "speculative")
referencing = LazyModule("referencing", "speculative")
deepspeed = LazyModule("deepspeed", "speculative")
vllm = LazyModule("vllm", "speculative")
shortuuid = LazyModule("shortuuid", "speculative")


# --- multimodal related lazy imports ---
qwen_vl_utils = LazyModule("qwen_vl_utils", "multimodal")
qwen_omni_utils = LazyModule("qwen_omni_utils", "multimodal")
