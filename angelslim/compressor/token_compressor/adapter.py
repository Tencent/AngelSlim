# Copyright 2026 Tencent Inc. All Rights Reserved.
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
from typing import Any, Dict, List, Tuple

import torch.nn as nn

from .base.config import TokenCompressorConfig
from .utils.config_utils import plan_pruning_execution


class UniversalPruningAdapter:
    """
    A metadata-driven adapter that transforms standard models into prunable models.
    The transformation sequence and targets are determined at initialization.
    """

    def __init__(
        self,
        model: nn.Module,
        strategy_config: TokenCompressorConfig,
        raw_map_data: List[Dict[str, Any]],
    ):
        """
        Args:
            model: The base HuggingFace model.
            strategy_config: User-defined compression and data requirements.
            raw_map_data: The ordered list of component mappings from YAML.
        """
        self.model = model

        # 1. Generate the immutable execution plan at initialization
        self.strategy_config, self.execution_plan = plan_pruning_execution(
            strategy_config=strategy_config,
            raw_map_data=raw_map_data,
            model_config=getattr(model, "config", None),
        )

        # 2. Initialize backup storage for original module pointers
        if not hasattr(self.model, "old_model"):
            self.model.old_model = {}

    def _get_parent_and_attr(self, path: str) -> Tuple[Any, str]:
        """Resolves a dot-separated string path to (parent_object, attribute_name)."""
        parts = path.split(".")
        current = self.model
        for part in parts[:-1]:
            current = getattr(current, part)
        return current, parts[-1]

    def _get_wrapper_class(self, module_path: str, class_name: str) -> Any:
        """
        Dynamically imports the specified wrapper class.
        """
        # Relative import logic: assumes we are inside the 'token_compressor'
        # package
        module = importlib.import_module(
            module_path,
            package="angelslim.compressor.token_compressor",
        )
        return getattr(module, class_name)

    def _expand_execution_step(self, step: Dict[str, Any]) -> List[Tuple[str, int]]:
        """
        Expands a plan step into physical module paths.
        Handles '[n]' by referencing the 'indices' field determined during planning.
        """
        path_template = step["path"]
        if "[n]" not in path_template:
            return [(path_template, -1)]

        prefix, suffix = path_template.split("[n]")
        suffix = suffix.lstrip(".")
        container_path = prefix.rstrip(".")

        parent, attr = self._get_parent_and_attr(container_path)
        container = getattr(parent, attr)

        # Use planned indices; if None, default to the entire range of the
        # container
        target_indices = step.get("indices")
        if target_indices is None:
            target_indices = range(len(container))

        expanded = []
        for i in target_indices:
            full_path = f"{container_path}.{i}"
            if suffix:
                full_path += f".{suffix}"
            expanded.append((full_path, i))
        return expanded

    def wrap_model(self) -> nn.Module:
        """
        Sequentially wraps model components according to the execution_plan.
        """
        for step in self.execution_plan:
            name = step["name"]
            wrapper_mod = step["wrapper_module"]
            wrapper_cls = step["wrapper_class"]

            WrapperClass = self._get_wrapper_class(wrapper_mod, wrapper_cls)
            targets = self._expand_execution_step(step)

            if name not in self.model.old_model:
                self.model.old_model[name] = {}

            print(f"targets: {targets}")

            for path, idx in targets:
                parent, attr_name = self._get_parent_and_attr(path)
                original_module = getattr(parent, attr_name)

                # Prevent double-wrapping
                if not isinstance(original_module, WrapperClass):
                    # Store original module for safe recovery
                    backup_key = idx if idx != -1 else "single"
                    self.model.old_model[name][backup_key] = original_module

                    # Instantiate and replace with the prunable wrapper
                    new_module = WrapperClass(original_module, self.strategy_config)

                    # Explicitly inject the layer index for collection
                    # components
                    if idx != -1:
                        new_module.layer_idx = idx

                    setattr(parent, attr_name, new_module)

            print(f"[UniversalAdapter] '{name}' wrapped successfully")

        return self.model

    def unwrap_model(self) -> nn.Module:
        """
        Restores the model to its original state by iterating the plan in REVERSE order.
        """
        if not hasattr(self.model, "old_model") or not self.model.old_model:
            return self.model

        # Order is reversed to restore nested modules from inside-out
        for step in reversed(self.execution_plan):
            name = step["name"]
            if name not in self.model.old_model:
                continue

            targets = self._expand_execution_step(step)
            backups = self.model.old_model[name]

            for path, idx in targets:
                backup_key = idx if idx != -1 else "single"
                if backup_key in backups:
                    parent, attr_name = self._get_parent_and_attr(path)
                    setattr(parent, attr_name, backups[backup_key])

            print(f"[UniversalAdapter] '{name}' successfully restored.")

        # Cleanup metadata
        self.model.old_model = {}
        if hasattr(self.model, "_pruning_adapter"):
            del self.model._pruning_adapter

        print("[UniversalAdapter] Model fully reverted to standard architecture.")
        return self.model
