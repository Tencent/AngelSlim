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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import yaml


@dataclass
class StrategyConfig:
    """Configuration for a specific pruning or merging algorithm."""

    strategy: str
    """The registered name of the compression algorithm (e.g., 'idpruner', 'random')."""

    params: Dict[str, Any] = field(default_factory=dict)
    """Hyperparameters to be passed to the strategy function."""

    model_related_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Model-specific parameters to be passed to the strategy function."""


@dataclass
class DataRequirements:
    """Capturing requirements stored in the PruningContext."""

    inputs_embeds: bool = False
    """Whether to capture multimodal embeddings."""

    vision_token_mask: bool = False
    """Whether to generate the visual token boolean mask."""

    feature_map: bool = False
    """Whether to capture hidden states for diversity analysis."""

    image_grid_thw: bool = False
    """Whether to capture image geometry metadata."""

    vit_q_layers: List[int] = field(default_factory=list)
    """Indices of Vision Tower layers to capture Query states from."""

    vit_k_layers: List[int] = field(default_factory=list)
    """Indices of Vision Tower layers to capture Key states from."""

    llm_q_layers: List[int] = field(default_factory=list)
    """Indices of Language Model layers to capture Query states from."""

    llm_k_layers: List[int] = field(default_factory=list)
    """Indices of Language Model layers to capture Key states from."""

    def needs_vit_q(self, layer_idx: int) -> bool:
        """Checks if a specific Vision Tower layer should capture its Query states."""
        return layer_idx in self.vit_q_layers

    def needs_vit_k(self, layer_idx: int) -> bool:
        """Checks if a specific Vision Tower layer should capture its Key states."""
        return layer_idx in self.vit_k_layers

    def needs_llm_q(self, layer_idx: int) -> bool:
        """Checks if a specific Language Model layer should capture its Query states."""
        return layer_idx in self.llm_q_layers

    def needs_llm_k(self, layer_idx: int) -> bool:
        """Checks if a specific Language Model layer should capture its Key states."""
        return layer_idx in self.llm_k_layers


@dataclass
class TokenCompressorConfig:
    """The root configuration object for token compression, containing all requirements
    and multi-stage strategies."""

    requirements: DataRequirements = field(default_factory=DataRequirements)
    """The centralized data capturing plan for the model adapter."""

    strategies: Dict[Union[str, int], StrategyConfig] = field(default_factory=dict)
    """A map of execution stages ('global' or layer indices)
    to their respective strategy configurations."""

    @classmethod
    def from_yaml(cls, path: str) -> "TokenCompressorConfig":
        """
        Parse YAML configuration file
        Return a structured TokenCompressorConfig object.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        compressor_data = raw_data.get("compressor", {})

        req_raw = compressor_data.get("requirements", {})
        requirements = DataRequirements(**req_raw)

        strategies_raw = compressor_data.get("strategies", {})
        strategies = {}
        for stage, conf in strategies_raw.items():
            if isinstance(stage, str) and stage.isdigit():
                key = int(stage)
            elif isinstance(stage, int):
                key = stage
            else:
                key = stage

            strategies[key] = StrategyConfig(
                strategy=conf["strategy"],
                params=conf.get("params", {}),
                model_related_params=conf.get("model_related_params", {}),
            )

        return cls(requirements=requirements, strategies=strategies)
