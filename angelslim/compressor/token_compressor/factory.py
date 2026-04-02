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

"""
Token Compression Strategy Factory module.
Responsible for registering and dispatching pruning and merging algorithms.
"""

from typing import Callable, Dict

from .algorithm.attention_based import special_token_based_attention_pruning

# Algorithm Imports
from .algorithm.basic import baseline_pruning, override_pruning, random_pruning
from .algorithm.dart import dart_pruning
from .algorithm.divprune import divprune
from .algorithm.hiprune import hiprune_pruning
from .algorithm.idpruner import idpruner
from .algorithm.scope import scope_pruning
from .algorithm.visionselector import vision_selector_pruning
from .algorithm.visionzip import visionzip
from .algorithm.vispruner import vispruner_pruning

# --- Strategy Registries ---

# 1. Pruning Strategies Registry
PRUNING_STRATEGIES: Dict[str, Callable] = {
    "baseline": baseline_pruning,
    "override": override_pruning,
    "random": random_pruning,
    "special_token_based_attention": (special_token_based_attention_pruning),
    "divprune": divprune,
    "dart": dart_pruning,
    "hiprune": hiprune_pruning,
    "vision_selector": vision_selector_pruning,
    "vispruner": vispruner_pruning,
    "scope": scope_pruning,
    "idpruner": idpruner,
}

# 2. Merging Strategies Registry
MERGING_STRATEGIES: Dict[str, Callable] = {
    "visionzip": visionzip,
}


def compression_strategy_factory(name: str) -> Callable:
    """
    Factory function to retrieve a registered token compression strategy.

    Args:
        name (str, required): The registered name of the compression strategy.

    Returns:
        Callable: The corresponding compression strategy function.
    """
    if name in PRUNING_STRATEGIES:
        return PRUNING_STRATEGIES[name]

    if name in MERGING_STRATEGIES:
        return MERGING_STRATEGIES[name]

    available = list(PRUNING_STRATEGIES.keys()) + list(MERGING_STRATEGIES.keys())
    raise ValueError(
        f"[AngelSlim Error] Unknown compression strategy: '{name}'. "
        f"Available strategies: {available}"
    )


def is_merging_strategy(name: str) -> bool:
    """
    Checks if a given strategy name belongs to merging algorithms.

    Args:
        name (str, required): The registered name of the strategy.

    Returns:
        bool: True if it is a merging strategy, False otherwise.
    """
    return name in MERGING_STRATEGIES
