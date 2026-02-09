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

from typing import Any, Dict, List, Set

from ..base.config import TokenCompressorConfig


def plan_pruning_execution(
    strategy_config: TokenCompressorConfig,
    raw_map_data: List[Dict[str, Any]],
    model_config: Any,
) -> List[Dict[str, Any]]:
    """
    Generates a tailored execution plan by matching DataRequirements and
    Pruning Strategies against the model's component mapping.

    This function also handles dynamic parameter routing based on the model path
    and resolves negative layer indices for the vision tower.

    Logic:
    1. Perform dynamic routing for 'model_related_params'
        by matching patterns against model path.
    2. Resolve negative indices in 'vit_q_layers' and
        'vit_k_layers' using vision depth.
    3. Determine which components need to be wrapped
        based on requirements and strategies.

    Args:
        strategy_config: Structured config containing requirements and strategies.
        raw_map_data: Sequential list of components from model_mappings.yaml.
        model_config: The native model configuration.

    Returns:
        List[Dict[str, Any]]: Refined execution steps with resolved 'indices'.
    """

    # --- 1. Dynamic Parameter Routing based on Model Path ---
    # Retrieve the model identifier path (standard HF attribute)
    model_path_str = getattr(model_config, "_name_or_path", "").lower()

    for _stage_key, strategy in strategy_config.strategies.items():
        # Check if the strategy defines parameters that depend on the model
        # path
        if hasattr(strategy, "model_related_params") and strategy.model_related_params:
            for (
                param_name,
                pattern_map,
            ) in strategy.model_related_params.items():
                resolved_value = None
                # Iterate through the pattern dictionary (e.g., {"7b": "...",
                # "3b": "..."})
                for pattern, value in pattern_map.items():
                    if pattern.lower() in model_path_str:
                        resolved_value = value
                        break

                # Strict principle: Raise error if a required mapping is
                # missing for the current model
                if resolved_value is None:
                    raise RuntimeError(
                        f"[AngelSlim Error] Strategy '{strategy.strategy}' "
                        f"requires dynamic parameter '{param_name}', "
                        f"but no matching pattern was found "
                        f"in model path: '{model_path_str}'. "
                        f"Available patterns: {list(pattern_map.keys())}"
                    )

                # Inject the resolved value into the active parameters
                # dictionary
                strategy.params[param_name] = resolved_value
                print(
                    f"[ConfigUtils] Resolved dynamic param '{param_name}'"
                    f" to '{resolved_value}' for model '{model_path_str}'"
                )

    # --- 2. Negative Index Resolution for Vision Tower ---
    # Extract total depth of the vision tower for relative index calculation
    vision_depth = 0
    vision_cfg = getattr(model_config, "vision_config", None)
    if vision_cfg is not None:
        # Compatibility check for both object and dict-like config
        if isinstance(vision_cfg, dict):
            vision_depth = vision_cfg.get("depth", None) or vision_cfg.get(
                "num_hidden_layers", None
            )
        else:
            vision_depth = getattr(vision_cfg, "depth", None) or getattr(
                vision_cfg, "num_hidden_layers", None
            )

    def resolve_negative_indices(layers: List[int], depth: int) -> List[int]:
        """Resolves indices like -1 to actual depth-based indices."""
        for i, layer in enumerate(layers):
            if layer < 0:
                layers[i] = depth + layer
        return layers

    requirements = strategy_config.requirements
    if vision_depth > 0:
        requirements.vit_q_layers = resolve_negative_indices(
            requirements.vit_q_layers, vision_depth
        )
        requirements.vit_k_layers = resolve_negative_indices(
            requirements.vit_k_layers, vision_depth
        )

    # --- 3. Execution Plan Construction ---
    # Resolve required Vision indices (Union of Q and K capture requirements)
    required_vit_indices: Set[int] = set(requirements.vit_q_layers) | set(
        requirements.vit_k_layers
    )

    # Resolve required LLM indices (Capture requirements + Layers with active
    # pruning)
    required_llm_indices: Set[int] = set(requirements.llm_q_layers) | set(
        requirements.llm_k_layers
    )

    # Add indices where per-layer pruning strategies are defined
    for stage_key in strategy_config.strategies.keys():
        if isinstance(stage_key, int):
            required_llm_indices.add(stage_key)

    execution_plan = []

    # The Vision Transformer wrapper is required if any internal vision layers
    # need to be hooked
    needs_vision_tower = len(required_vit_indices) > 0

    for entry in raw_map_data:
        name = entry["name"]
        step = entry.copy()

        # Branch 1: Core management modules (Always wrapped)
        if name in ["vl_model", "text_model"]:
            step["indices"] = None
            execution_plan.append(step)

        # Branch 2: Vision Transformer wrapper (Conditional)
        elif name == "vision_transformer":
            if needs_vision_tower:
                step["indices"] = None
                execution_plan.append(step)

        # Branch 3: Vision Attention hooks (Specific Layers)
        elif name == "vision_attn":
            if required_vit_indices:
                step["indices"] = sorted(list(required_vit_indices))
                execution_plan.append(step)

        # Branch 4: LLM Attention hooks (Specific Layers)
        elif name == "llm_attn":
            if required_llm_indices:
                step["indices"] = sorted(list(required_llm_indices))
                execution_plan.append(step)

        else:
            # Skip mapping entries not utilized by the pruning logic
            continue

    return strategy_config, execution_plan
