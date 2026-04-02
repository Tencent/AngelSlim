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

from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache


class PruningCache(DynamicCache):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
        **kwargs
    ):
        super().__init__(
            config=config,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
            **kwargs
        )
        # Stores pruning info. Key can be layer index (int) or stage name
        # (e.g., "global").
        self.pruning_info: Dict[Union[int, str], Dict[str, Any]] = {}
        # Tracks update counts per layer to distinguish between prefill and
        # decoding.
        self.update_counts: Dict[int, int] = {}

    def is_prefill_stage(self, layer_idx: int, timing: str = "after_update") -> bool:
        count = self.update_counts.get(layer_idx, 0)
        if timing == "before_update":
            return count == 0
        elif timing == "after_update":
            return count == 1
        else:
            raise ValueError("Timing must be either 'before_update' or 'after_update'.")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        updated_key, updated_value = super().update(
            key_states, value_states, layer_idx, cache_kwargs
        )
        self.update_counts[layer_idx] = self.update_counts.get(layer_idx, 0) + 1
        return updated_key, updated_value

    def set_pruning_mask_for_layer(self, stage_key: Union[int, str], mask: torch.Tensor):
        if mask.dim() != 1:
            mask = mask.view(-1)

        self.pruning_info[stage_key] = {
            "pruned": True,
            "mask": mask,
            "pruned_tokens": mask.size(0) - mask.sum().item(),
        }

    def get_pruning_mask_for_layer(
        self, stage_key: Union[int, str]
    ) -> Tuple[bool, Optional[torch.Tensor], int]:
        """
        Retrieves the pruning status, mask tensor,
        and pruned token count for a specific stage or layer.
        """
        # Support for smart index -1 to get the last registered pruning stage
        if stage_key == -1:
            numeric_keys = [k for k in self.pruning_info.keys() if isinstance(k, int)]
            if not numeric_keys:
                return False, None, 0
            stage_key = max(numeric_keys)

        info = self.pruning_info.get(stage_key, {})
        return (
            info.get("pruned", False),
            info.get("mask"),
            info.get("pruned_tokens", 0),
        )

    @property
    def is_prefill(self) -> bool:
        """Global flag to check if the overall model is in prefill."""
        return self.is_prefill_stage(0, timing="before_update")

    def reset(self):
        """Full reset of the cache, pruning history, and stage counters."""
        super().reset()
        self.pruning_info = {}
        self.update_counts = {}
