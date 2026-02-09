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

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


class LayerTensorMap(dict):
    """
    A dictionary subclass for storing layer-wise tensors
    that supports relative indexing.

    Negative keys (e.g., -1 for the last layer) are automatically resolved to absolute
    indices using the 'total_layers' attribute.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_layers: Optional[int] = None

    def __getitem__(self, key: int) -> torch.Tensor:
        """
        Retrieves a tensor with support for negative indexing.

        Args:
            key (int): The layer index. Supports standard Python negative indexing.

        Returns:
            torch.Tensor: The activation tensor for the resolved layer index.

        Raises:
            RuntimeError: If negative indexing is used but total_layers is not set.
            KeyError: If the resolved index does not exist in the map.
        """
        if key < 0:
            if self.total_layers is None:
                # Principle 6: Fail fast if metadata required for indexing is
                # missing
                raise RuntimeError(
                    f"[AngelSlim Error] Attempted negative indexing ({key}), "
                    f"but total_layers is None. Ensure 'llm_layer_num' or "
                    f"'vit_layer_num' is set in the PruningContext."
                )
            # Resolve relative index (e.g., -2 -> total_layers - 2)
            key = self.total_layers + key

        return super().__getitem__(key)

    def get(self, key: int, default: Any = None) -> Any:
        """Safely retrieve a tensor with resolution for negative indexing."""
        try:
            return self.__getitem__(key)
        except (KeyError, RuntimeError):
            return default


@dataclass
class PruningContext:
    """
    The runtime execution state containing
    intermediate tensors and metadata for pruning.

    This class synchronizes layer counts to internal LayerTensorMaps automatically
    via property setters to support dynamic updates during the inference lifecycle.
    """

    input_ids: torch.Tensor
    """Full sequence input token IDs of shape [batch_size, sequence_length]."""

    inputs_embeds: torch.Tensor
    """Multimodal embeddings after the projector
    of shape [batch_size, sequence_length, hidden_dim]."""

    vit_q: LayerTensorMap = field(default_factory=LayerTensorMap)
    """Query states from Vision Tower layers."""

    vit_k: LayerTensorMap = field(default_factory=LayerTensorMap)
    """Key states from Vision Tower layers."""

    llm_q: LayerTensorMap = field(default_factory=LayerTensorMap)
    """Query states from Language Model layers."""

    llm_k: LayerTensorMap = field(default_factory=LayerTensorMap)
    """Key states from Language Model layers."""

    feature_map: Optional[torch.Tensor] = None
    """Hidden states used for diversity or semantic calculations."""

    image_grid_thw: Optional[torch.Tensor] = None
    """Grid dimensions [N, 3] for images."""

    video_grid_thw: Optional[torch.Tensor] = None
    """Grid dimensions [N, 3] for videos."""

    cu_seqlens_full: Optional[torch.Tensor] = None
    """Cumulative sequence lengths for packed sequences."""

    spatial_merge_size: int = 2
    """Pooling factor for spatial patches."""

    reverse_indices: Optional[torch.Tensor] = None
    """Indices to restore spatial order."""

    window_index: Optional[torch.Tensor] = None
    """Indices for window-based attention mapping."""

    vision_token_mask: Optional[torch.Tensor] = None
    """Boolean mask identifying visual token positions."""

    keep_mask: Optional[torch.Tensor] = None
    """Pruning mask produced by a specific layer."""

    kv_token_scale: Optional[torch.Tensor] = None
    """Tensors for soft-pruning attention scaling."""

    model_config: Any = None
    """Configuration object of the base model."""

    # Internal backing fields for properties
    _vit_layer_num: Optional[int] = None
    _llm_layer_num: Optional[int] = None

    @property
    def vit_layer_num(self) -> Optional[int]:
        """Getter for the number of layers in the Vision Tower."""
        return self._vit_layer_num

    @vit_layer_num.setter
    def vit_layer_num(self, value: int):
        """Setter that synchronizes the total count to vision-related maps."""
        self._vit_layer_num = value
        self.vit_q.total_layers = value
        self.vit_k.total_layers = value

    @property
    def llm_layer_num(self) -> Optional[int]:
        """Getter for the number of layers in the Language Model."""
        return self._llm_layer_num

    @llm_layer_num.setter
    def llm_layer_num(self, value: int):
        """Setter that synchronizes the total count to LLM-related maps."""
        self._llm_layer_num = value
        self.llm_q.total_layers = value
        self.llm_k.total_layers = value

    def __post_init__(self):
        """
        Performs post-initialization synchronization.
        Ensures that if layer numbers are passed to the constructor, they are
        propagated to the LayerTensorMap instances.
        """
        if self.vit_layer_num is not None:
            # Trigger property setter
            self.vit_layer_num = self.vit_layer_num

        if self.llm_layer_num is not None:
            # Trigger property setter
            self.llm_layer_num = self.llm_layer_num

    @property
    def vision_indices(self) -> torch.Tensor:
        """Returns absolute indices of visual tokens."""
        if self.vision_token_mask is None:
            return torch.tensor([], dtype=torch.long, device=self.input_ids.device)
        return torch.where(self.vision_token_mask)[0]

    @property
    def non_vision_indices(self) -> torch.Tensor:
        """Returns absolute indices of non-visual tokens."""
        if self.vision_token_mask is None:
            return torch.tensor([], dtype=torch.long, device=self.input_ids.device)
        return torch.where(~self.vision_token_mask)[0]

    def __repr__(self):
        return (
            f"PruningContext(vit_layers={self.vit_layer_num}, "
            f"llm_layers={self.llm_layer_num}, "
            f"vit_q_keys={list(self.vit_q.keys())}, "
            f"llm_q_keys={list(self.llm_q.keys())})"
        )

    def __str__(self):
        return self.__repr__()
