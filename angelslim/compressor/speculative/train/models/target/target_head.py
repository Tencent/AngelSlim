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
import os

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig

from angelslim.utils import decide_device_for_distributed


class TargetHead(nn.Module):
    """
    Target Head for computing logits from hidden states in offline EAGLE3 training.

    This module takes the last hidden states from the target model and projects them
    to vocabulary logits, which are used as training targets for the draft model.
    """

    def __init__(self, lm_head: nn.Module):
        """
        Initialize the TargetHead.

        Args:
            lm_head: Language model head (typically nn.Linear) that projects
                    hidden states to vocabulary logits. This should be loaded
                    from the target model.
        """
        super().__init__()
        self.lm_head = lm_head

        # Freeze the lm_head parameters since we only use it for inference
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logits from hidden states.

        Args:
            hidden_states: Hidden states from target model with shape
                          (batch_size, seq_length, hidden_size)

        Returns:
            Logits with shape (batch_size, seq_length, vocab_size)
        """
        # Project hidden states to vocabulary logits
        logits = self.lm_head(hidden_states)
        return logits

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        lm_head_key: str = "lm_head.weight",
        sub_config_name=None,
    ):
        """
        Load TargetHead from a pretrained model efficiently.

        This method only loads the lm_head weights using safetensors index,
        which is more memory-efficient than loading the entire model.

        Args:
            model_name_or_path: Path to pretrained model or model identifier
            **kwargs: Additional arguments for model loading (e.g., torch_dtype,
                     trust_remote_code, device_map)

        Returns:
            TargetHead instance with loaded lm_head
        """
        # Load model config to get architecture info
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        if hasattr(config, sub_config_name):
            config = getattr(config, sub_config_name)
        else:
            raise ValueError(
                f"Config {config} has no sub-config named {sub_config_name}"
            )

        # Get model dimensions
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size

        # Initialize lm_head
        lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Load lm_head weights from safetensors
        try:
            # Read safetensors index to locate lm_head weights
            index_path = os.path.join(
                model_name_or_path, "model.safetensors.index.json"
            )

            if not os.path.exists(index_path):
                raise FileNotFoundError(
                    f"model.safetensors.index.json not found in {model_name_or_path}. "
                    "Please ensure the model is saved in safetensors "
                    "format with sharding."
                )

            # Model is sharded, use index to find lm_head
            with open(index_path, "r") as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"][lm_head_key]

            # Load lm_head weights using safetensors
            safetensors_file = os.path.join(model_name_or_path, head_path)
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                tensor_slice = f.get_slice(lm_head_key)
                _, hidden_dim = tensor_slice.get_shape()
                tensor = tensor_slice[:, :hidden_dim]
            lm_head.weight.data = tensor

        except Exception as e:
            raise RuntimeError(
                f"Failed to load lm_head weights from {model_name_or_path}. "
                f"Error: {str(e)}"
            )

        # Create TargetHead instance
        target_head = cls(lm_head)

        device = decide_device_for_distributed()
        target_head.to(device)
        target_head.eval()

        return target_head
