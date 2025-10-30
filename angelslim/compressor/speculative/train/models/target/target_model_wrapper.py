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

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class BaseBackend(ABC):
    """Base class for model backends"""

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self):
        """Load the backend model"""
        pass

    @abstractmethod
    def get_hidden_states_and_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get hidden states and logits from model"""
        pass


class TransformersBackend(BaseBackend):
    """HuggingFace Transformers backend"""

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        default_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        default_kwargs.update(self.kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **default_kwargs
        )

        # Freeze the base model
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def get_hidden_states_and_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            outputs = self.model(
                input_ids, attention_mask, output_hidden_states=True, output_logits=True
            )

        aux_hidden_states_layer_ids = kwargs.get("aux_hidden_states_layer_ids", None)
        if aux_hidden_states_layer_ids is None:
            out_hidden_nums = len(outputs.hidden_states)
            aux_hidden_states_layer_ids = [
                1,
                out_hidden_nums // 2 - 1,
                out_hidden_nums - 4,
            ]

        embed_offset = 1
        low_hidden_states = outputs.hidden_states[
            aux_hidden_states_layer_ids[0] + embed_offset
        ]
        mid_hidden_states = outputs.hidden_states[
            aux_hidden_states_layer_ids[1] + embed_offset
        ]
        high_hidden_states = outputs.hidden_states[
            aux_hidden_states_layer_ids[2] + embed_offset
        ]
        hidden_states = torch.cat(
            [low_hidden_states, mid_hidden_states, high_hidden_states], dim=-1
        )

        target = outputs.logits
        return hidden_states, target.to(input_ids.device)


class TargetModelWrapper:
    """
    Target model wrapper for Eagle3 training.

    Supports three backends:
    - hf: HuggingFace Transformers AutoModelForCausalLM
    """

    BACKENDS = {
        "hf": TransformersBackend,
    }

    def __init__(self, backend: str, model_path: str, **kwargs):
        """
        Initialize TargetModel with specified backend

        Args:
            backend: One of ["hf", "vllm_local", "vllm_serving"]
            model_path: Path to model or serving endpoint
            **kwargs: Additional arguments for backend initialization
        """
        if backend not in self.BACKENDS:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                f"Available backends: {list(self.BACKENDS.keys())}"
            )

        self.backend_name = backend
        self.backend = self.BACKENDS[backend](model_path, **kwargs)
        self.backend.load_model()

    def get_hidden_states_and_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get hidden states and logits from target model

        Args:
            input_ids: Input token ids, shape [batch_size, seq_len]
            attention_mask: Attention mask, shape [batch_size, seq_len]
            position_ids: Position ids, shape [batch_size, seq_len]
            past_key_values: Past key values for generation

        Returns:
            Tuple of (hidden_states, logits)
            - hidden_states: shape [batch_size, seq_len, hidden_size]
            - logits: shape [batch_size, seq_len, vocab_size]
        """
        return self.backend.get_hidden_states_and_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    @property
    def model(self):
        """Access underlying model"""
        return self.backend.model

    @property
    def tokenizer(self):
        """Access underlying tokenizer"""
        if not hasattr(self.backend, "tokenizer"):
            raise AttributeError(
                f"Backend '{self.backend_name}' does not have a tokenizer attribute"
            )
        if self.backend.tokenizer is None:
            raise ValueError(f"Backend '{self.backend_name}' does not have a tokenizer")
        return self.backend.tokenizer


def create_target_model(
    backend: str,
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    **extra_kwargs,
) -> TargetModelWrapper:
    """
    Factory function to create target model with appropriate backend configuration.

    Args:
        backend: Backend type, one of ["hf"]
        model_path: Path to model or serving endpoint URL
        torch_dtype: Data type for model weights (for HF backend)
        trust_remote_code: Whether to trust remote code
        tokenizer_path: Path to tokenizer
        **extra_kwargs: Additional backend-specific arguments

    Returns:
        TargetModelWrapper instance
    """
    # Prepare common kwargs
    kwargs = {"trust_remote_code": trust_remote_code, **extra_kwargs}

    # Add backend-specific kwargs
    if backend == "hf":
        kwargs.update(
            {
                "dtype": torch_dtype,
            }
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return TargetModelWrapper(backend=backend, model_path=model_path, **kwargs)
