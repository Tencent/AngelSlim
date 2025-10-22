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
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get hidden states and logits from model"""
        pass


class TransformersBackend(BaseBackend):
    """HuggingFace Transformers backend"""

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        default_kwargs = {
            "torch_dtype": torch.bfloat16,
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
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
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


class VLLMLocalBackend(BaseBackend):
    """vLLM local model backend"""

    def load_model(self):
        from vllm import LLM

        self.model = LLM(
            model=self.model_path,
            tensor_parallel_size=self.kwargs.get("tensor_parallel_size", 1),
            trust_remote_code=self.kwargs.get("trust_remote_code", True),
            **self.kwargs,
        )
        self.tokenizer = self.model.get_tokenizer()

    def get_hidden_states_and_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Access vLLM internal model runner
        model_runner = self.model.llm_engine.model_executor.driver_worker.model_runner
        device = next(model_runner.model.parameters()).device

        # Process batch
        token_ids_list = input_ids.cpu().tolist()
        hidden_states_list = []
        logits_list = []

        for token_ids in token_ids_list:
            outputs = model_runner.model(
                input_ids=torch.tensor([token_ids]).to(device),
                output_hidden_states=True,
            )
            hidden_states_list.append(outputs.hidden_states[-1])
            logits_list.append(outputs.logits)

        hidden_states = torch.cat(hidden_states_list, dim=0)
        logits = torch.cat(logits_list, dim=0)

        return hidden_states, logits


class VLLMServingBackend(BaseBackend):
    """vLLM serving endpoint backend"""

    def load_model(self):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.kwargs.get("api_key", "EMPTY"), base_url=self.model_path
        )
        self.model_name = self.kwargs.get("model_name", "default")

        # Try to get tokenizer if available
        try:
            from transformers import AutoTokenizer

            tokenizer_path = self.kwargs.get("tokenizer_path", self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:  # noqa: F841
            self.tokenizer = None

    def get_hidden_states_and_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available for vLLM serving backend")

        # Convert input_ids to text
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        # Call serving API
        responses = []
        for text in texts:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=text,
                max_tokens=1,
                logprobs=True,
                echo=True,
                extra_body={"return_hidden_states": True},
            )
            responses.append(response)

        # Parse responses
        hidden_states = self._parse_hidden_states(responses)
        logits = self._parse_logits(responses)

        return hidden_states, logits

    def _parse_hidden_states(self, responses) -> torch.Tensor:
        """Parse hidden states from serving responses"""
        raise NotImplementedError(
            "Hidden states parsing from vLLM serving requires custom implementation"
        )

    def _parse_logits(self, responses) -> torch.Tensor:
        """Parse logits from serving responses"""
        raise NotImplementedError(
            "Logits parsing from vLLM serving requires custom implementation"
        )


class TargetModelWrapper:
    """
    Target model wrapper for Eagle3 training.

    Supports three backends:
    - hf: HuggingFace Transformers AutoModelForCausalLM
    - vllm_local: vLLM local model
    - vllm_serving: vLLM serving endpoint
    """

    BACKENDS = {
        "hf": TransformersBackend,
        "vllm_local": VLLMLocalBackend,
        "vllm_serving": VLLMServingBackend,
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
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
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
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    @property
    def model(self):
        """Access underlying model"""
        return self.backend.model

    @property
    def tokenizer(self):
        """Access tokenizer"""
        return self.backend.tokenizer


def create_target_model(
    backend: str,
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    tensor_parallel_size: int = 1,
    api_key: str = "EMPTY",
    model_name: str = "default",
    tokenizer_path: Optional[str] = None,
    **extra_kwargs,
) -> TargetModelWrapper:
    """
    Factory function to create target model with appropriate backend configuration.

    Args:
        backend: Backend type, one of ["hf", "vllm_local", "vllm_serving"]
        model_path: Path to model or serving endpoint URL
        torch_dtype: Data type for model weights (for HF backend)
        trust_remote_code: Whether to trust remote code
        tensor_parallel_size: Tensor parallel size (for vLLM local backend)
        api_key: API key (for vLLM serving backend)
        model_name: Model name (for vLLM serving backend)
        tokenizer_path: Path to tokenizer (for vLLM serving backend)
        **extra_kwargs: Additional backend-specific arguments

    Returns:
        TargetModelWrapper instance

    Examples:
        >>> # HuggingFace backend
        >>> model = create_target_model(
        ...     backend="hf",
        ...     model_path="/path/to/model"
        ... )

        >>> # vLLM local backend
        >>> model = create_target_model(
        ...     backend="vllm_local",
        ...     model_path="/path/to/model",
        ...     tensor_parallel_size=2
        ... )

        >>> # vLLM serving backend
        >>> model = create_target_model(
        ...     backend="vllm_serving",
        ...     model_path="http://localhost:8000/v1",
        ...     model_name="my-model",
        ...     tokenizer_path="/path/to/tokenizer"
        ... )
    """
    # Prepare common kwargs
    kwargs = {"trust_remote_code": trust_remote_code, **extra_kwargs}

    # Add backend-specific kwargs
    if backend == "hf":
        kwargs.update(
            {
                "torch_dtype": torch_dtype,
            }
        )
    elif backend == "vllm_local":
        kwargs.update(
            {
                "tensor_parallel_size": tensor_parallel_size,
            }
        )
    elif backend == "vllm_serving":
        kwargs.update(
            {
                "api_key": api_key,
                "model_name": model_name,
            }
        )
        if tokenizer_path:
            kwargs["tokenizer_path"] = tokenizer_path

    return TargetModelWrapper(backend=backend, model_path=model_path, **kwargs)
