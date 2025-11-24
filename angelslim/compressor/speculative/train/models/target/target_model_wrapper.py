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
from typing import List, Optional, Tuple

import torch

from angelslim.utils import decide_device_for_distributed, print_with_rank


class BaseBackend(ABC):
    """
    Base class for model backends.

    This abstract class defines the interface that all backend implementations
    must follow to ensure consistent behavior across different model serving frameworks.
    """

    def __init__(self, model_path: str, **kwargs):
        """
        Initialize the backend.

        Args:
            model_path: Path to the model checkpoint or serving endpoint
            **kwargs: Additional backend-specific configuration parameters
        """
        self.model_path = model_path
        self.kwargs = kwargs
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the backend model and tokenizer.

        This method should initialize self.model and self.tokenizer.
        Implementations should handle device placement and model configuration.
        """
        pass

    @abstractmethod
    def get_hidden_states_and_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Extract hidden states and logits from the model.

        Args:
            input_ids: Input token IDs, shape [batch_size, seq_len]
            attention_mask: Attention mask, shape [batch_size, seq_len]
            **kwargs: Additional model-specific arguments

        Returns:
            Tuple of (hidden_states, logits):
                - hidden_states: Concatenated auxiliary hidden states,
                  shape [batch_size, seq_len, hidden_size * num_layers]
                - logits: Model output logits, shape [batch_size, seq_len, vocab_size]
        """
        pass

    @abstractmethod
    def get_aux_and_target_hiddens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Extract auxiliary and target hidden states from the model.

        Args:
            input_ids: Input token IDs, shape [batch_size, seq_len]
            attention_mask: Attention mask, shape [batch_size, seq_len]
            **kwargs: Additional model-specific arguments

        Returns:
            Tuple of (aux_hidden_states, target_hidden_states):
                - aux_hidden_states: Concatenated auxiliary hidden states
                    from multiple layers
                - target_hidden_states: Final layer hidden states
        """
        pass

    def _get_default_aux_layer_ids(self, total_layers: int) -> List[int]:
        """
        Calculate default auxiliary hidden state layer indices.

        Selects three representative layers: early, middle, and late in the model.

        Args:
            total_layers: Total number of hidden state layers (including embedding)

        Returns:
            List of three layer indices [low, mid, high]
        """
        return [
            1,  # Early layer
            total_layers // 2 - 1,  # Middle layer
            total_layers - 4,  # Late layer (before final layers)
        ]

    def _extract_auxiliary_hidden_states(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        aux_layer_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Extract and concatenate auxiliary hidden states from specified layers.

        Args:
            hidden_states: Tuple of hidden states from all layers
            aux_layer_ids: List of layer indices to extract.
                If None, uses default layers.

        Returns:
            Concatenated hidden states, shape [batch_size, seq_len, hidden_size * 3]
        """
        if aux_layer_ids is None:
            aux_layer_ids = self._get_default_aux_layer_ids(len(hidden_states))

        # Offset by 1 to skip embedding layer
        embed_offset = 1

        selected_hiddens = [
            hidden_states[layer_id + embed_offset] for layer_id in aux_layer_ids
        ]

        return torch.cat(selected_hiddens, dim=-1)


class TransformersBackend(BaseBackend):
    """
    HuggingFace Transformers backend implementation.

    This backend uses the transformers library's AutoModelForCausalLM
    for model loading and inference.
    """

    def load_model(self) -> None:
        """Load model and tokenizer using HuggingFace Transformers."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Determine device based on distributed environment
        device = decide_device_for_distributed()
        print_with_rank(f"Loading model to device: {device}")

        # Prepare model loading configuration
        model_kwargs = self._prepare_model_kwargs(device)

        # Load and configure model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **model_kwargs
        )
        self._freeze_model_parameters()
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def _prepare_model_kwargs(self, device: str) -> dict:
        """
        Prepare keyword arguments for model loading.

        Args:
            device: Target device for model placement

        Returns:
            Dictionary of model loading arguments
        """
        default_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": device,
            "trust_remote_code": True,
        }
        default_kwargs.update(self.kwargs)
        return default_kwargs

    def _freeze_model_parameters(self) -> None:
        """Freeze all model parameters to prevent training."""
        for param in self.model.parameters():
            param.requires_grad = False

    def get_hidden_states_and_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract hidden states and logits using Transformers backend.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: May contain 'aux_hidden_states_layer_ids' to specify custom layers

        Returns:
            Tuple of (concatenated_hidden_states, logits)
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_logits=True,
            )

        # Extract auxiliary hidden states
        aux_layer_ids = kwargs.get("aux_hidden_states_layer_ids", None)
        hidden_states = self._extract_auxiliary_hidden_states(
            outputs.hidden_states, aux_layer_ids
        )

        # Return hidden states and logits on the same device as input
        return hidden_states, outputs.logits.to(input_ids.device)

    def get_aux_and_target_hiddens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Extract auxiliary and final layer hidden states.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: May contain 'aux_hidden_states_layer_ids' to specify custom layers

        Returns:
            Tuple of (auxiliary_hidden_states, final_hidden_states)
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_logits=True,
            )

        # Extract auxiliary hidden states
        aux_layer_ids = kwargs.get("aux_hidden_states_layer_ids", None)
        aux_hidden_states = self._extract_auxiliary_hidden_states(
            outputs.hidden_states, aux_layer_ids
        )

        # Get final layer hidden states
        target_hidden_states = outputs.hidden_states[-1]

        # hidden_states: B, N, 3*D
        # target_hiddens: B, N, D
        return {
            "hidden_states": aux_hidden_states,
            "target_hiddens": target_hidden_states,
        }


class VLMForwardWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        inputs_embeds = None
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            inputs_embeds = kwargs["inputs_embeds"]
        elif len(args) > 2 and args[2] is not None:
            inputs_embeds = args[2]

        outputs = self.model.forward(*args, **kwargs)
        return outputs, inputs_embeds


class VLMTransformersBackend(BaseBackend):
    """VLM HuggingFace Transformers backend"""

    def load_model(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        default_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        default_kwargs.update(self.kwargs)

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, **default_kwargs
        )
        # self.model = VLMForwardWrapper(self.model)

        # Freeze the base model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.tokenizer = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def get_hidden_states_and_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Extract hidden states and logits using Transformers backend.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: May contain 'aux_hidden_states_layer_ids' to specify custom layers

        Returns:
            Tuple of (concatenated_hidden_states, logits)
        """
        inputs_embeds_list, position_ids_list = [], []

        def hook(module, args, kwargs):
            if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
                inputs_embeds_list.append(
                    kwargs["inputs_embeds"].clone().detach().cpu()
                )
            if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                position_ids_list.append(kwargs["position_ids"].clone().detach().cpu())
            return args, kwargs

        handle = self.model.language_model.register_forward_pre_hook(
            hook, with_kwargs=True
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_logits=True,
            )

        handle.remove()
        inputs_embeds = inputs_embeds_list[0].to(input_ids.device)
        position_ids = position_ids_list[0].to(input_ids.device)

        # Extract auxiliary hidden states
        aux_layer_ids = kwargs.get("aux_hidden_states_layer_ids", None)
        hidden_states = self._extract_auxiliary_hidden_states(
            outputs.hidden_states, aux_layer_ids
        )

        # Return hidden states and logits on the same device as input
        return (
            hidden_states,
            outputs.logits.to(input_ids.device),
            inputs_embeds,
            position_ids,
        )

    def get_aux_and_target_hiddens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Extract auxiliary and final layer hidden states.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: May contain 'aux_hidden_states_layer_ids' to specify custom layers

        Returns:
            Tuple of (auxiliary_hidden_states, final_hidden_states)
        """
        inputs_embeds_list, position_ids_list = [], []

        def hook(module, args, kwargs):
            if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
                inputs_embeds_list.append(
                    kwargs["inputs_embeds"].clone().detach().cpu()
                )
            if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                position_ids_list.append(kwargs["position_ids"].clone().detach().cpu())
            return args, kwargs

        handle = self.model.language_model.register_forward_pre_hook(
            hook, with_kwargs=True
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_logits=True,
            )

        handle.remove()
        inputs_embeds = inputs_embeds_list[0].to(input_ids.device)
        position_ids = position_ids_list[0].to(input_ids.device)

        # Extract auxiliary hidden states
        aux_layer_ids = kwargs.get("aux_hidden_states_layer_ids", None)
        aux_hidden_states = self._extract_auxiliary_hidden_states(
            outputs.hidden_states, aux_layer_ids
        )

        # Get final layer hidden states
        target_hidden_states = outputs.hidden_states[-1]

        # hidden_states: B, N, 3*D
        # target_hiddens: B, N, D
        # inputs_embeds: B, N, D
        # position_ids: 3, N
        return {
            "hidden_states": aux_hidden_states,
            "target_hiddens": target_hidden_states,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
        }


class TargetModelWrapper:
    """
    Unified wrapper for target models in Eagle3 training.

    This wrapper provides a consistent interface across
    different backend implementations, allowing seamless switching
    between model serving frameworks.

    Supported backends:
        - hf: HuggingFace Transformers (AutoModelForCausalLM)
    Supported modal types:
        - LLM: Large Language Models
        - VLM: Vision-Language Models

    Example:
        >>> wrapper = TargetModelWrapper(
        ...     backend="hf",
        ...     modal_type="LLM",
        ...     model_path="/path/to/model",
        ...     dtype=torch.bfloat16
        ... )
        >>> hidden_states, logits = wrapper.get_hidden_states_and_logits(input_ids)
    """

    BACKENDS = {
        ("hf", "LLM"): TransformersBackend,
        ("hf", "VLM"): VLMTransformersBackend,
    }

    def __init__(
        self, model_path: str, modal_type: str = "LLM", backend: str = "hf", **kwargs
    ):
        """
        Initialize TargetModel with specified backend

        Args:
            backend: One of ["hf"]
            model_path: Path to model
            **kwargs: Additional arguments for backend initialization
        """
        if (backend, modal_type) not in self.BACKENDS:
            raise ValueError(
                f"Unsupported backend: {(backend, modal_type)}. "
                f"Available backends: {list(self.BACKENDS.keys())}"
            )

        self.backend_name = backend
        self.backend = self.BACKENDS[(backend, modal_type)](model_path, **kwargs)
        self.backend.load_model()

    def get_hidden_states_and_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Get hidden states and logits from target model

        Args:
            input_ids: Input token ids, shape [batch_size, seq_len]
            attention_mask: Attention mask, shape [batch_size, seq_len]

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

    def get_aux_and_target_hiddens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Get auxiliary and target hidden states from model.

        Args:
            input_ids: Input token IDs, shape [batch_size, seq_len]
            attention_mask: Attention mask, shape [batch_size, seq_len]
            **kwargs: Additional backend-specific arguments

        Returns:
            Tuple of (aux_hidden_states, target_hidden_states)
        """
        return self.backend.get_aux_and_target_hiddens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    @property
    def model(self):
        """
        Access the underlying model instance.

        Returns:
            The backend's model object
        """
        return self.backend.model

    @property
    def tokenizer(self):
        """
        Access the underlying tokenizer instance.

        Returns:
            The backend's tokenizer object

        Raises:
            AttributeError: If backend doesn't support tokenizers
            ValueError: If tokenizer is not initialized
        """
        if not hasattr(self.backend, "tokenizer"):
            raise AttributeError(
                f"Backend '{self.backend_name}' does not support tokenizers"
            )
        if self.backend.tokenizer is None:
            raise ValueError(
                f"Tokenizer not initialized for backend '{self.backend_name}'"
            )
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

    This function provides a convenient way to instantiate a TargetModelWrapper
    with commonly used default settings.

    Args:
        backend: Backend type, one of ["hf"]
        model_path: Path to model checkpoint or serving endpoint URL
        torch_dtype: Data type for model weights (for HF backend)
        trust_remote_code: Whether to trust and execute remote code
        **extra_kwargs: Additional backend-specific arguments

    Returns:
        Configured TargetModelWrapper instance

    Raises:
        ValueError: If backend is not supported

    Example:
        >>> model = create_target_model(
        ...     backend="hf",
        ...     model_path="/path/to/llama-7b",
        ...     torch_dtype=torch.float16
        ... )
    """
    # Prepare common configuration
    kwargs = {
        "trust_remote_code": trust_remote_code,
        **extra_kwargs,
    }

    # Add backend-specific configuration
    if backend == "hf":
        kwargs["dtype"] = torch_dtype
    else:
        raise ValueError(
            f"Unsupported backend: '{backend}'. "
            f"Use one of: {list(TargetModelWrapper.BACKENDS.keys())}"
        )

    return TargetModelWrapper(backend=backend, model_path=model_path, **kwargs)
