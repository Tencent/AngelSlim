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

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
from huggingface_hub import snapshot_download

from angelslim.utils import decide_device_for_distributed, print_with_rank

from .cosyvoice3_llm import CosyVoice3LM


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
            if hasattr(self.model.config, "num_hidden_layers"):
                num_layers = self.model.config.num_hidden_layers
            elif hasattr(self.model.config.text_config, "num_hidden_layers"):
                num_layers = self.model.config.text_config.num_hidden_layers
            else:
                raise ValueError(
                    "Failed to set aux hidden states layers as model config. "
                    f"{self.model.config} does not have num_hidden_layers"
                )
            aux_layer_ids = self._get_default_aux_layer_ids(num_layers)

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


class VLMTransformersBackend(BaseBackend):
    """VLM HuggingFace Transformers backend"""

    SUPPORT_MODEL_TYPE = ["hunyuan_vl", "qwen3_vl"]

    def load_model(self):
        if (
            self.target_model_type is None
            or self.target_model_type not in self.SUPPORT_MODEL_TYPE
        ):
            raise ValueError(f"{self.target_model_type} is not supported now!")

        if self.target_model_type == "hunyuan_vl":
            from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

            device = decide_device_for_distributed()
            print_with_rank(f"Loading model to device: {device}")

            # Prepare model loading configuration
            model_kwargs = self._prepare_model_kwargs(device)

            self.model = HunYuanVLForConditionalGeneration.from_pretrained(
                self.model_path, **model_kwargs
            )
            self.model.eval()

            # Load processor
            self.tokenizer = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        elif self.target_model_type == "qwen3_vl":
            from transformers import AutoModelForImageTextToText, AutoProcessor

            device = decide_device_for_distributed()
            print_with_rank(f"Loading model to device: {device}")

            # Prepare model loading configuration
            model_kwargs = self._prepare_model_kwargs(device)

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path, **model_kwargs
            )

            # Freeze the base model
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

            # Load processor
            self.tokenizer = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unsupported target model type: {self.target_model_type}")

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

        if self.target_model_type == "qwen3_vl":
            handle = self.model.language_model.register_forward_pre_hook(
                hook, with_kwargs=True
            )
        elif self.target_model_type == "hunyuan_vl":
            handle = self.model.model.register_forward_pre_hook(hook, with_kwargs=True)
        else:
            raise ValueError(f"Unsupported target model type: {self.target_model_type}")
        pixel_values = kwargs.get("pixel_values", None)
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        input_position_ids = kwargs.get("input_position_ids", None)
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=input_position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                output_logits=True,
            )

        handle.remove()
        inputs_embeds = (
            inputs_embeds_list[0].to(input_ids.device) if inputs_embeds_list else None
        )

        if self.target_model_type == "hunyuan_vl":
            position_ids = (
                position_ids_list[0][:, 0, :].to(input_ids.device)
                if position_ids_list
                else None
            )
        else:
            position_ids = (
                position_ids_list[0].to(input_ids.device) if position_ids_list else None
            )

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

        if self.target_model_type == "qwen3_vl":
            handle = self.model.language_model.register_forward_pre_hook(
                hook, with_kwargs=True
            )
        elif self.target_model_type == "hunyuan_vl":
            handle = self.model.model.register_forward_pre_hook(hook, with_kwargs=True)
        else:
            raise ValueError(f"Unsupported target model type: {self.target_model_type}")

        pixel_values = kwargs.get("pixel_values", None)
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        input_position_ids = kwargs.get("input_position_ids", None)
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                pixel_values=pixel_values,
                position_ids=input_position_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_logits=True,
            )

        handle.remove()
        inputs_embeds = (
            inputs_embeds_list[0].to(input_ids.device) if inputs_embeds_list else None
        )
        if self.target_model_type == "hunyuan_vl":
            position_ids = (
                position_ids_list[0][:, 0, :].to(input_ids.device)
                if position_ids_list
                else None
            )
        else:
            position_ids = (
                position_ids_list[0].to(input_ids.device) if position_ids_list else None
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
        # inputs_embeds: B, N, D
        # position_ids: 3, N
        return {
            "hidden_states": aux_hidden_states,
            "target_hiddens": target_hidden_states,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
        }


class AudioTransformersBackend(BaseBackend):
    """Audio HuggingFace Transformers backend"""

    SUPPORT_MODEL_TYPE = ["qwen2_audio"]

    def load_model(self):
        if (
            self.target_model_type is None
            or self.target_model_type not in self.SUPPORT_MODEL_TYPE
        ):
            raise ValueError(f"{self.target_model_type} is not supported now!")

        if self.target_model_type == "qwen2_audio":
            from transformers import (
                Qwen2AudioForConditionalGeneration,
                Qwen2AudioProcessor,
            )

            device = decide_device_for_distributed()
            print_with_rank(f"Loading model to device: {device}")

            # Prepare model loading configuration
            model_kwargs = self._prepare_model_kwargs(device)

            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_path, **model_kwargs
            )

            # Freeze the base model
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

            self.tokenizer = Qwen2AudioProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        else:
            raise ValueError(f"Unsupported target model type: {self.target_model_type}")

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
        input_features = kwargs.get("input_features", None)
        feature_attention_mask = kwargs.get("feature_attention_mask", None)
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        handle.remove()

        inputs_embeds = (
            inputs_embeds_list[0].to(input_ids.device) if inputs_embeds_list else None
        )
        position_ids = (
            position_ids_list[0].to(input_ids.device) if position_ids_list else None
        )

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
        input_features = kwargs.get("input_features", None)
        feature_attention_mask = kwargs.get("feature_attention_mask", None)
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        handle.remove()
        inputs_embeds = (
            inputs_embeds_list[0].to(input_ids.device) if inputs_embeds_list else None
        )
        position_ids = (
            position_ids_list[0].to(input_ids.device) if position_ids_list else None
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
        # inputs_embeds: B, N, D
        # position_ids: 3, N
        return {
            "hidden_states": aux_hidden_states,
            "target_hiddens": target_hidden_states,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
        }


class TTSTransformersBackend(TransformersBackend):
    """
    HuggingFace Transformers backend implementation.

    """

    def load_model(self) -> None:
        # Load and configure model
        if not os.path.exists(self.model_path):
            self.model_path = snapshot_download(self.model_path)

        # Determine device based on distributed environment
        self.device = decide_device_for_distributed()
        print_with_rank(f"Loading model to device: {self.device}")

        # Load model
        if os.path.exists(os.path.join(self.model_path, "cosyvoice3.yaml")):
            self.model_name = "cosyvoice3"
            self._load_cosyvoice3()
        else:
            raise NotImplementedError("This model is not implemented")

        self._freeze_model_parameters()
        self.model.eval()

    def _load_cosyvoice3(self) -> None:
        """Load text tokenizer using HuggingFace Transformers."""

        self.model = CosyVoice3LM(
            self.model_path,
            llm_input_size=896,
            llm_output_size=896,
            speech_token_size=6561,
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.model_path, "llm.pt"), map_location=self.device
            ),
            strict=True,
        )

        # Load tokenizer
        self.tokenizer = self.model.tokenizer

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
        if self.model_name == "cosyvoice3":
            with torch.no_grad():
                outputs = self.model(
                    **input_ids,
                    output_hidden_states=True,
                )
            return outputs


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
        ("hf", "TTS"): TTSTransformersBackend,
        ("hf", "Audio"): AudioTransformersBackend,
    }

    def __init__(
        self,
        model_path: str,
        modal_type: str = "LLM",
        backend: str = "hf",
        target_model_type: str = None,
        **kwargs,
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
        self.backend.target_model_type = target_model_type
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
    target_model_type: str = None,
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

    return TargetModelWrapper(
        backend=backend,
        model_path=model_path,
        target_model_type=target_model_type,
        **kwargs,
    )
