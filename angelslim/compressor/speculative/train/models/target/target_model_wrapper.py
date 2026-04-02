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

# ============================================================================
# Picklable callable classes for vLLM apply_model (TP > 1 compatibility)
#
# When tensor_parallel_size > 1, vLLM uses multiproc_executor which
# serializes functions via pickle through shared memory broadcast.
# Local closures / nested functions cannot be pickled by standard pickle.
# These module-level classes replace the original closures so that they
# can be serialized and sent to vLLM worker processes.
# ============================================================================


class _VLMSetupHooksFn:
    """Picklable callable that registers forward hooks inside a vLLM VLM worker."""

    def __init__(self, cache_attr: str, tp_size: int, lm_module_name: str):
        self.cache_attr = cache_attr
        self.tp_size = tp_size
        self.lm_module_name = lm_module_name

    def __call__(self, model):
        cache_attr = self.cache_attr
        lm_module_name = self.lm_module_name
        handles = []
        # Use lists to accumulate chunks from vLLM's chunked prefill.
        # When a long sequence is split into multiple chunks, each chunk
        # triggers the hooks separately. We collect all chunks and
        # concatenate them in _VLMCollectAndCleanupFn.
        setattr(
            model,
            cache_attr,
            {
                "all_hidden_states": [],
                "inputs_embeds_chunks": [],
                "position_ids_chunks": [],
                "normed_hidden_states_chunks": [],
            },
        )
        cache = getattr(model, cache_attr)

        tp_rank = 0
        tp_world_size = self.tp_size
        if tp_world_size > 1:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            tp_rank = get_tensor_model_parallel_rank()
            tp_world_size = get_tensor_model_parallel_world_size()

        lm = getattr(model, lm_module_name, None)
        if lm is None:
            raise AttributeError(
                f"Model does not have attribute '{lm_module_name}'. "
                f"Available attributes: {list(model.__dict__.keys())}"
            )

        def pre_hook(module, args, hook_kwargs):
            if "inputs_embeds" in hook_kwargs and hook_kwargs["inputs_embeds"] is not None:
                embeds = hook_kwargs["inputs_embeds"].clone().detach()
                if tp_rank == 0:
                    cache["inputs_embeds_chunks"].append(embeds.cpu())

            pos_key = None
            if "position_ids" in hook_kwargs and hook_kwargs["position_ids"] is not None:
                pos_key = "position_ids"
            elif "positions" in hook_kwargs and hook_kwargs["positions"] is not None:
                pos_key = "positions"

            pos_tensor = None
            if pos_key is not None:
                pos_tensor = hook_kwargs[pos_key]
            elif len(args) >= 2 and args[1] is not None:
                if isinstance(args[1], torch.Tensor):
                    pos_tensor = args[1]

            if pos_tensor is not None and tp_rank == 0:
                cache["position_ids_chunks"].append(pos_tensor.clone().detach().cpu())
            return args, hook_kwargs

        pre_hook_target = getattr(lm, "model", lm)
        h = pre_hook_target.register_forward_pre_hook(pre_hook, with_kwargs=True)
        handles.append(h)

        layers = None
        for attr in ["layers", "decoder_layers", "h", "blocks"]:
            layers = getattr(lm, attr, None)
            if layers is not None:
                break
        if layers is None and hasattr(lm, "model"):
            for attr in ["layers", "decoder_layers", "h", "blocks"]:
                layers = getattr(lm.model, attr, None)
                if layers is not None:
                    break

        if layers is not None:
            # Each element is a list of chunk tensors for that layer.
            # After all chunks are processed, they will be concatenated
            # in _VLMCollectAndCleanupFn.
            cache["all_hidden_states"] = [[] for _ in range(len(layers))]

            for layer_idx, layer in enumerate(layers):

                def make_layer_hook(idx):
                    def layer_hook(module, args, output):
                        if isinstance(output, tuple) and len(output) == 2:
                            hidden_states_out, residual_out = output
                            if residual_out is not None:
                                hidden = (hidden_states_out + residual_out).clone().detach()
                            else:
                                hidden = hidden_states_out.clone().detach()
                        else:
                            hidden = (
                                output.clone().detach()
                                if not isinstance(output, tuple)
                                else output[0].clone().detach()
                            )
                        if tp_rank == 0:
                            cache["all_hidden_states"][idx].append(hidden.cpu())

                        return output

                    return layer_hook

                h = layer.register_forward_hook(make_layer_hook(layer_idx))
                handles.append(h)

            first_layer = layers[0]

            def _first_layer_pre_hook(module, args, hook_kwargs):
                return args, hook_kwargs

            h = first_layer.register_forward_pre_hook(_first_layer_pre_hook, with_kwargs=True)
            handles.append(h)

        # Register hook on final norm to capture hidden states after RMSNorm
        # Try to find norm in lm.model or lm directly
        norm_module = None
        if hasattr(lm, "model") and hasattr(lm.model, "norm"):
            norm_module = lm.model.norm
        elif hasattr(lm, "norm"):
            norm_module = lm.norm

        if norm_module is not None:
            print_with_rank(
                f"[VLM] Registering norm hook on {type(norm_module).__name__} "
                f"(tp_rank={tp_rank})"
            )

            def vlm_norm_hook(module, args, output):
                if isinstance(output, tuple):
                    normed_hs = output[0]
                else:
                    normed_hs = output
                if tp_rank == 0:
                    cache["normed_hidden_states_chunks"].append(normed_hs.clone().detach().cpu())
                return output

            h = norm_module.register_forward_hook(vlm_norm_hook)
            handles.append(h)
        else:
            print_with_rank(
                f"[VLM] WARNING: norm module is None, norm hook NOT registered! "
                f"lm type: {type(lm).__name__}"
            )

        cache["_handles"] = handles
        return True


class _VLMCollectAndCleanupFn:
    """Picklable callable that reads hook data and cleans up.

    Handles vLLM's chunked prefill by concatenating chunk tensors collected
    during multiple forward passes into single contiguous tensors.
    """

    def __init__(self, cache_attr: str):
        self.cache_attr = cache_attr

    @staticmethod
    def _concat_chunks(chunks):
        """Concatenate a list of chunk tensors along the sequence dimension.

        Args:
            chunks: list of tensors, each with shape (seq_chunk, ...) or
                    (batch, seq_chunk, ...)

        Returns:
            Concatenated tensor or None if chunks is empty.
        """
        if not chunks:
            return None
        # Deduplicate chunks that may have been captured by multiple hooks
        # for the same forward pass. Use a simple heuristic: if consecutive
        # chunks have the same shape and identical content, skip duplicates.
        deduped = [chunks[0]]
        for c in chunks[1:]:
            if c.shape == deduped[-1].shape and torch.equal(c, deduped[-1]):
                continue
            deduped.append(c)
        if len(deduped) == 1:
            return deduped[0]
        # Concatenate along the sequence dimension (dim=0 for 1D/2D, dim=1 for 3D+)
        cat_dim = 0 if deduped[0].dim() <= 2 else 1
        return torch.cat(deduped, dim=cat_dim)

    def __call__(self, model):
        cache = getattr(model, self.cache_attr, None)
        if cache is None:
            return None
        for h in cache.get("_handles", []):
            h.remove()

        all_hidden_states_raw = cache["all_hidden_states"]

        # Concatenate chunked hidden states for each layer
        all_hidden_states = []
        for layer_chunks in all_hidden_states_raw:
            if isinstance(layer_chunks, list):
                if not layer_chunks:
                    all_hidden_states.append(None)
                elif len(layer_chunks) == 1:
                    all_hidden_states.append(layer_chunks[0])
                else:
                    # Concatenate chunks along sequence dimension
                    cat_dim = 0 if layer_chunks[0].dim() <= 2 else 1
                    all_hidden_states.append(torch.cat(layer_chunks, dim=cat_dim))
            else:
                # Backward compatibility: already a single tensor
                all_hidden_states.append(layer_chunks)

        # Concatenate chunked position_ids, inputs_embeds, normed_hidden_states
        position_ids = self._concat_chunks(cache.get("position_ids_chunks", []))
        inputs_embeds = self._concat_chunks(cache.get("inputs_embeds_chunks", []))
        normed_hidden_states = self._concat_chunks(cache.get("normed_hidden_states_chunks", []))

        result = {
            "all_hidden_states": all_hidden_states,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "normed_hidden_states": normed_hidden_states,
        }
        delattr(model, self.cache_attr)
        return result


class _LLMSetupHooksFn:
    """Picklable callable that registers forward hooks inside a vLLM LLM worker."""

    def __init__(self, cache_attr: str, tp_size: int):
        self.cache_attr = cache_attr
        self.tp_size = tp_size

    def __call__(self, model):
        cache_attr = self.cache_attr
        handles = []
        # Use lists to accumulate chunks from vLLM's chunked prefill.
        # When a long sequence is split into multiple chunks, each chunk
        # triggers the hooks separately. We collect all chunks and
        # concatenate them in _LLMCollectAndCleanupFn.
        setattr(
            model,
            cache_attr,
            {
                "all_hidden_states": [],
                "inputs_embeds_chunks": [],
                "position_ids_chunks": [],
                "normed_hidden_states_chunks": [],
            },
        )
        cache = getattr(model, cache_attr)

        tp_rank = 0
        tp_world_size = self.tp_size
        if tp_world_size > 1:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            tp_rank = get_tensor_model_parallel_rank()
            tp_world_size = get_tensor_model_parallel_world_size()

        inner_model = getattr(model, "model", None)
        if inner_model is None:
            raise AttributeError(
                f"Model does not have attribute 'model'. "
                f"Available attributes: {list(model.__dict__.keys())}"
            )

        def pre_hook(module, args, hook_kwargs):
            if "inputs_embeds" in hook_kwargs and hook_kwargs["inputs_embeds"] is not None:
                embeds = hook_kwargs["inputs_embeds"].clone().detach()
                if tp_rank == 0:
                    cache["inputs_embeds_chunks"].append(embeds.cpu())

            pos_key = None
            if "position_ids" in hook_kwargs and hook_kwargs["position_ids"] is not None:
                pos_key = "position_ids"
            elif "positions" in hook_kwargs and hook_kwargs["positions"] is not None:
                pos_key = "positions"

            pos_tensor = None
            if pos_key is not None:
                pos_tensor = hook_kwargs[pos_key]
            elif len(args) >= 2 and args[1] is not None:
                if isinstance(args[1], torch.Tensor):
                    pos_tensor = args[1]

            if pos_tensor is not None and tp_rank == 0:
                cache["position_ids_chunks"].append(pos_tensor.clone().detach().cpu())
            return args, hook_kwargs

        h = inner_model.register_forward_pre_hook(pre_hook, with_kwargs=True)
        handles.append(h)

        layers = None
        for attr in ["layers", "decoder_layers", "h", "blocks"]:
            layers = getattr(inner_model, attr, None)
            if layers is not None:
                break

        if layers is not None:
            # Each element is a list of chunk tensors for that layer.
            # After all chunks are processed, they will be concatenated
            # in _LLMCollectAndCleanupFn.
            num_layers = len(layers)
            cache["all_hidden_states"] = [[] for _ in range(num_layers)]

            for layer_idx, layer in enumerate(layers):

                def make_layer_hook(idx):
                    def layer_hook(module, args, output):
                        if isinstance(output, tuple) and len(output) == 2:
                            hidden_states_out, residual_out = output
                            if residual_out is not None:
                                hidden = (hidden_states_out + residual_out).clone().detach()
                            else:
                                hidden = hidden_states_out.clone().detach()
                        else:
                            hidden = (
                                output.clone().detach()
                                if not isinstance(output, tuple)
                                else output[0].clone().detach()
                            )
                        if tp_rank == 0:
                            cache["all_hidden_states"][idx].append(hidden.cpu())

                        return output

                    return layer_hook

                h = layer.register_forward_hook(make_layer_hook(layer_idx))
                handles.append(h)

            first_layer = layers[0]

            def _first_layer_pre_hook(module, args, hook_kwargs):
                return args, hook_kwargs

            h = first_layer.register_forward_pre_hook(_first_layer_pre_hook, with_kwargs=True)
            handles.append(h)

        # Register hook on final norm (inner_model.norm) to capture
        # hidden states after RMSNorm, which is needed for correct
        # target_hiddens when using TargetHead (lm_head projection).
        norm_module = getattr(inner_model, "norm", None)
        if norm_module is not None:
            print_with_rank(
                f"[LLM] Registering norm hook on {type(norm_module).__name__} "
                f"(tp_rank={tp_rank})"
            )

            def norm_hook(module, args, output):
                # vLLM RMSNorm returns (normed_hidden_states, residual)
                # or just normed_hidden_states depending on implementation
                if isinstance(output, tuple):
                    normed_hs = output[0]
                else:
                    normed_hs = output
                if tp_rank == 0:
                    cache["normed_hidden_states_chunks"].append(normed_hs.clone().detach().cpu())
                return output

            h = norm_module.register_forward_hook(norm_hook)
            handles.append(h)
        else:
            print_with_rank(
                "[LLM] WARNING: inner_model.norm is None, norm hook NOT registered! "
                f"inner_model type: {type(inner_model).__name__}"
            )

        cache["_handles"] = handles
        return True


class _LLMCollectAndCleanupFn:
    """Picklable callable that reads hook data and cleans up (for LLM backend).

    Handles vLLM's chunked prefill by concatenating chunk tensors collected
    during multiple forward passes into single contiguous tensors.
    """

    def __init__(self, cache_attr: str):
        self.cache_attr = cache_attr

    @staticmethod
    def _concat_chunks(chunks):
        """Concatenate a list of chunk tensors along the sequence dimension.

        Args:
            chunks: list of tensors, each with shape (seq_chunk, ...) or
                    (batch, seq_chunk, ...)

        Returns:
            Concatenated tensor or None if chunks is empty.
        """
        if not chunks:
            return None
        # Deduplicate chunks that may have been captured by multiple hooks
        # for the same forward pass (e.g., position_ids captured by both
        # pre_hook and layer_hook). Use a simple heuristic: if consecutive
        # chunks have the same shape and identical content, skip duplicates.
        deduped = [chunks[0]]
        for c in chunks[1:]:
            if c.shape == deduped[-1].shape and torch.equal(c, deduped[-1]):
                continue
            deduped.append(c)
        if len(deduped) == 1:
            return deduped[0]
        # Concatenate along the sequence dimension (dim=0 for 1D/2D, dim=1 for 3D+)
        cat_dim = 0 if deduped[0].dim() <= 2 else 1
        return torch.cat(deduped, dim=cat_dim)

    def __call__(self, model):
        cache = getattr(model, self.cache_attr, None)
        if cache is None:
            return None
        for h in cache.get("_handles", []):
            h.remove()

        all_hidden_states_raw = cache["all_hidden_states"]

        # Concatenate chunked hidden states for each layer
        all_hidden_states = []
        for layer_chunks in all_hidden_states_raw:
            if isinstance(layer_chunks, list):
                if not layer_chunks:
                    all_hidden_states.append(None)
                elif len(layer_chunks) == 1:
                    all_hidden_states.append(layer_chunks[0])
                else:
                    # Concatenate chunks along sequence dimension
                    cat_dim = 0 if layer_chunks[0].dim() <= 2 else 1
                    all_hidden_states.append(torch.cat(layer_chunks, dim=cat_dim))
            else:
                # Backward compatibility: already a single tensor
                all_hidden_states.append(layer_chunks)

        # Concatenate chunked position_ids, inputs_embeds, normed_hidden_states
        position_ids = self._concat_chunks(cache.get("position_ids_chunks", []))
        inputs_embeds = self._concat_chunks(cache.get("inputs_embeds_chunks", []))
        normed_hidden_states = self._concat_chunks(cache.get("normed_hidden_states_chunks", []))

        result = {
            "all_hidden_states": all_hidden_states,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "normed_hidden_states": normed_hidden_states,
        }
        delattr(model, self.cache_attr)
        return result


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
            total_layers: Total number of decoder layers (excluding embedding)

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

        selected_hiddens = [hidden_states[layer_id + embed_offset] for layer_id in aux_layer_ids]

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
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        self._freeze_model_parameters()
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def _prepare_model_kwargs(self, device: str) -> dict:
        """
        Prepare keyword arguments for model loading.

        Args:
            device: Target device for model placement

        Returns:
            Dictionary of model loading arguments
        """
        default_kwargs = {
            "torch_dtype": torch.bfloat16,
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
                use_cache=False,  # match SpecForge: no KV-cache during training
            )

        # Extract auxiliary hidden states
        aux_layer_ids = kwargs.get("aux_hidden_states_layer_ids", None)
        hidden_states = self._extract_auxiliary_hidden_states(outputs.hidden_states, aux_layer_ids)

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

    SUPPORT_MODEL_TYPE = ["hunyuan_vl", "qwen3_vl", "qwen2_5_vl"]

    def load_model(self):
        if self.target_model_type is None or self.target_model_type not in self.SUPPORT_MODEL_TYPE:
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
            self.tokenizer = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        elif self.target_model_type in ("qwen3_vl", "qwen2_5_vl"):
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
                inputs_embeds_list.append(kwargs["inputs_embeds"].clone().detach())
            if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                position_ids_list.append(kwargs["position_ids"].clone().detach())
            return args, kwargs

        if self.target_model_type in ("qwen3_vl", "qwen2_5_vl"):
            handle = self.model.model.language_model.register_forward_pre_hook(
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
        inputs_embeds = inputs_embeds_list[0].to(input_ids.device) if inputs_embeds_list else None

        if self.target_model_type == "hunyuan_vl":
            position_ids = (
                position_ids_list[0][:, 0, :].to(input_ids.device) if position_ids_list else None
            )
        else:
            position_ids = position_ids_list[0].to(input_ids.device) if position_ids_list else None

        # Extract auxiliary hidden states
        aux_layer_ids = kwargs.get("aux_hidden_states_layer_ids", None)
        hidden_states = self._extract_auxiliary_hidden_states(outputs.hidden_states, aux_layer_ids)

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
                inputs_embeds_list.append(kwargs["inputs_embeds"].clone().detach())
            if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                position_ids_list.append(kwargs["position_ids"].clone().detach())
            return args, kwargs

        if self.target_model_type in ("qwen3_vl", "qwen2_5_vl"):
            handle = self.model.model.language_model.register_forward_pre_hook(
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
        inputs_embeds = inputs_embeds_list[0].to(input_ids.device) if inputs_embeds_list else None
        if self.target_model_type == "hunyuan_vl":
            position_ids = (
                position_ids_list[0][:, 0, :].to(input_ids.device) if position_ids_list else None
            )
        else:
            position_ids = position_ids_list[0].to(input_ids.device) if position_ids_list else None
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


class VLMVLLMBackend(BaseBackend):
    """VLM vLLM backend, use vLLM for inference and extract hidden states.

    Register forward hook on vLLM model's language_model to capture
    inputs_embeds and position_ids, and extract hidden states via apply_model.

    Supported model types:
        - qwen3_vl: Qwen3-VL series vision-language models
        - qwen2_5_vl: Qwen2.5-VL series vision-language models
        - hunyuan_vl: HunYuan-VL series vision-language models
    """

    SUPPORT_MODEL_TYPE = ["qwen3_vl", "qwen2_5_vl", "hunyuan_vl"]

    def load_model(self) -> None:
        """Load VLM model using vLLM.

        Only supported in Ray actor or standalone (non-torchrun) environments.
        Ray actor processes do not have torchrun environment variables;
        CUDA_VISIBLE_DEVICES is managed by Ray automatically, and vLLM
        can freely use NCCL to create process groups without any conflicts.

        For torchrun-based VLM inference, use VLMTransformersBackend instead.
        """
        from vllm import LLM

        if self.target_model_type is None or self.target_model_type not in self.SUPPORT_MODEL_TYPE:
            raise ValueError(
                f"{self.target_model_type} is not supported. "
                f"Supported types: {self.SUPPORT_MODEL_TYPE}"
            )

        # Extract vllm-related parameters from kwargs
        tp_size = self.kwargs.get("tensor_parallel_size", 1)
        self.tp_size = (
            tp_size  # Save TP size for hook functions to decide whether to collect from tp_rank 0
        )
        max_model_len = self.kwargs.get("max_model_len", 8192)
        gpu_memory_utilization = self.kwargs.get("gpu_memory_utilization", 0.9)
        print_with_rank(f"gpu_memory_utilization: {gpu_memory_utilization}")
        enforce_eager = self.kwargs.get("enforce_eager", True)
        max_num_seqs = self.kwargs.get("max_num_seqs", 8)
        limit_mm_per_prompt = self.kwargs.get("limit_mm_per_prompt", {"image": 10, "video": 10})
        print_with_rank(f"limit_mm_per_prompt: {limit_mm_per_prompt}")

        if tp_size > 1:
            distributed_executor_backend = self.kwargs.get("distributed_executor_backend", "mp")
        else:
            distributed_executor_backend = self.kwargs.get("distributed_executor_backend", None)

        # apply_model() passes closure functions (setup_hooks_fn, collect_and_cleanup_fn)
        # to the vLLM EngineCore subprocess. vLLM's default safe serializer cannot
        # handle `function` objects, so we enable pickle-based fallback serialization.
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        from transformers import AutoProcessor

        self.tokenizer = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        _max_pixels = os.environ.get("MAX_PIXELS")
        _min_pixels = os.environ.get("MIN_PIXELS", "1024")
        _max_pixels = int(_max_pixels) if _max_pixels is not None else None
        _min_pixels = int(_min_pixels) if _min_pixels is not None else None
        print_with_rank(f"_max_pixels: {_max_pixels}, _min_pixels: {_min_pixels}")
        mm_processor_kwargs = {}
        if _max_pixels is not None or _min_pixels is not None:
            from angelslim.compressor.speculative.train.data.data_utils import (
                build_image_processor_kwargs,
            )

            mm_processor_kwargs = build_image_processor_kwargs(
                self.tokenizer.image_processor,
                max_pixels=_max_pixels,
                min_pixels=_min_pixels,
            )

        print_with_rank(f"Loading VLM model with vLLM backend: {self.model_path}")
        print_with_rank(
            f"  tensor_parallel_size={tp_size}, max_model_len={max_model_len}, "
            f"distributed_executor_backend={distributed_executor_backend}"
        )
        if mm_processor_kwargs:
            print_with_rank(f"  mm_processor_kwargs={mm_processor_kwargs}")

        self.model = LLM(
            model=self.model_path,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
            distributed_executor_backend=distributed_executor_backend,
            trust_remote_code=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            mm_processor_kwargs=mm_processor_kwargs or None,
            # Disable prefix caching to ensure forward hooks capture hidden
            # states for the FULL sequence (see VLLMBackend for details).
            enable_prefix_caching=False,
        )

    def _get_language_model_module_name(self) -> str:
        """Return the language model sub-module name based on model type."""
        if self.target_model_type in ("qwen3_vl", "qwen2_5_vl"):
            return "language_model"
        elif self.target_model_type == "hunyuan_vl":
            return "model"
        else:
            raise ValueError(f"Unsupported target model type: {self.target_model_type}")

    @staticmethod
    def _collapse_consecutive_image_pad(ids: list, image_pad_token_id: int) -> list:
        """Collapse runs of consecutive image_pad tokens into a single token.

        online_dataset_builder._process_single_conversation expands each
        <|image_pad|> placeholder into N consecutive image_pad tokens based on
        the image's pixel count.  vLLM, however, expects only **one**
        image_pad token per image and performs its own expansion internally
        via _apply_prompt_updates.  If we pass the already-expanded N tokens,
        vLLM replaces only the first one and leaves N-1 stale tokens in the
        prompt, corrupting the sequence length.

        This helper compresses each consecutive run of image_pad tokens back
        to a single token so that vLLM can expand it correctly.

        Args:
            ids: list of token ids (already valid-length truncated)
            image_pad_token_id: the integer token id for <|image_pad|>

        Returns:
            New list with consecutive image_pad runs collapsed to one token.
        """
        if image_pad_token_id not in ids:
            return ids
        collapsed: list = []
        prev_is_pad = False
        for tok in ids:
            if tok == image_pad_token_id:
                if not prev_is_pad:
                    collapsed.append(tok)
                prev_is_pad = True
            else:
                collapsed.append(tok)
                prev_is_pad = False
        return collapsed

    def _build_vllm_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> list:
        """Convert batched tensor inputs to a vLLM PromptType list.

        vLLM does not accept batched tensor inputs; each sample must be
        converted to an independent prompt dict.

        vLLM's multi_modal_data only accepts the following modality keys:
        "audio", "image", "video", "vision_chunk".  Metadata tensors such as
        image_grid_thw / video_grid_thw must NOT be placed as top-level keys;
        vLLM's internal processor will compute them automatically from the
        raw images / videos.

        **Important**: When images are present, the input_ids from
        _process_single_conversation contain N consecutive image_pad tokens
        per image (pre-expanded for HF Transformers backend).  vLLM expects
        only **one** image_pad token per image and will expand it internally.
        This method collapses consecutive image_pad tokens before passing to
        vLLM.

        Args:
            input_ids: shape [batch_size, seq_len]
            attention_mask: shape [batch_size, seq_len], used to determine valid length
            **kwargs: may contain:
                - raw_images: list of PIL Image lists (one list per sample)
                - raw_videos: list of video data (one per sample)
                - pixel_values, pixel_values_videos: pre-processed tensors
                  (used only when raw images/videos are not available)

        Returns:
            List of vLLM PromptType, one element per sample
        """
        from vllm import TokensPrompt

        batch_size = input_ids.shape[0]
        raw_images = kwargs.get("raw_images", None)
        raw_videos = kwargs.get("raw_videos", None)

        # Resolve image_pad token id for collapsing consecutive runs
        image_pad_token_id = None
        has_multimodal = (raw_images is not None and any(imgs for imgs in raw_images)) or (
            raw_videos is not None and any(vids for vids in raw_videos)
        )
        if has_multimodal and self.tokenizer is not None:
            # Qwen VL models: image_token attribute or fall back to tokenizer vocab
            _image_token = getattr(self.tokenizer, "image_token", "<|image_pad|>")
            _tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            _vocab = _tok.get_vocab() if hasattr(_tok, "get_vocab") else {}
            image_pad_token_id = _vocab.get(_image_token)

        prompts = []
        for i in range(batch_size):
            # Truncate to valid tokens based on attention_mask
            if attention_mask is not None:
                valid_len = int(attention_mask[i].sum().item())
                ids = input_ids[i, :valid_len].tolist()
            else:
                ids = input_ids[i].tolist()

            # Collapse consecutive image_pad tokens so vLLM can re-expand correctly
            if image_pad_token_id is not None:
                ids = self._collapse_consecutive_image_pad(ids, image_pad_token_id)

            prompt: dict = {"prompt_token_ids": ids}

            # Attach multimodal data (image and/or video)
            # vLLM expects raw PIL Images (or ndarray), NOT pre-processed
            # pixel_values tensors.  It will run its own image processor
            # internally, which also computes image_grid_thw automatically.
            mm_data = {}
            if raw_images is not None and raw_images[i]:
                images_for_sample = raw_images[i]
                # vLLM accepts a single image or a list of images
                if len(images_for_sample) == 1:
                    mm_data["image"] = images_for_sample[0]
                else:
                    mm_data["image"] = images_for_sample

            if raw_videos is not None and raw_videos[i]:
                mm_data["video"] = raw_videos[i]

            if mm_data:
                prompt["multi_modal_data"] = mm_data

            prompts.append(TokensPrompt(**prompt))

        return prompts

    def get_hidden_states_and_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """get hidden states and logits from vLLM backend.

        Args:
            input_ids: shape [batch_size, seq_len]
            attention_mask: shape [batch_size, seq_len]
            **kwargs: pixel_values, image_grid_thw, aux_hidden_states_layer_ids

        Returns:
            Tuple of (hidden_states, logits, inputs_embeds, position_ids)
        """
        raise NotImplementedError(
            "get_hidden_states_and_logits is not implemented for VLMVLLMBackend. "
            "Please use get_aux_and_target_hiddens instead."
        )

    def get_aux_and_target_hiddens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Extract auxiliary and target hidden states using the vLLM backend.

        Registers forward hooks inside vLLM workers via apply_model, stores the
        collected hidden states as a temporary attribute on the model, then reads
        and cleans up the data after inference via a second apply_model call.

        Note: This method requires vLLM to run with enforce_eager=True (no CUDA
        graph) so that forward hooks fire correctly.

        Args:
            input_ids: Input token IDs, shape [batch_size, seq_len]
            attention_mask: Attention mask, shape [batch_size, seq_len]
            **kwargs: May contain:
                - pixel_values: image pixel values
                - image_grid_thw: image grid dimensions
                - aux_hidden_states_layer_ids: list of auxiliary layer indices

        Returns:
            dict containing:
                - hidden_states: concatenated auxiliary hidden states,
                  shape [batch_size, seq_len, hidden_size * 3]
                - target_hiddens: final-layer hidden states,
                  shape [batch_size, seq_len, hidden_size]
                - inputs_embeds: input embeddings,
                  shape [batch_size, seq_len, hidden_size]
                - position_ids: position encoding IDs
        """
        lm_module_name = self._get_language_model_module_name()
        aux_layer_ids = kwargs.get("aux_hidden_states_layer_ids", None)
        # Temporary attribute name used to store hook data inside the worker
        _CACHE_ATTR = "_vlm_vllm_hook_cache"
        # TP size from model init
        _tp_size = getattr(self, "tp_size", 1)

        # Use picklable callable classes instead of local closures so that
        # vLLM's multiproc_executor can serialize them via pickle when TP > 1.
        setup_hooks_fn = _VLMSetupHooksFn(_CACHE_ATTR, _tp_size, lm_module_name)
        collect_and_cleanup_fn = _VLMCollectAndCleanupFn(_CACHE_ATTR)

        # Register hooks inside the worker; include in try-finally so that
        # partially-registered hooks are always cleaned up on failure.
        try:
            self.model.apply_model(setup_hooks_fn)

            # Build vLLM inputs
            prompts = self._build_vllm_inputs(input_ids, attention_mask, **kwargs)

            # Run inference (vLLM internally triggers the hooks)
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=0,
            )
            _ = self.model.generate(prompts, sampling_params=sampling_params)

        finally:
            # Read data from the worker and clean up hooks
            worker_results = self.model.apply_model(collect_and_cleanup_fn)

        # apply_model returns a list of results, one per worker.
        # When TP > 1, only the TP rank-0 worker holds complete data
        # (other workers have None elements in all_hidden_states).
        # Pick the first result whose all_hidden_states contains non-None entries.
        collected = None
        for result in worker_results:
            if (
                result is not None
                and result.get("all_hidden_states")
                and any(h is not None for h in result["all_hidden_states"])
            ):
                collected = result
                break

        if collected is None:
            raise RuntimeError(
                "Failed to collect hidden states from vLLM model. "
                "apply_model returned no valid results."
            )

        all_hs = collected["all_hidden_states"]
        inputs_embeds = collected["inputs_embeds"]
        position_ids = collected["position_ids"]
        normed_hidden_states = collected.get("normed_hidden_states")

        if not all_hs or all(h is None for h in all_hs):
            raise RuntimeError(
                "Failed to collect hidden states from vLLM model. "
                "Please check that the model architecture is supported and "
                "enforce_eager=True is set."
            )

        # Determine auxiliary layer indices
        if aux_layer_ids is None:
            num_layers = len(all_hs)
            aux_layer_ids = self._get_default_aux_layer_ids(num_layers)

        # Extract and concatenate auxiliary-layer hidden states
        selected_hiddens = [all_hs[layer_id] for layer_id in aux_layer_ids]
        aux_hidden_states = torch.cat(selected_hiddens, dim=-1)

        # Use normed hidden states (after final RMSNorm) as target_hiddens
        # for correct logit computation via TargetHead.
        if normed_hidden_states is not None:
            target_hidden_states = normed_hidden_states
            print_with_rank(
                f"[VLM] Using normed_hidden_states as target_hiddens, "
                f"shape={normed_hidden_states.shape}, "
                f"mean={normed_hidden_states.float().mean().item():.6f}, "
                f"std={normed_hidden_states.float().std().item():.6f}"
            )
        else:
            print_with_rank(
                "[VLM] WARNING: normed_hidden_states not available from vLLM hook. "
                "Falling back to pre-norm last-layer hidden states. "
                "This may cause incorrect target_logits computation. "
                f"collected keys: {list(collected.keys())}"
            )
            target_hidden_states = all_hs[-1]

        # vLLM internally uses packed/flattened format, so hook-captured hidden
        # states are 2D (seq_len, hidden_size) without a batch dimension.
        # Add batch dimension to match the expected (B, N, D) format used by
        # the rest of the pipeline (data collator, trainer, etc.).
        if aux_hidden_states.dim() == 2:
            aux_hidden_states = aux_hidden_states.unsqueeze(0)
        if target_hidden_states.dim() == 2:
            target_hidden_states = target_hidden_states.unsqueeze(0)
        if inputs_embeds is not None and inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # Handle position_ids for hunyuan_vl (take the first dimension)
        if self.target_model_type == "hunyuan_vl" and position_ids is not None:
            if position_ids.dim() == 3:
                position_ids = position_ids[:, 0, :]

        # For MRoPE models (e.g. Qwen3-VL), hook-captured position_ids have
        # shape (3, num_tokens) without a batch dimension.  Add batch dim to
        # get (3, 1, num_tokens) so the data collator can pad/stack correctly.
        if position_ids is not None and position_ids.dim() == 2:
            # Check if this is MRoPE format (first dim == 3) vs regular (B, N)
            if position_ids.shape[0] == 3:
                # MRoPE: (3, num_tokens) -> (3, 1, num_tokens)
                position_ids = position_ids.unsqueeze(1)
            # else: regular 2D (B, N), keep as-is

        # Truncate to input_ids length to exclude decode-step hidden states.
        # vLLM's generate() runs prefill (all input tokens) then decode
        # (max_tokens=1 new token). Our hooks capture both stages, so the
        # concatenated hidden states may be longer than the input sequence.
        # We only need the prefill hidden states (matching input_ids length).
        if attention_mask is not None:
            expected_len = int(attention_mask.sum(dim=-1).max().item())
        else:
            expected_len = input_ids.shape[-1]

        if aux_hidden_states.shape[1] > expected_len:
            aux_hidden_states = aux_hidden_states[:, :expected_len, :]
        if target_hidden_states.shape[1] > expected_len:
            target_hidden_states = target_hidden_states[:, :expected_len, :]
        if inputs_embeds is not None and inputs_embeds.shape[1] > expected_len:
            inputs_embeds = inputs_embeds[:, :expected_len, :]
        if position_ids is not None:
            if position_ids.dim() == 1 and position_ids.shape[0] > expected_len:
                position_ids = position_ids[:expected_len]
            elif position_ids.dim() >= 2 and position_ids.shape[-1] > expected_len:
                position_ids = position_ids[..., :expected_len]

        # Move results to the same device as input_ids
        device = input_ids.device
        aux_hidden_states = aux_hidden_states.to(device)
        target_hidden_states = target_hidden_states.to(device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)

        return {
            "hidden_states": aux_hidden_states,
            "target_hiddens": target_hidden_states,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
        }


class VLLMBackend(BaseBackend):
    """LLM vLLM backend, use vLLM for inference and extract hidden states.

    Register forward hooks on vLLM model to capture inputs_embeds,
    position_ids, and extract hidden states from each decoder layer.

    This backend is designed for pure text LLMs (not VLMs), where the
    top-level model is a CausalLM (e.g. Qwen2ForCausalLM, Qwen3ForCausalLM)
    with a `model` attribute containing the decoder layers.

    Supported model types:
        - qwen2.5: Qwen2.5 series text LLMs
        - qwen3: Qwen3 series text LLMs
    """

    SUPPORT_MODEL_TYPE = ["qwen2.5", "qwen3"]

    def load_model(self) -> None:
        """Load LLM model using vLLM.

        Only supported in Ray actor or standalone (non-torchrun) environments.
        Ray actor processes do not have torchrun environment variables;
        CUDA_VISIBLE_DEVICES is managed by Ray automatically, and vLLM
        can freely use NCCL to create process groups without any conflicts.

        For torchrun-based LLM inference, use TransformersBackend instead.
        """
        from vllm import LLM

        if self.target_model_type is None or self.target_model_type not in self.SUPPORT_MODEL_TYPE:
            raise ValueError(
                f"{self.target_model_type} is not supported. "
                f"Supported types: {self.SUPPORT_MODEL_TYPE}"
            )

        # Extract vllm-related parameters from kwargs
        tp_size = self.kwargs.get("tensor_parallel_size", 1)
        self.tp_size = tp_size
        max_model_len = self.kwargs.get("max_model_len", 8192)
        gpu_memory_utilization = self.kwargs.get("gpu_memory_utilization", 0.9)
        print_with_rank(f"gpu_memory_utilization: {gpu_memory_utilization}")
        enforce_eager = self.kwargs.get("enforce_eager", True)
        max_num_seqs = self.kwargs.get("max_num_seqs", 8)

        if tp_size > 1:
            distributed_executor_backend = self.kwargs.get("distributed_executor_backend", "mp")
        else:
            distributed_executor_backend = self.kwargs.get("distributed_executor_backend", None)

        # apply_model() passes closure functions to the vLLM EngineCore subprocess.
        # vLLM's default safe serializer cannot handle `function` objects,
        # so we enable pickle-based fallback serialization.
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        print_with_rank(f"Loading LLM model with vLLM backend: {self.model_path}")
        print_with_rank(
            f"  tensor_parallel_size={tp_size}, max_model_len={max_model_len}, "
            f"distributed_executor_backend={distributed_executor_backend}"
        )

        self.model = LLM(
            model=self.model_path,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
            distributed_executor_backend=distributed_executor_backend,
            trust_remote_code=True,
            # Disable prefix caching to ensure forward hooks capture hidden
            # states for the FULL sequence. With prefix caching enabled,
            # vLLM reuses cached KV for shared prefixes and only runs the
            # model forward on the uncached suffix, causing hooks to miss
            # the prefix portion of hidden states.
            enable_prefix_caching=False,
        )

    def _build_vllm_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> list:
        """Convert batched tensor inputs to a vLLM TokensPrompt list.

        Args:
            input_ids: shape [batch_size, seq_len]
            attention_mask: shape [batch_size, seq_len], used to determine valid length

        Returns:
            List of vLLM TokensPrompt, one element per sample
        """
        from vllm import TokensPrompt

        batch_size = input_ids.shape[0]
        prompts = []
        for i in range(batch_size):
            if attention_mask is not None:
                valid_len = int(attention_mask[i].sum().item())
                ids = input_ids[i, :valid_len].tolist()
            else:
                ids = input_ids[i].tolist()

            prompts.append(TokensPrompt(prompt_token_ids=ids))

        return prompts

    def get_hidden_states_and_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """get hidden states and logits from vLLM backend.

        Args:
            input_ids: shape [batch_size, seq_len]
            attention_mask: shape [batch_size, seq_len]
            **kwargs: aux_hidden_states_layer_ids

        Returns:
            Tuple of (hidden_states, logits)
        """
        raise NotImplementedError(
            "get_hidden_states_and_logits is not implemented for VLLMBackend. "
            "Please use get_aux_and_target_hiddens instead."
        )

    def get_aux_and_target_hiddens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Extract auxiliary and target hidden states using the vLLM backend.

        Registers forward hooks inside vLLM workers via apply_model, stores the
        collected hidden states as a temporary attribute on the model, then reads
        and cleans up the data after inference via a second apply_model call.

        Note: This method requires vLLM to run with enforce_eager=True (no CUDA
        graph) so that forward hooks fire correctly.

        Args:
            input_ids: Input token IDs, shape [batch_size, seq_len]
            attention_mask: Attention mask, shape [batch_size, seq_len]
            **kwargs: May contain:
                - aux_hidden_states_layer_ids: list of auxiliary layer indices

        Returns:
            dict containing:
                - hidden_states: concatenated auxiliary hidden states,
                  shape [batch_size, seq_len, hidden_size * 3]
                - target_hiddens: final-layer hidden states,
                  shape [batch_size, seq_len, hidden_size]
                - inputs_embeds: input embeddings,
                  shape [batch_size, seq_len, hidden_size]
                - position_ids: position encoding IDs
        """
        aux_layer_ids = kwargs.get("aux_hidden_states_layer_ids", None)
        # Temporary attribute name used to store hook data inside the worker
        _CACHE_ATTR = "_llm_vllm_hook_cache"
        # TP size from model init
        _tp_size = getattr(self, "tp_size", 1)

        # Use picklable callable classes instead of local closures so that
        # vLLM's multiproc_executor can serialize them via pickle when TP > 1.
        setup_hooks_fn = _LLMSetupHooksFn(_CACHE_ATTR, _tp_size)
        collect_and_cleanup_fn = _LLMCollectAndCleanupFn(_CACHE_ATTR)

        # Register hooks inside the worker
        try:
            self.model.apply_model(setup_hooks_fn)

            # Build vLLM inputs (pure text, no multimodal data)
            prompts = self._build_vllm_inputs(input_ids, attention_mask)

            # Run inference (vLLM internally triggers the hooks)
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=0,
            )
            _ = self.model.generate(prompts, sampling_params=sampling_params)

        finally:
            worker_results = self.model.apply_model(collect_and_cleanup_fn)

        # Pick the first result whose all_hidden_states contains non-None entries
        collected = None
        for result in worker_results:
            if (
                result is not None
                and result.get("all_hidden_states")
                and any(h is not None for h in result["all_hidden_states"])
            ):
                collected = result
                break

        if collected is None:
            raise RuntimeError(
                "Failed to collect hidden states from vLLM model. "
                "apply_model returned no valid results."
            )

        all_hs = collected["all_hidden_states"]
        inputs_embeds = collected["inputs_embeds"]
        position_ids = collected["position_ids"]
        normed_hidden_states = collected.get("normed_hidden_states")

        if not all_hs or all(h is None for h in all_hs):
            raise RuntimeError(
                "Failed to collect hidden states from vLLM model. "
                "Please check that the model architecture is supported and "
                "enforce_eager=True is set."
            )

        # Determine auxiliary layer indices
        if aux_layer_ids is None:
            num_layers = len(all_hs)
            aux_layer_ids = self._get_default_aux_layer_ids(num_layers)

        # Extract and concatenate auxiliary-layer hidden states
        selected_hiddens = [all_hs[layer_id] for layer_id in aux_layer_ids]
        aux_hidden_states = torch.cat(selected_hiddens, dim=-1)

        # Use normed hidden states (after final RMSNorm) as target_hiddens
        # for correct logit computation via TargetHead.
        # vLLM decoder layer hooks capture hidden states BEFORE the final
        # RMSNorm, which leads to incorrect target_logits when used with
        # TargetHead (lm_head projection). The norm hook captures the
        # post-norm hidden states that match HF Transformers' behavior.
        if normed_hidden_states is not None:
            target_hidden_states = normed_hidden_states
            print_with_rank(
                f"[LLM] Using normed_hidden_states as target_hiddens, "
                f"shape={normed_hidden_states.shape}, "
                f"mean={normed_hidden_states.float().mean().item():.6f}, "
                f"std={normed_hidden_states.float().std().item():.6f}"
            )
        else:
            # Fallback to last layer hidden states (pre-norm) if norm hook
            # did not fire (e.g., unsupported architecture)
            print_with_rank(
                "[LLM] WARNING: normed_hidden_states not available from vLLM hook. "
                "Falling back to pre-norm last-layer hidden states. "
                "This may cause incorrect target_logits computation. "
                f"collected keys: {list(collected.keys())}"
            )
            target_hidden_states = all_hs[-1]

        # vLLM internally uses packed/flattened format, so hook-captured hidden
        # states are 2D (seq_len, hidden_size) without a batch dimension.
        # Add batch dimension to match the expected (B, N, D) format.
        if aux_hidden_states.dim() == 2:
            aux_hidden_states = aux_hidden_states.unsqueeze(0)
        if target_hidden_states.dim() == 2:
            target_hidden_states = target_hidden_states.unsqueeze(0)
        if inputs_embeds is not None and inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # Truncate to input_ids length to exclude decode-step hidden states.
        # vLLM's generate() runs prefill (all input tokens) then decode
        # (max_tokens=1 new token). Our hooks capture both stages, so the
        # concatenated hidden states may be longer than the input sequence.
        # We only need the prefill hidden states (matching input_ids length).
        if attention_mask is not None:
            expected_len = int(attention_mask.sum(dim=-1).max().item())
        else:
            expected_len = input_ids.shape[-1]

        if aux_hidden_states.shape[1] > expected_len:
            aux_hidden_states = aux_hidden_states[:, :expected_len, :]
        if target_hidden_states.shape[1] > expected_len:
            target_hidden_states = target_hidden_states[:, :expected_len, :]
        if inputs_embeds is not None and inputs_embeds.shape[1] > expected_len:
            inputs_embeds = inputs_embeds[:, :expected_len, :]
        if position_ids is not None:
            if position_ids.dim() == 1 and position_ids.shape[0] > expected_len:
                position_ids = position_ids[:expected_len]
            elif position_ids.dim() >= 2 and position_ids.shape[-1] > expected_len:
                position_ids = position_ids[..., :expected_len]

        # Move results to the same device as input_ids
        device = input_ids.device
        aux_hidden_states = aux_hidden_states.to(device)
        target_hidden_states = target_hidden_states.to(device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)

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
        if self.target_model_type is None or self.target_model_type not in self.SUPPORT_MODEL_TYPE:
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
                inputs_embeds_list.append(kwargs["inputs_embeds"].clone().detach())
            if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                position_ids_list.append(kwargs["position_ids"].clone().detach())
            return args, kwargs

        handle = self.model.language_model.register_forward_pre_hook(hook, with_kwargs=True)
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

        inputs_embeds = inputs_embeds_list[0].to(input_ids.device) if inputs_embeds_list else None
        position_ids = position_ids_list[0].to(input_ids.device) if position_ids_list else None

        # Extract auxiliary hidden states
        aux_layer_ids = kwargs.get("aux_hidden_states_layer_ids", None)
        hidden_states = self._extract_auxiliary_hidden_states(outputs.hidden_states, aux_layer_ids)

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
                inputs_embeds_list.append(kwargs["inputs_embeds"].clone().detach())
            if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                position_ids_list.append(kwargs["position_ids"].clone().detach())
            return args, kwargs

        handle = self.model.language_model.register_forward_pre_hook(hook, with_kwargs=True)
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
        inputs_embeds = inputs_embeds_list[0].to(input_ids.device) if inputs_embeds_list else None
        position_ids = position_ids_list[0].to(input_ids.device) if position_ids_list else None

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
            torch.load(os.path.join(self.model_path, "llm.pt"), map_location=self.device),
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
        ("vllm", "LLM"): VLLMBackend,
        ("vllm", "VLM"): VLMVLLMBackend,
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
            raise AttributeError(f"Backend '{self.backend_name}' does not support tokenizers")
        if self.backend.tokenizer is None:
            raise ValueError(f"Tokenizer not initialized for backend '{self.backend_name}'")
        return self.backend.tokenizer


def create_target_model(
    backend: str,
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    target_model_type: str = None,
    modal_type: str = "LLM",
    **extra_kwargs,
) -> TargetModelWrapper:
    """
    Factory function to create target model with appropriate backend configuration.

    This function provides a convenient way to instantiate a TargetModelWrapper
    with commonly used default settings.

    Args:
        backend: Backend type, one of ["hf", "vllm"]
        model_path: Path to model checkpoint or serving endpoint URL
        torch_dtype: Data type for model weights (for HF backend)
        trust_remote_code: Whether to trust and execute remote code
        target_model_type: Specific model type, e.g. "qwen3_vl", "hunyuan_vl"
        modal_type: Modal type, one of ["LLM", "VLM", "TTS", "Audio"]
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
        kwargs["torch_dtype"] = torch_dtype
    elif backend == "vllm":
        # vllm backend does not use the torch_dtype parameter; other extra_kwargs are kept
        pass
    else:
        raise ValueError(
            f"Unsupported backend: '{backend}'. "
            f"Use one of: {list(TargetModelWrapper.BACKENDS.keys())}"
        )

    return TargetModelWrapper(
        backend=backend,
        model_path=model_path,
        modal_type=modal_type,
        target_model_type=target_model_type,
        **kwargs,
    )
