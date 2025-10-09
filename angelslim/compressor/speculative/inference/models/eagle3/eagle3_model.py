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
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from ....utils import (
    EWMAScorePredictor,
    MeanScorePredictor,
    MomentumScorePredictor,
    evaluate_posterior,
    initialize_past_key_values,
    initialize_tree,
    prepare_logits_processor,
    reset_tree_mode,
    tree_decoding,
    update_inference_inputs,
)
from .configuration_eagle3_model import Eagle3Config
from .draft import Llama3Eagle3Drafter
from .target import LlamaForCausalLM as KVLlamaForCausalLM
from .target import Qwen3ForCausalLM as KVQwen3ForCausalLM


@dataclass
class GenerationConfig:
    """Configuration for text generation"""

    temperature: float = 0.0
    top_p: float = 0.0
    top_k: float = 0.0
    max_new_tokens: int = 512
    max_length: int = 2048
    log: bool = False
    is_llama3: bool = False


@dataclass
class GenerationState:
    """State management for generation process"""

    stop_token_id: Optional[int]
    logits_processor: Optional[Any]
    input_ids: torch.Tensor
    past_key_values: Any
    input_len: int
    new_token: int = 0


class ModelLoader:
    """Handles loading of base models and EAGLE components"""

    SUPPORTED_ARCHITECTURES = {
        "LlamaForCausalLM": KVLlamaForCausalLM,
        "Qwen3ForCausalLM": KVQwen3ForCausalLM,
    }

    @classmethod
    def load_base_model(cls, base_model_path: str, **kwargs) -> nn.Module:
        """Load base model based on architecture"""
        config = AutoConfig.from_pretrained(base_model_path)
        if not getattr(config, "architectures", None):
            raise ValueError("Base model config missing 'architectures' field")

        arch = config.architectures[0]
        if arch not in cls.SUPPORTED_ARCHITECTURES:
            raise NotImplementedError(f"Model {arch} not supported")

        model_class = cls.SUPPORTED_ARCHITECTURES[arch]
        return model_class.from_pretrained(base_model_path, **kwargs)

    @classmethod
    def ensure_config_path(cls, eagle_model_path: str) -> str:
        """Ensure config file exists and return its path"""
        config_path = os.path.join(eagle_model_path, "config.json")
        if not os.path.exists(config_path):
            config_path = hf_hub_download(eagle_model_path, "config.json")
        return config_path

    @classmethod
    def load_eagle_state_dict(cls, eagle_model_path: str, device: torch.device) -> dict:
        """Load EAGLE model state dictionary"""
        candidates = ["pytorch_model.bin", "model.safetensors"]

        for fname in candidates:
            fpath = os.path.join(eagle_model_path, fname)
            if not os.path.exists(fpath):
                try:
                    fpath = hf_hub_download(eagle_model_path, fname)
                except Exception:
                    continue

            if fname.endswith(".bin"):
                return torch.load(fpath, map_location=device)
            elif fname.endswith(".safetensors"):
                return load_file(fpath)

        raise FileNotFoundError(f"EAGLE model weights not found in {eagle_model_path}")


class PerformanceBenchmark:
    """Handles performance benchmarking for optimal token selection"""

    @staticmethod
    def auto_select_total_token(base_model: nn.Module, vocab_size: int) -> int:
        """Automatically select optimal total token count based on performance"""
        device = next(base_model.parameters()).device
        candidates = [40, 48, 50, 56, 60]
        factors = [1, 1.05, 1.07, 1.1, 1.13]
        times = []

        for length, factor in zip(candidates, factors):
            input_ids = torch.randint(0, vocab_size - 200, (1, length), device=device)
            torch.cuda.synchronize()
            start_time = time.time()

            for _ in range(20):
                torch.cuda.synchronize()
                with torch.no_grad():
                    _ = base_model(input_ids)
                torch.cuda.synchronize()

            end_time = time.time()
            times.append((end_time - start_time) / factor)

        return candidates[times.index(min(times))]


class GenerationManager:
    """Manages the generation process and stopping conditions"""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self._padding_token = None

    def prepare_generation(
        self, model: "Eagle3Model", input_ids: torch.Tensor, config: GenerationConfig
    ) -> GenerationState:
        """Prepare all necessary components for generation"""
        stop_token_id = None
        if config.is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        logits_processor = (
            prepare_logits_processor(
                temperature=config.temperature, top_p=config.top_p, top_k=config.top_k
            )
            if config.temperature > 1e-5
            else None
        )

        input_ids = input_ids.clone()
        model.eagle_layer.reset_kv()

        if hasattr(model, "past_key_values"):
            past_key_values = model.past_key_values
            model.current_length_data.zero_()
        else:
            past_key_values, past_key_values_data, current_length_data = (
                initialize_past_key_values(
                    model.base_model, max_length=config.max_length
                )
            )
            model.past_key_values = past_key_values
            model.past_key_values_data = past_key_values_data
            model.current_length_data = current_length_data

        reset_tree_mode(model)

        return GenerationState(
            stop_token_id=stop_token_id,
            logits_processor=logits_processor,
            input_ids=input_ids,
            past_key_values=past_key_values,
            input_len=input_ids.shape[1],
        )

    def should_stop(
        self,
        input_ids: torch.Tensor,
        input_len: int,
        new_token: int,
        config: GenerationConfig,
        stop_token_id: Optional[int],
    ) -> bool:
        """Check if generation should stop"""
        if stop_token_id is not None:
            if torch.any(input_ids[0, input_len:] == stop_token_id):
                return True

        if torch.any(input_ids[0, input_len:] == self.tokenizer.eos_token_id):
            return True

        if new_token > config.max_new_tokens:
            return True

        if input_ids.shape[1] > config.max_length:
            return True

        return False

    def get_padding_token(self, device: torch.device) -> torch.Tensor:
        """Get or create padding token"""
        if self._padding_token is None or self._padding_token.device != device:
            self._padding_token = (torch.zeros(1, 1, dtype=torch.long) - 1).to(device)
        return self._padding_token


class Eagle3Model(nn.Module):
    """
    EAGLE3 Model for speculative decoding with improved structure and maintainability
    """

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: AutoTokenizer,
        eagle_layer: nn.Module,
        early_stop_method: Optional[str] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.eagle_layer = eagle_layer
        self.generation_manager = GenerationManager(tokenizer)
        self.early_stop_method = early_stop_method

    @classmethod
    def from_pretrained(
        cls,
        base_model_path: Optional[str] = None,
        eagle_model_path: Optional[str] = None,
        total_token: int = 60,
        depth: int = 7,
        top_k: int = 10,
        threshold: float = 1.0,
        enable_benchmark: bool = False,
        early_stop_method: Optional[str] = None,
        stop_think_token: str = "</think>",
        step_split_tokens: Optional[List[str]] = None,
        **kwargs,
    ) -> "Eagle3Model":
        """Create Eagle3Model from pretrained components"""
        # Load base model and tokenizer
        if not step_split_tokens:
            step_split_tokens = [
                "\n\n",
                "\n\n\n",
                ".\n\n",
                ".\n\n\n",
                " \n\n",
                " \n\n\n",
            ]
        base_model = ModelLoader.load_base_model(base_model_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
        tokenizer.stop_think_id = tokenizer.encode(
            stop_think_token, add_special_tokens=False
        )[0]
        tokenizer.step_split_ids = []
        for s in step_split_tokens:
            t = tokenizer.encode(s, add_special_tokens=False)
            if len(t) > 1:
                continue
            tokenizer.step_split_ids.append(t[0])

        # Load configuration
        config_path = ModelLoader.ensure_config_path(eagle_model_path)
        config = Eagle3Config.from_pretrained(config_path)

        # Initialize EAGLE layer
        device = next(base_model.parameters()).device
        eagle_state_dict = ModelLoader.load_eagle_state_dict(eagle_model_path, device)

        # TODO: Implement factory pattern for different drafter types
        eagle_layer = Llama3Eagle3Drafter(
            config,
            total_tokens=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
            path=base_model_path,
            load_emb=True,
            early_stop_method=early_stop_method,
        )

        # Clean up unused components
        if config.vocab_size == config.draft_vocab_size:
            del eagle_layer.d2t
            del eagle_layer.t2d

        eagle_layer.load_state_dict(eagle_state_dict, strict=False)
        eagle_layer.to(device=device, dtype=base_model.dtype)
        eagle_layer.init_tree()

        # Auto-select optimal token count if needed
        if total_token == -1 and enable_benchmark:
            total_token = PerformanceBenchmark.auto_select_total_token(
                base_model, config.vocab_size
            )
            eagle_layer.total_tokens = total_token - 1

        return cls(base_model, tokenizer, eagle_layer, early_stop_method)

    def get_tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer"""
        return self.tokenizer

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        output_orig: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[Any, torch.Tensor], Tuple[Any, torch.Tensor, torch.Tensor]]:
        """Forward pass through the model"""
        with torch.inference_mode():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

            hidden_states = outputs[0]

            if output_orig:
                orig = self.base_model.lm_head(hidden_states)
                return outputs, orig, hidden_states
            else:
                return outputs, hidden_states

    @torch.no_grad()
    def eagle_generate(
        self,
        input_ids: torch.Tensor,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: float = 0.0,
        max_new_tokens: int = 512,
        max_length: int = 2048,
        log: bool = False,
        is_llama3: bool = False,
        early_stop_smooth_type: str = "ewma",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, int, int, List[int]]]:
        """Generate text using EAGLE speculative decoding"""
        config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            log=log,
            is_llama3=is_llama3,
        )

        state = self.generation_manager.prepare_generation(self, input_ids, config)
        padding = self.generation_manager.get_padding_token(input_ids.device)

        # Prefill phase
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, _, _ = (
            initialize_tree(
                input_ids, self, state.past_key_values, state.logits_processor
            )
        )

        accept_length_list = []
        max_decode_steps = config.max_length - self.eagle_layer.total_tokens - 10

        if early_stop_smooth_type == "ewma":
            predict_class = EWMAScorePredictor
            kwargs = {"alpha": 0.1}
        elif early_stop_smooth_type == "momentum":
            predict_class = MomentumScorePredictor
            kwargs = {"window_size": 10}
        else:
            predict_class = MeanScorePredictor
            kwargs = {"window_size": 10}

        predictors = []
        if self.early_stop_method is not None:
            for _ in range(self.early_stop_method.count("_") + 1):
                predictors.append(predict_class(**kwargs))
        is_thinking = True

        for step in range(max_decode_steps):  # noqa: B007
            # Ensure tensors are on correct device
            draft_tokens = draft_tokens.to(input_ids.device)
            tree_position_ids = tree_position_ids.to(input_ids.device)

            self.base_model.model.tree_mask = tree_mask

            # Target model forward pass
            logits, hidden_state_new, _ = tree_decoding(
                self,
                draft_tokens,
                state.past_key_values,
                tree_position_ids,
                state.input_ids,
                retrieve_indices,
            )

            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]

            # Verification phase
            best_candidate, accept_length, sample_token = evaluate_posterior(
                logits, candidates, state.logits_processor
            )

            new_token_ids = (
                candidates[None, best_candidate, : accept_length + 1].view(-1).tolist()
            )
            if is_thinking and self.tokenizer.stop_think_id in new_token_ids:
                is_thinking = False
            if is_thinking and self.early_stop_method:
                can_stop = False
                for split_pos in range(len(new_token_ids)):
                    if new_token_ids[split_pos] in self.tokenizer.step_split_ids:
                        can_stop = True
                        break
                if can_stop:
                    scores = []
                    for p, m in zip(predictors, self.early_stop_method.split("_")):
                        score = p.predict_next_score()
                        scores.append(score)
                        if m == "confidence":
                            can_stop = can_stop and score and score > 0.8
                        elif m == "progress":
                            can_stop = can_stop and score and score > 0.3
                        elif m == "remain":
                            can_stop = can_stop and score and score < 200
                    if early_stop_smooth_type == "paragraph_mean":
                        for p in predictors:
                            p.clear_before()
                if can_stop:
                    accept_length = split_pos
                    sample_token[..., -1] = self.tokenizer.stop_think_id
                    is_thinking = False
                    print(f"Early Stop: scores={scores}")

            accept_length_list.append(
                accept_length.item()
                if torch.is_tensor(accept_length)
                else accept_length
            )

            # Update inference inputs
            (
                state.input_ids,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                state.new_token,
                early_stop_signal,
            ) = update_inference_inputs(
                input_ids=state.input_ids,
                candidates=candidates,
                best_candidate=best_candidate,
                accept_length=accept_length,
                retrieve_indices=retrieve_indices,
                logits_processor=state.logits_processor,
                new_token=state.new_token,
                past_key_values_data_list=self.past_key_values_data,
                current_length_data=self.current_length_data,
                model=self,
                hidden_state_new=hidden_state_new,
                sample_token=sample_token,
            )
            if is_thinking and early_stop_signal is not None:
                early_stop_signal_cpu = early_stop_signal.tolist()
                for p, s in zip(predictors, early_stop_signal_cpu):
                    p.add_score(s)

            if self.generation_manager.should_stop(
                state.input_ids,
                state.input_len,
                state.new_token,
                config,
                state.stop_token_id,
            ):
                break

        return (
            (state.input_ids, state.new_token, step, accept_length_list)
            if log
            else state.input_ids
        )

    @torch.no_grad()
    def naive_generate(
        self,
        input_ids: torch.Tensor,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: float = 0.0,
        max_new_tokens: int = 512,
        max_length: int = 2048,
        log: bool = False,
        is_llama3: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, int, int]]:
        """Generate text using naive (non-speculative) decoding"""
        config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            log=log,
            is_llama3=is_llama3,
        )

        state = self.generation_manager.prepare_generation(self, input_ids, config)

        outputs = self.base_model(
            state.input_ids, past_key_values=state.past_key_values, use_cache=True
        )

        max_decode_steps = config.max_length - self.eagle_layer.total_tokens - 10

        for step in range(max_decode_steps):  # noqa: B007
            if state.logits_processor is not None:
                logits = state.logits_processor(None, outputs.logits[:, -1])
                probs = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probs, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(
                input_id, use_cache=True, past_key_values=state.past_key_values
            )
            state.input_ids = torch.cat([state.input_ids, input_id], dim=-1)
            state.new_token += 1

            if self.generation_manager.should_stop(
                state.input_ids,
                state.input_len,
                state.new_token,
                config,
                state.stop_token_id,
            ):
                break

        return (state.input_ids, state.new_token, step) if log else state.input_ids
