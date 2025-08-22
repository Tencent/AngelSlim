import os
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from ....utils import (
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


class Eagle3Model(nn.Module):
    def __init__(self, base_model, tokenizer, eagle_layer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.eagle_layer = eagle_layer

    @classmethod
    def from_pretrained(
        cls,
        base_model_path=None,
        eagle_model_path=None,
        total_token=60,
        depth=7,
        top_k=10,
        threshold=1.0,
        enable_benchmark=False,
        **kwargs,
    ):
        # Load base model
        base_model = cls._load_base_model(base_model_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

        # Load config
        config_path = cls._ensure_config_path(eagle_model_path)
        config = Eagle3Config.from_pretrained(config_path)

        # Initilize eagle_layer
        device = next(base_model.parameters()).device
        eagle_state_dict = cls._load_eagle_state_dict(eagle_model_path, device)

        # TODO: load different types of drafters from factories
        eagle_layer = Llama3Eagle3Drafter(
            config,
            total_tokens=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
            path=base_model_path,
            load_emb=True,
        )

        # Delete d2t/t2d if not used
        if config.vocab_size == config.draft_vocab_size:
            del eagle_layer.d2t
            del eagle_layer.t2d

        eagle_layer.load_state_dict(eagle_state_dict, strict=False)
        eagle_layer.to(device=device, dtype=base_model.dtype)
        eagle_layer.init_tree()

        # Auto select total_token if needed
        if total_token == -1 and enable_benchmark:
            total_token = cls._auto_select_total_token(base_model, config.vocab_size)
            eagle_layer.total_tokens = total_token - 1

        return cls(base_model, tokenizer, eagle_layer)

    def get_tokenizer(self):
        return self.tokenizer

    @staticmethod
    def _load_base_model(base_model_path, **kwargs):
        config = AutoConfig.from_pretrained(base_model_path)
        if not getattr(config, "architectures", None):
            raise ValueError("Base model config missing 'architectures' field")

        arch = config.architectures[0]
        if arch == "LlamaForCausalLM":
            return KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
        elif arch == "Qwen3ForCausalLM":
            return KVQwen3ForCausalLM.from_pretrained(base_model_path, **kwargs)
        else:
            raise NotImplementedError(f"Model {arch} not supported")

    @staticmethod
    def _ensure_config_path(eagle_model_path):
        config_path = os.path.join(eagle_model_path, "config.json")
        if not os.path.exists(config_path):
            config_path = hf_hub_download(eagle_model_path, "config.json")
        return config_path

    @staticmethod
    def _load_eagle_state_dict(eagle_model_path, device):
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

    @staticmethod
    def _auto_select_total_token(base_model, vocab_size):
        device = next(base_model.parameters()).device
        cans = [40, 48, 50, 56, 60]
        factors = [1, 1.05, 1.07, 1.1, 1.13]
        times = []

        for length, factor in zip(cans, factors):
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

        return cans[times.index(min(times))]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
    ):
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def _prepare_generation(
        self, input_ids, temperature, top_p, top_k, max_length, is_llama3
    ):
        stop_token_id = None
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        logits_processor = (
            prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
            if temperature > 1e-5
            else None
        )

        input_ids = input_ids.clone()
        self.eagle_layer.reset_kv()

        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            past_key_values, past_key_values_data, current_length_data = (
                initialize_past_key_values(self.base_model, max_length=max_length)
            )
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        reset_tree_mode(self)
        return stop_token_id, logits_processor, input_ids, past_key_values

    def _should_stop(
        self, input_ids, input_len, new_token, max_new_tokens, max_length, stop_token_id
    ):
        if stop_token_id is not None:
            if torch.any(input_ids[0, input_len:] == stop_token_id):
                return True
        if torch.any(input_ids[0, input_len:] == self.tokenizer.eos_token_id):
            return True
        if new_token > max_new_tokens:
            return True
        if input_ids.shape[1] > max_length:
            return True
        return False

    @torch.no_grad()
    def eagle_generate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
        is_llama3=False,
    ):
        stop_token_id, logits_processor, input_ids, past_key_values = (
            self._prepare_generation(
                input_ids, temperature, top_p, top_k, max_length, is_llama3
            )
        )

        padding = getattr(self, "_padding_token", None)
        if padding is None or padding.device != input_ids.device:
            padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
            self._padding_token = padding

        input_len = input_ids.shape[1]
        # prefill
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, _, _ = (
            initialize_tree(input_ids, self, past_key_values, logits_processor)
        )

        new_token = 0
        accept_length_list = []
        max_decode_steps = max_length - self.eagle_layer.total_tokens - 10

        for step in range(max_decode_steps):  # noqa: B007
            self.base_model.model.tree_mask = tree_mask

            if draft_tokens.device != input_ids.device:
                draft_tokens = draft_tokens.to(input_ids.device)

            if tree_position_ids.device != input_ids.device:
                tree_position_ids = tree_position_ids.to(input_ids.device)

            # Target model forward, get logits
            logits, hidden_state_new, _ = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )

            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]

            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            if torch.is_tensor(accept_length):
                accept_length_list.append(accept_length.item())
            else:
                accept_length_list.append(accept_length)

            # Adjusting the input sequence, draft model forward
            (
                input_ids,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                new_token,
                _,
                _,
            ) = update_inference_inputs(
                input_ids=input_ids,
                candidates=candidates,
                best_candidate=best_candidate,
                accept_length=accept_length,
                retrieve_indices=retrieve_indices,
                logits_processor=logits_processor,
                new_token=new_token,
                past_key_values_data_list=self.past_key_values_data,
                current_length_data=self.current_length_data,
                model=self,
                hidden_state_new=hidden_state_new,
                sample_p=sample_p,
            )

            if self._should_stop(
                input_ids,
                input_len,
                new_token,
                max_new_tokens,
                max_length,
                stop_token_id,
            ):
                break

        return (input_ids, new_token, step, accept_length_list) if log else input_ids

    @torch.no_grad()
    def naive_generate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
        is_llama3=False,
    ):
        stop_token_id, logits_processor, input_ids, past_key_values = (
            self._prepare_generation(
                input_ids, temperature, top_p, top_k, max_length, is_llama3
            )
        )

        input_len = input_ids.shape[1]
        outputs = self.base_model(
            input_ids, past_key_values=past_key_values, use_cache=True
        )

        new_token = 0
        max_decode_steps = max_length - self.eagle_layer.total_tokens - 10

        for step in range(max_decode_steps):  # noqa: B007
            if logits_processor is not None:
                logits = logits_processor(None, outputs.logits[:, -1])
                probs = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probs, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(
                input_id, use_cache=True, past_key_values=past_key_values
            )
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if self._should_stop(
                input_ids,
                input_len,
                new_token,
                max_new_tokens,
                max_length,
                stop_token_id,
            ):
                break

        return (input_ids, new_token, step) if log else input_ids
