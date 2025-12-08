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
from abc import ABC, abstractmethod
from collections import Counter

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import PreTrainedModel

from ...data.data_utils import process_token_dict_to_mappings
from ..model_utils import expand_mask, make_causal_mask


class Eagle3BaseDraftModel(PreTrainedModel, ABC):
    @abstractmethod
    def compute_logits(self, input_ids, attention_mask):
        pass

    @abstractmethod
    def encode_layers(self, input_ids, attention_mask):
        pass

    @abstractmethod
    def combine_hidden_states(self, hidden_states):
        pass

    @abstractmethod
    def get_input_embeddings(self, input_ids):
        pass

    def freeze_embed_weights(self):
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def load_embed_weights(self, target_model_name_or_path, embed_weight_key):
        """
        Load embedding weights from pretrained model.

        Args:
            target_model_name_or_path: Local path or
            HuggingFace model identifier (e.g., 'Qwen/Qwen2-7B')
            embed_weight_key: Key for the embedding weights in the model file
        """
        # Handle HuggingFace model identifier
        if not os.path.exists(target_model_name_or_path):
            # print(f"Downloading model from HuggingFace: {target_model_name_or_path}")
            target_model_name_or_path = snapshot_download(
                repo_id=target_model_name_or_path
            )

        # Try loading embedding weights
        tensor = self._load_from_safetensors(
            target_model_name_or_path, embed_weight_key
        )
        if tensor is None:
            tensor = self._load_from_pytorch_bin(
                target_model_name_or_path, embed_weight_key
            )

        if tensor is None:
            raise FileNotFoundError(
                f"Could not find embedding weights in {target_model_name_or_path}. "
                "Expected either safetensors or pytorch_model.bin format."
            )

        with torch.no_grad():
            self.embed_tokens.weight.copy_(tensor)

    def _load_from_safetensors(
        self, model_path, embed_weight_key="model.embed_tokens.weight"
    ):
        """Load embedding weights from safetensors format."""
        try:
            index_file = os.path.join(model_path, "model.safetensors.index.json")
            if not os.path.exists(index_file):
                return None

            with open(index_file, "r") as f:
                index_json = json.load(f)

            if embed_weight_key in index_json["weight_map"]:
                emb_path = index_json["weight_map"][embed_weight_key]
            else:
                raise KeyError("Embedding weights key not found in index.")

            safetensors_file = os.path.join(model_path, emb_path)

            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                tensor_slice = f.get_slice(embed_weight_key)
                _, hidden_dim = tensor_slice.get_shape()
                tensor = tensor_slice[:, :hidden_dim].float()

            return tensor
        except Exception as e:
            print(f"Failed to load from safetensors: {e}")
            return None

    def _load_from_pytorch_bin(
        self, model_path, embed_weight_key="model.embed_tokens.weight"
    ):
        """Load embedding weights from pytorch_model.bin format."""
        try:
            index_file = os.path.join(model_path, "pytorch_model.bin.index.json")
            if not os.path.exists(index_file):
                return None

            with open(index_file, "r") as f:
                index_json = json.load(f)

            if embed_weight_key in index_json["weight_map"]:
                emb_path = index_json["weight_map"][embed_weight_key]
            else:
                raise KeyError("Embedding weights key not found in index.")

            bin_file = os.path.join(model_path, emb_path)

            weights = torch.load(bin_file, map_location="cpu")
            tensor = weights[embed_weight_key].float()

            return tensor
        except Exception as e:
            print(f"Failed to load from pytorch_model.bin: {e}")
            return None

    def build_vocab_mapping(self, dataset, cache_path):
        """
        Build vocab mapping from full vocabulary to draft vocabulary
        based on token frequency.

        Args:
            dataset: Preprocessed dataset containing 'input_ids' field
            cache_path: Path to save/load the token mapping cache
            num_processes: Number of processes for parallel processing
        """
        if not os.path.exists(cache_path):
            # we first count the frequency of effective tokens in the dataset
            token_dict = Counter()
            print(f"vocab len(dataset)={len(dataset)} type(dataset)={type(dataset)}")
            # for item in tqdm(dataset, desc=f"Counting tokens for vocab mapping"):

            for _, item in enumerate(dataset):
                input_ids = item["input_ids"]
                loss_mask = item["loss_mask"]
                masked_ids = input_ids[loss_mask == 1]
                unique_ids, counts = masked_ids.unique(return_counts=True)
                batch_token_dict = dict(zip(unique_ids.tolist(), counts.tolist()))
                token_dict.update(batch_token_dict)

            # generate the d2t and t2d mapping
            d2t, t2d = process_token_dict_to_mappings(
                token_dict,
                self.draft_vocab_size,
                self.vocab_size,
            )

            vocab_mapping = {
                "d2t": d2t,
                "t2d": t2d,
            }

            cache_parent_dir = os.path.dirname(cache_path)
            os.makedirs(cache_parent_dir, exist_ok=True)
            torch.save(vocab_mapping, cache_path)
            print(f"Saved vocab mapping to: {cache_path}")
        else:
            # Load from cache
            cache = torch.load(cache_path)
            d2t = cache["d2t"]
            t2d = cache["t2d"]

        self.t2d.copy_(t2d)
        self.d2t.copy_(d2t)
