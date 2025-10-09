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
import math
import os
from abc import ABC
from typing import Any, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import Tensor, nn


class BaseEagle3Drafter(nn.Module, ABC):
    """Base class for Eagle3 drafter model with speculative decoding capability."""

    def __init__(
        self,
        config: Any,
        load_emb: bool = False,
        path: Optional[str] = None,
        total_tokens: int = 63,
        depth: int = 5,
        top_k: int = 8,
        threshold: float = 1.0,
        early_stop_method: Optional[str] = None,
    ):
        """
        Initialize the drafter model.

        Args:
            config: Model configuration object
            load_emb: Whether to load embeddings from path
            path: Path to load embeddings from
            total_tokens: Total tokens to generate in speculative decoding
            depth: Depth of the speculative decoding tree
            top_k: Number of top candidates to consider at each step
            threshold: Probability threshold for speculative decoding
        """
        super().__init__()
        self.config = config

        # Model architecture components
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.early_stop_method = early_stop_method

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )

        # Handle different hidden sizes between target and draft models
        hidden_size_multiplier = 3
        fc_output_size = config.hidden_size
        if early_stop_method is not None:
            fc_output_size += early_stop_method.count("_") + 1
        if hasattr(config, "target_hidden_size"):
            self.fc = nn.Linear(
                config.target_hidden_size * hidden_size_multiplier,
                fc_output_size,
                bias=False,
            )
        else:
            self.fc = nn.Linear(
                config.hidden_size * hidden_size_multiplier,
                fc_output_size,
                bias=False,
            )

        self.logsoftmax = nn.LogSoftmax(dim=-1)

        # Vocabulary mapping buffers
        self.register_buffer(
            "d2t", torch.zeros(config.draft_vocab_size, dtype=torch.long)
        )
        self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.bool))

        # Speculative decoding parameters
        self.top_k = top_k
        self.total_tokens = total_tokens - 1  # Exclude the initial token
        self.depth = depth
        self.threshold = math.log(threshold)

        # Load embeddings if requested
        if load_emb and not hasattr(config, "target_hidden_size"):
            self.load_embed(path)

        # Freeze embedding parameters
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def load_embed(self, path: str) -> None:
        """Load token embeddings from either safetensors or pytorch format."""
        try:
            # First try safetensors format
            self._load_safetensors_embed(path)
        except Exception:
            # Fall back to pytorch format
            self._load_pytorch_embed(path)

    def _load_safetensors_embed(self, path: str) -> None:
        """Load embeddings from safetensors format."""
        index_json_path = os.path.join(path, "model.safetensors.index.json")
        if not os.path.exists(index_json_path):
            index_json_path = hf_hub_download(path, "model.safetensors.index.json")

        with open(index_json_path, "r") as f:
            index_json = json.loads(f.read())
            emb_path = index_json["weight_map"]["model.embed_tokens.weight"]

        local_emb_path = os.path.join(path, emb_path)
        if not os.path.exists(local_emb_path):
            local_emb_path = hf_hub_download(path, emb_path)

        with safe_open(local_emb_path, framework="pt", device="cpu") as f:
            tensor_slice = f.get_slice("model.embed_tokens.weight")
            _, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()

        self.embed_tokens.weight.data = tensor

    def _load_pytorch_embed(self, path: str) -> None:
        """Load embeddings from pytorch format."""
        index_json_path = os.path.join(path, "pytorch_model.bin.index.json")
        if not os.path.exists(index_json_path):
            index_json_path = hf_hub_download(path, "pytorch_model.bin.index.json")

        with open(index_json_path, "r") as f:
            index_json = json.loads(f.read())
            emb_path = index_json["weight_map"]["model.embed_tokens.weight"]

        local_emb_path = os.path.join(path, emb_path)
        if not os.path.exists(local_emb_path):
            local_emb_path = hf_hub_download(path, emb_path)

        weights = torch.load(local_emb_path)
        tensor = weights["model.embed_tokens.weight"].float()
        self.embed_tokens.weight.data = tensor

    def init_tree(self) -> None:
        """Initialize tree structures for speculative decoding."""
        self.tree_mask_init = torch.eye(
            self.top_k, device=self.embed_tokens.weight.device
        )[None, None]
        self.position_ids = torch.zeros(
            self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long
        )
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)

    def reset(self) -> None:
        """Reset tree mask for new sequence generation."""
        self.tree_mask = None

    def reset_kv(self) -> None:
        """Reset key-value cache for new sequence generation."""
        self.stable_kv = None

    @torch.no_grad()
    def topK_genrate(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        logits_processor: Optional[Any] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Generate tokens using top-K speculative decoding.

        Args:
            hidden_states: Hidden states from the target model
            input_ids: Input token IDs
            logits_processor: Optional logits processor

        Returns:
            Tuple containing:
            - draft_tokens: Generated draft tokens
            - retrieve_indices: Indices for retrieving tokens from tree
            - tree_mask: Mask for tree attention
            - tree_position_ids: Position IDs in the tree
        """
        # Initialize data structures
        scores_list = []
        parents_list = []
        ss_token = []

        # Prepare input and initialize tree
        input_ids = input_ids.to(hidden_states.device)
        sample_token = input_ids[:, -1]
        input_ids = input_ids[:, 1:]
        self.initial_position_id = input_ids.shape[1]

        self.reset()

        # Generate initial hidden states and tokens
        last_hidden, past_key_values, early_stop_signal = self._get_initial_hidden(
            hidden_states, input_ids
        )
        self.stable_kv = past_key_values

        # Generate first level of tokens
        topk_index, scores = self._get_topk_tokens(last_hidden)
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))

        # Handle vocabulary mapping if needed
        if self.config.vocab_size == self.config.draft_vocab_size:
            ss_token.append(topk_index)
            input_ids = topk_index
        else:
            mapped_tokens = topk_index + self.d2t[topk_index]
            ss_token.append(mapped_tokens)
            input_ids = mapped_tokens

        # Prepare for tree traversal
        input_hidden = last_hidden[None].repeat(1, self.top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(self.top_k, device=self.embed_tokens.weight.device)

        # Traverse the tree depth levels
        for i in range(self.depth):
            (
                tree_mask,
                input_hidden,
                input_ids,
                scores,
                topk_cs_index,
                past_key_values,
            ) = self._process_tree_level(
                i,
                tree_mask,
                input_hidden,
                input_ids,
                scores,
                topk_cs_index,
                scores_list,
                parents_list,
                ss_token,
                past_key_values,
            )
        # Process the final results
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = (
            self._finalize_results(
                scores_list, ss_token, sample_token, parents_list, logits_processor
            )
        )

        # Delete some used lists and variables to free memory
        del scores_list, parents_list, ss_token

        return (
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            early_stop_signal,
        )

    def _get_initial_hidden(
        self, hidden_states: Tensor, input_ids: Tensor
    ) -> Tuple[Tensor, Any]:
        """Get initial hidden states and past key values."""
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            outputs = self(
                hidden_states,
                input_ids=input_ids[:, kv_len:],
                past_key_values=self.stable_kv,
                use_cache=True,
            )
        else:
            outputs = self(hidden_states, input_ids=input_ids, use_cache=True)
        if self.early_stop_method is not None:
            out_hidden, past_key_values, early_stop_signal = outputs
        else:
            out_hidden, past_key_values = outputs
            early_stop_signal = None
        return out_hidden[:, -1], past_key_values, early_stop_signal

    def _get_topk_tokens(self, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        """Get top-k tokens from hidden states."""
        logits = self.lm_head(self.norm(hidden))
        probs = self.logsoftmax(logits)
        topk = torch.topk(probs, self.top_k, dim=-1)
        return topk.indices, topk.values[0]

    def _process_tree_level(
        self,
        level: int,
        tree_mask: Tensor,
        input_hidden: Tensor,
        input_ids: Tensor,
        scores: Tensor,
        topk_cs_index: Tensor,
        scores_list: list,
        parents_list: list,
        ss_token: list,
        past_key_values,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Process one level of the speculative decoding tree."""
        self.tree_mask = tree_mask
        position_ids = self.position_ids + self.initial_position_id

        # Get next level hidden states
        out_hidden, past_key_values, early_stop_signal = self(
            input_hidden,
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True,
        )
        self.initial_position_id += 1

        # Calculate parent indices
        bias1 = self.top_k if level > 0 else 0
        bias2 = max(0, level - 1)
        bias = 1 + self.top_k**2 * bias2 + bias1
        parents = topk_cs_index + bias
        parents_list.append(parents)

        # Get top-k tokens for this level
        topk_index, topk_p = self._get_topk_tokens(out_hidden[0])
        cu_scores = topk_p + scores[:, None]

        # Select best candidates
        topk_cs = torch.topk(cu_scores.view(-1), self.top_k, dim=-1)
        topk_cs_index, scores = topk_cs.indices, topk_cs.values

        # Update data structures
        out_ids = topk_cs_index // self.top_k
        input_hidden = out_hidden[:, out_ids]
        input_ids = topk_index.view(-1)[topk_cs_index][None]

        # Handle vocabulary mapping if needed
        if self.config.vocab_size == self.config.draft_vocab_size:
            ss_token.append(topk_index)
        else:
            input_ids = input_ids + self.d2t[input_ids]
            ss_token.append(topk_index + self.d2t[topk_index])

        scores_list.append(cu_scores)
        tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

        return (
            tree_mask,
            input_hidden,
            input_ids,
            scores,
            topk_cs_index,
            past_key_values,
        )

    def _finalize_results(
        self,
        scores_list: list,
        ss_token: list,
        sample_token: Tensor,
        parents_list: list,
        logits_processor: Optional[Any],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Finalize the speculative decoding results."""
        # Combine all scores and tokens
        all_scores = torch.cat(scores_list, dim=0).view(-1)
        all_tokens = torch.cat(ss_token, dim=0).view(-1)

        # Select top tokens
        top_scores = torch.topk(all_scores, self.total_tokens, dim=-1)
        top_indices = torch.sort(top_scores.indices).values
        draft_tokens = all_tokens[top_indices]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        # Build tree structure
        tree_mask, tree_position_ids = self._build_tree_mask(top_indices, parents_list)

        # Generate retrieval indices
        retrieve_indices = self._generate_retrieve_indices(
            tree_position_ids, top_indices, parents_list
        )

        # Apply logits processor if provided
        if logits_processor is not None:
            retrieve_indices = self._apply_logits_processor(retrieve_indices)

        return draft_tokens[None], retrieve_indices, tree_mask, tree_position_ids

    def _build_tree_mask(
        self, top_indices: Tensor, parents_list: list
    ) -> Tuple[Tensor, Tensor]:
        """Build the tree attention mask and position IDs."""
        all_parents = torch.cat(parents_list, dim=0)[top_indices // self.top_k].long()

        # Find parent-child relationships
        mask_index = torch.searchsorted(top_indices, all_parents - 1, right=False)
        mask_index[all_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()

        # Build tree mask
        tree_mask = torch.eye(self.total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(self.total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        # Calculate position IDs
        tree_position_ids = torch.sum(tree_mask, dim=1) - 1
        tree_mask = tree_mask.float()[None, None]
        return tree_mask, tree_position_ids

    def _generate_retrieve_indices(
        self, tree_position_ids: Tensor, top_indices: Tensor, parents_list: list
    ) -> Tensor:
        """Generate indices for retrieving tokens from the tree."""
        all_parents = torch.cat(parents_list, dim=0)[top_indices // self.top_k].long()
        mask_index = torch.searchsorted(top_indices, all_parents - 1, right=False)
        mask_index[all_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()

        # Identify leaf nodes
        noleaf_index = torch.unique(mask_index).tolist()
        leaf_num = self.total_tokens - (len(noleaf_index) - 1)

        # Build retrieval paths for leaves
        retrieve_indices = (
            torch.zeros(
                leaf_num, torch.max(tree_position_ids).item() + 1, dtype=torch.long
            )
            - 1
        )
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(self.total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        return torch.tensor(retrieve_indices, dtype=torch.long)

    def _apply_logits_processor(self, retrieve_indices: Tensor) -> Tensor:
        """Apply logits processor to retrieval indices if provided."""
        maxitem = self.total_tokens + 5

        def custom_sort(lst):
            return [x if x >= 0 else maxitem for x in lst]

        return torch.tensor(
            sorted(retrieve_indices.tolist(), key=custom_sort), dtype=torch.long
        )
