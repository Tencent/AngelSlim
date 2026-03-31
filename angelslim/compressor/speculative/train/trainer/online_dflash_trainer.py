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

"""Online DFlash Trainer for speculative decoding training.

DFlash uses block-parallel cross-attention rather than Eagle3's
iterative autoregressive approach, so it overrides compute_loss
with its own block-wise CE loss logic.
"""

import gc
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig

from .eagle3_trainer import Eagle3Trainer
from .trainer_factory import Eagle3TrainerFactory

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
):
    """Construct Flex Attention BlockMask for DFlash training.

    KV: [Context (S tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos).
      2. Intra-block attention is bidirectional.
      3. Different blocks are invisible to each other.
      4. Invalid blocks (block_keep_mask=False) see nothing.
    """

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[b, q_block_id]

        is_context = kv_idx < S
        # Strictly less than: matches inference where target_hidden[anchor_pos]
        # is not available as context.
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    return create_block_mask(
        dflash_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


class TargetEmbeddingsAndHead(nn.Module):
    """Efficiently loads only the embedding layer and lm_head from a pretrained model.

    Handles safetensors slicing and Weight Tying correctly.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        embed_key: Optional[str] = None,
        lm_head_key: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
    ) -> "TargetEmbeddingsAndHead":

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        instance = cls(config)

        if embed_key is None:
            embed_key = "model.embed_tokens.weight"
        if lm_head_key is None:
            lm_head_key = "lm_head.weight"

        tie_weights = getattr(config, "tie_word_embeddings", False)
        instance._load_weights(model_path, embed_key, lm_head_key, tie_weights)

        instance.to(device=device, dtype=dtype)
        instance.eval()
        instance.requires_grad_(False)

        return instance

    def _load_weights(self, model_path: str, embed_key: str, lm_head_key: str, tie_weights: bool):
        index_files = glob.glob(os.path.join(model_path, "*.index.json"))
        files_to_load = {}

        if index_files:
            with open(index_files[0], "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})

            if embed_key in weight_map:
                files_to_load[embed_key] = weight_map[embed_key]
            else:
                raise ValueError(f"Embedding key '{embed_key}' not found in weight map.")

            if not tie_weights:
                if lm_head_key in weight_map:
                    files_to_load[lm_head_key] = weight_map[lm_head_key]
        else:
            safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
            bins = glob.glob(os.path.join(model_path, "*.bin"))
            target_file = safetensors[0] if safetensors else (bins[0] if bins else None)

            if not target_file:
                raise FileNotFoundError("No checkpoint found.")

            files_to_load[embed_key] = os.path.basename(target_file)
            if not tie_weights:
                files_to_load[lm_head_key] = os.path.basename(target_file)

        file_to_keys_map = {}
        for key, filename in files_to_load.items():
            full_path = os.path.join(model_path, filename)
            if full_path not in file_to_keys_map:
                file_to_keys_map[full_path] = []
            file_to_keys_map[full_path].append(key)

        for file_path, keys in file_to_keys_map.items():
            self._load_file_content(file_path, keys, embed_key, lm_head_key)

        if tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

    def _load_file_content(
        self,
        file_path: str,
        keys_to_extract: list,
        target_embed_key: str,
        target_head_key: str,
    ):
        state_dict_part = {}

        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                for k in keys_to_extract:
                    if k in f.keys():
                        state_dict_part[k] = f.get_tensor(k)
        else:
            full_state = torch.load(file_path, map_location="cpu")
            for k in keys_to_extract:
                if k in full_state:
                    state_dict_part[k] = full_state[k]
            del full_state
            gc.collect()

        for k, tensor in state_dict_part.items():
            if k == target_embed_key:
                self.embed_tokens.weight.data.copy_(tensor)
            elif k == target_head_key:
                if tensor.shape == self.lm_head.weight.data.shape:
                    self.lm_head.weight.data.copy_(tensor)


@Eagle3TrainerFactory.register("online", "DFlash")
class OnlineDFlashTrainer(Eagle3Trainer):
    """Online DFlash Trainer for speculative decoding training.

    Uses block-parallel cross-attention and anchor-based CE loss
    rather than Eagle3's iterative autoregressive training loop.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        length: int,
        draft_model_config: Dict[str, Any],
        **kwargs,
    ):
        """
        Initialize the OnlineDFlashTrainer.

        Args:
            draft_model: DFlash draft model
            target_model: Target model for generating hidden states
            length: Not used for DFlash (kept for interface compatibility)
            draft_model_config: Configuration dictionary for draft model,
                must contain dflash-specific fields
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(draft_model=draft_model, length=length, **kwargs)
        self.target_model = target_model
        self._aux_hidden_states_layer_ids = getattr(
            draft_model_config, "aux_hidden_states_layer_ids", None
        )

        # Extract DFlash-specific config
        dflash_config = getattr(draft_model_config, "dflash_config", {}) or {}
        self.block_size = getattr(draft_model_config, "block_size", 16)
        self.num_anchors = getattr(draft_model_config, "num_anchors", 512)
        self.loss_decay_gamma = getattr(draft_model_config, "loss_decay_gamma", None)
        self.attention_backend = getattr(draft_model_config, "attention_backend", "flex_attention")
        self.mask_token_id = dflash_config.get(
            "mask_token_id",
            getattr(draft_model_config, "mask_token_id", None),
        )

        # Load target model's lm_head and embed_tokens
        # In offline mode target_model may be None; fall back to config path.
        target_model_path = None
        if target_model is not None:
            target_model_path = getattr(target_model, "model_path", None)
        if target_model_path is None:
            target_model_path = getattr(draft_model_config, "target_model_name_or_path", None)
        embed_weight_key = getattr(
            draft_model_config, "embed_weight_key", "model.embed_tokens.weight"
        )
        lm_head_key = getattr(draft_model_config, "lm_head_key", "lm_head.weight")
        trust_remote_code = getattr(draft_model_config, "trust_remote_code", True)

        if target_model_path is not None:
            target_components = TargetEmbeddingsAndHead.from_pretrained(
                target_model_path,
                embed_key=embed_weight_key,
                lm_head_key=lm_head_key,
                device="cuda",
                trust_remote_code=trust_remote_code,
            )
            self.target_lm_head = target_components.lm_head
            self.target_embed_tokens = target_components.embed_tokens
        else:
            raise ValueError(
                "target_model_name_or_path must be set in draft_model_config "
                "or target_model.model_path for DFlash training."
            )

    def prepare_data_for_draft_model(self, inputs):
        """Prepare data for DFlash training.

        Extracts hidden states from the target model. DFlash needs
        multi-layer hidden states concatenated as context features.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        loss_mask = inputs["loss_mask"]

        # Get hidden states from target model
        hidden_states, _ = self.target_model.get_hidden_states_and_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aux_hidden_states_layer_ids=self._aux_hidden_states_layer_ids,
        )

        return {
            "input_ids": input_ids,
            "hidden_states": hidden_states,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
        }

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask).

        Returns (None, None) when the batch has no valid anchors (too-short or
        loss_mask-empty sequences), which is handled gracefully in forward().
        """
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_valid = int(valid_counts.max().item())

        # Need at least 2 valid positions (anchor + at least one prediction target)
        if max_valid <= 1:
            return None, None

        max_n = min(self.num_anchors, max_valid - 1)

        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(valid, indices, torch.tensor(seq_len + 1, device=device))

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(
            1
        ).clamp(max=max_n)
        anchors = torch.where(keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device))

        return anchors, keep_mask

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full((bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device)

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.target_embed_tokens(noise_ids)

    def _compute_dflash_loss_and_accuracy(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        """Core DFlash block-parallel loss logic (shared by train + eval).

        Steps:
          1. Sample anchor positions from valid loss_mask positions.
          2. Build noise embedding (anchor token is real, rest are MASK).
          3. Build DFlash BlockMask (context-causal + intra-block bidirectional).
          4. Run draft model forward → logits.
          5. Compute weighted CE loss with optional exponential decay.
          6. Compute accuracy (no-decay mask).

        Returns:
            (loss, accuracy) — both scalar tensors.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # ── 1. Anchor sampling ────────────────────────────────────────────────
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )

        # No valid anchors → return zero loss connected to model params (DDP-safe)
        if anchor_positions is None:
            zero_loss = sum(p.sum() * 0.0 for p in model.parameters() if p.requires_grad)
            return zero_loss, torch.tensor(0.0, device=device)

        # ── 2. Noise embedding ────────────────────────────────────────────────
        noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)

        # ── 3. Position IDs  [B, S + N*block_size] ───────────────────────────
        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        # ── 4. Attention mask (DFlash BlockMask) ─────────────────────────────
        dflash_attn_mask = create_dflash_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            S=seq_len,
            block_size=self.block_size,
            device=device,
        )

        # ── 5. Draft model forward → logits  [B, N*bs, vocab] ────────────────
        model_dtype = next(model.parameters()).dtype
        noise_embedding = noise_embedding.to(model_dtype)
        hidden_states = hidden_states.to(model_dtype)

        output_hidden = model(
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
            position_ids=full_position_ids,
        )

        output_hidden = output_hidden.to(self.target_lm_head.weight.dtype)
        logits = self.target_lm_head(output_hidden)

        # ── 6. Labels: position k in block predicts token at (anchor + k) ────
        bs = self.block_size
        label_offsets = torch.arange(0, bs, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            dim=2,
            index=safe_label_indices,
        )  # [B, N, bs]

        # ── 7. Weight mask: valid block × in-bounds × skip anchor × loss_mask ─
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, bs).float()
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(bs, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()  # skip pos 0 (anchor)

        gathered_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            dim=2,
            index=safe_label_indices,
        )
        weight_mask = weight_mask * gathered_loss_mask

        binary_eval_mask = weight_mask.view(-1)  # no decay, used for accuracy

        # ── 8. Exponential decay: exp(-(k-1)/γ), k=1 gets weight 1.0 ─────────
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(bs, device=device).view(1, 1, -1)
            decay = torch.exp(-(k - 1).clamp(min=0).float() / self.loss_decay_gamma)
            weight_mask = weight_mask * decay

        # ── 9. Cross-entropy loss ─────────────────────────────────────────────
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        loss = (loss_per_token * flat_weights).sum() / (flat_weights.sum() + 1e-6)

        # ── 10. Accuracy (no gradient) ────────────────────────────────────────
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            accuracy = correct.sum().float() / (binary_eval_mask.sum() + 1e-6)

        return loss, accuracy

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[int] = None,
        return_outputs: bool = False,
    ) -> torch.Tensor:
        """Compute the DFlash training loss.

        Unlike Eagle3's iterative multi-step loss, DFlash computes a single
        block-parallel cross-entropy loss over all sampled anchor positions.
        """
        data = self.prepare_data_for_draft_model(inputs)

        loss, accuracy = self._compute_dflash_loss_and_accuracy(
            model=model,
            input_ids=data["input_ids"],
            hidden_states=data["hidden_states"],
            loss_mask=data["loss_mask"],
        )

        self.log(
            {
                "train/loss": round(float(loss.item()), 4),
                "train/accuracy": round(float(accuracy.item()), 4),
            }
        )

        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Perform an evaluation step."""
        data = self.prepare_data_for_draft_model(inputs)

        with torch.no_grad():
            loss, accuracy = self._compute_dflash_loss_and_accuracy(
                model=model,
                input_ids=data["input_ids"],
                hidden_states=data["hidden_states"],
                loss_mask=data["loss_mask"],
            )

        self.log(
            {
                "eval/loss": round(float(loss.item()), 4),
                "eval/accuracy": round(float(accuracy.item()), 4),
            }
        )

        return loss, None, None
