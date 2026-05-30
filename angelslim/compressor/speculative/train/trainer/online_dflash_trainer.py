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


class _FP32StateAdamW(torch.optim.Optimizer):
    """AdamW with fp32 master weights (AngelSlim DFlash optimizer).

    Maintains fp32 master copies of all parameters (in optimizer state).
    On each step:
      1. Cast bf16 gradients to fp32.
      2. Clip fp32 grad norm.
      3. Adam update on fp32 master weights.
      4. Copy fp32 master -> bf16 model params.

    Key properties:
      * Accumulation in fp32 (no precision loss from bf16 quantization).
      * Grad clipping on fp32 gradients.
      * Only the final copy-back introduces bf16 quantization.

    Compatible with FSDP + accelerate + HF Trainer (operates on the SAME
    parameter objects required for FSDP state_dict).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        max_grad_norm=1.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        super().__init__(params, defaults)

        # Eagerly initialize all master parameters at construction so all
        # ranks start from synchronized bf16 params before any training step.
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["master_param"] = p.data.detach().clone().to(torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        """Full fp32 master-weight update step (AngelSlim DFlash optimizer)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Phase 1: Cast all bf16 grads to fp32 (kept temporarily in state).
        all_fp32_grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Ensure states are fp32 (handles resume from checkpoint).
                if state["exp_avg"].dtype != torch.float32:
                    state["exp_avg"] = state["exp_avg"].to(torch.float32)
                if state["exp_avg_sq"].dtype != torch.float32:
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(torch.float32)
                if state["master_param"].dtype != torch.float32:
                    state["master_param"] = state["master_param"].to(torch.float32)

                fp32_grad = p.grad.detach().to(torch.float32)
                state["_fp32_grad"] = fp32_grad
                all_fp32_grads.append(fp32_grad)

        # Phase 2: Clip fp32 grad norm.
        # Manual clipping because all_fp32_grads holds plain tensors (not Parameters).
        # In FSDP SHARD_GRAD_OP + use_orig_params=True, p.grad is the full
        # all-reduced gradient (same on all ranks), so per-rank clipping is correct.
        if self.max_grad_norm > 0 and all_fp32_grads:
            total_norm_sq = sum(g.norm().pow(2) for g in all_fp32_grads)
            total_norm = total_norm_sq.sqrt()
            clip_coef = self.max_grad_norm / (total_norm + 1e-6)
            clip_coef_clamped = min(clip_coef.item(), 1.0)
            if clip_coef_clamped < 1.0:
                for g in all_fp32_grads:
                    g.mul_(clip_coef_clamped)

        # Phase 3: Adam update on fp32 master weights, then copy back to bf16.
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = state.pop("_fp32_grad")

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                master_param = state["master_param"]

                state["step"] += 1
                step_t = state["step"].item()

                # Decoupled weight decay on fp32 master (AdamW style).
                if weight_decay != 0:
                    master_param.mul_(1.0 - lr * weight_decay)

                # Adam update in fp32.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step_t
                bias_correction2 = 1 - beta2**step_t

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)

                master_param.addcdiv_(exp_avg, denom, value=-step_size)

                # Copy fp32 master -> bf16 model param (only quantization point).
                p.data.copy_(master_param.to(p.dtype))
                p.grad = None

        return loss


class _FP32MasterWeightOptimizer(torch.optim.Optimizer):
    """Thin wrapper around any torch optimizer that maintains fp32 master weights.

    AngelSlim DFlash fp32 master-weight pattern:
      1. At construction, clone bf16 model params -> fp32 master copies.
      2. On every step():
         a. Copy bf16 grads -> fp32 master grads (cast up), clear bf16 grads.
         b. Clip fp32 grad norm.
         c. Run inner optimizer step on fp32 params.
         d. Copy updated fp32 values -> bf16 model params (cast down).

    Used as ``self.optimizer`` inside HF Trainer for the DDP code path.
    HF Trainer calls ``self.optimizer.step()`` / ``self.optimizer.zero_grad()``
    directly, so placing the sync logic inside the optimizer is the only
    reliable way to ensure it actually runs.

    Inherits from torch.optim.Optimizer so isinstance checks in HF Trainer's
    lr_scheduler creation (LambdaLR.__init__) pass correctly.
    """

    def __init__(
        self,
        bf16_params: List[torch.Tensor],
        inner_optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 1.0,
    ):
        self._bf16_params = bf16_params

        # Build fp32 master copies and replace the optimizer's param groups.
        self._fp32_params: List[torch.Tensor] = [
            p.detach().clone().to(torch.float32).requires_grad_(True) for p in bf16_params
        ]
        assert len(inner_optimizer.param_groups) == 1, (
            "_FP32MasterWeightOptimizer expects a single param group; "
            "extend if/when multiple groups (e.g. LoRA, lr split) are needed."
        )
        inner_optimizer.param_groups[0]["params"] = self._fp32_params
        # Re-initialise state dict for the new param objects.
        from collections import defaultdict

        inner_optimizer.state = defaultdict(dict)

        self._inner = inner_optimizer
        self.max_grad_norm = max_grad_norm

        # Call torch.optim.Optimizer.__init__ so that isinstance(self, Optimizer)
        # returns True. The _initializing flag prevents add_param_group from
        # delegating to self._inner during super().__init__ (which would
        # create a duplicate param group in the inner optimizer).
        self._initializing = True
        super().__init__(self._fp32_params, inner_optimizer.defaults)
        self._initializing = False

        # Redirect param_groups and state to inner optimizer's versions so
        # lr_scheduler / lr logging always see the correct param groups.
        self.param_groups = self._inner.param_groups
        self.state = self._inner.state

    # ------------------------------------------------------------------ #
    # Core step / zero_grad — called directly by HF Trainer               #
    # ------------------------------------------------------------------ #

    def step(self, closure=None):
        """Full fp32 master-weight update step."""
        with torch.no_grad():
            # (a) Copy bf16 grads -> fp32 master grads.
            for bf16_p, fp32_p in zip(self._bf16_params, self._fp32_params):
                if bf16_p.grad is not None:
                    fp32_p.grad = bf16_p.grad.detach().to(torch.float32)
                    bf16_p.grad = None
                else:
                    fp32_p.grad = None

            # (b) Clip fp32 grad norm.
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._fp32_params, self.max_grad_norm)

        # (c) Optimizer step on fp32 params.
        loss = self._inner.step(closure)

        # (d) Copy fp32 -> bf16 model params.
        with torch.no_grad():
            for bf16_p, fp32_p in zip(self._bf16_params, self._fp32_params):
                bf16_p.data.copy_(fp32_p.data.to(bf16_p.dtype))

        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients on both bf16 model params and fp32 master params."""
        for bf16_p in self._bf16_params:
            if set_to_none:
                bf16_p.grad = None
            elif bf16_p.grad is not None:
                bf16_p.grad.zero_()
        for fp32_p in self._fp32_params:
            if set_to_none:
                fp32_p.grad = None
            elif fp32_p.grad is not None:
                fp32_p.grad.zero_()

    # ------------------------------------------------------------------ #
    # Delegate everything else to the inner optimizer                      #
    # ------------------------------------------------------------------ #

    def state_dict(self):
        return self._inner.state_dict()

    def load_state_dict(self, state_dict):
        return self._inner.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        # During super().__init__ the inner optimizer is not yet assigned,
        # so fall back to the default Optimizer behaviour.
        if getattr(self, "_initializing", True):
            return super().add_param_group(param_group)
        return self._inner.add_param_group(param_group)

    def __repr__(self):
        return f"_FP32MasterWeightOptimizer({self._inner})"


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
        # Gamma warmup: gradually increase loss_decay_gamma per epoch
        # (AngelSlim DFlash gamma-warmup schedule).
        self._gamma_init = self.loss_decay_gamma
        self.gamma_warmup = getattr(draft_model_config, "gamma_warmup", False)
        self._gamma_step = getattr(draft_model_config, "gamma_warmup_step", 0.5)
        self.attention_backend = getattr(draft_model_config, "attention_backend", "flex_attention")
        self.mask_token_id = dflash_config.get(
            "mask_token_id",
            getattr(draft_model_config, "mask_token_id", None),
        )

        # Sync _attn_implementation on the draft model so its attention layers
        # dispatch to the correct backend (eager vs flex_attention vs sdpa).
        if self.attention_backend == "eager":
            draft_model.config._attn_implementation = "eager"
        elif self.attention_backend == "flex_attention":
            draft_model.config._attn_implementation = "flex_attention"
        else:
            draft_model.config._attn_implementation = self.attention_backend

        # fp32 master weights optimizer — set by create_optimizer() (DDP path).
        # FSDP path uses _FP32StateAdamW directly as self.optimizer instead.
        self._fp32_optimizer: Optional["_FP32MasterWeightOptimizer"] = None

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

    def create_optimizer(self, model=None):
        """Create optimizer for DFlash training.

        Three branches:
          * DeepSpeed: defer to HF Trainer's default optimizer creation.
          * FSDP: AdamW with fp32 optimizer states (``_FP32StateAdamW``),
            using the AngelSlim DFlash fp32-master pattern. Critical because
            bf16 momentum and variance only have 7-bit mantissa, which causes
            training quality degradation after a few thousand steps.
          * DDP / single GPU: ``_FP32MasterWeightOptimizer`` wrapping AdamW for
            fp32 master weight updates.
        """
        if self.is_deepspeed_enabled:
            return super().create_optimizer(model)

        if self.is_fsdp_enabled:
            args = self.args
            param_groups = [{"params": [p for p in self.model.parameters() if p.requires_grad]}]
            optimizer = _FP32StateAdamW(
                param_groups,
                lr=args.learning_rate,
                betas=(
                    getattr(args, "adam_beta1", 0.9),
                    getattr(args, "adam_beta2", 0.999),
                ),
                eps=getattr(args, "adam_epsilon", 1e-8),
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
            )
            self.optimizer = optimizer
            return self.optimizer

        bf16_params: List[torch.Tensor] = [p for p in self.model.parameters() if p.requires_grad]
        if not bf16_params:
            return super().create_optimizer(model)

        from torch.optim import AdamW

        args = self.args
        inner_optimizer = AdamW(
            # Placeholder — _FP32MasterWeightOptimizer will replace param_groups
            # with fp32 copies immediately after construction.
            bf16_params,
            lr=args.learning_rate,
            betas=(
                getattr(args, "adam_beta1", 0.9),
                getattr(args, "adam_beta2", 0.999),
            ),
            eps=getattr(args, "adam_epsilon", 1e-8),
            weight_decay=args.weight_decay,
        )

        fp32_opt = _FP32MasterWeightOptimizer(
            bf16_params=bf16_params,
            inner_optimizer=inner_optimizer,
            max_grad_norm=args.max_grad_norm,
        )
        self._fp32_optimizer = fp32_opt
        self.optimizer = fp32_opt
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Create LR scheduler: AngelSlim DFlash CosineAnnealingWarmupLR.

        AngelSlim warmup formula:  lr = base_lr * (step + 1) / warmup_steps
        HF Trainer warmup formula: lr = base_lr * step / warmup_steps

        The +1 offset means step 0 yields lr = base_lr / warmup_steps instead
        of 0. After warmup, both use identical cosine annealing.
        """
        import math

        from torch.optim.lr_scheduler import LambdaLR

        if optimizer is None:
            optimizer = self.optimizer

        warmup_steps = self.args.get_warmup_steps(num_training_steps)

        def angelslim_cosine_schedule(current_step: int) -> float:
            """LR multiplier for AngelSlim DFlash CosineAnnealingWarmupLR."""
            if current_step < warmup_steps:
                # AngelSlim: (last_epoch + 1) / warmup_epochs * base_lr
                # After N step() calls last_epoch = N, so first lr = 1/warmup_steps.
                return float(current_step + 1) / float(max(1, warmup_steps))
            # Cosine decay phase — identical to HF.
            progress = float(current_step - warmup_steps) / float(
                max(1, num_training_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.lr_scheduler = LambdaLR(optimizer, angelslim_cosine_schedule)
        return self.lr_scheduler

    def _clip_grad_norm(self, *args, **kwargs):
        """Skip HF Trainer's built-in grad clipping when running our fp32 optimizers.

        Both ``_FP32MasterWeightOptimizer`` (DDP) and ``_FP32StateAdamW`` (FSDP)
        clip gradients internally on fp32 grads (AngelSlim DFlash fp32-master
        clipping). HF Trainer's Accelerator-based clip_grad_norm_ would
        otherwise run on bf16 grads (incorrect precision for clipping), causing
        DOUBLE CLIPPING that significantly slows down training.
        """
        if self._fp32_optimizer is not None:
            # DDP path — clipped inside _FP32MasterWeightOptimizer.step()
            return torch.tensor(0.0)

        # FSDP path: _FP32StateAdamW clips internally on fp32 grads.
        # self.optimizer may be wrapped by AcceleratedOptimizer, so unwrap once.
        optimizer = self.optimizer
        if hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        if isinstance(optimizer, _FP32StateAdamW):
            return torch.tensor(0.0)

        return super()._clip_grad_norm(*args, **kwargs)

    def save_optimizer_and_scheduler(self, output_dir, **kwargs):
        """Override to handle fp32 master weight optimizer with FSDP.

        FSDP's built-in optim_state_dict() cannot handle our custom fp32
        master-weight optimizers because their fp32 params are not registered
        in the FSDP module's parameter graph. Save optimizer/scheduler state
        directly instead.
        """
        self._save_optimizer_and_scheduler(output_dir)

    def _save_optimizer_and_scheduler(self, output_dir):
        """Bypass FSDP's optim_state_dict for our custom fp32 optimizers."""
        optimizer = self.optimizer
        if hasattr(optimizer, "optimizer"):
            # Unwrap AcceleratedOptimizer
            optimizer = optimizer.optimizer

        if isinstance(optimizer, (_FP32StateAdamW, _FP32MasterWeightOptimizer)):
            if self.args.should_save:
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(output_dir, "optimizer.pt"),
                )
                if self.lr_scheduler is not None:
                    torch.save(
                        self.lr_scheduler.state_dict(),
                        os.path.join(output_dir, "scheduler.pt"),
                    )
        else:
            super()._save_optimizer_and_scheduler(output_dir)

    def _update_gamma_warmup(self):
        """Update loss_decay_gamma: gamma = gamma_init + step * epoch.

        AngelSlim DFlash gamma-warmup schedule:
            current_gamma = loss_decay_gamma + step * float(epoch)
        """
        if not self.gamma_warmup or self._gamma_init is None:
            return
        current_epoch = int(self.state.epoch) if hasattr(self.state, "epoch") else 0
        self.loss_decay_gamma = self._gamma_init + self._gamma_step * float(current_epoch)

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
        # Update gamma if warmup is enabled (no-op when gamma_warmup=False)
        self._update_gamma_warmup()

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
