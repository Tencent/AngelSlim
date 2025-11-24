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
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import Trainer

from angelslim.utils.lazy_imports import deepspeed

from ...utils import padding


class Eagle3Trainer(Trainer, ABC):
    """
    EAGLE3 Trainer for speculative decoding training.

    Implements training logic for EAGLE3 model using a draft model to predict
    tokens based on hidden states from a target model.
    """

    def __init__(self, draft_model: nn.Module, length: int, **kwargs):
        """
        Initialize the OnlineEagle3Trainer.

        Args:
            draft_model: Draft model for token prediction
            length: Number of speculative decoding steps
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(model=draft_model, **kwargs)
        self.length = length

    @property
    def draft_model(self) -> nn.Module:
        """Get the draft model."""
        return self.model

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[int] = None,
        return_outputs: bool = False,
    ) -> Tuple[List[torch.Tensor], List, List[float]]:
        """
        Compute the training loss for the model.

        Args:
            model: The model for which to compute the loss
            inputs: Input data dictionary with input_ids, attention_mask,
                loss_mask, position_ids
            num_items_in_batch: Number of items in batch (unused)
            return_outputs: Whether to return model outputs (unused)

        Returns:
            Tuple of (prediction_losses, value_losses, accuracies) for each step
        """
        data_for_draft_model = self.prepare_data_for_draft_model(inputs)

        attention_mask = data_for_draft_model["attention_mask"]  # Batch x Seq
        position_ids = data_for_draft_model["position_ids"]  # Batch x Seq
        input_ids = data_for_draft_model["input_ids"]  # Batch x Seq
        target_logits = data_for_draft_model["target_logits"]  # Batch x Seq x Vocab
        loss_mask = data_for_draft_model["loss_mask"]  # Batch x Seq x 1
        hidden_states = data_for_draft_model["hidden_states"]  # Batch x Seq x Hidden

        hidden_states = self.down_project_hidden_states(hidden_states)
        attention_mask, position_ids = self.prepare_attention_mask_and_position_ids(
            hidden_states, attention_mask, position_ids
        )
        loss = self.draft_model_training_time_test(
            input_ids,
            hidden_states,
            attention_mask,
            position_ids,
            target_logits,
            loss_mask,
        )

        return loss

    @abstractmethod
    def prepare_data_for_draft_model(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare data for draft model training.
        """
        pass

    def down_project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Down project hidden states for draft model training.
        """
        # Step 4: Prepare hidden states with gradient tracking
        if not hidden_states.requires_grad:
            hidden_states.requires_grad = True
        hidden_states = self.draft_model.combine_hidden_states(hidden_states)
        return hidden_states

    def prepare_attention_mask_and_position_ids(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare attention mask for draft model training.
        """
        # Step 5: Prepare attention mask and position IDs
        batch_size, seq_length, _ = hidden_states.shape
        device = hidden_states.device

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=device
            )

        attention_mask = self.draft_model.prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, 0
        )

        return attention_mask, position_ids

    def draft_model_training_time_test(
        self,
        input_ids,
        hidden_states,
        attention_mask,
        position_ids,
        target_logits,
        loss_mask,
    ):
        _, seq_length, _ = hidden_states.shape

        # Step 6: Initialize containers for losses, accuracies and cache
        plosses, acces = [], []
        cache_hidden = [[], []]

        # Step 7: Iterative speculative decoding training loop
        for idx in range(self.length):
            # Step 7.1: Get input embeddings with gradient tracking
            inputs_embeds = self.draft_model.get_input_embeddings(input_ids)
            if not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad = True

            # Step 7.2: Encode through draft model layers
            hidden_states = self.draft_model.encode_layers(
                inputs_embeds=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )

            # Step 7.3: Compute logits from hidden states
            logits = self.draft_model.compute_logits(hidden_states)

            # Step 7.4: Compute target distribution and position mask
            with torch.no_grad():
                target_max_token = target_logits.argmax(-1)
                target_mask = self.draft_model.t2d[target_max_token][..., None].int()
                position_mask = target_mask * loss_mask

                target_head = target_logits[..., self.draft_model.t2d].float()
                target_p = nn.Softmax(dim=2)(target_head).detach()

            # Step 7.5: Compute loss
            out_logp = nn.LogSoftmax(dim=2)(logits)
            loss = -torch.sum(position_mask * target_p * out_logp, dim=2).mean()

            # Step 7.6: Compute accuracy
            with torch.no_grad():
                correct = (
                    logits.argmax(-1) == target_p.argmax(-1)
                ) * position_mask.squeeze(-1)
                accuracy = correct.sum().item() / (loss_mask.sum().item() + 1e-6)

            # Step 7.7: Store loss and accuracy
            plosses.append(loss)
            acces.append(accuracy)

            # Step 7.8: Update inputs for next iteration (skip on last step)
            if idx < self.length - 1:
                input_ids = padding(input_ids, left=False)
                target_logits = padding(target_logits, left=False)
                loss_mask = padding(loss_mask, left=False)

                # Update attention mask to prevent attending to future positions
                ind = torch.arange(seq_length, device=attention_mask.device)
                attention_mask[:, :, ind[idx:], ind[: seq_length - idx]] = torch.finfo(
                    attention_mask.dtype
                ).min

        # Step 8: Compute weighted loss
        ploss_weight = [0.8**i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])

        log = {f"train/acc_{i}": round(float(acces[i]), 3) for i in range(len(acces))}
        log.update(
            {
                f"train/ploss_{i}": round(float(plosses[i].item()), 3)
                for i in range(len(plosses))
            }
        )
        self.log(log)

        # Step 9: Return loss
        return ploss

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        """
        Override save_model to handle DeepSpeed ZeRO-3 model saving.

        Args:
            output_dir: Directory to save the model. If None, uses self.args.output_dir
            _internal_call: Internal flag used by Trainer
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        # Check if using DeepSpeed ZeRO-3
        is_deepspeed_zero3 = (
            self.is_deepspeed_enabled
            and hasattr(self.accelerator.state, "deepspeed_plugin")
            and self.accelerator.state.deepspeed_plugin.zero_stage == 3
        )

        if is_deepspeed_zero3:
            # Handle ZeRO-3 model saving
            self._save_zero3_model(output_dir, _internal_call)
        else:
            # Fall back to parent class save_model
            super().save_model(output_dir, _internal_call)

    def _save_zero3_model(self, output_dir: str, _internal_call: bool = False):
        """
        Save model with DeepSpeed ZeRO-3 specific logic.

        Args:
            output_dir: Directory to save the model
            _internal_call: Internal flag used by Trainer
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save with DeepSpeed's state_dict gathering
        # All processes must participate in parameter gathering to avoid deadlock
        with deepspeed.zero.GatheredParameters(self.model.parameters()):
            state_dict = self.model.state_dict()

        draft_model_state_dict = {
            k: v for k, v in state_dict.items() if "embed" not in k
        }

        # Only main process saves the model
        if self.args.should_save and self.accelerator.is_main_process:
            self.model.save_pretrained(
                output_dir,
                is_main_process=True,
                state_dict=draft_model_state_dict,
                save_function=torch.save,
            )

            # Save training arguments
            from transformers.trainer import TRAINING_ARGS_NAME

            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # Wait for all processes
        self.accelerator.wait_for_everyone()
