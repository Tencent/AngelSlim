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

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from ...utils import padding
from .eagle3_trainer import Eagle3Trainer
from .trainer_factory import Eagle3TrainerFactory


@Eagle3TrainerFactory.register("online", "VLM")
class OnlineVLMEagle3Trainer(Eagle3Trainer):
    """
    Online EAGLE3 Trainer for speculative decoding training.
    Implements training logic for EAGLE3 model using a draft model to predict
    tokens based on hidden states from a target model.
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
        Initialize the OnlineEagle3Trainer.
        Args:
            draft_model: Draft model for token prediction
            target_model: Target model for generating hidden states
            length: Number of speculative decoding steps
            draft_model_config: Configuration dictionary for draft model
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(draft_model=draft_model, length=length, **kwargs)
        self.target_model = target_model
        self._aux_hidden_states_layer_ids = getattr(
            draft_model_config, "aux_hidden_states_layer_ids", None
        )

    def prepare_data_for_draft_model(
        self, input_ids, attention_mask, loss_mask, **kwargs
    ):
        # Get hidden states and logits from target model
        hidden_states, target_logits, inputs_embeds, position_ids = (
            self.target_model.get_hidden_states_and_logits(
                input_ids=input_ids,
                attention_mask=attention_mask,
                aux_hidden_states_layer_ids=self._aux_hidden_states_layer_ids,
                **kwargs,
            )
        )

        # Apply right padding and move tensors to correct device
        target_logits = padding(target_logits, left=False).to(input_ids.device)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None].to(input_ids.device)

        return {
            "hidden_states": hidden_states,
            "target_logits": target_logits,
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            **kwargs,
        }

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
        data_for_draft_model = self.prepare_data_for_draft_model(**inputs)

        attention_mask = data_for_draft_model["attention_mask"]
        inputs_embeds = data_for_draft_model["inputs_embeds"]
        position_ids = data_for_draft_model.get("position_ids", None)
        input_ids = data_for_draft_model["input_ids"]
        target_logits = data_for_draft_model["target_logits"]
        loss_mask = data_for_draft_model["loss_mask"]
        hidden_states = data_for_draft_model["hidden_states"]

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
            inputs_embeds,
        )

        return loss

    def draft_model_training_time_test(
        self,
        input_ids,
        hidden_states,
        attention_mask,
        position_ids,
        target_logits,
        loss_mask,
        inputs_embeds,
    ):
        _, seq_length, _ = hidden_states.shape

        # Initialize containers for losses, accuracies and cache
        plosses, acces = [], []
        cache_hidden = [[], []]

        # Iterative speculative decoding training loop
        for idx in range(self.length):
            # Get input embeddings with gradient tracking
            if inputs_embeds is None:
                inputs_embeds = self.draft_model.get_input_embeddings(input_ids)
            if not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad = True

            # Encode through draft model layers
            hidden_states = self.draft_model.encode_layers(
                inputs_embeds=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )

            # Compute logits from hidden states
            logits = self.draft_model.compute_logits(hidden_states)

            # Compute target distribution and position mask
            with torch.no_grad():
                target_max_token = target_logits.argmax(-1)
                target_mask = self.draft_model.t2d[target_max_token][..., None].int()
                position_mask = target_mask * loss_mask

                target_head = target_logits[..., self.draft_model.t2d].float()
                target_p = nn.Softmax(dim=2)(target_head).detach()

            # Compute loss
            out_logp = nn.LogSoftmax(dim=2)(logits)
            loss = -torch.sum(position_mask * target_p * out_logp, dim=2).mean()

            # Compute accuracy
            with torch.no_grad():
                correct = (
                    logits.argmax(-1) == target_p.argmax(-1)
                ) * position_mask.squeeze(-1)
                accuracy = correct.sum().item() / (loss_mask.sum().item() + 1e-6)

            # Store loss and accuracy
            plosses.append(loss)
            acces.append(accuracy)

            # Update inputs for next iteration (skip on last step)
            if idx < self.length - 1:
                input_ids = padding(input_ids, left=False)
                target_logits = padding(target_logits, left=False)
                loss_mask = padding(loss_mask, left=False)

                # Update attention mask to prevent attending to future positions
                ind = torch.arange(seq_length, device=attention_mask.device)
                attention_mask[:, :, ind[idx:], ind[: seq_length - idx]] = torch.finfo(
                    attention_mask.dtype
                ).min

        # Compute weighted loss
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

        # Return loss
        return ploss
