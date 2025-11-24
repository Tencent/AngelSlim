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

from typing import Any, Dict

from torch import nn

from ...utils import padding
from .eagle3_trainer import Eagle3Trainer
from .trainer_factory import Eagle3TrainerFactory


@Eagle3TrainerFactory.register("online", "LLM")
class OnlineEagle3Trainer(Eagle3Trainer):
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

    def prepare_data_for_draft_model(self, inputs):
        # Step 1: Extract input tensors
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        loss_mask = inputs["loss_mask"]
        position_ids = inputs.get("position_ids", None)

        # Step 2: Get hidden states and logits from target model
        hidden_states, target_logits = self.target_model.get_hidden_states_and_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aux_hidden_states_layer_ids=self._aux_hidden_states_layer_ids,
        )

        # Step 3: Apply right padding and move tensors to correct device
        target_logits = padding(target_logits, left=False).to(input_ids.device)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None].to(input_ids.device)

        return {
            "hidden_states": hidden_states,
            "target_logits": target_logits,
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
