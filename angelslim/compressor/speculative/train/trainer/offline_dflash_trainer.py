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

from .online_dflash_trainer import OnlineDFlashTrainer
from .trainer_factory import Eagle3TrainerFactory


@Eagle3TrainerFactory.register("offline", "DFlash")
class OfflineDFlashTrainer(OnlineDFlashTrainer):
    """
    DFlash trainer for offline (pre-computed hidden states) training.

    The main difference vs online: hidden_states are loaded directly from the
    pre-computed .ckpt files, so prepare_data_for_draft_model() just unpacks
    the batch instead of running a target-model forward pass.
    """

    def prepare_data_for_draft_model(self, inputs):
        """
        Unpack pre-computed hidden states from the offline batch.

        Expected batch keys (from OfflineDFlashDataset):
            input_ids      [B, S]
            hidden_states  [B, S, D*L]
            loss_mask      [B, S]
            attention_mask [B, S]
        """
        return {
            "input_ids": inputs["input_ids"],
            "hidden_states": inputs["hidden_states"],
            "loss_mask": inputs["loss_mask"],
            "attention_mask": inputs["attention_mask"],
        }
