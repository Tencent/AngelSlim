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

from .offline_eagle3_trainer import OfflineEagle3Trainer, OfflineVLMEagle3Trainer
from .online_eagle3_trainer import OnlineEagle3Trainer, OnlineVLMEagle3Trainer
from .trainer_factory import Eagle3TrainerFactory

__all__ = [
    "Eagle3TrainerFactory",
    "OnlineEagle3Trainer",
    "OnlineVLMEagle3Trainer",
    "OfflineEagle3Trainer",
    "OfflineVLMEagle3Trainer",
]
