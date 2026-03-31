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

from ....utils import print_info


class TrainerFactory:
    _trainers = {}

    @classmethod
    def register(cls, name: str):
        def decorator(trainer_class):
            if name in cls._trainers:
                print_info(f"Warning: Trainer '{name}' is already registered.")
            cls._trainers[name] = trainer_class
            return trainer_class

        return decorator

    @classmethod
    def create(cls, training_mode, quant_model, config, plugin_manager):
        normalized = training_mode.lower().replace("-", "").replace("_", "")
        if normalized not in cls._trainers:
            raise ValueError(
                f"[TrainerFactory] Unsupported training mode: '{training_mode}'. "
                f"Supported: {', '.join(cls._trainers.keys())}."
            )
        return cls._trainers[normalized](quant_model, config, plugin_manager)

    @classmethod
    def get_available_trainers(cls):
        return list(cls._trainers.keys())
