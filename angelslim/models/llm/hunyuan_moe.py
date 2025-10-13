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

import re

from ...compressor.quant.core import PTQSaveVllmHF
from ...utils.utils import find_layers
from ..base_model import BaseLLMModel
from ..model_factory import SlimModelFactory


@SlimModelFactory.register
class HunyuanMoE(BaseLLMModel):
    def __init__(
        self,
        model=None,
        deploy_backend="vllm",
    ):
        super().__init__(
            model=model,
            deploy_backend=deploy_backend,
        )
        self.block_name = "model.layers"

    def get_observer_layers(self):
        names = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "shared_mlp.gate_proj",
            "shared_mlp.up_proj",
            "shared_mlp.down_proj",
        ]
        expert_pattern = [
            r"model\.layers\.\d+\.mlp\.experts\.\d+\.gate_proj",
            r"model\.layers\.\d+\.mlp\.experts\.\d+\.up_proj",
            r"model\.layers\.\d+\.mlp\.experts\.\d+\.down_proj",
        ]

        observer_layers_dict = find_layers(
            self.model, layers=self.observer_layer_classes
        )

        compiled_patterns = [re.compile(pattern) for pattern in expert_pattern]

        observer_layers_dict = {
            k: v
            for k, v in observer_layers_dict.items()
            if k.startswith(self.block_name)
            and (
                any(name in k for name in names)
                or any(pattern.search(k) for pattern in compiled_patterns)
            )
        }

        if self.quant_config.custom_observe_layers_names != "default":
            for custom_observe_name in self.quant_config.custom_observe_layers_names:
                for default_name in observer_layers_dict.keys():
                    if custom_observe_name not in default_name:
                        observer_layers_dict.pop(default_name)
        return observer_layers_dict

    def get_parent_dict(self, observer_layers_dict):
        parent_mapping = {r"experts\.\d+": "experts"}
        parent_dict = {}
        for layer_name in observer_layers_dict.keys():
            parent_name = layer_name
            for k, v in parent_mapping.items():
                parent_name = re.sub(k, v, layer_name)
            if parent_name != layer_name:
                parent_dict[layer_name] = parent_name
        return parent_dict

    def get_save_func(self):
        if self.deploy_backend in ["vllm", "huggingface"]:
            return PTQSaveVllmHF
        else:
            raise NotImplementedError(
                f"deploy_backend {self.deploy_backend} is not supported for saving."
            )
