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
from ..base_model import BaseLLMModel
from ..model_factory import SlimModelFactory


@SlimModelFactory.register
class SeedOss(BaseLLMModel):
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
            "k_proj",
            "v_proj",
            "q_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
        observer_layers_dict = {}
        layers_dict = self.find_layers(self.model, layers=self.observer_layer_classes)

        ignore_layers = self.skip_layer_names()
        for name, module in layers_dict.items():
            if name.startswith(self.block_name) and name.split(".")[-1] in names:
                observer_layers_dict[name] = module
            else:
                ignore_layers.append(name)
        self.quant_config.quant_algo_info["ignore_layers"] = ignore_layers

        if self.quant_config.custom_observe_layers_names != "default":
            for custom_observe_name in self.quant_config.custom_observe_layers_names:
                for default_name in observer_layers_dict.keys():
                    if custom_observe_name not in default_name:
                        observer_layers_dict.pop(default_name)
        return observer_layers_dict

    def get_smooth_mapping_layers(self, smooth_config, mappings=None):
        if mappings is None:
            mappings = [
                (["q_proj", "k_proj", "v_proj"], "input_layernorm"),
                (["gate_proj", "up_proj"], "post_attention_layernorm"),
            ]
        print(f"smooth mappings={mappings}")
        assert len(mappings) == 2
        assert smooth_config.smooth_first_linears or smooth_config.smooth_last_linears
        # TODO: support smooth_last_linears
        return super().get_smooth_mapping_layers(smooth_config, mappings)

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
