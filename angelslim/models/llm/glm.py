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

import torch.nn as nn

from ...compressor.quant.core import PTQSaveVllmHF
from ...utils.utils import find_layers
from ..base_model import BaseLLMModel
from ..model_factory import SlimModelFactory


@SlimModelFactory.register
class GLM(BaseLLMModel):
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
        obs_layers = [nn.Linear]
        observer_layers_dict = {}
        layers_dict = find_layers(self.model, layers=obs_layers)

        ignore_layers = self.skip_layer_names()
        for name, module in layers_dict.items():
            if name.startswith(self.block_name) and name.split(".")[-1] in names:
                observer_layers_dict[name] = module
            else:
                ignore_layers.append(name)
        ignore_layers = sorted(list(set(ignore_layers)))
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

    def fuse_observer_amax(self, sub_layer, name):
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            prefix = name.rsplit(".", 1)[0]
            q_name = f"{prefix}.q_proj"
            k_name = f"{prefix}.k_proj"
            v_name = f"{prefix}.v_proj"

            weight_scales = []
            for key in [q_name, k_name, v_name]:
                tensor = self.weight_observer_amax_dict[key]
                weight_scales.append(tensor)
            weight_observer_amax = max(weight_scales)

            act_scales = []
            for key in [q_name, k_name, v_name]:
                tensor = self.input_observer_amax_dict[key]
                act_scales.append(tensor)
            input_observer_amax = max(act_scales)
        elif "gate_proj" in name or "up_proj" in name:
            prefix = name.rsplit(".", 1)[0]
            gate_name = f"{prefix}.gate_proj"
            up_name = f"{prefix}.up_proj"

            weight_scales = []
            for key in [gate_name, up_name]:
                tensor = self.weight_observer_amax_dict[key]
                weight_scales.append(tensor)
            weight_observer_amax = max(weight_scales)

            act_scales = []
            for key in [gate_name, up_name]:
                tensor = self.input_observer_amax_dict[key]
                act_scales.append(tensor)
            input_observer_amax = max(act_scales)
        else:
            weight_observer_amax = self.weight_observer_amax_dict[name]
            input_observer_amax = self.input_observer_amax_dict[name]

        return weight_observer_amax, input_observer_amax
