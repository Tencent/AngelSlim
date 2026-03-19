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

import torch

from ....utils import print_info, set_op_by_name
from ..modules.quant import QuantLinear
from .base_plugin import BasePlugin
from .plugin_manager import PluginManager


@PluginManager.plugin("learnable_scale")
class LearnableScalePlugin(BasePlugin):
    def __init__(self, quant_info, **kwargs):
        super().__init__(**kwargs)
        self.quant_info = quant_info
        self.ignore_layers = self.config["compress_config"].quantization.ignore_layers

    def before_train(self, **kwargs):
        qc = self.config["training_config"].plugin_config["quant_config"]
        for name, module in self.quant_model.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if any(ig in name for ig in self.ignore_layers):
                    continue
                q_linear = QuantLinear(
                    module,
                    self.config,
                    self.quant_info,
                    qc.get("use_weight_quant", True),
                    qc.get("use_activation_quant", True),
                )
                set_op_by_name(self.quant_model.model, name, q_linear)

        print_info(self.quant_model.model)

        if qc.get("use_activation_quant", True):
            self._lazy_init(**kwargs)

        set_quant_parameters(self.quant_model.model, requires_grad=True)
        set_weight_parameters(self.quant_model.model, requires_grad=False)

    def _lazy_init(self, **kwargs):
        for _, module in self.quant_model.model.named_modules():
            if isinstance(module, QuantLinear):
                if module.act_quantizer.dynamic:
                    return
                module.set_quant_state(weight_quant=False, act_quant=True)

        init_samples = (
            self.config["training_config"]
            .plugin_config["quant_config"]
            .get("lazy_init_samples", 10)
        )
        for i in range(init_samples):
            batch = kwargs["train_dataset"][i]
            inputs = {
                k: torch.tensor(v).unsqueeze(0).to(self.quant_model.model.device)
                for k, v in batch.items()
                if k != "labels"
            }
            with torch.no_grad():
                self.quant_model.model(**inputs)

        for _, module in self.quant_model.model.named_modules():
            if isinstance(module, QuantLinear):
                module.act_quantizer.init = True
                module.set_quant_state(weight_quant=True, act_quant=True)


def set_quant_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        if n.find("scale") > -1 or n.find("zero_point") > -1:
            m.requires_grad = requires_grad
    return iter(params)


def quant_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find("scale") > -1 or n.find("zero_point") > -1:
            params.append(m)
    return iter(params)


def set_weight_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        if n.find("weight") > -1 and not (n.find("scale") > -1 or n.find("zero_point") > -1):
            m.requires_grad = requires_grad
    return iter(params)


def weight_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find("weight") > -1 and not (n.find("scale") > -1 or n.find("zero_point") > -1):
            params.append(m)
    return iter(params)


def trainable_parameters(model):
    params = []
    for _, m in model.named_parameters():
        if m.requires_grad:
            params.append(m)
    return iter(params)


@torch.no_grad()
def quant_inplace(model):
    for _, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight.data = module.weight_quantizer(module.weight.data)
