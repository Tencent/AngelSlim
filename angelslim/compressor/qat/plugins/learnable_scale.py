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
from tqdm import tqdm

from ....utils import print_info, set_op_by_name
from ..modules.quantizer import QuantLinear
from .base_plugin import BasePlugin
from .plugin_manager import PluginManager


@PluginManager.plugin("learnable_scale")
class LearnableScalePlugin(BasePlugin):
    def __init__(self, quant_info=None, ignore_layers=None, resume_ckpt_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.quant_info = quant_info
        self.ignore_layers = ignore_layers
        self.resume_ckpt_dir = resume_ckpt_dir
        self.use_weight_quant = self.config.get("use_weight_quant", False)
        self.use_activation_quant = self.config.get("use_activation_quant", False)

    def before_train(self, **kwargs):
        for name, module in self.quant_model.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if any(ig in name for ig in self.ignore_layers):
                    continue
                q_linear = QuantLinear(
                    module,
                    self.config,
                    self.quant_info,
                    self.use_weight_quant,
                    self.use_activation_quant,
                    resume=self.resume_ckpt_dir is not None,
                )
                set_op_by_name(self.quant_model.model, name, q_linear)

        print_info(self.quant_model.model)

        if (
            self.use_activation_quant
            and not q_linear.act_quantizer.dynamic
            and self.resume_ckpt_dir is None
        ):
            self._lazy_init(**kwargs)

        set_quant_parameters(self.quant_model.model, requires_grad=True)
        set_weight_parameters(self.quant_model.model, requires_grad=False)

    def after_train(self):
        if self.use_weight_quant:
            quant_inplace(self.quant_model.model)
            set_quant_state(
                self.quant_model.model, weight_quant=False, act_quant=self.use_activation_quant
            )

    def _lazy_init(self, **kwargs):
        set_quant_state(self.quant_model.model, weight_quant=False, act_quant=True)

        init_samples = self.config.get("lazy_init_samples", 10)
        for i in tqdm(range(init_samples), desc="Lazy init"):
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
                if not module.act_quantizer.dynamic and not isinstance(
                    module.act_quantizer.scale, torch.nn.Parameter
                ):  # for MoE
                    if not hasattr(module.act_quantizer, "overall_scale"):
                        scale = torch.tensor(
                            1.0, dtype=module.weight.dtype, device=module.weight.device
                        )
                        zp = (
                            torch.tensor(0.0, dtype=scale.dtype, device=scale.device)
                            if not module.act_quantizer.is_sym
                            else None
                        )
                        module.act_quantizer._set_quant_parameters(scale, zp)
                        print_info(
                            f"Not calibrate, init scale: {module.act_quantizer.scale.item()}"
                        )
                    else:
                        max_scale = max(module.act_quantizer.overall_scale)
                        max_idx = module.act_quantizer.overall_scale.index(max_scale)
                        zp = (
                            module.act_quantizer.overall_zero_point[max_idx]
                            if not module.act_quantizer.is_sym
                            else None
                        )
                        module.act_quantizer._set_quant_parameters(max_scale, zp)
                        print_info(
                            f"Lazy init done, scale: {module.act_quantizer.scale.item()}, samples: {module.act_quantizer.calib_count}"  # noqa: E501
                        )
                        del (
                            module.act_quantizer.overall_scale,
                            module.act_quantizer.overall_zero_point,
                            module.act_quantizer.calib_count,
                        )

        set_quant_state(self.quant_model.model, weight_quant=self.use_weight_quant, act_quant=True)


def set_quant_state(model, weight_quant=False, act_quant=False):
    for module in model.modules():
        if isinstance(module, QuantLinear):
            module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)


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
