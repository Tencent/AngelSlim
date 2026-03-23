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

import os

import torch

from ...utils import print_info, set_op_by_name
from ..compressor_factory import CompressorFactory
from ..quant.modules.helper_layer import QDQModule
from .modules.quantizer import QuantLinear
from .plugins.plugin_manager import PluginManager
from .trainers.trainer_factory import TrainerFactory

__all__ = ["QAT"]


@CompressorFactory.register
class QAT:
    def __init__(self, model, slim_config=None):
        self.quant_model = model
        self.config = slim_config
        self.training_mode = slim_config["compress_config"].QAT.training_mode.lower()
        self.save_fmt = slim_config["compress_config"].QAT.save_format
        self.plugin_config = slim_config["compress_config"].QAT.plugin_config
        self.quant_model.init_ptq(slim_config)
        self.quant_info = self.quant_model.quant_config
        self.plugin_manager = PluginManager()
        self._init_plugins()
        self._init_trainer()

    def _init_plugins(self):
        # Register learnable rotation plugin
        if self.plugin_config.get("enable_rotation", False):
            self.plugin_manager.register_plugin(
                "learnable_rotation",
                config=self.plugin_config.get("rotation_config", {}),
                quant_model=self.quant_model,
            )

        # Register learnable scale plugin
        if self.plugin_config.get("enable_scale", False):
            self.plugin_manager.register_plugin(
                "learnable_scale",
                quant_info=self.quant_info,
                ignore_layers=self.config["compress_config"].quantization.ignore_layers,
                resume_ckpt_dir=self.config["compress_config"].QAT.resume_ckpt_dir,
                config=self.plugin_config.get("quant_config", {}),
                quant_model=self.quant_model,
            )

    def _init_trainer(self):
        self.trainer = TrainerFactory.create(
            training_mode=self.training_mode,
            quant_model=self.quant_model,
            config=self.config,
            plugin_manager=self.plugin_manager,
        )

    def run(self, dataloader):
        self.trainer.run(dataloader)

    def convert(self):
        if self.save_fmt != "real":
            return

        print_info("Start QAT convert: replacing QuantLinear with QDQModule...")
        quant_algo = self.quant_info.quant_algo

        quant_linear_modules = [
            (name, module)
            for name, module in self.quant_model.model.named_modules()
            if isinstance(module, QuantLinear)
        ]

        for name, module in quant_linear_modules:
            weight_scale = None
            if hasattr(module, "weight_quantizer"):
                weight_scale = module.weight_quantizer.scale.data.clone()

            input_scale = None
            if module.use_act_quant and hasattr(module, "act_quantizer"):
                act_quantizer = module.act_quantizer
                if hasattr(act_quantizer, "scale") and act_quantizer.scale is not None:
                    input_scale = act_quantizer.scale.data.clone()

            qdq_module = QDQModule(
                quant_algo=quant_algo,
                weight=module.weight,
                weight_scale=weight_scale,
                bias=module.bias,
                group_size=(
                    module.weight_quantizer.group_size
                    if hasattr(module.weight_quantizer, "group_size")
                    else 128
                ),
                input_scale=input_scale,
            )
            set_op_by_name(self.quant_model.model, name, qdq_module)

    def save(self, save_path: str):
        if self.save_fmt == "fake":
            parts = save_path.rsplit("/")
            save_path = os.path.join("/".join(parts[:-1]), parts[-1] + "_fake_quant_model.pt")
            print_info(f"Start save QAT fake ckpt to: {save_path}")

            cpu_state = self.trainer.external_trainer.model.state_dict()
            torch.save(cpu_state, save_path)

        elif self.save_fmt == "real":
            save_func = self.quant_model.get_save_func()(self.quant_model)
            save_func.save(os.path.join(save_path, "final_quant_checkpoint"))

        else:
            print_info("Save format not specified, skip save.")
