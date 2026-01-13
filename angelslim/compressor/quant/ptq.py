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

import json
import os
import warnings

import torch
from safetensors.torch import load_file
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

from angelslim.compressor.quant.core.quant_func import (
    get_fp_maxval,
    quantize_weight_per_tensor_fp8,
)

from ...utils import find_parent_layer_and_sub_name, print_info
from ..compressor_factory import CompressorFactory
from .core import PTQHook
from .modules import AWQ, FP8, GPTQ, INT8, NVFP4, LeptoFP8, SmoothQuant

__all__ = ["PTQ"]


@CompressorFactory.register
class PTQ:
    def __init__(self, model, slim_config=None):
        """
        Args:
            model(nn.Moudle, required): the model to be quant.
            slim_config(dict, required): the configuration for quantization.
                - compress_config: the configuration for compression.
                - global_config: the global configuration for the model.
        """
        self.quant_model = model
        # init ptq config of model
        self.quant_model.init_ptq(slim_config)
        self.absolute_model_path = slim_config["global_config"].absolute_model_path
        self.quant_algo = self.quant_model.quant_config.quant_algo
        self.quant_helpers = self.quant_model.quant_config.quant_helpers
        if (
            "fp8" in self.quant_algo
            or "int8" in self.quant_algo
            or "nvfp4" in self.quant_algo
        ):
            # Add ptq observer hook
            self.ptq_hook = PTQHook(self.quant_model)
            self.ptq_hook.apply_hook()

        if "gptq" in self.quant_algo or "gptaq" in self.quant_algo:
            max_seq_length = self.quant_model.quant_config.max_seq_length
            hidden_size = self.quant_model.quant_config.hidden_size
            self.gptq = GPTQ(
                self.quant_model, seq_length=max_seq_length, hidden_size=hidden_size
            )
        elif "awq" in self.quant_algo:
            max_seq_length = self.quant_model.quant_config.max_seq_length
            hidden_size = self.quant_model.quant_config.hidden_size
            model_arch_type = self.quant_model.quant_config.model_arch_type
            self.awq = AWQ(
                self.quant_model,
                seq_length=max_seq_length,
                hidden_size=hidden_size,
                model_arch_type=model_arch_type,
                mse_range=self.quant_model.quant_config.quant_algo_info["mse_range"],
                observer_layer_classes=self.quant_model.observer_layer_classes,
                low_memory=self.quant_model.quant_config.low_memory,
            )
        elif "fp8" in self.quant_algo:
            max_seq_length = self.quant_model.quant_config.max_seq_length
            hidden_size = self.quant_model.quant_config.hidden_size
            model_arch_type = self.quant_model.quant_config.model_arch_type
            if "lepto" in self.quant_algo:
                self.fp8 = LeptoFP8(
                    self.ptq_hook,
                    self.quant_model,
                    seq_length=max_seq_length,
                    hidden_size=hidden_size,
                    model_arch_type=model_arch_type,
                    low_memory=self.quant_model.quant_config.low_memory,
                )
            else:
                self.fp8 = FP8(
                    self.quant_model,
                    seq_length=max_seq_length,
                    hidden_size=hidden_size,
                    model_arch_type=model_arch_type,
                    low_memory=self.quant_model.quant_config.low_memory,
                )
        elif "int8" in self.quant_algo:
            max_seq_length = self.quant_model.quant_config.max_seq_length
            hidden_size = self.quant_model.quant_config.hidden_size
            model_arch_type = self.quant_model.quant_config.model_arch_type
            self.int8 = INT8(
                self.quant_model,
                seq_length=max_seq_length,
                hidden_size=hidden_size,
                model_arch_type=model_arch_type,
                low_memory=self.quant_model.quant_config.low_memory,
            )
        elif "nvfp4" in self.quant_algo:
            self.nvfp4 = NVFP4(self.quant_model)
        else:
            raise NotImplementedError(
                f"[AngelSlim Error] algo {self.quant_algo} is not support"
            )

        if "smooth" in self.quant_helpers:
            self.smooth = SmoothQuant(
                self.quant_model,
                self.ptq_hook,
                alpha=self.quant_model.quant_config.smooth_alpha,
            )

    def calibrate(self, dataloader):
        if "gptq" in self.quant_algo or "gptaq" in self.quant_algo:
            self.gptq.run(dataloader)
        elif "awq" in self.quant_algo:
            self.awq.run(dataloader)
        elif "fp8" in self.quant_algo:
            self.fp8.run(dataloader)
        elif "int8" in self.quant_algo:
            self.int8.run(dataloader)
        elif "nvfp4" in self.quant_algo:
            self.nvfp4.run(dataloader)
        else:
            raise AssertionError(
                f"[AngelSlim Error] algo {self.quant_algo} is not support calibrate"
            )

    def convert(self):
        """
        Saves scales and inserts QDQ modules.
        """
        print_info("Start convert model...")
        if "gptq" in self.quant_algo or "gptaq" in self.quant_algo:
            self.gptq.convert()
        elif "awq" in self.quant_algo:
            self.awq.convert()
        elif "lepto" in self.quant_algo:
            self.fp8.convert()
        else:
            if "smooth" in self.quant_helpers:
                self.smooth.convert()
            self._convert()
        print_info("convert model done.")

    def save(self, save_path: str):
        """
        Save PTQ scales or ckpt.
        """
        if (
            hasattr(self.quant_model.quant_config, "quant_analyse")
            and self.quant_model.quant_config.quant_analyse
        ):
            # scale analyse
            for k in self.quant_model.act_scales_dict.keys():
                act_scales_data = self.quant_model.act_scales_dict[k].data
                if act_scales_data > 1.5:
                    print_info(
                        f"[AngelSlim Warning] Act_scales {k}: "
                        f"The weight is too high:{act_scales_data}. "
                        f"It is recommended to clip it to 1.5 "
                    )
            for k in self.quant_model.weight_scales_dict.keys():
                weight_scales_data = self.quant_model.weight_scales_dict[k].data
                if weight_scales_data > 1.5:
                    print_info(
                        f"[AngelSlim Warning] Weight_scales {k}: "
                        f"The weight is too high:{weight_scales_data}. "
                        f"It is recommended to clip it to 1.5 "
                    )

        print_info("Start save PTQ ckpt to: {}".format(save_path))
        if "gptq" in self.quant_algo or "gptaq" in self.quant_algo:
            self.gptq.save(save_path)
        elif "awq" in self.quant_algo:
            self.awq.save(save_path)
        else:
            save_func = self.quant_model.get_save_func()(self.quant_model)
            save_func.save(save_path)

    def _convert(self):
        # 1. get act, weight and kv-cache scale
        for name, sub_layer in self.ptq_hook.quant_layers_dict.items():
            if (
                getattr(  # noqa: B009
                    self.ptq_hook.observer_dict[sub_layer], "act_observer"
                )
                is not None
            ):
                try:
                    self.quant_model.act_scales_dict[name] = (
                        self.ptq_hook.observer_dict[sub_layer].act_observer.scales()
                    )
                except ValueError:
                    self.quant_model.act_scales_dict[name] = torch.tensor(
                        1.0, device=torch.cuda.current_device()
                    )
                    warnings.warn(
                        f"Not calibrated for {name}. Using default act scale 1.0.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if (
                getattr(  # noqa: B009
                    self.ptq_hook.observer_dict[sub_layer], "kv_cache_observer"
                )
                is not None
            ):
                self.quant_model.kv_cache_scales_dict[name] = (
                    self.ptq_hook.observer_dict[sub_layer].kv_cache_observer.scales()
                )
            if (
                getattr(  # noqa: B009
                    self.ptq_hook.observer_dict[sub_layer], "weight_observer"
                )
                is not None
            ):
                if sub_layer.weight.device.type == "meta":
                    with open(
                        os.path.join(
                            self.absolute_model_path, "model.safetensors.index.json"
                        ),
                        "r",
                    ) as f:
                        model_index = json.load(f)
                    orign_w_file = os.path.join(
                        self.absolute_model_path,
                        model_index["weight_map"][name + ".weight"],
                    )
                    orign_w = load_file(orign_w_file, device="cpu")
                    print_info(f"Load meta weight {name} from file {orign_w_file}")
                    sub_layer.to_empty(device="cpu")
                    sub_layer.weight.data = orign_w[name + ".weight"]

                    if hasattr(sub_layer, "bias"):
                        if (name + ".bias") in model_index["weight_map"]:
                            orign_b_file = os.path.join(
                                self.absolute_model_path,
                                model_index["weight_map"][name + ".bias"],
                            )
                            orign_b = load_file(orign_b_file, device="cpu")
                            print_info(
                                f"Load meta bias {name} from file {orign_b_file}"
                            )
                            sub_layer.bias.data = orign_b[name + ".bias"]
                        else:
                            print_info(f"{name + '.bias'} not found. Set bias to None.")
                            sub_layer.bias = None

                weight_scales = self.quant_model.get_weight_scales(
                    sub_layer, self.ptq_hook.observer_dict[sub_layer].weight_observer
                )
                self.quant_model.weight_scales_dict[name] = weight_scales

        self.ptq_hook.remove_hook()
        torch.cuda.empty_cache()

        self.ptq_hook.post_process()

        quant_convert_module = self.quant_model.get_quant_convert_module()
        if "nvfp4" in self.quant_algo:
            self.quant_model.get_observer_values()
        # 2. insert qdq module
        for name, sub_layer in self.ptq_hook.quant_layers_dict.items():
            parent_layer, sub_name = find_parent_layer_and_sub_name(
                quant_convert_module, name
            )

            if self.quant_model.quant_config.cpu_convert:
                sub_layer = sub_layer.to("cpu")
                print_info(f"Convert layer {name} on cpu")
            if "nvfp4" in self.quant_algo:
                self.nvfp4.post_process(sub_layer, name)
                qdq_module = self.quant_model.get_nvfp4_qdq_module(sub_layer, name)
            else:
                qdq_module = self.quant_model.get_qdq_module(sub_layer, name)

            if qdq_module is not sub_layer:
                setattr(parent_layer, sub_name, qdq_module)

        maxval = get_fp_maxval(bits=8)
        for name, sub_layer in self.quant_model.model.named_modules():
            if isinstance(sub_layer, Qwen3VLMoeTextExperts):
                parent_layer, sub_name = find_parent_layer_and_sub_name(
                    quant_convert_module, name
                )
                gate_up_act_max = sub_layer.gateupobservers.scales()
                down_act_max = sub_layer.downobservers.scales()
                gate_up_act_dtype = gate_up_act_max.dtype
                down_act_dtype = down_act_max.dtype
                gate_up_act_scale = gate_up_act_max / maxval.type(gate_up_act_dtype)
                down_act_scale = down_act_max / maxval.type(down_act_dtype)

                gate_proj, up_proj = sub_layer.gate_up_proj.chunk(2, dim=-1)
                abs_inputs = torch.abs(gate_proj)
                batch_size = abs_inputs.shape[0]
                abs_inputs_flat = abs_inputs.view(batch_size, -1)
                gate_weight_max, _ = torch.max(abs_inputs_flat, dim=1, keepdim=True)

                abs_inputs = torch.abs(up_proj)
                batch_size = abs_inputs.shape[0]
                abs_inputs_flat = abs_inputs.view(batch_size, -1)
                up_weight_max, _ = torch.max(abs_inputs_flat, dim=1, keepdim=True)

                abs_inputs = torch.abs(sub_layer.down_proj)
                batch_size = abs_inputs.shape[0]
                abs_inputs_flat = abs_inputs.view(batch_size, -1)
                down_weight_max, _ = torch.max(abs_inputs_flat, dim=1, keepdim=True)

                gate_weight_dtype = gate_proj.dtype
                up_weight_dtype = up_proj.dtype
                down_weight_dtype = sub_layer.down_proj.dtype
                gate_weight_scale = gate_weight_max / maxval.type(gate_weight_dtype)
                up_weight_scale = up_weight_max / maxval.type(up_weight_dtype)
                down_weight_scale = down_weight_max / maxval.type(down_weight_dtype)

                q_linear = MyQDQModule(
                    gate_proj=gate_proj.cpu(),
                    up_proj=up_proj.cpu(),
                    down_proj=sub_layer.down_proj.cpu(),
                    gate_proj_weight_scale=gate_weight_scale.cpu(),
                    up_proj_weight_scale=up_weight_scale.cpu(),
                    down_proj_weight_scale=down_weight_scale.cpu(),
                    gate_up_proj_input_scale=gate_up_act_scale.cpu(),
                    down_proj_input_scale=down_act_scale.cpu(),
                )
                setattr(parent_layer, sub_name, q_linear)
        self.quant_model.quantized = True

    def __getattr__(self, item):
        return super().__getattr__(item)


class MyQDQModule(torch.nn.Module):
    def __init__(
        self,
        gate_proj: torch.nn.Parameter,
        up_proj: torch.nn.Parameter,
        down_proj: torch.nn.Parameter,
        gate_proj_weight_scale: torch.nn.Parameter,
        up_proj_weight_scale: torch.nn.Parameter,
        down_proj_weight_scale: torch.nn.Parameter,
        gate_up_proj_input_scale: torch.nn.Parameter,
        down_proj_input_scale: torch.nn.Parameter,
    ):
        super().__init__()
        quant_gate_weight, _ = quantize_weight_per_tensor_fp8(
            gate_proj, gate_proj_weight_scale
        )
        quant_up_weight, _ = quantize_weight_per_tensor_fp8(
            up_proj, up_proj_weight_scale
        )
        quant_down_weight, _ = quantize_weight_per_tensor_fp8(
            down_proj, down_proj_weight_scale
        )
        quant_gate_up_weight = torch.cat([quant_gate_weight, quant_up_weight], dim=-1)

        self.gate_up_proj = torch.nn.Parameter(
            quant_gate_up_weight, requires_grad=False
        )
        self.down_proj = torch.nn.Parameter(quant_down_weight, requires_grad=False)

        gate_proj_weight_scale = (
            gate_proj_weight_scale.view(-1)
            if gate_proj_weight_scale.ndim == 0
            else gate_proj_weight_scale
        )
        up_proj_weight_scale = (
            up_proj_weight_scale.view(-1)
            if up_proj_weight_scale.ndim == 0
            else up_proj_weight_scale
        )
        down_proj_weight_scale = (
            down_proj_weight_scale.view(-1)
            if down_proj_weight_scale.ndim == 0
            else down_proj_weight_scale
        )
        gate_up_proj_weight_scale = torch.cat(
            [gate_proj_weight_scale, up_proj_weight_scale], dim=-1
        )

        self.gate_up_proj_weight_scale = torch.nn.Parameter(
            gate_up_proj_weight_scale, requires_grad=False
        )
        self.down_proj_weight_scale = torch.nn.Parameter(
            down_proj_weight_scale, requires_grad=False
        )

        down_proj_input_scale = (
            down_proj_input_scale.view(-1)
            if down_proj_input_scale.ndim == 0
            else down_proj_input_scale.squeeze()
        )
        gate_up_proj_input_scale = (
            gate_up_proj_input_scale.view(-1)
            if gate_up_proj_input_scale.ndim == 0
            else gate_up_proj_input_scale.squeeze()
        )

        self.gate_up_proj_input_scale = torch.nn.Parameter(
            gate_up_proj_input_scale, requires_grad=False
        )
        self.down_proj_input_scale = torch.nn.Parameter(
            down_proj_input_scale, requires_grad=False
        )

    def forward(self, x):
        pass
