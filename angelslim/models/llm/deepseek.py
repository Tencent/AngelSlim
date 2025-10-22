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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.deepseek_v3 import DeepseekV3Config

from ...compressor.quant.core import (
    DeepSeekV3PTQSaveMulti,
    DeepSeekV3PTQSaveSingle,
    weight_dequant,
)
from ...compressor.quant.modules import QDQModule
from ...utils import find_layers, print_info
from ..base_model import BaseLLMModel
from ..model_factory import SlimModelFactory
from .modeling_deepseek import (
    ColumnParallelLinear,
    DeepseekV3ForCausalLM,
    Linear,
    RowParallelLinear,
)


@SlimModelFactory.register
class DeepSeek(BaseLLMModel):
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
        self.column_parallel_linear_class = ColumnParallelLinear
        self.row_parallel_linear_class = RowParallelLinear
        self.observer_layer_classes = [
            nn.Linear,
            Linear,
            ColumnParallelLinear,
            RowParallelLinear,
        ]
        torch.set_default_dtype(torch.bfloat16)

    def from_pretrained(
        self,
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
        using_multi_nodes=False,
    ):
        if torch_dtype == "fp8":
            print_info("[Slim] Loading DeepSeek with fp8")
            config = DeepseekV3Config.from_pretrained(model_path)
            if hasattr(config, "quantization_config"):
                delattr(config, "quantization_config")
            if hasattr(config, "use_cache"):
                config.use_cache = use_cache
            self.model = DeepseekV3ForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype="auto",
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=low_cpu_mem_usage,
                using_multi_nodes=using_multi_nodes,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=low_cpu_mem_usage,
                use_cache=use_cache,
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

    def get_observer_layers(self):
        names = self.quant_config.quant_algo_info["ignore_layers"]
        observer_layers_dict = find_layers(
            self.model, layers=self.observer_layer_classes
        )
        observer_layers_dict = {
            k: v
            for k, v in observer_layers_dict.items()
            if k.startswith(self.block_name) and not any(name in k for name in names)
        }
        if self.quant_config.custom_observe_layers_names != "default":
            for custom_observe_name in self.quant_config.custom_observe_layers_names:
                for default_name in observer_layers_dict.keys():
                    if custom_observe_name not in default_name:
                        observer_layers_dict.pop(default_name)
        return observer_layers_dict

    def get_weight_scales(self, layer, weight_observer):
        assert layer.weight.dtype in [torch.bfloat16, torch.float8_e4m3fn]
        if layer.weight.dtype == torch.bfloat16:
            weight = layer.weight.clone().detach()
            weight_observer(weight)
            return weight_observer.scales()
        else:
            if not self.model.using_multi_nodes:
                layer = layer.to("cuda")
                weight = weight_dequant(layer.weight, layer.weight_scale_inv)
                weight = weight.to("cpu")
                layer = layer.to("cpu")
                weight_observer(weight)
                del weight
                return weight_observer.scales()
            else:
                weight_bf16 = weight_dequant(layer.weight, layer.weight_scale_inv)
                weight_observer(weight_bf16)
                scales = weight_observer.scales()
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                if isinstance(layer, self.column_parallel_linear_class):
                    all_scales = [torch.empty_like(scales) for _ in range(world_size)]
                    dist.all_gather(all_scales, scales)
                    merged_scales = torch.cat(all_scales, dim=0)
                    return merged_scales
                elif isinstance(layer, self.row_parallel_linear_class):
                    all_scales = [torch.empty_like(scales) for _ in range(world_size)]
                    dist.all_gather(all_scales, scales)
                    merged_scales = torch.cat(all_scales, dim=-1)
                    return merged_scales
                else:
                    return scales

    def get_qdq_module(self, layer, name):
        if self.model.using_multi_nodes:
            return layer
        act_scale, weight_scale = None, None
        if name in self.act_scales_dict:
            act_scale = self.act_scales_dict[name]
        if name in self.weight_scales_dict:
            weight_scale = self.weight_scales_dict[name]

        if layer.weight.dtype == torch.bfloat16:
            weight = layer.weight.clone().detach()
        elif layer.weight.dtype == torch.float8_e4m3fn:
            layer = layer.to("cuda")
            weight = weight_dequant(layer.weight, layer.weight_scale_inv)
            weight = weight.to("cpu")
            layer = layer.to("cpu")
        q_linear = QDQModule(
            quant_algo=self.quant_config.quant_algo,
            weight=weight,
            weight_scale=weight_scale,
            group_size=self.quant_config.quant_algo_info.get("w_group_size", 128),
            bias=layer.bias,
            input_scale=act_scale,
        )
        del weight
        return q_linear

    def model_forward(self, dataloader, **kwargs):
        if self.quant_config.low_memory:
            device = "cpu"
        else:
            device = torch.cuda.current_device()
        calibrated_cnt = 0
        if dataloader is not None:
            with torch.no_grad():
                for batch in tqdm(
                    dataloader, desc="calibrating...", total=len(dataloader)
                ):
                    inputs = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    try:
                        _, logits = self.model(inputs)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                            reduction="none",
                        )
                        attention_mask = (
                            attention_mask.view(-1).to(logits.device).float()
                        )
                        loss = loss * attention_mask
                        avg_loss = loss.mean()
                        ppl = torch.exp(avg_loss)

                        print_info(f"ppl is : {ppl:.4f}")

                        calibrated_cnt += 1
                    except ValueError:
                        calibrated_cnt += 1
                        pass

    def get_save_func(self):
        if self.deploy_backend in ["vllm", "trtllm"]:
            if self.model.using_multi_nodes:
                return DeepSeekV3PTQSaveMulti
            return DeepSeekV3PTQSaveSingle
        else:
            raise NotImplementedError(
                f"deploy_backend {self.deploy_backend} is not supported for saving."
            )
