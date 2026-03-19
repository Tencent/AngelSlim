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
import gc

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeExperts,
    Qwen3_5MoeTopKRouter,
)

from ...compressor.quant.core import LossFilter, PTQVLMSaveVllmHF
from ...utils import print_info
from ...utils.utils import find_layers, find_parent_layer_and_sub_name
from ..llm.qwen import Qwen, QwenMoeExpertsWithLinear
from ..model_factory import SlimModelFactory


@SlimModelFactory.register
class Qwen3_5(Qwen):
    def __init__(
        self,
        model=None,
        deploy_backend="vllm",
    ):
        super().__init__(
            model=model,
            deploy_backend=deploy_backend,
        )
        self.modal_type = "VLM"
        self.block_name = "model.language_model.layers"
        self.vit_block_name = "model.visual.blocks"
        self.pre_transformer_module_names = [
            "visual",
            "language_model.embed_tokens",
            "language_model.norm",
            "language_model.rotary_emb",
        ]

        self.observer_layer_classes = [torch.nn.Linear, Qwen3_5MoeTopKRouter]
        self.observed_names = [
            "k_proj",
            "v_proj",
            "q_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "in_proj_qkv",
            "out_proj",
        ]

    def replace_moe(self):
        for name, module in self.model.named_modules():
            if isinstance(module, Qwen3_5MoeExperts) and not isinstance(
                module, QwenMoeExpertsWithLinear
            ):
                print(name)
                parent_layer, sub_name = find_parent_layer_and_sub_name(self.model, name)
                moe_linear = QwenMoeExpertsWithLinear(module)
                del module
                setattr(parent_layer, sub_name, moe_linear)

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
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

    def get_observer_layers(self):

        if hasattr(self.quant_config, "quant_vit") and self.quant_config.quant_vit:
            vit_names = ["attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"]
            self.observed_names.extend(vit_names)

        observer_layers_dict = {}
        layers_dict = find_layers(self.model, layers=self.observer_layer_classes)
        ignore_layers = self.skip_layer_names()
        for name, module in layers_dict.items():
            block_condition = name.startswith(self.block_name) or (
                hasattr(self.quant_config, "quant_vit")
                and self.quant_config.quant_vit
                and name.startswith(self.vit_block_name)
            )
            result = name.split(".")[-1]
            if block_condition and result in self.observed_names:
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

    def model_forward(self, dataloader, **kwargs):
        self.model.use_cache = False

        calibrated_cnt = 0
        if (
            "gptq" in self.quant_config.quant_algo
            or "awq" in self.quant_config.quant_algo
            or "gptaq" in self.quant_config.quant_algo
        ):
            device = "cuda:0"
        else:
            device = self.model.device
        print_info(f"device is {device}")
        if dataloader is not None:
            loss_filter = LossFilter(processor=self.processor)
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="calibrating...", total=len(dataloader)):
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    inputs["use_cache"] = False
                    labels = batch["labels"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    try:
                        outputs = self.model(**inputs)
                        logits = outputs.logits.float()

                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                            reduction="none",
                        )

                        attention_mask = attention_mask.view(-1).to(logits.device).float()
                        loss = loss * attention_mask
                        loss = loss_filter.filter_loss(
                            loss=loss, labels=labels, model_type="Qwen3_5"
                        )
                        avg_loss = loss.mean()
                        ppl = torch.exp(avg_loss)

                        print_info(f"ppl is : {ppl:.4f}")

                        calibrated_cnt += 1
                    except ValueError:
                        calibrated_cnt += 1
                        pass
                    inputs = {
                        k: v.to("cpu") if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }
                    attention_mask = attention_mask.to("cpu")
                    labels = labels.to("cpu")
                    del outputs, inputs
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()

    def get_quant_module(self):
        """
        Returns the module that will be quantized.
        This is typically the main transformer module of the model.
        """
        return self.model.model.language_model.layers

    def get_save_func(self):
        if self.deploy_backend in ["vllm", "huggingface"]:
            return PTQVLMSaveVllmHF
        else:
            raise NotImplementedError(
                f"deploy_backend {self.deploy_backend} is not supported for saving."
            )
