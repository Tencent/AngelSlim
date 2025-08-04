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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from ...compressor.quant.core import PTQVLMSaveVllmHF
from ...utils import print_info
from ..base_model import BaseLLMModel
from ..model_factory import SlimModelFactory


@SlimModelFactory.register
class QwenVL(BaseLLMModel):
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

    def from_pretrained(
        self,
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
    ):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_cache=use_cache,
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
        names = [
            "k_proj",
            "v_proj",
            "q_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]

        if hasattr(self.quant_config, "quant_vit") and self.quant_config.quant_vit:
            vit_names = ["qkv", "proj"]
            names.extend(vit_names)

        obs_layers = [nn.Linear]
        observer_layers_dict = {}
        layers_dict = self.find_layers(self.model, layers=obs_layers)

        ignore_layers = self.skip_layer_names()
        for name, module in layers_dict.items():
            block_condition = name.startswith(self.block_name) or (
                hasattr(self.quant_config, "quant_vit")
                and self.quant_config.quant_vit
                and name.startswith(self.vit_block_name)
            )
            if block_condition and name.split(".")[-1] in names:
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
        ):
            device = "cuda:0"
        else:
            device = self.model.device
        print_info(f"device is {device}")
        if dataloader is not None:
            with torch.no_grad():
                for batch in tqdm(
                    dataloader, desc="calibrating...", total=len(dataloader)
                ):
                    inputs = {
                        "input_ids": batch["input_ids"].to(device),
                        "attention_mask": batch["attention_mask"].to(device),
                        "pixel_values": batch["pixel_values"].to(device),
                        "image_grid_thw": batch["image_grid_thw"].to(device),
                    }
                    inputs = {k: v.to(device) for k, v in inputs.items()}
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
        if self.deploy_backend in ["vllm", "huggingface"]:
            return PTQVLMSaveVllmHF
        else:
            raise NotImplementedError(
                f"deploy_backend {self.deploy_backend} is not supported for saving."
            )
