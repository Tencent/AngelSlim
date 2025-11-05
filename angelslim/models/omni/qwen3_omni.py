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
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3OmniMoeForConditionalGeneration,
)

from ...compressor.quant.core import PTQVLMSaveVllmHF
from ...utils import find_layers, print_info
from ..base_model import BaseLLMModel
from ..model_factory import SlimModelFactory


@SlimModelFactory.register
class Qwen_Omni(BaseLLMModel):
    def __init__(
        self,
        model=None,
        deploy_backend="vllm",
    ):
        super().__init__(
            model=model,
            deploy_backend=deploy_backend,
        )
        self.modal_type = "Omni"
        self.block_name = ["thinker.model.layers", "talker.model.layers"]

    def from_pretrained(
        self,
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        use_audio_in_video=False,
        attn_implementation="default",
    ):
        self.use_audio_in_video = use_audio_in_video
        if attn_implementation == "default":
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        else:
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
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

        observer_layers_dict = {}
        layers_dict = find_layers(self.model, layers=self.observer_layer_classes)

        ignore_layers = self.skip_layer_names()
        for name, module in layers_dict.items():
            block_condition = any(name.startswith(block) for block in self.block_name)
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

    def get_kvcache_observer_layers_names(self, observe_names):
        names = ["self_attn.k_proj", "self_attn.v_proj"]
        return [
            k
            for k in observe_names
            if any(k.startswith(block) for block in self.block_name)
            and k.split(".")[-2] + "." + k.split(".")[-1] in names
        ]

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
            with torch.no_grad():
                for batch in tqdm(
                    dataloader, desc="calibrating...", total=len(dataloader)
                ):
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    try:
                        text_ids, audio = self.model.generate(
                            **inputs, use_audio_in_video=self.use_audio_in_video
                        )
                        calibrated_cnt += 1
                    except ValueError:
                        calibrated_cnt += 1
                        pass

    def get_quant_module(self):
        """
        Returns the module that will be quantized.
        This is typically the main transformer module of the model.
        """
        return self.model.thinker.model.layers

    def get_save_func(self):
        if self.deploy_backend in ["vllm", "huggingface"]:
            return PTQVLMSaveVllmHF
        else:
            raise NotImplementedError(
                f"deploy_backend {self.deploy_backend} is not supported for saving."
            )
