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
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerTextExperts,
    Qwen3OmniMoeTalkerTextTopKRouter,
    Qwen3OmniMoeThinkerTextExperts,
    Qwen3OmniMoeThinkerTextTopKRouter,
)

from ...compressor.quant.core import PTQVLMSaveVllmHF
from ...utils import find_layers, find_parent_layer_and_sub_name, print_info
from ..base_model import BaseLLMModel
from ..llm.qwen import QwenMoeExpertsWithLinear
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
        self.thinker_block_name = "thinker.model.layers"
        self.talker_block_name = "talker.model.layers"
        self.observer_layer_classes = [
            torch.nn.Linear,
            Qwen3OmniMoeThinkerTextTopKRouter,
            Qwen3OmniMoeTalkerTextTopKRouter,
        ]
        self.observed_names = [
            "k_proj",
            "v_proj",
            "q_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    def replace_moe(self):
        for name, module in self.model.thinker.named_modules():
            if isinstance(module, Qwen3OmniMoeThinkerTextExperts) and not isinstance(
                module, QwenMoeExpertsWithLinear
            ):
                print(name)
                parent_layer, sub_name = find_parent_layer_and_sub_name(self.model.thinker, name)
                moe_linear = QwenMoeExpertsWithLinear(module)
                del module
                setattr(parent_layer, sub_name, moe_linear)

        for name, module in self.model.talker.named_modules():
            if isinstance(module, Qwen3OmniMoeTalkerTextExperts) and not isinstance(
                module, QwenMoeExpertsWithLinear
            ):
                print(name)
                parent_layer, sub_name = find_parent_layer_and_sub_name(self.model.talker, name)
                moe_linear = QwenMoeExpertsWithLinear(module)
                del module
                setattr(parent_layer, sub_name, moe_linear)

    def _patch_inputs_embeds_generate_device(self, module):
        if module is None or getattr(module, "_angelslim_generate_device_patch", False):
            return

        original_generate = module.generate
        skip_keys = {"past_key_values", "encoder_outputs"}

        def move_to_target_device(value, target_device):
            if isinstance(value, torch.Tensor):
                if value.device.type == "meta" or value.device == target_device:
                    return value
                return value.to(target_device)
            if isinstance(value, tuple):
                return tuple(move_to_target_device(item, target_device) for item in value)
            if isinstance(value, list):
                return [move_to_target_device(item, target_device) for item in value]
            if isinstance(value, dict):
                return {
                    key: item if key in skip_keys else move_to_target_device(item, target_device)
                    for key, item in value.items()
                }
            return value

        def generate_on_module_device(*args, **kwargs):
            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is not None:
                target_device = getattr(module, "device", inputs_embeds.device)
                if target_device.type == "meta":
                    target_device = inputs_embeds.device

                kwargs = {
                    key: value if key in skip_keys else move_to_target_device(value, target_device)
                    for key, value in kwargs.items()
                }

            return original_generate(*args, **kwargs)

        module.generate = generate_on_module_device
        module._angelslim_generate_device_patch = True

    def _patch_omni_generate_devices(self):
        talker = getattr(self.model, "talker", None)
        self._patch_inputs_embeds_generate_device(talker)
        self._patch_inputs_embeds_generate_device(getattr(talker, "code_predictor", None))

    def init_ptq(self, slim_config):
        super().init_ptq(slim_config)
        self.replace_moe()

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
        self._patch_omni_generate_devices()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

    def _get_quant_block_names(self):
        block_names = [self.thinker_block_name]
        if getattr(self.quant_config, "quant_talker", True):
            block_names.append(self.talker_block_name)
        return block_names

    def get_observer_layers(self):
        observer_layers_dict = {}
        layers_dict = find_layers(self.model, layers=self.observer_layer_classes)
        block_names = self._get_quant_block_names()

        ignore_layers = self.skip_layer_names()
        for name, module in layers_dict.items():
            block_condition = any(name.startswith(block) for block in block_names)
            if block_condition and name.split(".")[-1] in self.observed_names:
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
        block_names = self._get_quant_block_names()
        return [
            k
            for k in observe_names
            if any(k.startswith(block) for block in block_names)
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
                for batch in tqdm(dataloader, desc="calibrating...", total=len(dataloader)):
                    try:
                        text_ids, audio = self.model.generate(
                            **batch, use_audio_in_video=self.use_audio_in_video
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
