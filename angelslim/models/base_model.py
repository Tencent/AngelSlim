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
from abc import ABCMeta, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..compressor.quant.core import QuantConfig
from ..compressor.quant.modules import QDQModule
from ..utils import common_prefix, print_info

__all__ = ["BaseLLMModel", "BaseDiffusionModel"]


class BaseLLMModel(metaclass=ABCMeta):
    """
    Base class for model compression, providing common functionalities
    such as initialization, quantization configuration, and model handling.
    Args:
        model (torch.nn.Module, optional): the model to be compressed.
            If not provided, the model will be built from `model_path`.
        deploy_backend (str, optional): deploy_backend for model compression,
            currently only supports "vllm".
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        deploy_backend: Optional[str] = "vllm",
    ):
        assert deploy_backend in [
            "vllm",
            "huggingface",
            "trtllm",
        ], f"Unsupported deploy backend {deploy_backend}"
        self.deploy_backend = deploy_backend
        self.model = model
        self.tokenizer = None
        self.modal_type = "LLM"
        self.pre_transformer_module_names = ["model.embed_tokens"]

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
        self.model = AutoModelForCausalLM.from_pretrained(
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

    def init_ptq(self, slim_config):
        """
        Initialize the model for post-training quantization (PTQ).
        Args:
            slim_config(dict, required): the configuration for quantization.
                - compress_config: the configuration for compression.
                - global_config: the global configuration for the model.
        """
        quant_config = QuantConfig(
            slim_config["compress_config"], slim_config["global_config"]
        )
        self.quant_config = quant_config
        self.act_scales_dict = {}
        self.weight_scales_dict = {}
        self.kv_cache_scales_dict = {}
        if hasattr(self.quant_config, "weight_observer"):
            self.quant_algo_dict = self.get_quant_config()
        else:
            self.quant_algo_dict = None
        self.quantized = False

    @abstractmethod
    def get_observer_layers(self):
        pass

    @abstractmethod
    def get_save_func(self):
        pass

    def skip_layer_names(self):
        return self.quant_config.quant_algo_info.get("ignore_layers", [])

    def get_model(self):
        return self.model

    def get_qdq_module(self, sub_layer, name):
        act_scale, weight_scale = None, None
        if name in self.act_scales_dict:
            act_scale = self.act_scales_dict[name]
        if name in self.weight_scales_dict:
            weight_scale = self.weight_scales_dict[name]
        if self.deploy_backend in ["vllm", "huggingface"]:
            q_linear = QDQModule(
                quant_algo=self.quant_config.quant_algo,
                weight=sub_layer.weight,
                weight_scale=weight_scale,
                bias=sub_layer.bias,
                input_scale=act_scale,
            )
        else:
            print_info(
                "[Slim] current {} deploy_backend not support".format(
                    self.deploy_backend
                )
            )
            raise NotImplementedError
        return q_linear

    def get_kvcache_observer_layers_names(self, observe_names):
        names = ["self_attn.k_proj", "self_attn.v_proj"]
        return [
            k
            for k in observe_names
            if k.startswith(self.block_name)
            and k.split(".")[-2] + "." + k.split(".")[-1] in names
        ]

    def get_quant_config(self):
        assert self.quant_config is not None
        kv_cache_observer = self.quant_config.kv_cache_observer
        act_observer = self.quant_config.act_observer
        weight_observer = self.quant_config.weight_observer

        if hasattr(self.quant_config, "smooth_observer"):
            smooth_observer = self.quant_config.smooth_observer
        else:
            smooth_observer = None

        # assert isinstance(self.quant_config.quant_algo, dict)
        w = self.quant_config.quant_algo_info.get("w", None)
        a = self.quant_config.quant_algo_info.get("a", None)
        c = self.quant_config.quant_algo_info.get("c", None)

        w_group_size = self.quant_config.quant_algo_info.get("w_group_size", -1)

        a_quant_algo = a.split("_")[0] if a is not None else None
        w_quant_algo = w.split("_")[0] if w is not None else None
        c_quant_algo = c.split("_")[0] if c is not None else None
        a_quant_bits = (
            int(re.search(r"\d+", a_quant_algo).group())
            if a_quant_algo is not None
            else None
        )
        w_quant_bits = (
            int(re.search(r"\d+", w_quant_algo).group())
            if w_quant_algo is not None
            else None
        )
        c_quant_bits = (
            int(re.search(r"\d+", c_quant_algo).group())
            if c_quant_algo is not None
            else None
        )
        a_quant_method = a.split("_")[1] if a is not None else None
        w_quant_method = w.split("_")[1] if w is not None else None
        c_quant_method = c.split("_")[1] if c is not None else None

        custom_observe_layers_names = self.quant_config.custom_observe_layers_names

        quant_algo_dict = {
            "act_observer": act_observer,
            "weight_observer": weight_observer,
            "kv_cache_observer": kv_cache_observer,
            "smooth_observer": smooth_observer,
            "a_quant_algo": a_quant_algo,
            "w_quant_algo": w_quant_algo,
            "c_quant_algo": c_quant_algo,
            "a_quant_bits": a_quant_bits,
            "w_quant_bits": w_quant_bits,
            "c_quant_bits": c_quant_bits,
            "w_group_size": w_group_size,
            "a_quant_method": a_quant_method,
            "w_quant_method": w_quant_method,
            "c_quant_method": c_quant_method,
            "all_reduce": self.is_all_reduce(),
            "custom_observe_layers_names": custom_observe_layers_names,
        }
        return quant_algo_dict

    def is_all_reduce(self):
        return False

    def build_hf_model(self, model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            use_flash_attention_2=True,
            trust_remote_code=True,
        )
        return model

    def find_layers(self, module, layers=None, name=""):
        if type(module) in layers and name not in self.skip_layer_names():
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(
                self.find_layers(
                    child,
                    layers=layers,
                    name=name + "." + name1 if name != "" else name1,
                )
            )
        return res

    def get_pre_transformer_modules(self):
        pre_transformer_modules_dict = {}
        for full_name in self.pre_transformer_module_names:
            current_module = self.model
            parts = full_name.split(".")
            for part in parts:
                if not hasattr(current_module, part):
                    current_module = None
                    break
                current_module = getattr(current_module, part)
            if current_module is not None:
                pre_transformer_modules_dict[full_name] = current_module
        return pre_transformer_modules_dict

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

        if dataloader is not None:
            with torch.no_grad():
                for batch in tqdm(
                    dataloader, desc="calibrating...", total=len(dataloader)
                ):
                    inputs = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    try:
                        outputs = self.model(inputs)
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

    def get_smooth_mapping_layers(self, smooth_config, mappings=None):
        assert mappings is not None, "mappings must be provided"
        smooth_mapping_layers = {}

        for to_balance_list, to_smooth in mappings:
            for smooth_name, smooth_layer in self.model.named_modules():
                if smooth_name.split(".")[-1] != to_smooth:
                    continue
                balance_layers_list = []

                for to_balance in to_balance_list:
                    longest_prefix = 0
                    balance_layers = []

                    # use common_prefix to support moe experts
                    for name, layer in self.model.named_modules():
                        if name.split(".")[-1] != to_balance:
                            continue
                        prefix = common_prefix(name, smooth_name)
                        if prefix.count(".") < longest_prefix:
                            continue
                        elif prefix.count(".") == longest_prefix:
                            balance_layers.append((name, layer))
                        else:
                            longest_prefix = prefix.count(".")
                            balance_layers = [(name, layer)]

                    if balance_layers:
                        balance_layers_list.extend(balance_layers)

                if balance_layers_list:
                    smooth_mapping_layers[smooth_name] = (
                        smooth_layer,
                        balance_layers_list,
                    )

        return smooth_mapping_layers

    def get_parent_dict(self, observer_layers_dict):
        return {}

    def get_weight_scales(self, layer, weight_observer):
        weight = layer.weight.clone().detach()
        weight_observer(weight)
        return weight_observer.scales()

    def __getattr__(self, item):
        return super().__getattr__(item)


class BaseDiffusionModel(metaclass=ABCMeta):
    """
    Base class for diffusion model compression, providing common functionalities
    such as initialization, quantization configuration, and model handling.
    Args:
        model (torch.nn.Module, optional): the model to be compressed.
            If not provided, the model will be built from `model_path`.
        deploy_backend (str, optional): deploy_backend for model compression.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        deploy_backend: Optional[str] = "torch",
    ):
        assert deploy_backend in [
            "torch",
            "tensorrt",
        ], f"Unsupported deploy backend {deploy_backend}"
        self.deploy_backend = deploy_backend
        self.model = model
        self.modal_type = "Diffusion"

    @staticmethod
    def from_pretrained(self, model_path, **kwargs):
        """
        Load a pretrained diffusion model.
        Args:
            model_path (str): Path to the pretrained model.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def get_observer_layers(self):
        pass

    @abstractmethod
    def get_save_func(self):
        pass
