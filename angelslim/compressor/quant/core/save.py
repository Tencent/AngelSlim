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
import json
import os
import re
import shutil
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from glob import glob
from typing import Dict

import torch
import torch.distributed as dist
from safetensors.torch import load_file, safe_open
from safetensors.torch import save_file as safe_save
from safetensors.torch import save_model
from tqdm import tqdm
from transformers.models.deepseek_v3 import DeepseekV3Config

from ....utils import print_info
from .packing_utils import pack_weight_to_int8
from .quant_func import Int8PerChannelQuantizer, fake_quant_dequant, weight_dequant

__all__ = ["PTQvLLMSaveHF"]


class PTQSaveBase(metaclass=ABCMeta):
    def __init__(self, quant_model):
        self.quant_model = quant_model

    @abstractmethod
    def save(self, save_path):
        pass


class PTQvLLMSaveHF(PTQSaveBase):
    def __init__(self, quant_model):
        super(PTQvLLMSaveHF, self).__init__(quant_model=quant_model.model)

    def save(self, save_path):
        """save quantized model and configs to local disk"""
        os.makedirs(save_path, exist_ok=True)

        state_dict = self.quant_model.state_dict()
        for name in list(state_dict.keys()):
            if "qweight" in name:
                pop_name = name.replace("qweight", "layer.weight")
                if pop_name in state_dict.keys():
                    state_dict.pop(pop_name)
                pop_name = name.replace("qweight", "layer.bias")
                if pop_name in state_dict.keys():
                    state_dict[name.replace("qweight", "bias")] = state_dict[pop_name]
                    state_dict.pop(pop_name)
        print_info("state_dict:{}".format(state_dict.keys()))
        model_base_name = "quant_model"
        model_save_name = model_base_name + ".safetensors"
        safetensors_metadata = {}
        safetensors_metadata["format"] = "pt"
        safe_save(
            state_dict, os.path.join(save_path, model_save_name), safetensors_metadata
        )
        self.quant_model.config.save_pretrained(save_path)


class PTQVLMSaveVllmHF(PTQSaveBase):
    def __init__(self, quant_model):
        super().__init__(quant_model=quant_model)

    def save(self, save_path):
        save_name = self.quant_model.quant_config.save_name
        ignore_field = (
            "ignore" if save_name == "compressed-tensors" else "ignored_layers"
        )

        w_quant_algo = self.quant_model.quant_config.quant_algo_info["w"]
        a_quant_algo = self.quant_model.quant_config.quant_algo_info["a"]
        is_dynamic = "dynamic" in a_quant_algo
        ignored_layers = self.quant_model.skip_layer_names()

        trtllm_config = {
            "quantization": {
                "exclude_modules": ignored_layers,
                "kv_cache_quant_algo": None,
            }
        }
        if "fp8" in self.quant_model.quant_config.quant_algo:
            quant_format = "naive-quantized"
            trtllm_config["quantization"]["quant_algo"] = "FP8"
            act_config = {
                "num_bits": 8,
                "strategy": re.search(r"per-([a-zA-Z]+)", a_quant_algo).group(1),
                "dynamic": is_dynamic,
                "type": "float",
            }
            weight_config = {
                "num_bits": 8,
                "strategy": re.search(r"per-([a-zA-Z]+)", w_quant_algo).group(1),
                "dynamic": False,
                "type": "float",
            }
        elif "int8" in self.quant_model.quant_config.quant_algo:
            quant_format = "int-quantized"
            trtllm_config["quantization"]["quant_algo"] = "INT8"
            act_config = {
                "num_bits": 8,
                "strategy": re.search(r"per-([a-zA-Z]+)", a_quant_algo).group(1),
                "dynamic": is_dynamic,
                "type": "int",
            }
            weight_config = {
                "num_bits": 8,
                "strategy": re.search(r"per-([a-zA-Z]+)", w_quant_algo).group(1),
                "dynamic": False,
                "type": "int",
            }
        elif "nvfp4" in self.quant_model.quant_config.quant_algo:
            quant_format = "naive-quantized"
            group_size = self.quant_model.quant_config.quant_algo_info["block_size"]
            trtllm_config["quantization"]["quant_algo"] = "NVFP4"
            trtllm_config["quantization"]["group_size"] = group_size
            act_config = {
                "num_bits": 4,
                "group_size": group_size,
                "dynamic": is_dynamic,
                "type": "float",
            }
            weight_config = {
                "num_bits": 4,
                "group_size": group_size,
                "dynamic": False,
                "type": "float",
            }
        else:
            raise ValueError(
                f"{self.quant_model.quant_config.quant_algo} not supported"
            )
        quantization_config = {"quant_method": save_name, ignore_field: ignored_layers}
        if save_name == "compressed-tensors":
            quantization_config.update(
                {
                    "config_groups": {
                        "group_0": {
                            "weights": weight_config,
                            "input_activations": act_config,
                            "output_activations": None,
                            "targets": ["Linear"],
                        }
                    },
                    "kv_cache_scheme": None,
                    "format": quant_format,
                    "quantization_status": "compressed",
                }
            )
        else:
            quantization_config["activation_scheme"] = (
                "dynamic" if is_dynamic else "static"
            )

        quant_dict = {"quantization_config": quantization_config}
        self.quant_model.get_model().config.update(quant_dict)
        print_info("Save quantization_config: {}".format(quant_dict))

        os.makedirs(save_path, exist_ok=True)

        self.quant_model.get_model().save_pretrained(save_path)
        self.quant_model.processor.save_pretrained(save_path)
        self.quant_model.tokenizer.save_pretrained(save_path)


class PTQSaveVllmHF(PTQSaveBase):
    def __init__(self, quant_model):
        super().__init__(quant_model=quant_model)

    def save(self, save_path):
        save_name = self.quant_model.quant_config.save_name
        ignore_field = (
            "ignore" if save_name == "compressed-tensors" else "ignored_layers"
        )
        w_quant_algo = self.quant_model.quant_config.quant_algo_info["w"]
        a_quant_algo = self.quant_model.quant_config.quant_algo_info["a"]
        is_dynamic = "dynamic" in a_quant_algo
        ignored_layers = self.quant_model.skip_layer_names()
        trtllm_config = {
            "quantization": {
                "exclude_modules": ignored_layers,
                "kv_cache_quant_algo": None,
            }
        }

        if "fp8" in self.quant_model.quant_config.quant_algo:
            quant_format = "naive-quantized"
            trtllm_config["quantization"]["quant_algo"] = "FP8"
            act_config = {
                "num_bits": 8,
                "strategy": re.search(r"per-([a-zA-Z]+)", a_quant_algo).group(1),
                "dynamic": is_dynamic,
                "type": "float",
            }
            weight_config = {
                "num_bits": 8,
                "strategy": re.search(r"per-([a-zA-Z]+)", w_quant_algo).group(1),
                "dynamic": False,
                "type": "float",
            }
        elif "int8" in self.quant_model.quant_config.quant_algo:
            quant_format = "int-quantized"
            trtllm_config["quantization"]["quant_algo"] = "INT8"
            act_config = {
                "num_bits": 8,
                "strategy": re.search(r"per-([a-zA-Z]+)", a_quant_algo).group(1),
                "dynamic": is_dynamic,
                "type": "int",
            }
            weight_config = {
                "num_bits": 8,
                "strategy": re.search(r"per-([a-zA-Z]+)", w_quant_algo).group(1),
                "dynamic": False,
                "type": "int",
            }
        elif "nvfp4" in self.quant_model.quant_config.quant_algo:
            quant_format = "naive-quantized"
            group_size = self.quant_model.quant_config.quant_algo_info["block_size"]
            trtllm_config["quantization"]["quant_algo"] = "NVFP4"
            trtllm_config["quantization"]["group_size"] = group_size
            act_config = {
                "num_bits": 4,
                "group_size": group_size,
                "dynamic": is_dynamic,
                "type": "float",
            }
            weight_config = {
                "num_bits": 4,
                "group_size": group_size,
                "dynamic": False,
                "type": "float",
            }
        else:
            raise ValueError(
                f"{self.quant_model.quant_config.quant_algo} not supported"
            )

        quantization_config = {"quant_method": save_name, ignore_field: ignored_layers}
        if save_name == "compressed-tensors":
            quantization_config.update(
                {
                    "config_groups": {
                        "group_0": {
                            "weights": weight_config,
                            "input_activations": act_config,
                            "output_activations": None,
                            "targets": ["Linear"],
                        }
                    },
                    "kv_cache_scheme": None,
                    "format": quant_format,
                    "quantization_status": "compressed",
                }
            )
        else:
            quantization_config["activation_scheme"] = (
                "dynamic" if is_dynamic else "static"
            )

        quant_dict = {"quantization_config": quantization_config}
        self.quant_model.get_model().config.update(quant_dict)
        print_info("Save quantization_config: {}".format(quant_dict))

        os.makedirs(save_path, exist_ok=True)
        self.quant_model.get_model().save_pretrained(save_path)

        with open(os.path.join(save_path, "hf_quant_config.json"), "w") as f:
            json.dump(trtllm_config, f, indent=4)

        self.quant_model.tokenizer.save_pretrained(save_path)


class PTQOnlyScaleSave(PTQSaveBase):
    def __init__(self, quant_model):
        super().__init__(quant_model=quant_model)

    def save(self, save_path):
        a_quant_algo = self.quant_model.quant_config.quant_algo_info["a"]
        ignored_layers = self.quant_model.skip_layer_names()

        static_q_dict = {
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": (
                    "dynamic" if "dynamic" in a_quant_algo else "static"
                ),
                "ignored_layers": ignored_layers,
            }
        }

        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "hf_quant_config.json"), "w") as f:
            json.dump(static_q_dict, f, indent=4)

        save_scales = {}
        new_model_index = {
            "metadata": {},
            "weight_map": {},
        }
        safetensor_name = "model-scales.safetensors"
        for name, value in self.quant_model.act_scales_dict.items():
            save_scales[name + ".input_scale"] = value
            new_model_index["weight_map"][name + ".input_scale"] = safetensor_name
        for name, value in self.quant_model.weight_scales_dict.items():
            save_scales[name + ".weight_scale"] = value
            new_model_index["weight_map"][name + ".weight_scale"] = safetensor_name

        safetensor_file = os.path.join(save_path, safetensor_name)
        safe_save(save_scales, safetensor_file)

        # update model index json
        new_model_index_file = os.path.join(save_path, "model.safetensors.index.json")
        with open(new_model_index_file, "w") as f:
            json.dump(new_model_index, f, indent=2)


class PTQTorchSave(PTQSaveBase):
    def __init__(self, quant_model):
        super(PTQTorchSave, self).__init__(quant_model=quant_model)

    def save(self, save_path):
        """save quantized model and configs to local disk"""
        os.makedirs(save_path, exist_ok=True)

        if self.quant_model.act_scales_dict:
            for k, v in self.quant_model.act_scales_dict.items():
                _save_path = os.path.join(save_path, "{}.act_scales.pt".format(k))
                torch.save(v, _save_path)
            print_info("save act scales done.")
        else:
            print_info("no act scales found.")

        if self.quant_model.weight_scales_dict:
            for k, v in self.quant_model.weight_scales_dict.items():
                _save_path = os.path.join(save_path, "{}.weight_scales.pt".format(k))
                torch.save(v, _save_path)
            print_info("save weight scales done.")
        else:
            print_info("no act scales found.")


class PTQPTMSave(PTQSaveBase):
    def __init__(self, quant_model):
        super(PTQPTMSave, self).__init__(quant_model=quant_model)

    def save(self, save_path):
        """save quantized model and configs to local disk"""
        os.makedirs(save_path, exist_ok=True)
        _index = torch.distributed.get_rank()
        if self.quant_model.act_scales_dict:
            for k, v in self.quant_model.act_scales_dict.items():
                _save_path = os.path.join(save_path, "{}.act_scales.pt".format(k))
                torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.MAX)
                if _index == 0:
                    torch.save(v, _save_path)
            print_info("save act scales done.")

        if self.quant_model.weight_scales_dict:
            for k, v in self.quant_model.weight_scales_dict.items():
                torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.MAX)
                _save_path = os.path.join(save_path, "{}.weight_scales.pt".format(k))
                if _index == 0:
                    torch.save(v, _save_path)
            print_info("save weight scales done.")


class DeepSeekV3PTQSaveMulti(PTQSaveBase):
    def __init__(self, quant_model, check_scales=False):
        super().__init__(quant_model=quant_model)
        self.moe_act_scales_dict = {}
        self.moe_weight_scales_dict = {}
        self.check_scales = check_scales
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.no_mp_key = [
            "input_layernorm",
            "post_attention_layernorm",
            ".q_a_proj.",
            "q_a_layernorm",
            ".kv_a_proj_with_mqa.",
            "kv_a_layernorm",
            ".gate.",
            "norm",
        ]
        self.mp_key = [
            ".q_proj.",
            ".q_b_proj.",
            ".kv_b_proj.",
            ".o_proj.",
            ".mlp.gate_proj",
            ".mlp.down_proj",
            ".mlp.up_proj",
            ".mlp.shared_experts.gate_proj",
            ".mlp.shared_experts.down_proj",
            ".mlp.shared_experts.up_proj",
        ]
        self.dim0_mp_key = [
            ".q_proj.",
            ".q_b_proj.",
            ".kv_b_proj.",
            ".mlp.gate_proj",
            ".mlp.up_proj",
            ".mlp.shared_experts.gate_proj",
            ".mlp.shared_experts.up_proj",
        ]
        self.dim1_mp_key = [
            ".o_proj.",
            ".mlp.down_proj",
            ".mlp.shared_experts.down_proj",
        ]

    def save(self, save_path):
        save_path = os.path.join(save_path, "scales")
        os.makedirs(save_path, exist_ok=True)
        _index = torch.cuda.current_device()

        # fuse scale
        fused_act_scale_dict = self._fuse_scale(self.quant_model.act_scales_dict)
        weight_scale_dict = deepcopy(self.quant_model.weight_scales_dict)
        fused_weight_fp8_scale_dict = self._fuse_scale(weight_scale_dict)

        if fused_act_scale_dict:
            for k, v in fused_act_scale_dict.items():
                _save_path = os.path.join(
                    save_path, "{}.input_scale.{}.pt".format(k, _index)
                )
                if "experts" in k and "shared_experts" not in k:
                    # handle Deepseek EP, do not all reduce
                    _save_path = os.path.join(
                        save_path, "{}.input_scale.{}.pt".format(k, self.rank)
                    )
                    torch.save(v, _save_path)
                else:
                    torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.MAX)
                    if self.rank == 0:
                        torch.save(v, _save_path)
            print_info("save act scales done.")

        if self.quant_model.weight_scales_dict:
            for k, v in self.quant_model.weight_scales_dict.items():
                max_value_group_wise = v
                # fp8 pertensor scale
                fused_max_value = fused_weight_fp8_scale_dict[k]

                # if weight quant is int4 and act quant is fp8, extra save int4 absmax
                if (
                    self.quant_model.quant_algo_dict["w_quant_algo"] == "int4"
                    and self.quant_model.quant_algo_dict["a_quant_algo"] == "fp8"
                ):
                    _save_path = os.path.join(
                        save_path, "{}.weight_scale.{}.{}.pt".format(k, "int4", _index)
                    )
                    scale_int4 = max_value_group_wise / 8

                    # save weigth-int4-pergroup scale
                    if "experts" in k and "shared_experts" not in k:
                        _save_path = os.path.join(
                            save_path,
                            "{}.weight_scale.{}.{}.pt".format(k, "int4", self.rank),
                        )
                        torch.save(scale_int4, _save_path)
                    else:
                        self._save_ckpt(
                            scale_int4,
                            _save_path,
                            self.quant_model.quant_algo_dict["all_reduce"],
                        )
                    scale = (fused_max_value.max() / 448.0).to(fused_max_value.dtype)
                elif self.quant_model.quant_algo_dict["w_quant_algo"] == "fp8":
                    scale = fused_max_value.max().to(fused_max_value.dtype)

                assert scale.numel() == 1

                if "experts" in k and "shared_experts" not in k:
                    _save_path = os.path.join(
                        save_path, "{}.weight_scale.{}.pt".format(k, self.rank)
                    )
                    torch.save(scale, _save_path)
                else:
                    print_info(f"before all reduce scale = {scale}")
                    torch.distributed.all_reduce(
                        scale, op=torch.distributed.ReduceOp.MAX
                    )
                    print_info(f"after all reduce scale = {scale}")
                    _save_path = os.path.join(
                        save_path, "{}.weight_scale.{}.pt".format(k, _index)
                    )
                    self._save_ckpt(
                        scale,
                        _save_path,
                        self.quant_model.quant_algo_dict["all_reduce"],
                    )

            print_info("save weight scales done.")

        tmp_path = os.path.join("/".join(save_path.split("/")[:-1]), "tmp")
        os.makedirs(tmp_path, exist_ok=True)
        self._update_and_save_weight(tmp_path, fused_weight_fp8_scale_dict)
        dist.barrier()
        dist.destroy_process_group()

        if self.rank == 0:
            save_model_path = os.path.join(
                "/".join(save_path.split("/")[:-1]), "checkpoint"
            )
            os.makedirs(save_model_path, exist_ok=True)

            self.convert_scales_to_safetensors(save_path, tmp_path)
            print_info("convert scales to safetensors done.")

            file_name = self.merge_model(
                tmp_path, save_model_path, mp=self.quant_model.model.world_size
            )
            print_info("merge model done.")

            self.add_mtp_weight(save_path=save_model_path, file_name=file_name)

            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            parent_dir = os.path.dirname(
                self.quant_model.model.ori_model_path.rstrip("/")
            )
            tp_model_path = os.path.join(
                parent_dir, f"ds_ckpt_tp{self.quant_model.model.world_size}"
            )
            if os.path.exists(tp_model_path):
                shutil.rmtree(tp_model_path)

    def _save_ckpt(self, scale, save_path, all_reduce=True):
        if all_reduce:
            if self.rank == 0:
                torch.save(scale, save_path)
        else:
            torch.save(scale, save_path)

    def _fuse_scale(self, scale_dict):
        """
        1. fuse q_a_proj and kv_a_proj scale
        2. fuse gate_proj(w1) and up_proj(w3) scale
        """
        if not scale_dict:
            return

        for layer_name in scale_dict:
            if "q_a_proj" in layer_name:
                q_a_scale = scale_dict[layer_name]
                kv_a_layer_name = layer_name.replace("q_a_proj", "kv_a_proj_with_mqa")
                kv_a_scale = scale_dict[kv_a_layer_name]
                fused_scale = torch.max(q_a_scale.max(), kv_a_scale.max())
                scale_dict[layer_name] = fused_scale
                scale_dict[kv_a_layer_name] = fused_scale

            if "gate_proj" in layer_name:
                w1_scale = scale_dict[layer_name]
                w3_layer_name = layer_name.replace("gate_proj", "up_proj")
                w3_scale = scale_dict[w3_layer_name]
                fused_scale = torch.max(w1_scale.max(), w3_scale.max())
                scale_dict[layer_name] = fused_scale
                scale_dict[w3_layer_name] = fused_scale

            scale_dict[layer_name] = scale_dict[layer_name].max()
        return scale_dict

    def _update_and_save_weight(self, save_model_path, fused_weight_fp8_scale_dict):
        self._update_fp8_weights(fused_weight_fp8_scale_dict)
        self._save_model(save_model_path)

    def _update_fp8_weights(self, fused_weight_fp8_scale_dict):
        # We always update fp8 weight in deepseek model,
        # cause we delivery fp8 per tensor scale.
        # It would change the distribution of origin bf16 weigth.
        if not self.quant_model.weight_scales_dict:
            return

        quant_layers_dict = self.quant_model.get_observer_layers()
        for name, layer in quant_layers_dict.items():
            print_info(f"*** update {name} weights...")
            assert hasattr(layer, "weight")
            if layer.weight.dtype == torch.float8_e4m3fn:
                weight_bf16 = weight_dequant(layer.weight, layer.weight_scale_inv)
                ori_bf16_weight = weight_bf16
                fused_max_value = fused_weight_fp8_scale_dict[name]
                if "w4a8" in self.quant_model.quant_config.quant_algo:
                    tensor_wise_scale = fused_max_value.max() / 448.0
                else:
                    tensor_wise_scale = fused_max_value.max()
                torch.distributed.all_reduce(
                    tensor_wise_scale, op=torch.distributed.ReduceOp.MAX
                )
                weight_fp8 = (
                    (weight_bf16 / tensor_wise_scale)
                    .clamp(-448, 448)
                    .to(torch.float8_e4m3fn)
                )
                weight_fp8 = weight_fp8.to(layer.weight.device)

                if "fp8" in self.quant_model.quant_config.quant_algo:
                    if "w4a8" in self.quant_model.quant_config.quant_algo:
                        new_weight_bf16 = (
                            weight_fp8.to(torch.bfloat16) * tensor_wise_scale
                        )
                        new_weight_bf16_qdq = fake_quant_dequant(
                            new_weight_bf16,
                            method="groupwise",
                            bits=4,
                            group_size=self.quant_model.quant_config.quant_algo_info[
                                "w_group_size"
                            ],
                        )
                        new_weight_fp8 = (
                            (new_weight_bf16_qdq / tensor_wise_scale)
                            .clamp(-448, 448)
                            .to(torch.float8_e4m3fn)
                        )
                        new_weight_fp8 = new_weight_fp8.to(layer.weight.device)
                        layer.weight.data = new_weight_fp8
                        bf16_weight_qdq = (
                            new_weight_fp8.to(torch.bfloat16) * tensor_wise_scale
                        )
                    else:
                        layer.weight.data = weight_fp8
                        bf16_weight_qdq = (
                            weight_fp8.to(torch.bfloat16) * tensor_wise_scale
                        )

                cos_sim = torch.cosine_similarity(
                    bf16_weight_qdq, ori_bf16_weight
                ).mean()
                print_info(f"qdq weight cos sim :{cos_sim}")
                if cos_sim < 0.95:
                    print_info("*** cos sim < 0.95 !!!")

    def _save_model(self, save_model_path):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        save_model(
            self.quant_model.model,
            os.path.join(
                save_model_path, f"model{self.rank}-mp{world_size}.safetensors"
            ),
        )
        print_info(f"save model{self.rank}-mp{world_size}.safetensors done.")

    def convert_scales_to_safetensors(self, input_path, save_path):
        state_dict = {}
        pt_files = list(glob(os.path.join(input_path, "*.pt")))
        pt_files.sort()
        for pt_file in tqdm(pt_files):
            scale_name = ".".join(pt_file.split("/")[-1].split(".")[:-2])
            scale = torch.load(pt_file)
            state_dict[scale_name] = scale.cpu()
        safetensor_file = os.path.join(save_path, "model-scales.safetensors")
        safe_save(state_dict, safetensor_file)

    def merge_model(self, input_path, save_model_path, mp=16):
        ori_state_dicts = [{} for _ in range(mp)]
        model_save_ind = 0
        localind = 0

        scale_path = os.path.join(input_path, "model-scales.safetensors")
        scales_dict = load_file(scale_path)

        for mpind in range(mp):
            file_path = os.path.join(input_path, f"model{mpind}-mp{mp}.safetensors")
            ori_state_dicts[localind] = safe_open(
                file_path, framework="pt", device="cpu"
            )
            localind += 1

        # process no_mp_key
        print_info("##no_mp_key##")
        num_layers = 61
        index_dict = {"weight_map": {}}

        # process model.norm.weight
        new_save_dict = {}
        filename = "model-" + "{:0>{}}".format(model_save_ind, 5) + ".safetensors"
        model_save_ind += 1
        for _, k in enumerate(ori_state_dicts[0].keys()):
            if "model.norm.weight" in k:
                param: torch.Tensor = ori_state_dicts[0].get_tensor(k)
                new_save_dict[k] = param
                index_dict["weight_map"][k] = str(filename)
        safe_save(new_save_dict, os.path.join(save_model_path, filename))

        # process model.encoder
        for nl in range(num_layers):
            new_save_dict = {}
            filename = "model-" + "{:0>{}}".format(model_save_ind, 5) + ".safetensors"
            model_save_ind += 1
            for _, k in enumerate(ori_state_dicts[0].keys()):
                if (
                    any(word if word in k else False for word in self.no_mp_key)
                    and "layers." + str(nl) + "." in k
                ):
                    param: torch.Tensor = ori_state_dicts[0].get_tensor(k)
                    self._transform_keys(
                        k,
                        param,
                        scales_dict,
                        new_save_dict,
                        index_dict,
                        filename,
                    )
            # process expert merge
            for mp_index in range(mp):
                for _, k in enumerate(ori_state_dicts[mp_index].keys()):
                    if "mlp.experts" in k and "layers." + str(nl) + "." in k:
                        param: torch.Tensor = ori_state_dicts[mp_index].get_tensor(k)
                        self._transform_keys(
                            k,
                            param,
                            scales_dict,
                            new_save_dict,
                            index_dict,
                            filename,
                        )
            safe_save(new_save_dict, os.path.join(save_model_path, filename))

        print_info("##mp_key##")
        # process mp_key
        filename = None
        new_save_dict = None

        # process embed_tokens, lm_head
        new_save_dict = {}
        filename = "model-" + "{:0>{}}".format(model_save_ind, 5) + ".safetensors"
        model_save_ind += 1
        for _, k in enumerate(ori_state_dicts[0].keys()):
            if any(
                word if word in k else False for word in ["embed_tokens", "lm_head"]
            ):
                param_list = []
                for i in range(mp):
                    param: torch.Tensor = ori_state_dicts[i].get_tensor(k)
                    param_list.append(param)
                newparam = torch.cat(param_list, dim=0)
                new_save_dict[k] = newparam
                index_dict["weight_map"][k] = str(filename)
        safe_save(new_save_dict, os.path.join(save_model_path, filename))
        # process others
        for nl in range(num_layers):
            new_save_dict = {}
            filename = "model-" + "{:0>{}}".format(model_save_ind, 5) + ".safetensors"
            model_save_ind += 1
            for _, k in enumerate(ori_state_dicts[0].keys()):
                if (
                    any(word if word in k else False for word in self.mp_key)
                    and "layers." + str(nl) + "." in k
                ):
                    param_list = []
                    for i in range(mp):
                        param: torch.Tensor = ori_state_dicts[i].get_tensor(k)
                        param_list.append(param)
                    if any(word if word in k else False for word in self.dim0_mp_key):
                        newparam = torch.cat(param_list, dim=0)
                    elif any(word if word in k else False for word in self.dim1_mp_key):
                        newparam = torch.cat(param_list, dim=1)
                    else:
                        raise AssertionError("Key should not in mp key!")
                    self._transform_keys(
                        k,
                        newparam,
                        scales_dict,
                        new_save_dict,
                        index_dict,
                        filename,
                    )
            safe_save(new_save_dict, os.path.join(save_model_path, filename))

        path = self.quant_model.model.ori_model_path
        for file_path in glob(os.path.join(path, "*token*")):
            new_file_path = os.path.join(save_model_path, os.path.basename(file_path))
            shutil.copyfile(file_path, new_file_path)
        for file_path in glob(os.path.join(path, "*conf*")):
            new_file_path = os.path.join(save_model_path, os.path.basename(file_path))
            try:
                shutil.copyfile(file_path, new_file_path)
            except IsADirectoryError:
                shutil.copytree(file_path, new_file_path)
        for file_path in glob(os.path.join(path, "*modeling_deepseek.py*")):
            new_file_path = os.path.join(save_model_path, os.path.basename(file_path))
            shutil.copyfile(file_path, new_file_path)

        if self.quant_model.model.config.model_type == "kimi_k2":
            file_path = os.path.join(input_path, "tiktoken.model")
            new_file_path = os.path.join(save_model_path, "tiktoken.model")
            shutil.copyfile(file_path, new_file_path)

        with open(
            os.path.join(save_model_path, "model.safetensors.index.json"), "w"
        ) as f:
            json.dump(index_dict, f, indent=4)

        # setting quantization config
        a_quant_algo = self.quant_model.quant_config.quant_algo_info["a"]
        if "fp8" in self.quant_model.quant_config.quant_algo:
            if "w4a8" in self.quant_model.quant_config.quant_algo:
                if self.quant_model.deploy_backend == "trtllm":
                    quant_dict = {
                        "quantization_config": {
                            "quant_method": "w4a8_awq",
                            "weight_group_size": 128,
                            "activation_scheme": (
                                "dynamic" if "dynamic" in a_quant_algo else "static"
                            ),
                            "kv_cache_quant_method": "fp8",
                            "ignored_layers": [
                                "*self_attn*",
                                "*gate_up_proj",
                                "*down_proj",
                                "*layers.61*",
                            ],
                            "ignored_quantization_config": {
                                "quant_method": "fp8",
                                "activation_scheme": "dynamic",
                                "fmt": "e4m3",
                                "kv_cache_quant_method": "fp8",
                                "weight_block_size": [128, 128],
                            },
                        }
                    }
                else:
                    raise NotImplementedError(
                        f"deploy_backend {self.quant_model.deploy_backend} \
                            is not supported for w4a8_fp8."
                    )
            else:
                ignore_layers = self.quant_model.quant_config.quant_algo_info[
                    "ignore_layers"
                ]
                if self.quant_model.deploy_backend == "vllm":
                    quant_dict = {
                        "quantization_config": {
                            "quant_method": "fp8",
                            "activation_scheme": (
                                "dynamic" if "dynamic" in a_quant_algo else "static"
                            ),
                            "ignored_layers": ignore_layers,
                        }
                    }
                else:
                    raise NotImplementedError(
                        f"deploy_backend {self.quant_model.deploy_backend} \
                            is not supported for fp8_static."
                    )

        config = DeepseekV3Config.from_pretrained(self.quant_model.model.ori_model_path)
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
        config.update(quant_dict)
        config.save_pretrained(save_model_path)
        print_info("Save quantization_config: {}".format(quant_dict))
        return "model-" + "{:0>{}}".format(model_save_ind, 5) + ".safetensors"

    def _transform_keys(
        self,
        param_name,
        param,
        scales_dict,
        new_save_dict,
        index_dict,
        filename,
    ):
        if "fp8" in self.quant_model.quant_config.quant_algo:
            if not any(
                substring in param_name
                for substring in self.quant_model.quant_config.quant_algo_info[
                    "ignore_layers"
                ]
            ):
                if param_name.endswith("weight_scale_inv"):
                    return
                weight_scale = scales_dict.get(f"{param_name}_scale", None)
                if weight_scale is not None:
                    new_save_dict[f"{param_name}_scale"] = weight_scale
                    new_save_dict[f"{param_name[:-7]}.input_scale"] = scales_dict[
                        f"{param_name[:-7]}.input_scale"
                    ]
                    index_dict["weight_map"][f"{param_name}_scale"] = str(filename)
                    index_dict["weight_map"][f"{param_name[:-7]}.input_scale"] = str(
                        filename
                    )
                    if "w4a8" in self.quant_model.quant_config.quant_algo:
                        param = self._packed_weight(
                            param_name,
                            param,
                            self.quant_model.quant_config.quant_algo_info[
                                "w_group_size"
                            ],
                            scales_dict,
                        )
                        new_save_dict[f"{param_name}_scale.int4"] = scales_dict[
                            f"{param_name}_scale.int4"
                        ]
                        index_dict["weight_map"][f"{param_name}_scale.int4"] = str(
                            filename
                        )
                        param_name = param_name.replace("weight", "qweight")

        new_save_dict[param_name] = param
        index_dict["weight_map"][param_name] = str(filename)

    def _packed_weight(self, weight_name, weight, block_wise, scales_dict):
        target_shape = (weight.shape[0] // block_wise, weight.shape[1] // block_wise)
        scale_inv = scales_dict[f"{weight_name}_scale"]
        scale_inv_padded = torch.full(target_shape, scale_inv.item()).float()

        weight = weight.cuda()
        scale_inv_padded = scale_inv_padded.cuda()
        bf16_weight = weight_dequant(weight, scale_inv_padded)
        weight = weight.cpu()
        scale_inv_padded = scale_inv_padded.cpu()
        bf16_weight = bf16_weight.cpu()

        int4_scale = scales_dict[f"{weight_name}_scale.int4"]
        int4_scale = torch.repeat_interleave(int4_scale, block_wise, dim=-1)
        assert int4_scale.shape == weight.shape
        quant_weight = torch.clamp(torch.round(bf16_weight / int4_scale), -8, 7)
        packed_weight = pack_weight_to_int8(quant_weight)
        print_info(
            f"Packing {weight_name}, packed weight dtype = {packed_weight.dtype}"
        )
        del bf16_weight
        return packed_weight

    def _read_weight_map(self, input_path):
        model_index_file = os.path.join(input_path, "model.safetensors.index.json")
        with open(model_index_file, "r") as f:
            model_index = json.load(f)
        weight_map = model_index["weight_map"]
        return weight_map

    def _get_tensor_from_safetensor(
        self, input_path, weight_name, safetensor_file, loaded_files
    ):
        if safetensor_file not in loaded_files:
            current_state_dict = load_file(
                os.path.join(input_path, safetensor_file), device="cpu"
            )
            loaded_files[safetensor_file] = current_state_dict
        weight = loaded_files[safetensor_file][weight_name]
        if len(loaded_files) > 4:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
        return weight

    def add_mtp_weight(self, input_path=None, save_path=None, file_name=None):
        if input_path is None:
            input_path = self.quant_model.model.ori_model_path
        weight_map = self._read_weight_map(input_path)

        state_dict = {}
        add_weight_map = {}
        loaded_files = {}
        for weight_name in weight_map:
            if "layers.61" in weight_name or "rotary_emb.inv_freq" in weight_name:
                print_info(f"- Add {weight_name}")
                safetensor_file = weight_map[weight_name]
                weight = self._get_tensor_from_safetensor(
                    input_path, weight_name, safetensor_file, loaded_files
                )
                state_dict[weight_name] = weight
                add_weight_map[weight_name] = file_name

        safe_save(state_dict, os.path.join(save_path, file_name))

        # update model index json
        new_model_index_file = os.path.join(save_path, "model.safetensors.index.json")
        with open(new_model_index_file, "r") as f:
            new_model_index = json.load(f)
        new_model_index["weight_map"].update(add_weight_map)
        with open(new_model_index_file, "w") as f:
            json.dump(new_model_index, f, indent=2)


class DeepSeekV3PTQSaveSingle(DeepSeekV3PTQSaveMulti):
    def __init__(self, quant_model):
        super().__init__(quant_model=quant_model)

    def save(self, save_path):
        # setting quantization config
        a_quant_algo = self.quant_model.quant_config.quant_algo_info["a"]
        if "fp8" in self.quant_model.quant_config.quant_algo:
            if "w4a8" in self.quant_model.quant_config.quant_algo:
                if self.quant_model.deploy_backend == "trtllm":
                    quant_dict = {
                        "quantization_config": {
                            "quant_method": "w4a8_awq",
                            "weight_group_size": 128,
                            "activation_scheme": (
                                "dynamic" if "dynamic" in a_quant_algo else "static"
                            ),
                            "kv_cache_quant_method": "fp8",
                            "ignored_layers": [
                                "*self_attn*",
                                "*gate_up_proj",
                                "*down_proj",
                                "*layers.61*",
                            ],
                            "ignored_quantization_config": {
                                "quant_method": "fp8",
                                "activation_scheme": "dynamic",
                                "fmt": "e4m3",
                                "kv_cache_quant_method": "fp8",
                                "weight_block_size": [128, 128],
                            },
                        }
                    }
                else:
                    raise NotImplementedError(
                        f"deploy_backend {self.quant_model.deploy_backend} \
                            is not supported for w4a8_fp8."
                    )
            else:
                ignore_layers = self.quant_model.quant_config.quant_algo_info[
                    "ignore_layers"
                ]
                if self.quant_model.deploy_backend == "vllm":
                    quant_dict = {
                        "quantization_config": {
                            "quant_method": "fp8",
                            "activation_scheme": (
                                "dynamic" if "dynamic" in a_quant_algo else "static"
                            ),
                            "ignored_layers": ignore_layers,
                        }
                    }
                else:
                    raise NotImplementedError(
                        f"deploy_backend {self.quant_model.deploy_backend} \
                            is not supported for fp8_static."
                    )

            os.makedirs(save_path, exist_ok=True)
            self.quant_model.get_model().config.update(quant_dict)
            print_info("Save quantization_config: {}".format(quant_dict))
            self.quant_model.get_model().save_pretrained(save_path)
            self.add_mtp_weight(save_path=save_path)
        else:
            raise ValueError(
                f"{self.quant_model.quant_config.quant_algo} not supported"
            )


class TPManager:
    def __init__(self, world_size: int):
        self.world_size = world_size

    @torch.no_grad()
    def gather(self, local: torch.Tensor, dim: int):
        if self.world_size == 1:
            return local.cpu()

        if dim != 0:
            local = local.transpose(0, dim).contiguous()

        shape = list(local.shape)
        shape[0] *= self.world_size

        full = torch.empty(
            shape,
            dtype=local.dtype,
            device=local.device,
        )

        dist.all_gather_into_tensor(full, local)

        if dim != 0:
            full = full.transpose(0, dim).contiguous()

        return full.cpu()


class MoEExpertGather:
    """EP gather using gather_object (CPU only)."""

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def gather(
        self,
        key: str,
        local_tensor: torch.Tensor,
        on_rank0_callback,
    ):
        obj = {
            "key": key,
            "tensor": local_tensor.cpu(),
        }
        gathered = [None] * self.world_size if self.rank == 0 else None

        dist.gather_object(
            obj,
            gathered,
            dst=0,
        )

        if self.rank == 0:
            for item in gathered:
                on_rank0_callback(
                    item["key"],
                    item["tensor"],
                )


class DeepSeekKeyRouter:
    def __init__(self):
        self.dim0_keys = [
            ".q_proj.",
            ".q_b_proj.",
            ".kv_b_proj.",
            ".mlp.gate_proj",
            ".mlp.up_proj",
            ".mlp.shared_experts.gate_proj",
            ".mlp.shared_experts.up_proj",
        ]

        self.dim1_keys = [
            ".o_proj.",
            ".mlp.down_proj",
            ".mlp.shared_experts.down_proj",
        ]

    def layer_id(self, key: str):
        if key.startswith("model.embed_tokens") or key.startswith("lm_head"):
            return -1
        if key.startswith("model.norm"):
            return -2
        m = re.search(r"model\.layers\.(\d+)\.", key)
        return int(m.group(1)) if m else None

    def is_moe_expert(self, key: str):
        return "mlp.experts" in key

    def tp_gather_dim(self, key: str):
        if any(x in key for x in self.dim0_keys):
            return 0
        if any(x in key for x in self.dim1_keys):
            return 1
        return None


class DeepSeekV3W4A8Int8Save(DeepSeekV3PTQSaveMulti):
    """
    DeepSeek R1 PTQ weight saver
    - TP layers: all_gather -> quantize -> save
    - EP (MoE experts): gather_object -> save
    """

    def __init__(self, quant_model, check_scales=False):
        super().__init__(quant_model=quant_model)

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.router = DeepSeekKeyRouter()
        self.quantizer = Int8PerChannelQuantizer()
        self.tp_mgr = TPManager(self.world_size)

    @torch.no_grad()
    def save(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        state = self.quant_model.model.state_dict()
        moe_gather = MoEExpertGather(self.rank, self.world_size)

        safetensors_index: Dict[str, str] = {}
        shard_idx = 1
        n_shards = 63

        def shard_name(i):
            return f"model-{i:05d}-of-{n_shards:05d}.safetensors"

        shard_idx = self._save_embeddings_and_norm(
            state, safetensors_index, save_path, shard_idx, shard_name
        )
        shard_idx = self._save_transformer_layers(
            state, safetensors_index, save_path, shard_idx, shard_name, moe_gather
        )

        if self.rank == 0:
            self._finalize(save_path, safetensors_index, shard_idx, shard_name)

    def _save_embeddings_and_norm(
        self,
        state: Dict[str, torch.Tensor],
        index: Dict[str, str],
        save_path: str,
        shard_idx: int,
        shard_name,
    ) -> int:
        if self.rank == 0:
            out = {}

        device = torch.device(
            f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
        )

        for k in list(state.keys()):
            lid = self.router.layer_id(k)
            if lid not in (-1, -2):
                continue

            if lid == -2:  # model.norm
                if self.rank == 0:
                    full = state[k]
            else:  # embed / lm_head (TP)
                v = state[k].to(device)
                full = self.tp_mgr.gather(v, 0)

            if self.rank == 0:
                out[k] = full
                index[k] = shard_name(shard_idx)

            del state[k]

        if self.rank == 0:
            safe_save(out, os.path.join(save_path, shard_name(shard_idx)))
            shard_idx += 1
            del out
            gc.collect()

        if dist.is_initialized():
            dist.barrier()

        return shard_idx

    def _save_transformer_layers(
        self,
        state: Dict[str, torch.Tensor],
        index: Dict[str, str],
        save_path: str,
        shard_idx: int,
        shard_name,
        moe_gather: MoEExpertGather,
    ) -> int:
        device = torch.device(
            f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
        )

        for lid in range(61):
            layer_keys = [k for k in state if self.router.layer_id(k) == lid]

            # ---------------- EP ----------------
            expert_out = {} if self.rank == 0 else None

            def on_moe(k, t):
                expert_out[k] = t  # noqa: B023

            for k in layer_keys:
                if self.router.is_moe_expert(k):
                    moe_gather.gather(k, state[k], on_moe)
                    del state[k]

            # ---------------- TP / local ----------------
            layer_out = {} if self.rank == 0 else None

            for k in layer_keys:
                if self.router.is_moe_expert(k):
                    continue

                v = state[k]

                # passthrough
                if any(
                    x in k
                    for x in ("layernorm", "gate.weight", "e_score_correction_bias")
                ):
                    if self.rank == 0:
                        layer_out[k] = v.cpu()
                    del state[k]
                    continue

                assert k.endswith("weight"), f"Unexpected param: {k}"

                dim = self.router.tp_gather_dim(k)
                if dim is None:
                    if self.rank == 0 and (
                        "q_a_proj" in k or "kv_a_proj_with_mqa" in k
                    ):
                        q, s = self.quantizer.quantize(v)
                        layer_out[k] = q
                        layer_out[k + "_scale"] = s
                    del state[k]
                    continue

                v = v.to(device)
                full = self.tp_mgr.gather(v, dim)
                del v

                if self.rank == 0:
                    q, s = self.quantizer.quantize(full)
                    layer_out[k] = q
                    layer_out[k + "_scale"] = s

                del state[k]
                del full
                torch.cuda.empty_cache()

            if self.rank == 0:
                if expert_out:
                    layer_out.update(expert_out)

                safe_save(layer_out, os.path.join(save_path, shard_name(shard_idx)))

                for k in layer_out:
                    index[k] = shard_name(shard_idx)

                shard_idx += 1
                del layer_out
                gc.collect()
                torch.cuda.empty_cache()

            if dist.is_initialized():
                dist.barrier()

        return shard_idx

    def _finalize(self, save_path, index, shard_idx, shard_name):
        self._save_quantization_config(save_path)
        self._copy_additional_files(self.quant_model.model.ori_model_path, save_path)

        with open(os.path.join(save_path, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": index}, f, indent=2)

        self.save_mtp_int8_from_fp8(
            self.quant_model.model.ori_model_path,
            save_path,
            shard_name(shard_idx),
        )

    def _save_quantization_config(self, save_path):
        quantization_config = {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "quantization_status": "compressed",
            "kv_cache_scheme": None,
            "ignore": ["lm_head"],
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "strategy": "channel",
                        "symmetric": True,
                        "type": "int",
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "strategy": "token",
                        "symmetric": True,
                        "type": "int",
                        "dynamic": True,
                    },
                },
                "group_1": {
                    "targets": [
                        "re:.*(self_attn|shared_experts).*",
                        "re:.*(mlp\\.gate_up_proj|mlp\\.down_proj).*",
                        "re:model\\.layers\\.61.*",
                    ],
                    "weights": {
                        "num_bits": 8,
                        "strategy": "channel",
                        "symmetric": True,
                        "type": "int",
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "strategy": "token",
                        "symmetric": True,
                        "type": "int",
                        "dynamic": True,
                    },
                },
            },
        }

        config = self.quant_model.model.config
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")

        config.update({"quantization_config": quantization_config})
        config.save_pretrained(save_path)

    def _copy_additional_files(self, source_path: str, target_path: str):
        """
        Copy additional files with specific suffixes,
        but NEVER override config.json generated by quantization.
        """
        import os
        import shutil

        os.makedirs(target_path, exist_ok=True)

        VALID_SUFFIX = (".py", ".json")
        SKIP_FILES = {"config.json"}

        for fname in os.listdir(source_path):
            if fname in SKIP_FILES:
                continue
            if not fname.endswith(VALID_SUFFIX):
                continue

            src = os.path.join(source_path, fname)
            dst = os.path.join(target_path, fname)

            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print_info(f"Copied {fname}")

    @torch.no_grad()
    def save_mtp_int8_from_fp8(
        self,
        input_path: str,
        save_path: str,
        shard_name: str,
    ):
        """
        Convert MTP (model.layers.61) fp8 weights into int8 per-channel
        and save as a new safetensors shard.
        """
        weight_map = self._read_weight_map(input_path)

        cache = {}
        out_state = {}
        new_weight_map = {}

        def is_mtp(name: str) -> bool:
            return name.startswith("model.layers.61.")

        def is_scale_inv(name: str) -> bool:
            return name.endswith("_scale_inv")

        def is_quantizable_weight(name: str) -> bool:
            """
            Only real GEMM weights should be quantized.
            """
            if not name.endswith(".weight"):
                return False

            skip_keywords = [
                "norm",
                "embed_tokens",
                "shared_head",
                "e_score_correction_bias",
                "eh_proj",
                "mlp.gate",
            ]
            return not any(k in name for k in skip_keywords)

        for weight_name, src_file in weight_map.items():
            if not is_mtp(weight_name):
                continue

            if is_scale_inv(weight_name):
                continue

            tensor = self._get_tensor_from_safetensor(
                input_path, weight_name, src_file, cache
            )

            if not is_quantizable_weight(weight_name):
                out_state[weight_name] = tensor
                new_weight_map[weight_name] = shard_name
                continue

            # ---------- fp8 weight ----------
            scale_inv_name = weight_name + "_scale_inv"
            assert scale_inv_name in weight_map, f"Missing {scale_inv_name}"

            scale_inv = self._get_tensor_from_safetensor(
                input_path,
                scale_inv_name,
                weight_map[scale_inv_name],
                cache,
            )

            # ---------- fp8 → bf16 (block dequant) ----------
            w_bf16 = weight_dequant(tensor.cuda(), scale_inv.cuda()).cpu()

            # ---------- bf16 → int8 per-channel ----------
            q, scale = self.quantizer.quantize(w_bf16)

            out_state[weight_name] = q
            out_state[weight_name + "_scale"] = scale

            new_weight_map[weight_name] = shard_name
            new_weight_map[weight_name + "_scale"] = shard_name

            del tensor, scale_inv, w_bf16, q, scale

        # ---------- save safetensors ----------
        safe_save(out_state, os.path.join(save_path, shard_name))

        # ---------- update index ----------
        index_file = os.path.join(save_path, "model.safetensors.index.json")
        with open(index_file, "r") as f:
            index = json.load(f)

        index["weight_map"].update(new_weight_map)

        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)

        print(f"[Done] Saved MTP int8 shard: {shard_name}")
