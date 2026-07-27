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
        safe_save(state_dict, os.path.join(save_path, model_save_name), safetensors_metadata)
        self.quant_model.config.save_pretrained(save_path)


class PTQVLMSaveVllmHF(PTQSaveBase):
    def __init__(self, quant_model):
        super().__init__(quant_model=quant_model)

    def save(self, save_path):
        save_name = self.quant_model.quant_config.save_name
        ignore_field = "ignore" if save_name == "compressed-tensors" else "ignored_layers"

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
            raise ValueError(f"{self.quant_model.quant_config.quant_algo} not supported")
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
            quantization_config["activation_scheme"] = "dynamic" if is_dynamic else "static"

        if (
            hasattr(self.quant_model.quant_config, "transform_config")
            and self.quant_model.quant_config.transform_config is not None
        ):
            quantization_config["transform_config"] = (
                self.quant_model.quant_config.transform_config
            )

        quant_dict = {"quantization_config": quantization_config}
        self.quant_model.get_model().config.update(quant_dict)
        print_info("Save quantization_config: {}".format(quant_dict))

        os.makedirs(save_path, exist_ok=True)

        self.quant_model.get_model().save_pretrained(save_path, max_shard_size="5GB")
        self.quant_model.processor.save_pretrained(save_path)
        self.quant_model.tokenizer.save_pretrained(save_path)


class PTQSaveVllmHF(PTQSaveBase):
    def __init__(self, quant_model):
        super().__init__(quant_model=quant_model)

    def save(self, save_path):
        save_name = self.quant_model.quant_config.save_name
        ignore_field = "ignore" if save_name == "compressed-tensors" else "ignored_layers"
        w_quant_algo = self.quant_model.quant_config.quant_algo_info.get("w", "")
        a_quant_algo = self.quant_model.quant_config.quant_algo_info.get("a", "")
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
            raise ValueError(f"{self.quant_model.quant_config.quant_algo} not supported")

        quantization_config = {"quant_method": save_name, ignore_field: ignored_layers}
        # Set kv_cache_scheme if kv_cache quantization is enabled
        c_quant_algo = self.quant_model.quant_config.quant_algo_info.get("c", None)
        if c_quant_algo is not None:
            kv_cache_scheme = {
                "num_bits": 8,
                "strategy": re.search(r"per-([a-zA-Z]+)", c_quant_algo).group(1),
                "type": "float",
            }
        else:
            kv_cache_scheme = None

        # W8A8C8 recipe (e.g. GLM-5): the model adapter attaches a rich
        # KV-cache scheme descriptor when it hooks the latent-KV / indexer-K
        # streams via ``apply_kvcache_observers``.  When present, override
        # the vanilla ``{num_bits, strategy, type}`` blob above with the
        # richer dict so downstream (vLLM / config.json) knows the
        # per-block-128 dynamic layout + RoPE-passthrough + indexer scheme.
        extra_kv_cache_scheme = getattr(
            self.quant_model, "_extra_kv_cache_scheme", None
        )
        if extra_kv_cache_scheme:
            kv_cache_scheme = dict(extra_kv_cache_scheme)

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
                    "kv_cache_scheme": kv_cache_scheme,
                    "format": quant_format,
                    "quantization_status": "compressed",
                }
            )
        else:
            quantization_config["activation_scheme"] = "dynamic" if is_dynamic else "static"
            if kv_cache_scheme is not None:
                quantization_config["kv_cache_scheme"] = "static"

        if (
            hasattr(self.quant_model.quant_config, "transform_config")
            and self.quant_model.quant_config.transform_config is not None
        ):
            quantization_config["transform_config"] = (
                self.quant_model.quant_config.transform_config
            )

        quant_dict = {"quantization_config": quantization_config}
        self.quant_model.get_model().config.update(quant_dict)
        print_info("Save quantization_config: {}".format(quant_dict))

        os.makedirs(save_path, exist_ok=True)
        # ------------------------------------------------------------------
        # Sanitize ``generation_config`` before save_pretrained().
        # Newer transformers (>= 4.50 and 5.x) call generation_config
        # .validate(strict=True) inside save_pretrained().  Some legacy
        # checkpoints (e.g. ChatGLM-5.2) ship a generation_config that sets
        # sampling-only fields such as ``top_p`` / ``top_k`` / ``typical_p``
        # without ``do_sample=True`` — that used to be a warning, now it's a
        # hard ValueError:
        #
        #     ValueError: GenerationConfig is invalid:
        #     - `top_p`: `do_sample` is not set to `True`. However, `top_p`
        #       is set to `0.95` ...
        #
        # We must not silently flip ``do_sample`` to True (that would change
        # the model's default decoding behaviour).  Instead we drop the
        # sampling-only fields when they are inconsistent with do_sample.
        gen_cfg = getattr(self.quant_model.get_model(), "generation_config", None)
        if gen_cfg is not None and not getattr(gen_cfg, "do_sample", False):
            for _sampling_key in ("top_p", "top_k", "typical_p", "epsilon_cutoff", "eta_cutoff"):
                if getattr(gen_cfg, _sampling_key, None) is not None:
                    print_info(
                        "[generation_config] drop `{}` (do_sample=False, "
                        "value was {}) to pass strict validate()".format(
                            _sampling_key, getattr(gen_cfg, _sampling_key)
                        )
                    )
                    try:
                        setattr(gen_cfg, _sampling_key, None)
                    except Exception:
                        pass
            # ``temperature`` is only meaningful with do_sample=True; the
            # default value 1.0 is safe to keep, but any other value would
            # also trip the strict validator.
            _tmp = getattr(gen_cfg, "temperature", None)
            if _tmp is not None and _tmp != 1.0:
                print_info(
                    "[generation_config] reset `temperature` "
                    "({} -> 1.0) to pass strict validate()".format(_tmp)
                )
                try:
                    gen_cfg.temperature = 1.0
                except Exception:
                    pass
        self.quant_model.get_model().save_pretrained(save_path, max_shard_size="5GB")

        with open(os.path.join(save_path, "hf_quant_config.json"), "w") as f:
            json.dump(trtllm_config, f, indent=4)

        self.quant_model.tokenizer.save_pretrained(save_path)
        # Save KV cache scales if available
        if (
            hasattr(self.quant_model, "kv_cache_scales_dict")
            and self.quant_model.kv_cache_scales_dict
        ):
            kv_scales_path = os.path.join(save_path, "kv_cache_scales.safetensors")
            kv_scales_dict = {}
            kv_scale_map = {}
            for name, scale in self.quant_model.kv_cache_scales_dict.items():
                kv_scales_dict[name] = scale
                kv_scale_map[name] = "kv_cache_scales.safetensors"
            safe_save(kv_scales_dict, kv_scales_path)
            print_info("Save KV cache scales to: {}".format(kv_scales_path))
            new_model_index_file = os.path.join(save_path, "model.safetensors.index.json")
            with open(new_model_index_file, "r") as f:
                new_model_index = json.load(f)
            new_model_index["weight_map"].update(kv_scale_map)
            with open(os.path.join(save_path, "model.safetensors.index.json"), "w") as f:
                json.dump(new_model_index, f, indent=2)


class PTQOnlyScaleSave(PTQSaveBase):
    def __init__(self, quant_model):
        super().__init__(quant_model=quant_model)

    def save(self, save_path):
        a_quant_algo = self.quant_model.quant_config.quant_algo_info["a"]
        ignored_layers = self.quant_model.skip_layer_names()

        static_q_dict = {
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": ("dynamic" if "dynamic" in a_quant_algo else "static"),
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
                _save_path = os.path.join(save_path, "{}.input_scale.{}.pt".format(k, _index))
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
                    torch.distributed.all_reduce(scale, op=torch.distributed.ReduceOp.MAX)
                    print_info(f"after all reduce scale = {scale}")
                    _save_path = os.path.join(save_path, "{}.weight_scale.{}.pt".format(k, _index))
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
            save_model_path = os.path.join("/".join(save_path.split("/")[:-1]), "checkpoint")
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
            parent_dir = os.path.dirname(self.quant_model.model.ori_model_path.rstrip("/"))
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
                torch.distributed.all_reduce(tensor_wise_scale, op=torch.distributed.ReduceOp.MAX)
                weight_fp8 = (
                    (weight_bf16 / tensor_wise_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                )
                weight_fp8 = weight_fp8.to(layer.weight.device)

                if "fp8" in self.quant_model.quant_config.quant_algo:
                    if "w4a8" in self.quant_model.quant_config.quant_algo:
                        new_weight_bf16 = weight_fp8.to(torch.bfloat16) * tensor_wise_scale
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
                        bf16_weight_qdq = new_weight_fp8.to(torch.bfloat16) * tensor_wise_scale
                    else:
                        layer.weight.data = weight_fp8
                        bf16_weight_qdq = weight_fp8.to(torch.bfloat16) * tensor_wise_scale

                cos_sim = torch.cosine_similarity(bf16_weight_qdq, ori_bf16_weight).mean()
                print_info(f"qdq weight cos sim :{cos_sim}")
                if cos_sim < 0.95:
                    print_info("*** cos sim < 0.95 !!!")

    def _save_model(self, save_model_path):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        save_model(
            self.quant_model.model,
            os.path.join(save_model_path, f"model{self.rank}-mp{world_size}.safetensors"),
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
            ori_state_dicts[localind] = safe_open(file_path, framework="pt", device="cpu")
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
            if any(word if word in k else False for word in ["embed_tokens", "lm_head"]):
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

        with open(os.path.join(save_model_path, "model.safetensors.index.json"), "w") as f:
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
                ignore_layers = self.quant_model.quant_config.quant_algo_info["ignore_layers"]
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
                for substring in self.quant_model.quant_config.quant_algo_info["ignore_layers"]
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
                    index_dict["weight_map"][f"{param_name[:-7]}.input_scale"] = str(filename)
                    if "w4a8" in self.quant_model.quant_config.quant_algo:
                        param = self._packed_weight(
                            param_name,
                            param,
                            self.quant_model.quant_config.quant_algo_info["w_group_size"],
                            scales_dict,
                        )
                        new_save_dict[f"{param_name}_scale.int4"] = scales_dict[
                            f"{param_name}_scale.int4"
                        ]
                        index_dict["weight_map"][f"{param_name}_scale.int4"] = str(filename)
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
        print_info(f"Packing {weight_name}, packed weight dtype = {packed_weight.dtype}")
        del bf16_weight
        return packed_weight

    def _read_weight_map(self, input_path):
        model_index_file = os.path.join(input_path, "model.safetensors.index.json")
        with open(model_index_file, "r") as f:
            model_index = json.load(f)
        weight_map = model_index["weight_map"]
        return weight_map

    def _get_tensor_from_safetensor(self, input_path, weight_name, safetensor_file, loaded_files):
        if safetensor_file not in loaded_files:
            current_state_dict = load_file(os.path.join(input_path, safetensor_file), device="cpu")
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
                ignore_layers = self.quant_model.quant_config.quant_algo_info["ignore_layers"]
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
            raise ValueError(f"{self.quant_model.quant_config.quant_algo} not supported")


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

        device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")

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
        device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")

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
                if any(x in k for x in ("layernorm", "gate.weight", "e_score_correction_bias")):
                    if self.rank == 0:
                        layer_out[k] = v.cpu()
                    del state[k]
                    continue

                assert k.endswith("weight"), f"Unexpected param: {k}"

                dim = self.router.tp_gather_dim(k)
                if dim is None:
                    if self.rank == 0 and ("q_a_proj" in k or "kv_a_proj_with_mqa" in k):
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

            tensor = self._get_tensor_from_safetensor(input_path, weight_name, src_file, cache)

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


class Glm5EPQuantSaver(PTQSaveVllmHF):
    """
    GLM-5 int8 (W8A8 / W8A8C8) saver under expert-parallel (EP).

    Under EP each rank only materialises its local 1/W expert slice; the
    shared layers (embed / lm_head / attention / norms / shared_experts) are
    held *完整* (identical) on every rank.  ``PTQSaveVllmHF`` calls
    ``model.save_pretrained`` which would only ever see the local 1/W experts
    on rank 0 and silently drop the other 15/16 — producing a corrupt
    checkpoint.

    This saver instead streams one transformer block at a time:
      * shared / TP layers  -> rank 0 takes its own (complete) copy
      * routed ``mlp.experts.*`` -> ``MoEExpertGather`` gathers every rank's
        local slice via ``gather_object`` onto rank 0
    and writes safetensors shards + a merged ``model.safetensors.index.json``.
    Rank 0 never holds the full 2.8 TB model in memory at once (peak =
    one layer), so the original CPU-OOM root cause stays avoided.

    The quantization format is identical to ``PTQSaveVllmHF`` (per-channel
    int8 ``weight`` + ``weight_scale`` + ``input_scale``, vLLM /
    compressed-tensors), because GLM-5's int8 ``convert`` already produced
    those ``QDQModule`` leaves before save — we only (re)shard, we do NOT
    re-quantize.
    """

    # GLMMoE experts live under model.layers.<lid>.mlp.experts.<eid>.*
    # For the MTP (Multi-Token Prediction) draft layer -- ``model.layers.<N>``
    # where N == num_hidden_layers (e.g. layer 78 in chatglm5.2) -- the MoE
    # is nested one level deeper under ``mtp_block.`` in the AngelSlim
    # modeling code (``model.layers.78.mtp_block.mlp.experts.<eid>.*``).
    # We accept both forms so that MTP experts also participate in EP
    # sharding / gathering, otherwise the entire MTP layer is silently
    # dropped from the exported checkpoint (index.json misses layer 78).
    _EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.(?:mtp_block\.)?mlp\.experts\.")
    _LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")

    @torch.no_grad()
    def save(self, save_path):
        # Fallback to the standard (non-EP) path when EP is not active.
        ep_active = getattr(self.quant_model, "expert_parallel_enabled", False)
        world = int(getattr(self.quant_model, "world_size", 1) or 1)
        if not ep_active or world <= 1:
            return super().save(save_path)

        rank = dist.get_rank() if dist.is_initialized() else 0
        os.makedirs(save_path, exist_ok=True)

        # ---- shared-filesystem sanity check ---------------------------------
        # Multi-node EP save assumes ``save_path`` is visible on every node
        # (each rank writes its own shard files there).  Detect the common
        # foot-gun -- someone points save_path at a node-local disk like
        # ``/dockerdata`` -- before we spend 20+ minutes writing shards that
        # will be silently dropped at merge time.
        if dist.is_initialized():
            sentinel = os.path.join(save_path, f"_shared_fs_sentinel_r{rank:02d}")
            with open(sentinel, "w") as f:
                f.write(str(rank))
            dist.barrier()
            missing = []
            for r in range(world):
                other = os.path.join(save_path, f"_shared_fs_sentinel_r{r:02d}")
                if not os.path.exists(other):
                    missing.append(r)
            dist.barrier()
            if rank == 0:
                # Rank 0 alone cleans up the sentinels.
                for r in range(world):
                    other = os.path.join(save_path, f"_shared_fs_sentinel_r{r:02d}")
                    try:
                        os.remove(other)
                    except FileNotFoundError:
                        pass
            if missing:
                local_prefixes = ("/dockerdata", "/data", "/tmp", "/root")
                local_hint = ""
                if any(save_path.startswith(pfx) for pfx in local_prefixes):
                    local_hint = (
                        f"  ROOT CAUSE HINT: ``save_path={save_path!r}`` "
                        f"is a node-local path.  Multi-node EP save requires "
                        f"a filesystem shared across all nodes (e.g. cephfs).  "
                        f"Point --save-path at ``/apdcephfs*`` or similar.\n"
                    )
                raise RuntimeError(
                    f"Glm5EPQuantSaver (rank {rank}): shared-filesystem check "
                    f"FAILED -- cannot see sentinel files for rank(s) "
                    f"{missing} at ``{save_path}``.  Aborting BEFORE spending "
                    f"time on a checkpoint that would end up half-saved.\n"
                    f"{local_hint}"
                )

        # ---- build int8 vLLM quantization_config (same as PTQSaveVllmHF) ----
        save_name = self.quant_model.quant_config.save_name
        ignore_field = "ignore" if save_name == "compressed-tensors" else "ignored_layers"
        w_quant_algo = self.quant_model.quant_config.quant_algo_info.get("w", "")
        a_quant_algo = self.quant_model.quant_config.quant_algo_info.get("a", "")
        is_dynamic = "dynamic" in a_quant_algo
        ignored_layers = self.quant_model.skip_layer_names()
        trtllm_config = {
            "quantization": {
                "exclude_modules": ignored_layers,
                "kv_cache_quant_algo": None,
            }
        }
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

        quantization_config = {"quant_method": save_name, ignore_field: ignored_layers}
        c_quant_algo = self.quant_model.quant_config.quant_algo_info.get("c", None)
        kv_cache_scheme = (
            {
                "num_bits": 8,
                "strategy": re.search(r"per-([a-zA-Z]+)", c_quant_algo).group(1),
                "type": "float",
            }
            if c_quant_algo is not None
            else None
        )
        extra_kv_cache_scheme = getattr(self.quant_model, "_extra_kv_cache_scheme", None)
        if extra_kv_cache_scheme:
            kv_cache_scheme = dict(extra_kv_cache_scheme)

        quantization_config["activation_scheme"] = "dynamic" if is_dynamic else "static"
        if kv_cache_scheme is not None:
            quantization_config["kv_cache_scheme"] = "static"

        if (
            hasattr(self.quant_model.quant_config, "transform_config")
            and self.quant_model.quant_config.transform_config is not None
        ):
            quantization_config["transform_config"] = (
                self.quant_model.quant_config.transform_config
            )

        quant_dict = {"quantization_config": quantization_config}
        self.quant_model.get_model().config.update(quant_dict)
        print_info("Save quantization_config: {}".format(quant_dict))

        if rank == 0:
            with open(os.path.join(save_path, "hf_quant_config.json"), "w") as f:
                json.dump(trtllm_config, f, indent=4)
            self.quant_model.tokenizer.save_pretrained(save_path)
            # ``config.update`` above only mutates the in-memory copy; vLLM
            # loads from disk, so we must ``save_pretrained`` the config
            # (this writes ``config.json`` + ``generation_config.json``).
            # We intentionally do NOT call ``model.save_pretrained`` here --
            # that would try to dump the state_dict, which under EP has only
            # 1/W experts on rank 0 and would corrupt the shard layout we
            # just built.  Only the *config* portion is needed.
            try:
                cfg = self.quant_model.get_model().config
                cfg.save_pretrained(save_path)
                # Emit generation config as well when present, so vLLM /
                # HuggingFace inference get default sampling parameters.
                gen_cfg = getattr(
                    self.quant_model.get_model(), "generation_config", None
                )
                if gen_cfg is not None:
                    try:
                        gen_cfg.save_pretrained(save_path)
                    except Exception as e:  # noqa: BLE001
                        print_info(
                            f"[Glm5EPQuantSaver] generation_config.save_pretrained "
                            f"skipped: {e}"
                        )
                print_info(
                    f"[Glm5EPQuantSaver] wrote config.json (with "
                    f"quantization_config) to {save_path}"
                )
            except Exception as e:  # noqa: BLE001
                print_info(
                    f"[Glm5EPQuantSaver] WARNING: config.save_pretrained failed: {e}. "
                    f"Please run scripts/ptq/fix_glm5_vllm_meta.py to patch."
                )

        # ---- stream + merge weights (EP) ----
        # rank0 owns experts [0, local), so every transformer block is present
        # in its local state_dict; we use that to enumerate layers.
        state = self.quant_model.get_model().state_dict()

        # ---- [DIAG] MTP layer coverage (layer == num_hidden_layers) ----
        # GLM-5 uses ``model.layers.<num_hidden_layers>`` for the MTP draft
        # layer (e.g. layer 78 in chatglm5.2).  If the exported index.json
        # is missing this layer, the root cause is almost always that
        # ``state_dict()`` on this rank does not contain any
        # ``model.layers.<N>.*`` keys for N == num_hidden_layers.  Emit a
        # one-shot diagnostic (rank 0 only) so users can immediately tell
        # whether the MTP block reached ``state_dict()`` at all.
        try:
            _hf_cfg = self.quant_model.get_model().config
            _mtp_lid = int(getattr(_hf_cfg, "num_hidden_layers", -1))
        except Exception:
            _mtp_lid = -1

        # ---- [DIAG-DEEP] Inspect the live ``model.model.layers[<MTP>]`` slot ----
        # This runs BEFORE the state_dict-based diagnostic so we still get
        # the module-tree probe even if state_dict is empty for MTP.  We
        # print:
        #   * type(layers[MTP]) and its ``repr`` (truncated)
        #   * every named sub-module class name (up to a cap)
        #   * every named parameter: shape / dtype / device / is_meta /
        #     requires_grad -- so we can tell whether the block exists but
        #     has meta-only tensors (which explains why ``state_dict()``
        #     returns no keys)
        #   * every named buffer: shape / dtype / device
        # As a control we also print the parameter summary of the LAST
        # main-stack layer (MTP - 1) so we can compare "empty" MTP to a
        # fully-loaded neighbour.
        if rank == 0 and _mtp_lid >= 0:
            try:
                _root = self.quant_model.get_model()
                _layers = _root.model.layers
                _n_layers_live = len(_layers)
                print(
                    f"[Glm5EPQuantSaver][DIAG-DEEP] len(model.model.layers) "
                    f"= {_n_layers_live}  (main stack expected {_mtp_lid}, "
                    f"+1 MTP -> {_mtp_lid + 1})",
                    flush=True,
                )
                for probe_lid in (_mtp_lid - 1, _mtp_lid):
                    if probe_lid < 0 or probe_lid >= _n_layers_live:
                        print(
                            f"[Glm5EPQuantSaver][DIAG-DEEP] layer {probe_lid} "
                            f"OUT OF RANGE (len={_n_layers_live}); skip.",
                            flush=True,
                        )
                        continue
                    _blk = _layers[probe_lid]
                    _repr = repr(_blk)
                    if len(_repr) > 400:
                        _repr = _repr[:400] + "...<truncated>"
                    print(
                        f"[Glm5EPQuantSaver][DIAG-DEEP] --- layer {probe_lid} "
                        f"({'MTP' if probe_lid == _mtp_lid else 'main-tail'}) "
                        f"type={type(_blk).__name__} module={type(_blk).__module__}",
                        flush=True,
                    )
                    print(
                        f"[Glm5EPQuantSaver][DIAG-DEEP]   repr: {_repr}",
                        flush=True,
                    )
                    # Sub-module class names (top-level children only, cap 40).
                    _kids = list(_blk.named_children())
                    print(
                        f"[Glm5EPQuantSaver][DIAG-DEEP]   named_children "
                        f"({len(_kids)}): "
                        f"{[(n, type(m).__name__) for n, m in _kids][:40]}",
                        flush=True,
                    )
                    # Parameter status.
                    _params = list(_blk.named_parameters(recurse=True))
                    _meta_ct = sum(
                        1 for _, p in _params
                        if getattr(p, "is_meta", False)
                        or (hasattr(p, "device") and p.device.type == "meta")
                    )
                    print(
                        f"[Glm5EPQuantSaver][DIAG-DEEP]   named_parameters "
                        f"total={len(_params)}  meta={_meta_ct}",
                        flush=True,
                    )
                    # Anonymise per-expert indices to shrink output.
                    _seen_pat = {}
                    for _pname, _p in _params:
                        _anon = re.sub(r"experts\.\d+\.", "experts.<*>.", _pname)
                        if _anon in _seen_pat:
                            _seen_pat[_anon] += 1
                            continue
                        _seen_pat[_anon] = 1
                        try:
                            _dev = str(_p.device)
                        except Exception:
                            _dev = "?"
                        _is_meta = (
                            getattr(_p, "is_meta", False)
                            or (hasattr(_p, "device") and _p.device.type == "meta")
                        )
                        print(
                            f"[Glm5EPQuantSaver][DIAG-DEEP]     P {_anon}  "
                            f"shape={tuple(_p.shape)}  dtype={_p.dtype}  "
                            f"device={_dev}  is_meta={_is_meta}  "
                            f"requires_grad={_p.requires_grad}",
                            flush=True,
                        )
                    # Report the per-expert bucket sizes we collapsed above.
                    for _anon, _cnt in _seen_pat.items():
                        if _cnt > 1:
                            print(
                                f"[Glm5EPQuantSaver][DIAG-DEEP]     "
                                f"(× {_cnt} occurrences of pattern {_anon})",
                                flush=True,
                            )
                    # Buffers (norms, RoPE inv_freq, etc.).
                    _buffers = list(_blk.named_buffers(recurse=True))
                    print(
                        f"[Glm5EPQuantSaver][DIAG-DEEP]   named_buffers "
                        f"total={len(_buffers)}",
                        flush=True,
                    )
                    _seen_buf = set()
                    for _bname, _b in _buffers:
                        _anon = re.sub(r"experts\.\d+\.", "experts.<*>.", _bname)
                        if _anon in _seen_buf:
                            continue
                        _seen_buf.add(_anon)
                        try:
                            _dev = str(_b.device)
                        except Exception:
                            _dev = "?"
                        _is_meta = (
                            getattr(_b, "is_meta", False)
                            or (hasattr(_b, "device") and _b.device.type == "meta")
                        )
                        print(
                            f"[Glm5EPQuantSaver][DIAG-DEEP]     B {_anon}  "
                            f"shape={tuple(_b.shape)}  dtype={_b.dtype}  "
                            f"device={_dev}  is_meta={_is_meta}",
                            flush=True,
                        )
            except Exception as _e:  # noqa: BLE001
                print(
                    f"[Glm5EPQuantSaver][DIAG-DEEP] probe FAILED: "
                    f"{type(_e).__name__}: {_e}",
                    flush=True,
                )

        if rank == 0 and _mtp_lid >= 0:
            _mtp_prefix = f"model.layers.{_mtp_lid}."
            _mtp_keys = [k for k in state if k.startswith(_mtp_prefix)]
            _mtp_expert_keys = [
                k for k in _mtp_keys if self._EXPERT_RE.search(k)
            ]
            print(
                f"[Glm5EPQuantSaver][DIAG] MTP layer id = {_mtp_lid}; "
                f"state_dict has {len(_mtp_keys)} keys with prefix "
                f"'{_mtp_prefix}' ({len(_mtp_expert_keys)} matched as "
                f"EP experts by _EXPERT_RE).",
                flush=True,
            )
            for _k in _mtp_keys[:20]:
                try:
                    _shape = tuple(state[_k].shape)
                except Exception:
                    _shape = "?"
                print(
                    f"[Glm5EPQuantSaver][DIAG]   {_k}  shape={_shape}",
                    flush=True,
                )
            if len(_mtp_keys) > 20:
                print(
                    f"[Glm5EPQuantSaver][DIAG]   ...({len(_mtp_keys) - 20} more)",
                    flush=True,
                )
            if not _mtp_keys:
                print(
                    f"[Glm5EPQuantSaver][DIAG] WARNING: state_dict() contains "
                    f"NO keys for MTP layer {_mtp_lid}. The exported "
                    f"checkpoint will be missing this layer entirely. "
                    f"Root cause is UPSTREAM (from_pretrained / EP split / "
                    f"defuse_moe_experts dropped the MTP block); this saver "
                    f"cannot recover it. Consider copying MTP tensors "
                    f"directly from the source HF checkpoint.",
                    flush=True,
                )

        # ---- IndexShare: drop indexer keys on shared layers ----
        # GLM-5 marks each attention layer's indexer as ``full`` or
        # ``shared`` in ``config.indexer_types``.  On ``shared`` layers the
        # ``self_attn.indexer`` ``nn.Module`` is instantiated by modeling
        # code but its parameters are NOT trained/saved (topk is reused
        # from the previous "full" layer).  Keeping them in the exported
        # checkpoint would just persist random-init noise and diverge from
        # the reference weight_map (which contains indexer weights only on
        # the 22 "full" layers: 0/1/2/6/10/.../74/78).
        shared_lids = set()
        get_shared = getattr(
            self.quant_model, "_shared_indexer_layer_ids", None
        )
        if callable(get_shared):
            try:
                shared_lids = set(get_shared())
            except Exception as e:  # noqa: BLE001
                print(
                    f"[Glm5EPQuantSaver] IndexShare detection failed: {e}; "
                    f"falling back to keeping every layer's indexer.",
                    flush=True,
                )
                shared_lids = set()
        if shared_lids:
            drop_re = re.compile(
                r"^model\.layers\.(\d+)\.self_attn\.indexer\."
            )
            dropped = 0
            for k in list(state.keys()):
                m = drop_re.match(k)
                if m and int(m.group(1)) in shared_lids:
                    del state[k]
                    dropped += 1
            if rank == 0:
                print(
                    f"[Glm5EPQuantSaver] IndexShare: dropped {dropped} "
                    f"indexer keys across {len(shared_lids)} shared layers "
                    f"({sorted(shared_lids)[:8]}...).",
                    flush=True,
                )

        safetensors_index = {}
        shard_idx = 1

        def shard_name(i):
            return f"model-{i:05d}.safetensors"

        # ---------- embed_tokens / lm_head / model.norm ----------
        # Identical on every rank under EP (only experts are sharded), so rank 0
        # just copies its own complete copy.
        head_keys = [
            k for k in state
            if k.startswith("model.embed_tokens")
            or k.startswith("lm_head")
            or k.startswith("model.norm")
        ]
        if rank == 0 and head_keys:
            head_out = {k: state[k].cpu() for k in head_keys}
            head_file = shard_name(shard_idx)
            safe_save(head_out, os.path.join(save_path, head_file))
            for k in head_out:
                safetensors_index[k] = head_file
            shard_idx += 1
            del head_out
            gc.collect()
        for k in head_keys:
            del state[k]
        if dist.is_initialized():
            dist.barrier()

        # ---------- per-layer: rank-independent shard writes (no NCCL) --------
        # NCCL/RDMA cannot ferry the ~576 MB per-rank per-layer expert dict
        # (ibv_reg_mr_iova2 hits the MR quota / gets EINVAL for buffers of
        # this size).  Instead every rank writes its OWN pieces directly to
        # the shared cephfs -- shared layers by rank 0, EP experts by their
        # owner -- with unique per-rank shard filenames.  Each rank drops a
        # partial weight_map JSON, rank 0 reunites them into the final
        # ``model.safetensors.index.json`` after a barrier.  Small, safe,
        # and resumable.
        layer_ids = sorted(
            {int(m.group(1)) for k in state if (m := self._LAYER_RE.match(k))}
        )
        # ranks 1..W each get their own shard id space so filenames never
        # collide with rank 0's or with each other's.
        my_shard_idx = 1
        for lid in layer_ids:
            layer_keys = [
                k for k in state
                if self._LAYER_RE.match(k)
                and int(self._LAYER_RE.match(k).group(1)) == lid
            ]
            expert_keys = [k for k in layer_keys if self._EXPERT_RE.search(k)]
            shared_keys = [k for k in layer_keys if not self._EXPERT_RE.search(k)]

            # 1) EP experts owned by THIS rank -> its own shard file.
            if expert_keys:
                local_experts = {k: state[k].cpu() for k in expert_keys}
                shard_file = f"model-r{rank:02d}-{my_shard_idx:05d}.safetensors"
                safe_save(local_experts, os.path.join(save_path, shard_file))
                for k in local_experts:
                    safetensors_index[k] = shard_file
                my_shard_idx += 1
                del local_experts
                gc.collect()

            # 2) Shared (non-expert) portion of this layer -> rank 0 only
            # (identical on every rank under EP; save one copy).
            if rank == 0 and shared_keys:
                shared_out = {k: state[k].cpu() for k in shared_keys}
                shard_file = f"model-r00-{my_shard_idx:05d}.safetensors"
                safe_save(shared_out, os.path.join(save_path, shard_file))
                for k in shared_out:
                    safetensors_index[k] = shard_file
                my_shard_idx += 1
                del shared_out
                gc.collect()

            # Free this layer's memory before the next.
            for k in layer_keys:
                del state[k]

        # Sync so every rank has flushed its shards + partial index before
        # rank 0 stitches them together.
        if dist.is_initialized():
            dist.barrier()

        # Each rank writes its own partial index JSON; rank 0 will merge them.
        partial_index_file = os.path.join(
            save_path, f"_partial_weight_map_r{rank:02d}.json"
        )
        with open(partial_index_file, "w") as f:
            json.dump(safetensors_index, f)

        if dist.is_initialized():
            dist.barrier()

        # ---------- KV cache scales + final index merge (rank 0 only) --------
        if rank == 0:
            # Merge every rank's partial weight_map JSON.  Under multi-node
            # EP, ``save_path`` MUST live on a filesystem shared across ALL
            # nodes (e.g. cephfs); otherwise this loop only sees the local
            # node's ranks and the resulting checkpoint silently drops the
            # experts owned by remote ranks.  We detect that case here and
            # hard-fail, because a "partially saved" 1/2-model looks the
            # same as a fully saved model until you try to load it into
            # vLLM.
            merged_index = {}
            missing_ranks = []
            for r in range(world):
                p = os.path.join(save_path, f"_partial_weight_map_r{r:02d}.json")
                try:
                    with open(p, "r") as f:
                        merged_index.update(json.load(f))
                except FileNotFoundError:
                    missing_ranks.append(r)
            if missing_ranks:
                # Hint at the most common root cause: node-local save_path.
                local_prefixes = ("/dockerdata", "/data", "/tmp", "/root")
                local_hint = ""
                if any(save_path.startswith(pfx) for pfx in local_prefixes):
                    local_hint = (
                        f"\n\n  ROOT CAUSE HINT: ``save_path={save_path!r}`` "
                        f"looks like a node-local path.  Under multi-node EP "
                        f"every rank writes its own shard files, so the save "
                        f"directory MUST be on a filesystem shared across all "
                        f"nodes (e.g. /apdcephfs*).  Move save_path to shared "
                        f"storage and re-run."
                    )
                raise RuntimeError(
                    f"Glm5EPQuantSaver: missing partial weight_map JSON for "
                    f"rank(s) {missing_ranks} (world_size={world}).  Each "
                    f"rank should have written "
                    f"``_partial_weight_map_rXX.json`` to ``{save_path}``.  "
                    f"The resulting checkpoint would be MISSING all experts "
                    f"owned by those ranks -- refusing to write a corrupt "
                    f"model.safetensors.index.json.{local_hint}"
                )

            # ---------- MTP layer offline-quantize fallback ----------
            # If the live model dropped the MTP block (``model.layers.<N>``
            # where N == num_hidden_layers) at ``from_pretrained`` time --
            # confirmed by the [DIAG] probe above showing 0 MTP keys in
            # ``state_dict()`` -- read layer N's weights straight from the
            # source HF checkpoint on disk, quantize them offline using the
            # same recipe as the main stack, and inject them into
            # ``merged_index`` before writing the final ``index.json``.
            # Rank 0 alone owns the entire MTP shard (plan B1).
            self._maybe_emit_mtp_shard_from_source(
                save_path=save_path,
                merged_index=merged_index,
                mtp_lid=_mtp_lid,
                shared_lids=shared_lids,
                user_ignore_patterns=tuple(
                    p for p in ignored_layers if p
                ),
            )

            if (
                hasattr(self.quant_model, "kv_cache_scales_dict")
                and self.quant_model.kv_cache_scales_dict
            ):
                kv_scales_path = os.path.join(save_path, "kv_cache_scales.safetensors")
                kv_scales_dict = {}
                kv_scale_map = {}
                for name, scale in self.quant_model.kv_cache_scales_dict.items():
                    kv_scales_dict[name] = scale
                    kv_scale_map[name] = "kv_cache_scales.safetensors"
                safe_save(kv_scales_dict, kv_scales_path)
                print_info("Save KV cache scales to: {}".format(kv_scales_path))
                merged_index.update(kv_scale_map)

            with open(os.path.join(save_path, "model.safetensors.index.json"), "w") as f:
                json.dump({"metadata": {}, "weight_map": merged_index}, f, indent=2)
            # Clean up the transient partial JSON files.
            for r in range(world):
                p = os.path.join(save_path, f"_partial_weight_map_r{r:02d}.json")
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
            print_info(
                f"GLM-5 EP int8 checkpoint merged and saved to {save_path} "
                f"(total weight_map entries: {len(merged_index)})"
            )

    # ------------------------------------------------------------------
    # MTP layer offline-quantize fallback (plan B1).
    # ------------------------------------------------------------------
    # Root cause it addresses:
    #   The upstream transformers ``GlmMoeDsaModel.__init__`` sizes
    #   ``self.layers`` from ``num_hidden_layers`` only, so the MTP draft
    #   block (``model.layers.<N>`` with N == num_hidden_layers) never
    #   gets a slot.  Every disk key under ``model.layers.<N>.*`` is
    #   silently discarded as "unexpected" at ``from_pretrained`` time,
    #   which is why ``state_dict()`` has 0 MTP keys and the exported
    #   index.json misses the whole layer.
    #
    # The fix here is a self-contained post-processing pass on rank 0:
    #   1. Consult the source HF checkpoint's ``model.safetensors.index.json``
    #      to find every key belonging to layer N.
    #   2. Stream those tensors from their respective source shards
    #      (safe_open lazy handle, ``get_tensor`` per key -- constant memory).
    #   3. For each key, decide by leaf-name / substring rules whether to
    #      quantize to INT8 per-out-channel (weight + weight_scale bf16,
    #      symmetric, same layout ``PTQSaveVllmHF`` writes for the main
    #      stack) or pass through as bf16.
    #   4. Dump one shard ``model-mtp-r00.safetensors`` and append the
    #      resulting key -> file mappings to ``merged_index`` so the
    #      final ``model.safetensors.index.json`` covers layer N too.
    #
    # Safeguards:
    #   * If ``merged_index`` already contains ANY key for layer N (i.e.
    #     upstream got fixed and the state_dict path already emitted it),
    #     this method is a no-op.
    #   * If the source checkpoint can't be read or has 0 keys for layer
    #     N, we log a warning and return -- the caller will still write
    #     ``index.json`` (missing MTP, same as before the patch).
    #   * ``user_ignore_patterns`` is the exact list the main stack used
    #     (from ``skip_layer_names()``), so quantize / skip decisions on
    #     the MTP linears stay 1-to-1 consistent with the main stack.
    def _maybe_emit_mtp_shard_from_source(
        self,
        save_path,
        merged_index,
        mtp_lid,
        shared_lids,
        user_ignore_patterns,
    ):
        """Offline-quantize the MTP layer from the source HF checkpoint.

        Rank 0 only.  Mutates ``merged_index`` in place with the new
        ``model.layers.<mtp_lid>.*`` -> shard-file mappings and writes a
        single safetensors shard ``model-mtp-r00.safetensors`` under
        ``save_path``.
        """
        if mtp_lid is None or mtp_lid < 0:
            print_info(
                "[Glm5EPQuantSaver][MTP] skipped: could not determine "
                "num_hidden_layers; nothing to do."
            )
            return

        mtp_prefix = f"model.layers.{mtp_lid}."

        # ---- guard 1: state_dict path already emitted MTP -----------------
        existing_mtp_keys = [k for k in merged_index if k.startswith(mtp_prefix)]
        if existing_mtp_keys:
            print_info(
                f"[Glm5EPQuantSaver][MTP] already have "
                f"{len(existing_mtp_keys)} keys with prefix '{mtp_prefix}' "
                f"in merged_index (upstream fixed?); skipping offline "
                f"backfill."
            )
            return

        # ---- guard 2: source checkpoint path available --------------------
        # Try several attribute paths in priority order so we work across
        # model classes.  ``modeling_deepseek.py`` sets it as a *class*
        # attribute on the HF model; ``GLM5.from_pretrained`` sets it as
        # an *instance* attribute on the model (and, as a last resort,
        # on the adapter).  Users can also pass ``--model-path`` via YAML
        # ``model.model_path`` -- reach into ``config._name_or_path`` as
        # a final safety net (HF stores it there after ``from_pretrained``).
        src_path = None
        _candidates = []
        _hf_model = self.quant_model.model
        _adapter = self.quant_model
        for _obj, _attr in (
            (_hf_model, "ori_model_path"),
            (_adapter, "ori_model_path"),
            (_adapter, "model_path"),
        ):
            _val = getattr(_obj, _attr, None)
            _candidates.append(
                f"{type(_obj).__name__}.{_attr}={_val!r}"
            )
            if isinstance(_val, str) and _val and os.path.isdir(_val):
                src_path = _val
                break
        if src_path is None:
            # HF stashes the load path on the model's
            # ``config._name_or_path`` after ``from_pretrained``; try that.
            try:
                _val = getattr(
                    getattr(_hf_model, "config", None), "_name_or_path", None
                )
                _candidates.append(
                    f"quant_model.model.config._name_or_path={_val!r}"
                )
                if isinstance(_val, str) and _val and os.path.isdir(_val):
                    src_path = _val
            except Exception:
                pass
        if src_path is None:
            print_info(
                f"[Glm5EPQuantSaver][MTP] ABORT: cannot locate source "
                f"checkpoint path on either quant_model.model or "
                f"quant_model.  Tried: {_candidates}.  Leaving MTP "
                f"layer missing from the exported checkpoint."
            )
            return

        src_index_path = os.path.join(src_path, "model.safetensors.index.json")
        if not os.path.exists(src_index_path):
            print_info(
                f"[Glm5EPQuantSaver][MTP] ABORT: source index.json not "
                f"found at {src_index_path}; leaving MTP layer missing."
            )
            return

        try:
            with open(src_index_path, "r") as f:
                src_weight_map = json.load(f).get("weight_map", {})
        except Exception as e:  # noqa: BLE001
            print_info(
                f"[Glm5EPQuantSaver][MTP] ABORT: cannot read "
                f"{src_index_path}: {e}"
            )
            return

        mtp_disk_keys = [k for k in src_weight_map if k.startswith(mtp_prefix)]
        if not mtp_disk_keys:
            print_info(
                f"[Glm5EPQuantSaver][MTP] ABORT: source checkpoint has 0 "
                f"keys with prefix '{mtp_prefix}'; nothing to backfill."
            )
            return

        # Group keys by source shard so we open each shard file once.
        from collections import defaultdict
        shard_to_keys = defaultdict(list)
        for k in mtp_disk_keys:
            shard_to_keys[src_weight_map[k]].append(k)

        # Lazy imports to avoid touching module-level torch state for
        # code paths that never trigger this method.
        from ....models.llm._glm5_skip_lists import (
            _QUANTIZABLE_LEAF_NAMES,
            _FORCED_SKIP_SUBSTRINGS,
        )

        # Precompute the "indexer is shared on MTP layer?" flag so the
        # decision matches how the main stack treats layer N.  On GLM-5
        # the MTP layer's indexer_types is 'full' (see ``_fix_hf_config``),
        # so this is normally False; we still respect ``shared_lids`` in
        # case a future config marks MTP as shared.
        mtp_indexer_is_shared = mtp_lid in (shared_lids or set())

        def _should_quantize(fqn: str) -> bool:
            """Return True if this linear should be INT8-quantized.

            Mirrors ``GLM5.get_observer_layers`` decision tree (see
            ``glm5.py``).  Any change there must be echoed here.
            """
            leaf = fqn.split(".")[-1]
            if leaf not in _QUANTIZABLE_LEAF_NAMES:
                return False
            if any(pat in fqn for pat in _FORCED_SKIP_SUBSTRINGS):
                return False
            if any(pat in fqn for pat in user_ignore_patterns):
                return False
            # IndexShare: skip indexer sub-linears on shared layers.
            if mtp_indexer_is_shared and ".self_attn.indexer." in fqn:
                return False
            return True

        def _quantize_int8_per_channel(w):
            """Symmetric per-out-channel INT8 quantization.

            Matches the layout ``PTQSaveVllmHF`` writes for the main
            stack under this recipe: ``weight`` is int8, ``weight_scale``
            is bf16 with shape ``(out, 1)``.  Activation is dynamic
            per-token so no ``input_scale`` is emitted.
            """
            # Work in fp32 for numerical safety; cast scale back to bf16
            # to match the main stack.
            w_fp32 = w.to(torch.float32)
            per_ch_absmax = w_fp32.abs().amax(dim=1, keepdim=True)
            # Clamp to avoid div-by-zero on all-zero rows (rare but real
            # for freshly-init MoE experts).
            per_ch_absmax = per_ch_absmax.clamp_min(1e-12)
            weight_scale = (per_ch_absmax / 127.0).to(torch.bfloat16)
            # Cast scale back to fp32 for the divide so we keep precision;
            # storing bf16 is only for the on-disk shard.
            w_int8 = (
                (w_fp32 / weight_scale.to(torch.float32))
                .round()
                .clamp(-128, 127)
                .to(torch.int8)
            )
            return w_int8, weight_scale

        shard_file = "model-mtp-r00.safetensors"
        shard_path = os.path.join(save_path, shard_file)
        out_state = {}
        n_quantized = 0
        n_passthrough = 0

        print_info(
            f"[Glm5EPQuantSaver][MTP] Backfilling layer {mtp_lid} from "
            f"source checkpoint ({src_path}); {len(mtp_disk_keys)} keys "
            f"across {len(shard_to_keys)} source shard file(s)."
        )

        for src_shard_name, keys in shard_to_keys.items():
            src_shard_path = os.path.join(src_path, src_shard_name)
            if not os.path.exists(src_shard_path):
                print_info(
                    f"[Glm5EPQuantSaver][MTP] WARNING: source shard "
                    f"{src_shard_path} not found; skipping {len(keys)} "
                    f"key(s)."
                )
                continue
            with safe_open(src_shard_path, framework="pt", device="cpu") as f:
                for k in keys:
                    try:
                        tensor = f.get_tensor(k)
                    except Exception as e:  # noqa: BLE001
                        print_info(
                            f"[Glm5EPQuantSaver][MTP] WARNING: cannot read "
                            f"{k} from {src_shard_path}: {e}; skipping."
                        )
                        continue

                    # Decide quantize vs pass-through.  Only *.weight of a
                    # whitelisted leaf linear participates in INT8; bias
                    # (there is none for these linears anyway), norms and
                    # scalar tensors pass through untouched.
                    is_weight = k.endswith(".weight") and tensor.ndim == 2
                    if is_weight and _should_quantize(k[: -len(".weight")]):
                        w_int8, w_scale = _quantize_int8_per_channel(tensor)
                        out_state[k] = w_int8
                        out_state[k[: -len(".weight")] + ".weight_scale"] = (
                            w_scale
                        )
                        n_quantized += 1
                    else:
                        # Preserve original dtype (bf16 for norms /
                        # eh_proj / enorm / hnorm / shared_head.norm /
                        # mlp.gate.* / indexer.*).
                        out_state[k] = tensor
                        n_passthrough += 1

        if not out_state:
            print_info(
                f"[Glm5EPQuantSaver][MTP] ABORT: no tensors produced "
                f"(all {len(mtp_disk_keys)} keys failed to read); "
                f"leaving MTP layer missing."
            )
            return

        safe_save(out_state, shard_path)
        for k in out_state:
            merged_index[k] = shard_file
        print_info(
            f"[Glm5EPQuantSaver][MTP] Wrote {shard_path}: "
            f"{n_quantized} linear(s) quantized (weight+weight_scale), "
            f"{n_passthrough} tensor(s) passthrough (bf16). "
            f"Total {len(out_state)} keys added to weight_map."
        )

