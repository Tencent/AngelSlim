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

from typing import List

from ..observers import (
    AbsMaxChannelWiseWeightObserver,
    AbsMaxGroupWiseWeightObserver,
    AbsmaxPerchannelObserver,
    AbsmaxPertensorObserver,
)

ACT_OBSERVERS_CLASS = {
    "per-tensor": AbsmaxPertensorObserver,
    "per-channel": AbsmaxPerchannelObserver,
}
WEIGHT_OBSERVERS_CLASS = {
    "per-tensor": AbsmaxPertensorObserver,
    "per-channel": AbsMaxChannelWiseWeightObserver,
    "per-group": AbsMaxGroupWiseWeightObserver,
}

KVCACHE_OBSERVERS_CLASS = {"per-channel": AbsmaxPerchannelObserver}


class QuantConfig:
    r"""
    Configure how to quantize a model or a part of the model. It will map each layer to
    an instance of observers by the settings.

    Args:
        config: The quant config.
    Examples:
        .. code-block:: python

            >>> from slim.quant import QuantConfig
            >>> q_config = QuantConfig(yaml_config)
    """

    def __init__(self, config, global_config=None):
        # quant_algo change
        self.act_observer = None
        self.weight_observer = None
        self.kv_cache_observer = None

        quantization_args = config.quantization
        self.quant_algo = quantization_args.name
        self.quant_bit = quantization_args.bits
        self.quant_helpers = quantization_args.quant_helpers
        act_quant_method = quantization_args.quant_method.get("activation", None)
        weight_quant_method = quantization_args.quant_method["weight"]
        self.cpu_convert = quantization_args.cpu_convert
        self.save_name = quantization_args.save_name

        if global_config:
            self.max_seq_length = global_config.max_seq_length
            self.hidden_size = global_config.hidden_size
            self.model_arch_type = global_config.model_arch_type

        if "fp8" in self.quant_algo:
            is_dynamic = "dynamic" if "dynamic" in self.quant_algo else "static"
            assert (
                is_dynamic or act_quant_method is not None
            ), "[Error] fp8_static need act_quant_method"
            self.act_observer = (
                ACT_OBSERVERS_CLASS[act_quant_method]
                if "static" in is_dynamic
                else None
            )
            self.weight_observer = WEIGHT_OBSERVERS_CLASS[weight_quant_method]
            self.kv_cache_observer = None

            if "w4a8" in self.quant_algo:
                group_size = (
                    128
                    if quantization_args.quant_method["group_size"] == -1
                    else quantization_args.quant_method["group_size"]
                )
                self.quant_algo_info = {
                    "w": f"int4_{weight_quant_method}",
                    "w_group_size": group_size,
                    "ignore_layers": quantization_args.ignore_layers,
                }
            else:
                self.quant_algo_info = {
                    "w": f"fp8_{weight_quant_method}",
                    "ignore_layers": quantization_args.ignore_layers,
                }

            if act_quant_method is not None:
                self.quant_algo_info["a"] = f"fp8_{act_quant_method}-{is_dynamic}"
            self.low_memory = config.quantization.low_memory
            self.quant_analyse = config.quantization.quant_analyse
            self.quant_vit = config.quantization.quant_vit
        elif "w4a8i8" in self.quant_algo:
            group_size = quantization_args.quant_method["group_size"]
            self.quant_algo_info = {
                "group_size": group_size,
                "ignore_layers": quantization_args.ignore_layers,
            }
            self.low_memory = config.quantization.low_memory
        elif "int8" in self.quant_algo:
            is_dynamic = "dynamic" if "dynamic" in self.quant_algo else "static"
            assert (
                is_dynamic or act_quant_method is not None
            ), "[Error] int8_static need act_quant_method"
            self.act_observer = (
                ACT_OBSERVERS_CLASS[act_quant_method]
                if "static" in is_dynamic
                else None
            )
            self.weight_observer = WEIGHT_OBSERVERS_CLASS[weight_quant_method]
            self.kv_cache_observer = None
            self.quant_algo_info = {
                "w": f"int8_{weight_quant_method}",
                "ignore_layers": quantization_args.ignore_layers,
            }
            if act_quant_method is not None:
                self.quant_algo_info["a"] = f"int8_{act_quant_method}-{is_dynamic}"
            self.low_memory = config.quantization.low_memory
            self.quant_analyse = config.quantization.quant_analyse
        elif "int4_awq" in self.quant_algo:
            self.act_observer = None
            self.weight_observer = None
            self.kv_cache_observer = None
            group_size = (
                128
                if quantization_args.quant_method["group_size"] == -1
                else quantization_args.quant_method["group_size"]
            )
            self.quant_algo_info = {
                "zero_point": quantization_args.quant_method["zero_point"],
                "group_size": int(group_size),
                "mse_range": quantization_args.quant_method["mse_range"],
            }
            self.low_memory = config.quantization.low_memory
        elif "int4_gptq" in self.quant_algo or "int4_gptaq" in self.quant_algo:
            self.act_observer = None
            self.weight_observer = None
            self.kv_cache_observer = None
            group_size = (
                128
                if quantization_args.quant_method["group_size"] == -1
                else quantization_args.quant_method["group_size"]
            )
            self.quant_algo_info = {
                "group_size": group_size,
                "ignore_layers": quantization_args.ignore_layers,
            }
        elif "nvfp4" in self.quant_algo:
            is_dynamic = "dynamic" if "dynamic" in self.quant_algo else "static"
            assert (
                is_dynamic or act_quant_method is not None
            ), "[Error] nvfp4 need act_quant_method"
            self.act_observer = (
                AbsmaxPertensorObserver if "static" in is_dynamic else None
            )
            self.weight_observer = AbsmaxPertensorObserver
            self.kv_cache_observer = None
            block_size = (
                16
                if quantization_args.quant_method["group_size"] == -1
                else quantization_args.quant_method["group_size"]
            )

            self.quant_algo_info = {
                "w": f"nvfp4_{weight_quant_method}",
                "ignore_layers": quantization_args.ignore_layers,
                "block_size": block_size,
            }

            if act_quant_method is not None:
                self.quant_algo_info["a"] = f"nvfp4_{act_quant_method}-{is_dynamic}"

        if "smooth" in self.quant_helpers:
            self.smooth_alpha = quantization_args.smooth_alpha
            self.smooth_observer = ACT_OBSERVERS_CLASS["per-channel"]
        self.custom_observe_layers_names = "default"

    def custom_observe_layers(
        self,
        names: List,
        act_observer="default",
        weight_observer="default",
        kv_cache_observer="default",
    ):
        """
        name supports fuzzy search.
        """
        self.custom_observe_layers_names = names
        self.act_observer = (
            act_observer if act_observer in ACT_OBSERVERS_CLASS else self.act_observer
        )
        self.weight_observer = (
            weight_observer
            if weight_observer in WEIGHT_OBSERVERS_CLASS
            else self.weight_observer
        )
        self.kv_cache_observer = (
            kv_cache_observer
            if kv_cache_observer in KVCACHE_OBSERVERS_CLASS
            else self.kv_cache_observer
        )
