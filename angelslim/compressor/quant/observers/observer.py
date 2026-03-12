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

import abc

import torch.nn as nn


class PTQObserver(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        layer,
        act_observer,
        weight_observer,
        kv_cache_observer,
        quant_algo_dict,
        smooth_act_observer=None,
        parent_observer=None,
        **kwargs,
    ):
        super(PTQObserver, self).__init__()
        self.layer = layer
        self.quant_algo_dict = quant_algo_dict
        self.act_observer = None
        self.smooth_act_observer = None
        self.weight_observer = None
        self.kv_cache_observer = None
        if act_observer:
            if parent_observer is not None:
                kwargs["parent_observer"] = parent_observer
            self.act_observer = act_observer(
                self.layer, quant_bits=quant_algo_dict["a_quant_bits"], **kwargs
            )
        if smooth_act_observer:
            if parent_observer is not None:
                kwargs["parent_observer"] = parent_observer
            self.smooth_act_observer = smooth_act_observer(
                self.layer, quant_bits=quant_algo_dict["a_quant_bits"], **kwargs
            )
        if kv_cache_observer:
            self.kv_cache_observer = kv_cache_observer(
                self.layer,
                quant_bits=quant_algo_dict["c_quant_bits"],
                quantization_wise=quant_algo_dict["c_quant_algo"],
            )
        if weight_observer:
            self.weight_observer = weight_observer(
                self.layer,
                quant_bits=quant_algo_dict["w_quant_bits"],
                group_size=quant_algo_dict.get("w_group_size", -1),
            )

    def forward(self, input, output):
        if self.act_observer is not None:
            self.act_observer(input)
        if self.kv_cache_observer is not None:
            self.kv_cache_observer(output)
        if self.smooth_act_observer is not None:
            self.smooth_act_observer(output)
