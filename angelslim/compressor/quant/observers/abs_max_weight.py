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

from .base_observer import BaseObserver


class AbsMaxChannelWiseWeightObserver(BaseObserver):
    def __init__(self, layer=None, quant_bits=8, group_size=-1):
        super(AbsMaxChannelWiseWeightObserver, self).__init__(quant_bits=quant_bits)
        assert group_size < 0, "ChannelWise quantization, group_size cannot > 0."
        self._layer = layer
        self._quant_bits = quant_bits
        self._scale = None
        self._zero_point = None
        self._min = None
        self._max = torch.tensor(1e-7, dtype=torch.float32)
        self.step = 0
        self.dtype = None

    def forward(self, inputs):
        """Calculate forward pass."""
        if not self.dtype:
            self.dtype = inputs.dtype
        self._min, self._max = self._cal_min_max(inputs)
        self.step += 1
        return inputs

    def _cal_min_max(self, inputs):
        new_inp = inputs
        # abs_max_values, _ = new_inp.abs().max(dim=self.quant_axis(), keepdim=True)
        # abs_max_values, _ = abs_max_values.max(dim=0)
        # abs_max_values = abs_max_values.squeeze()
        if len(new_inp.shape) > 2:
            new_inp = new_inp.flatten(1)
        abs_max_values, _ = new_inp.abs().max(dim=self.quant_axis())
        return 0, abs_max_values

    def cal_thresholds(self):
        """Compute thresholds for MAX function."""
        if self._scale is None:
            self._scale = self._max
        self._zero_point = torch.zeros_like(self._scale)

    def quant_axis(self):
        """Return quantization axis."""
        return -1

    def scales(self):
        """Return output scales."""
        if self.step == 0:
            raise ValueError("AbsMaxChannelWiseWeightObserver scales must calibrate data first!")
        if self._scale is None:
            self.cal_thresholds()
        if self.dtype:
            self._scale = self._scale.type(self.dtype)
        return self._scale

    def zero_points(self):
        """Return output zero points."""
        if self._zero_point is None:
            self.cal_thresholds()
        return self._zero_point
