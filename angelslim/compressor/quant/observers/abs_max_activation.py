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

__all__ = [
    "AbsmaxPertensorObserver",
    "AbsMaxTokenWiseActObserver",
    "AbsmaxPerchannelObserver",
]


class AbsmaxPertensorObserver(BaseObserver):
    def __init__(self, layer=None, quant_bits=8, **kwargs):
        super(AbsmaxPertensorObserver, self).__init__(quant_bits=quant_bits)
        self._layer = layer
        self._quant_bits = quant_bits
        self._scale = None
        self._zero_point = None
        self._min = None
        self._max = torch.tensor(1e-7, dtype=torch.float32)
        self.step = 0
        self.dtype = None
        self.parent_observer = (
            kwargs["parent_observer"]
            if kwargs and "parent_observer" in kwargs
            else None
        )

    def forward(self, inputs):
        """Calculate forward pass."""

        self.step += 1
        if not self.dtype:
            self.dtype = inputs.dtype
        if inputs.numel() > 0:
            self._min, self._max = self._cal_min_max(inputs)
            if self.parent_observer is not None:
                self.parent_observer.update(self._min, self._max, self.step)
        else:
            assert self.parent_observer is not None
            self._update_min_max(self.parent_observer.min, self.parent_observer.max)
        return inputs

    def _cal_min_max(self, inputs):
        abs_max_val = torch.max(torch.abs(inputs))
        # abs_max_val = torch.maximum(abs_max_val, self._max)
        if abs_max_val.data < self._max.data:
            abs_max_val = self._max
        # torch.cuda.empty_cache()
        return 0, abs_max_val.to(inputs.device)

    def _update_min_max(self, min, max):
        if min is not None and max is not None:
            if self._min is None or min < self._min:
                self._min = min
            if self._max is None or max > self._max:
                self._max = max

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
        if self.step == 0 and self.parent_observer is not None:
            self._update_min_max(self.parent_observer.min, self.parent_observer.max)
            self.step = self.parent_observer.step
        if self.step == 0:
            raise ValueError(
                "AbsmaxPertensorObserver scales must calibrate data first!"
            )
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


class AbsMaxTokenWiseActObserver(BaseObserver):
    def __init__(
        self,
        layer=None,
        quant_bits=8,
    ):
        super(AbsMaxTokenWiseActObserver, self).__init__(quant_bits=quant_bits)
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
        return inputs

    def _cal_min_max(self, inputs):
        new_inp = inputs
        if len(new_inp.shape) == 3:
            n_token = new_inp.shape[2]
            new_inp = new_inp.permute([2, 0, 1]).reshape([n_token, -1])
            abs_max_values, _ = new_inp.abs().max(dim=self.quant_axis())
        else:
            abs_max_val = torch.max(torch.abs(new_inp))
            abs_max_values = torch.maximum(abs_max_val, self._max)
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
        if self._scale is None:
            self.cal_thresholds()
        self._scale = self._scale.type(self.dtype)
        return self._scale

    def zero_points(self):
        """Return output zero points."""
        if self._zero_point is None:
            self.cal_thresholds()
        return self._zero_point


class AbsmaxPerchannelObserver(BaseObserver):
    def __init__(self, layer, quant_bits=8, **kwargs):
        super(AbsmaxPerchannelObserver, self).__init__(quant_bits=quant_bits)
        self._layer = layer
        self._quant_bits = quant_bits
        self._scale = None
        self._zero_point = None
        self._min = None
        self._max = (
            torch.zeros((self._layer.weight.shape[0])) - torch.inf
        )  # per-outchannel
        self.step = 0
        self.dtype = None

    def forward(self, inputs):
        """Calculate forward pass."""
        if not self.dtype:
            self.dtype = inputs.dtype
        x = inputs.clone()
        self._min, self._max = self._cal_min_max(x)
        del x
        self.step += 1
        return inputs

    def _cal_min_max(self, tensor, dim=0):
        hidden_dim = tensor.shape[-1]
        tensor_x = tensor.view(-1, hidden_dim).abs().detach()
        comming_max_x = torch.max(tensor_x, dim=dim)[0].float().cpu()
        return 0, torch.max(self._max, comming_max_x)

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
            raise ValueError(
                "AbsmaxPerchannelObserver scales must calibrate data first!"
            )
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
