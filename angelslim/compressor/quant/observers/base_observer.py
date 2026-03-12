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
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ....utils import print_info


class BaseObserver(nn.Module, metaclass=abc.ABCMeta):
    """
    Customized observers should extend this base observer
    and implement abstract methods.
    """

    def __init__(
        self,
        quant_bits=8,
        sign=True,
        symmetric=True,
    ):
        super(BaseObserver, self).__init__()
        self._quant_bits = quant_bits
        self._sign = sign
        self._symmetric = symmetric

        self._min = None
        self._max = None
        self._qmin = None
        self._qmax = None

        self._scale = None
        self._zero_point = None

    @abc.abstractmethod
    def cal_thresholds(self):
        pass

    @abc.abstractmethod
    def scales(self):
        pass

    def min_value(self):
        return self._min

    def max_value(self):
        return self._max

    def bit_length(self):
        """Return the bit length of quantized data."""
        return self._quant_bits

    @property
    def qmin_qmax(self):
        """Calculate the range of the quantized integer based on the specified
        quant_bits, sign, and symmetric properties."""
        if self._sign:
            self._qmin = -(2 ** (self.bit_length() - 1))
            self._qmax = 2 ** (self.bit_length() - 1) - 1
        else:
            self._qmin = 0
            self._qmax = 2 ** self.bit_length()
        return self._qmin, self._qmax

    def cal_scales_zero_points(self):
        """Calculate the scales and zero points based on the min_value and max_value."""
        # assert self.min_value() is not None and self.max_value() is not None
        if self.min_value() is None or self.max_value() is None:
            print_info("[Warning] scales is None, Please check calibration and model.")
            return None, None
        _qmin, _qmax = self.qmin_qmax
        if self._min == 0:
            self._min = torch.tensor(0.0)
        _min = torch.minimum(self.min_value(), torch.tensor(0.0))
        _max = torch.maximum(self.max_value(), torch.tensor(0.0))

        if self._symmetric:
            self._scale = torch.maximum(-_min, _max)
            if self._sign:
                self._zero_point = 0
            else:
                self._zero_point = (_qmax + _qmin) / 2
        else:
            self._scale = (_max - _min) / float(_qmax - _qmin)
            self._zero_point = _qmin - round(_min / self._scale)
            self._zero_point = np.clip(self._zero_point, _qmin, _qmax)
        return self._scale, self._zero_point


@dataclass
class ParentObserver:
    min = None
    max = None
    step = 0

    def update(self, min, max, step=None):
        if min is not None and max is not None:
            if self.min is None or min < self.min:
                self.min = min
            if self.max is None or max > self.max:
                self.max = max
        if step is not None:
            self.step = step
