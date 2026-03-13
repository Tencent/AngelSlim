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

import numpy as np
import torch

from .abs_max_activation import AbsmaxPertensorObserver


class EMAObserver(AbsmaxPertensorObserver):
    def __init__(
        self,
        layer=None,
        quant_bits=8,
    ):
        super(EMAObserver, self).__init__(quant_bits=quant_bits)
        self._layer = layer
        self._quant_bits = quant_bits
        self._scale = None
        self._zero_point = None
        self._min = None
        self._max = torch.tensor(1e-7, dtype=torch.float32)
        self.step = 0
        self.dtype = None
        self.ema_beta = 0.98
        self.ema_step = 0
        self.sampled = None

    def forward(self, inputs):
        """Calculate forward pass."""
        if not self.dtype:
            self.dtype = inputs.dtype
        x = inputs.clone()
        self._min, self._max = self._cal_min_max(x)
        del x
        self.step += 1
        torch.cuda.empty_cache()
        return inputs

    def _cal_min_max(self, inputs):
        abs_max_val = torch.max(torch.abs(inputs)).item()
        if not self.ema_step:
            self.sampled = (1 - self.ema_beta) * abs_max_val
            v_ema_corr = self.sampled
        else:
            v_ema = self.ema_beta * self.sampled + (1.0 - self.ema_beta) * abs_max_val
            self.sampled = v_ema
            v_ema_corr = v_ema / float((1.0 - np.power(self.ema_beta, self.ema_step + 1.0)))
        self.ema_step += 1
        return 0, torch.tensor(v_ema_corr, dtype=self.dtype).to(inputs.device)
