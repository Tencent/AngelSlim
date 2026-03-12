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


class MultiStepSampler:
    def __init__(self):
        pass

    def sample(self, x, sampled_x=None, layer_name=None):
        return torch.cat([x, sampled_x], dim=1)


class EMASampler:
    def __init__(self):
        self.ema_beta = 0.98
        self.ema_step = {}
        self.sampled = {}

    def sample(self, x, sampled_x=None, layer_name=None):
        if layer_name not in self.ema_step:
            self.sampled[layer_name] = (1 - self.ema_beta) * x
            self.ema_step[layer_name] = 0
            return self.sampled[layer_name]
        else:
            v_ema = self.ema_beta * self.sampled[layer_name] + (1.0 - self.ema_beta) * x
            self.sampled[layer_name] = v_ema
            v_ema_corr = v_ema / float(
                (1.0 - np.power(self.ema_beta, self.ema_step[layer_name] + 1.0))
            )
            self.ema_step[layer_name] += 1
            return v_ema_corr
