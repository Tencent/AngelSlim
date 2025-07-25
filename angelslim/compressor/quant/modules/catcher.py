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

__all__ = ["Catcher"]


class Catcher(torch.nn.Module):
    def __init__(self, module, inps, cache):
        super().__init__()
        self.module = module
        self.inps = inps
        self.cache = cache
        self.layer_kwargs = {}

    def forward(self, inp, **kwargs):
        for i in range(inp.shape[0]):
            self.inps[self.cache["i"], :, :] = inp[i, :, :]
            self.cache["i"] += 1
        self.layer_kwargs.update(kwargs)
        raise ValueError

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
