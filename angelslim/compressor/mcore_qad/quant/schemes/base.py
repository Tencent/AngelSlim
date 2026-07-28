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

"""Axis 2 -- Scale Scheme: how scales are organized over a tensor.

A scheme is an nn.Module that OWNS its scale storage (ScaleStores) and implements
the fake-quant math for its granularity:
  * quantize(x, fmt)   -> fake-quantized x, with scale obtained from its store(s).
                          The scheme computes the data-derived reference
                          (amax / qmax) and lets the store decide learnable vs
                          dynamic vs calibrated. Multi-level schemes (NVFP4) own
                          multiple stores (block + global) internally.
  * parallel_spec(host)-> how the scale must be sharded/replicated/synced (mcore).

Because the scheme owns parameters, a fresh scheme instance is built per Quantizer.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor

from angelslim.compressor.mcore_qad.parallel.parallel_spec import ParallelSpec
from angelslim.compressor.mcore_qad.quant.formats.base import QuantFormat
from angelslim.compressor.mcore_qad.quant.registry import Registry

SCHEME_REGISTRY: Registry["ScaleScheme"] = Registry("scheme")


class HostInfo:
    """Description of the hosting module the scheme needs for parallel placement."""

    def __init__(
        self,
        parallel_mode: str,
        shard_dim: int | None,
        is_expert: bool,
        sequence_parallel: bool,
        context_parallel: bool,
    ) -> None:
        self.parallel_mode = parallel_mode  # "column"|"row"|"replicated"|"expert"
        self.shard_dim = shard_dim
        self.is_expert = is_expert
        self.sequence_parallel = sequence_parallel
        self.context_parallel = context_parallel


class ScaleScheme(torch.nn.Module):
    @abstractmethod
    def quantize(self, x: Tensor, fmt: QuantFormat) -> Tensor: ...

    @abstractmethod
    def parallel_spec(self, host: HostInfo) -> ParallelSpec: ...
