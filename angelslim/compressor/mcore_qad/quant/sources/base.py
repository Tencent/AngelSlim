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

"""Axis 3 -- Scale Source: where a scale value comes from & whether it trains.

A ScaleStore turns a data-derived *reference* scale (computed by the scheme as
amax / qmax) into the scale actually used:
  * LearnableScale  : a trainable parameter, lazily initialized from the reference
                      on first use, then learned (LSQ). The ONLY trainable params
                      in the framework (weights are frozen).
  * CalibratedScale : a frozen buffer (loaded from outside); ignores the reference.
  * DynamicScale    : returns the reference itself (recomputed every forward), used
                      for activation per-(token,block) scales.

Keeping the "reference = amax/qmax" convention in the scheme means every store is
shape-agnostic in its math and the three behaviours differ only in storage.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor

from angelslim.compressor.mcore_qad.quant.registry import Registry

SOURCE_REGISTRY: Registry["ScaleStore"] = Registry("source")


class ScaleStore(torch.nn.Module):
    @abstractmethod
    def forward(self, ref: Tensor) -> Tensor:
        """Return the scale to use, given the data-derived reference scale `ref`."""

    def is_learnable(self) -> bool:
        return False

    def init_value(self, value: Tensor) -> None:
        """Seed from an external scale (PTQ / checkpoint). Default: no-op."""


def build_store(kind: str, shape) -> ScaleStore:
    """Factory: dynamic stores need no shape; learnable/calibrated do."""
    if kind == "dynamic":
        return SOURCE_REGISTRY.create("dynamic")
    if kind in ("learnable", "calibrated"):
        if shape is None:
            raise ValueError(
                f"source '{kind}' requires a known scale shape "
                f"(got None); only 'dynamic' may omit it."
            )
        return SOURCE_REGISTRY.create(kind, shape)
    raise KeyError(f"unknown source kind: {kind!r}")
