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

"""Low-level autograd primitives for fake quantization (pure torch, backend-agnostic).

These define how gradients flow through the (non-differentiable) rounding/snapping
ops. We use the modern detach-based straight-through formulation rather than custom
autograd.Function classes -- it is simpler, composes with any grid, and yields the
correct LSQ scale-gradient automatically (see note below).

Why plain STE + clamp already gives the LSQ gradient w.r.t. scale:
    dequant = snap_ste(x / s) * s
    with snap_ste having identity gradient, d(dequant)/ds = snap(x/s) - x/s for the
    in-range elements, and = +/-qmax for the clamp-saturated elements (because the
    clamped value is constant in x but linear in s). That is exactly the LSQ
    estimator. The only extra LSQ ingredient is a gradient-magnitude rescale on the
    scale parameter, provided here by `grad_scale`.
"""

from __future__ import annotations

import torch
from torch import Tensor


def round_ste(x: Tensor) -> Tensor:
    """Round to nearest integer; identity gradient (straight-through)."""
    return (torch.round(x) - x).detach() + x


def ste(quantized: Tensor, x: Tensor) -> Tensor:
    """Straight-through wrapper: forward returns `quantized`, backward flows to `x`.

    Use as ``return ste(hard_quantized_value, differentiable_input)``.
    """
    return x + (quantized - x).detach()


def grad_scale(x: Tensor, factor: float) -> Tensor:
    """Identity in forward, multiply gradient by `factor` in backward.

    Used to apply the LSQ scale-gradient rescale (1/sqrt(numel*qmax)) so the scale
    parameter trains at a stable magnitude relative to the weights/activations.
    """
    return (x - x * factor).detach() + x * factor


def nearest_level(magnitude: Tensor, levels: Tensor) -> Tensor:
    """Snap non-negative `magnitude` to the nearest value in sorted 1-D `levels`.

    No gradient (callers wrap with `ste`). `levels` must be ascending and start at 0.
    Implemented via midpoint bucketize for a non-uniform grid (e.g. E2M1).
    """
    mids = (levels[:-1] + levels[1:]) * 0.5
    idx = torch.bucketize(magnitude, mids)
    return levels[idx]


def lsq_grad_scale(numel: int, qmax: float) -> float:
    """LSQ recommended scale-gradient rescale factor 1/sqrt(numel * qmax)."""
    return 1.0 / max((numel * qmax) ** 0.5, 1e-12)
