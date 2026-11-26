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

__all__ = ["gptq_sweep"]


def gptq_sweep(weight, hinv, group_size, group_scale, quantize, code_dtype, static_groups=False):
    """GPTQ of ``weight`` [.., n, k] with upper-Cholesky ``hinv`` [.., k, k].
    ``group_scale(block) -> (scale, eff)`` reads a group's scale from the compensated weights,
    ``quantize(w, eff) -> (codes, dequant)`` rounds columns; returns (codes, group scales).
    ``static_groups`` rounds a whole group at once and compensates only later groups."""
    k = weight.shape[-1]
    if k % group_size:
        raise ValueError(f"input dim {k} is not a multiple of the group size {group_size}")
    if hinv.shape[-1] != k:
        raise ValueError(f"hinv {tuple(hinv.shape)} does not match weight {tuple(weight.shape)}")
    W = weight.to(device=hinv.device, dtype=torch.float32, copy=True)
    codes = torch.empty(W.shape, dtype=code_dtype, device=W.device)
    inv_diag = 1.0 / hinv.diagonal(dim1=-2, dim2=-1).clamp(min=1e-12)
    scales = []
    for c0 in range(0, k, group_size):
        c1 = c0 + group_size
        scale, eff = group_scale(W[..., c0:c1])
        scales.append(scale)
        if static_groups:
            block = W[..., c0:c1]
            code, dequant = quantize(block, eff)
            codes[..., c0:c1] = code
            if c1 < k:
                err = (block - dequant) * inv_diag[..., c0:c1].unsqueeze(-2)
                W[..., c1:] -= err @ hinv[..., c0:c1, c1:]
            continue
        err = torch.zeros(W.shape[:-1] + (group_size,), dtype=torch.float32, device=W.device)
        for j in range(c0, c1):
            w = W[..., j : j + 1]
            code, dequant = quantize(w, eff)
            codes[..., j : j + 1] = code
            resid = (w - dequant) * inv_diag[..., j, None, None]
            err[..., j - c0 : j - c0 + 1] = resid
            if j + 1 < c1:
                W[..., j + 1 : c1] -= resid * hinv[..., j, None, j + 1 : c1]
        if c1 < k:
            W[..., c1:] -= err @ hinv[..., c0:c1, c1:]
    return codes, scales
