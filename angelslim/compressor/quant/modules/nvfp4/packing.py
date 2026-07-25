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

"""FOCUS NVFP4 packing primitives for compressed-tensors export."""

from __future__ import annotations

import torch

from ..helper_layer import nvfp4_cast_to_e2m1, nvfp4_dequantize_e2m1

NVFP4_BLOCK_SIZE = 16
NVFP4_E2M1_MAX = 6.0


def _reshape_blocks(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    if weight.ndim < 2:
        raise ValueError(f"NVFP4 weight must have at least two dimensions, got {weight.shape}")
    if block_size != NVFP4_BLOCK_SIZE:
        raise ValueError(f"compressed-tensors NVFP4 requires block_size=16, got {block_size}")
    if weight.shape[-1] % block_size:
        raise ValueError(
            f"NVFP4 deployment weight width must be divisible by {block_size}, "
            f"got {weight.shape[-1]}"
        )
    return weight.float().reshape(*weight.shape[:-1], -1, block_size)


def nvfp4_quantize_pack(
    weight: torch.Tensor,
    max_scale: torch.Tensor,
    scale_2: torch.Tensor,
    *,
    block_size: int = NVFP4_BLOCK_SIZE,
    quant_max_scale: torch.Tensor | None = None,
    num_sub: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack FOCUS NVFP4 weights into compressed-tensors layout.

    Returns ``(weight_packed, weight_scale, weight_global_scale)``.
    ``quant_max_scale`` is used only for E2M1 code assignment and is discarded.
    """
    blocks = _reshape_blocks(weight, block_size)
    scale_shape = blocks.shape[:-1]
    learned_scale = max_scale.detach().float().reshape(scale_shape).to(weight.device)
    tensor_scale = scale_2.detach().float().reshape(-1).to(weight.device)
    if tensor_scale.numel() != 1:
        raise ValueError(f"NVFP4 scale_2 must be scalar, got {tuple(scale_2.shape)}")
    if (
        not torch.isfinite(learned_scale).all()
        or not (learned_scale > 0).all()
        or not torch.isfinite(tensor_scale).all()
        or not (tensor_scale > 0).all()
    ):
        raise ValueError("FOCUS NVFP4 scales must be finite and positive")

    max_value = blocks.abs().amax(dim=-1) * learned_scale
    local_scale = max_value / NVFP4_E2M1_MAX / tensor_scale
    local_scale = torch.where(max_value == 0, torch.ones_like(local_scale), local_scale)
    finfo = torch.finfo(torch.float8_e4m3fn)
    local_scale_fp8 = local_scale.clamp(min=finfo.tiny, max=finfo.max).to(torch.float8_e4m3fn)
    local_scale_float = local_scale_fp8.float()

    if quant_max_scale is not None:
        if num_sub is None:
            if quant_max_scale.numel() % local_scale.numel():
                raise ValueError("Cannot infer NVFP4 num_sub from quant_max_scale shape")
            num_sub = quant_max_scale.numel() // local_scale.numel()
        if num_sub <= 0 or block_size % num_sub:
            raise ValueError(f"block_size ({block_size}) must be divisible by num_sub ({num_sub})")
        expected_numel = local_scale.numel() * num_sub
        if quant_max_scale.numel() != expected_numel:
            raise ValueError(
                f"quant_max_scale has {quant_max_scale.numel()} elements, "
                f"expected {expected_numel}"
            )
        coefficient = torch.sigmoid(quant_max_scale.detach().float().to(weight.device)).reshape(
            *local_scale.shape, num_sub
        )
        subgroup_blocks = blocks.reshape(
            *local_scale.shape,
            num_sub,
            block_size // num_sub,
        )
        divisor = (
            local_scale_float.unsqueeze(-1).unsqueeze(-1)
            * tensor_scale
            * coefficient.unsqueeze(-1)
        )
        scaled = (subgroup_blocks / divisor).reshape(blocks.shape)
    else:
        effective_scale = local_scale_float * tensor_scale
        scaled = blocks / effective_scale.unsqueeze(-1)

    codes = nvfp4_cast_to_e2m1(scaled.clamp(-NVFP4_E2M1_MAX, NVFP4_E2M1_MAX)).reshape(
        *weight.shape[:-1], weight.shape[-1]
    )
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    weight_global_scale = 1.0 / tensor_scale
    return (
        packed.contiguous(),
        local_scale_fp8.contiguous(),
        weight_global_scale.cpu(),
    )


def nvfp4_unpack_dequantize(
    packed_weight: torch.Tensor,
    local_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Unpack compressed-tensors NVFP4 weights for parity validation."""
    if packed_weight.dtype != torch.uint8:
        raise TypeError(f"NVFP4 packed weight must be uint8, got {packed_weight.dtype}")
    if local_scale.dtype != torch.float8_e4m3fn:
        raise TypeError(f"NVFP4 local scale must be float8_e4m3fn, got {local_scale.dtype}")
    if packed_weight.shape[-1] * 2 % block_size:
        raise ValueError("Packed NVFP4 width is incompatible with block_size")

    codes = torch.empty(
        *packed_weight.shape[:-1],
        packed_weight.shape[-1] * 2,
        dtype=torch.uint8,
        device=packed_weight.device,
    )
    codes[..., 0::2] = packed_weight & 0x0F
    codes[..., 1::2] = packed_weight >> 4
    values = nvfp4_dequantize_e2m1(codes, dtype=torch.float32)
    blocks = values.reshape(*values.shape[:-1], -1, block_size)
    global_scale = weight_global_scale.float().reshape(-1).to(values.device)
    if global_scale.numel() != 1:
        raise ValueError("NVFP4 weight_global_scale must be scalar")
    dequant = blocks * (local_scale.float() / global_scale).unsqueeze(-1)
    return dequant.reshape(values.shape).to(dtype)
