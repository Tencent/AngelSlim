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

"""OCP MXFP4 packing primitives used by FOCUS deployment export.

The serialized layout matches vLLM compressed-tensors
``mxfp4-pack-quantized``:

* two E2M1 values per uint8 byte (first value in the low nibble);
* one E8M0 scale bit-pattern per group of 32 values;
* no tensor-level/global scale.
"""

from __future__ import annotations

import torch

MXFP4_GROUP_SIZE = 32
MXFP4_E2M1_MAX = 6.0
FP32_MIN_NORMAL = 2.0**-126

_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _require_e8m0_dtype() -> torch.dtype:
    dtype = getattr(torch, "float8_e8m0fnu", None)
    if dtype is None:
        raise RuntimeError("MXFP4 export requires torch.float8_e8m0fnu support")
    return dtype


def mxfp4_cast_to_e2m1(value: torch.Tensor) -> torch.Tensor:
    """Round pre-scaled values to E2M1 using round-half-to-even."""
    bounds = _E2M1_BOUNDS.to(device=value.device, dtype=value.dtype)
    tie_round_up = torch.tensor(
        [False, True, False, True, False, True, False],
        device=value.device,
    )
    magnitude = value.abs()
    ordinal = torch.searchsorted(bounds, magnitude, out_int32=True).to(torch.uint8)
    round_up = torch.any(
        (magnitude.unsqueeze(-1) == bounds) & tie_round_up,
        dim=-1,
    )
    sign = (value < 0).to(torch.uint8) << 3
    return (sign | (ordinal + round_up.to(torch.uint8))).to(torch.uint8)


def mxfp4_dequantize_e2m1(code: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    values = _E2M1_VALUES.to(device=code.device, dtype=dtype)
    return values[code.to(torch.long) & 0x0F]


def encode_e8m0_scale(scale: torch.Tensor) -> torch.Tensor:
    """Return the uint8 bit pattern of an E8M0 scale tensor."""
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("MXFP4 E8M0 scales must be finite and positive")
    return scale.to(_require_e8m0_dtype()).view(torch.uint8)


def decode_e8m0_scale(encoded: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if encoded.dtype != torch.uint8:
        raise TypeError(f"MXFP4 encoded scale must be uint8, got {encoded.dtype}")
    return encoded.contiguous().view(_require_e8m0_dtype()).to(dtype)


def _reshape_blocks(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    if weight.ndim < 2:
        raise ValueError(f"MXFP4 weight must have at least two dimensions, got {weight.shape}")
    if group_size != MXFP4_GROUP_SIZE:
        raise ValueError(f"vLLM compressed-tensors MXFP4 requires group_size=32, got {group_size}")
    if weight.shape[-1] % group_size:
        raise ValueError(
            f"MXFP4 deployment weight width must be divisible by {group_size}, "
            f"got {weight.shape[-1]}"
        )
    return weight.float().reshape(*weight.shape[:-1], -1, group_size)


def compute_mxfp4_scale(
    weight: torch.Tensor,
    max_scale: torch.Tensor,
    group_size: int = MXFP4_GROUP_SIZE,
) -> torch.Tensor:
    """Compute the learned FOCUS block scale before E8M0 serialization."""
    blocks = _reshape_blocks(weight, group_size)
    expected_shape = blocks.shape[:-1]
    learned_scale = max_scale.detach().float().reshape(expected_shape).to(weight.device)
    if not torch.isfinite(learned_scale).all() or not (learned_scale > 0).all():
        raise ValueError("FOCUS max_scale must be finite and positive for real MXFP4 export")

    max_value = blocks.abs().amax(dim=-1) * learned_scale
    shared_exp = torch.where(
        max_value == 0,
        torch.ones_like(max_value),
        torch.ceil(
            torch.log2(
                max_value / MXFP4_E2M1_MAX + FP32_MIN_NORMAL * (max_value == 0).to(max_value.dtype)
            )
        ),
    )
    shared_exp = shared_exp.clamp(min=-127, max=127)
    return torch.pow(2.0, shared_exp)


def mxfp4_quantize_pack(
    weight: torch.Tensor,
    max_scale: torch.Tensor,
    group_size: int = MXFP4_GROUP_SIZE,
    quant_max_scale: torch.Tensor | None = None,
    num_sub: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a FOCUS weight tensor into vLLM's packed MXFP4 layout.

    ``quant_max_scale`` is an offline-only subgroup coefficient. It changes
    E2M1 code assignment but is deliberately not serialized; dequantization
    continues to use the parent E8M0 scale only.
    """
    blocks = _reshape_blocks(weight, group_size)
    scale = compute_mxfp4_scale(weight, max_scale, group_size)
    if quant_max_scale is not None:
        if num_sub is None:
            if quant_max_scale.numel() % scale.numel():
                raise ValueError("Cannot infer MXFP4 num_sub from quant_max_scale shape")
            num_sub = quant_max_scale.numel() // scale.numel()
        if num_sub <= 0 or group_size % num_sub:
            raise ValueError(f"group_size ({group_size}) must be divisible by num_sub ({num_sub})")
        expected_numel = scale.numel() * num_sub
        if quant_max_scale.numel() != expected_numel:
            raise ValueError(
                f"quant_max_scale has {quant_max_scale.numel()} elements, "
                f"expected {expected_numel}"
            )
        coefficient = torch.sigmoid(quant_max_scale.detach().float().to(weight.device)).reshape(
            *scale.shape, num_sub
        )
        subgroup_blocks = blocks.reshape(
            *scale.shape,
            num_sub,
            group_size // num_sub,
        )
        quant_divisor = scale.unsqueeze(-1).unsqueeze(-1) * coefficient.unsqueeze(-1)
        scaled = (subgroup_blocks / quant_divisor).reshape(blocks.shape)
    else:
        scaled = blocks / scale.unsqueeze(-1)

    scaled = scaled.clamp(-MXFP4_E2M1_MAX, MXFP4_E2M1_MAX)
    code = mxfp4_cast_to_e2m1(scaled).reshape(*weight.shape[:-1], weight.shape[-1])
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed.contiguous(), encode_e8m0_scale(scale).contiguous()


def mxfp4_unpack_dequantize(
    packed_weight: torch.Tensor,
    encoded_scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    group_size: int = MXFP4_GROUP_SIZE,
) -> torch.Tensor:
    """Unpack and dequantize a compressed-tensors MXFP4 weight."""
    if packed_weight.dtype != torch.uint8:
        raise TypeError(f"MXFP4 packed weight must be uint8, got {packed_weight.dtype}")
    if group_size != MXFP4_GROUP_SIZE:
        raise ValueError(f"vLLM compressed-tensors MXFP4 requires group_size=32, got {group_size}")

    code = torch.empty(
        *packed_weight.shape[:-1],
        packed_weight.shape[-1] * 2,
        dtype=torch.uint8,
        device=packed_weight.device,
    )
    code[..., 0::2] = packed_weight & 0x0F
    code[..., 1::2] = packed_weight >> 4
    values = mxfp4_dequantize_e2m1(code, dtype=torch.float32)

    if values.shape[-1] % group_size:
        raise ValueError(
            f"Unpacked MXFP4 width must be divisible by {group_size}, got {values.shape[-1]}"
        )
    blocks = values.reshape(*values.shape[:-1], -1, group_size)
    scale = decode_e8m0_scale(encoded_scale, dtype=torch.float32)
    if scale.shape != blocks.shape[:-1]:
        raise ValueError(
            f"MXFP4 scale shape {tuple(scale.shape)} does not match "
            f"weight blocks {tuple(blocks.shape[:-1])}"
        )
    return (blocks * scale.unsqueeze(-1)).reshape(values.shape).to(dtype)
