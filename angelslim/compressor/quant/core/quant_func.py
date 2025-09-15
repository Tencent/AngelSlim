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

from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .metrics import mse_loss


@torch.no_grad()
def pseudo_quantize_tensor(
    w, w_bit=4, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**w_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (w_bit - 1) - 1
        min_int = -(2 ** (w_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


def quantize_weight_per_tensor_fp8(
    tensor: torch.Tensor, scale: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    finfo = torch.finfo(torch.float8_e4m3fn)

    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float()
    return qweight, scale


def quantize_activation_per_tensor_fp8(
    tensor: torch.Tensor, scale: float
) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fn)


def gemm_fp8(act, act_scale, weight, weight_scale, bias, out_dtype):
    if act.numel() == 0:
        # Deal with empty tensors (triggeted by empty MoE experts)
        return torch.empty(
            size=(0, weight.shape[0]), dtype=out_dtype, device=act.device
        )

    # TODO: Disable native fp8 gemm for now, always just dequantize
    # native_fp8_support = (
    #     torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
    # )
    native_fp8_support = False
    if native_fp8_support:
        need_reshape = act.dim() == 3
        if need_reshape:
            batch_size = act.shape[0]
            act_input = act.reshape(-1, act.shape[-1])
        else:
            batch_size = None
            act_input = act
        output, _ = torch._scaled_mm(
            act_input,
            weight.t(),
            out_dtype=out_dtype,
            scale_a=act_scale,
            scale_b=weight_scale,
            bias=bias,
        )
        if need_reshape:
            output = output.reshape(
                batch_size, output.shape[0] // batch_size, output.shape[1]
            )
    else:
        output = torch.nn.functional.linear(
            act.to(out_dtype) * act_scale.to(out_dtype),
            weight.to(out_dtype) * weight_scale.to(out_dtype),
            bias=bias,
        )
    return output


@torch.no_grad()
def quantize_weight_int(
    x: torch.Tensor, scales: torch.Tensor, bits=8
) -> Tuple[torch.Tensor, float]:
    if scales.ndim == 2:  # weight group-wise
        scales = torch.repeat_interleave(scales, x.shape[1] // scales.shape[1], dim=-1)
    bnt = (1 << (bits - 1)) - 1

    while scales.ndim < x.ndim:
        scales = scales.unsqueeze(-1)
    scales.div_(bnt)
    x.div_(scales).round_().clamp_(-bnt - 1, bnt)
    return x, scales


def tensor_quant_dequant_int(x, scales, bits=8):
    if scales.ndim == 2:  # weight group-wise
        scales = torch.repeat_interleave(scales, x.shape[1] // scales.shape[1], dim=-1)
    bnt = (1 << (bits - 1)) - 1

    while scales.ndim < x.ndim:
        scales = scales.unsqueeze(-1)
    scales = scales.div(bnt)
    quant_x = torch.clamp(torch.round(x / scales), -bnt - 1, bnt)
    quant_dequant_x = quant_x * scales
    return quant_dequant_x


def tensor_quant(x, scales, bits=8):
    if len(scales.shape) == 2:  # weight group-wise
        scales = torch.repeat_interleave(scales, x.shape[1] // scales.shape[1], dim=-1)
    bnt = (1 << (bits - 1)) - 1
    quant_scale = scales
    for _ in range(len(x.shape) - len(scales.shape)):
        quant_scale = quant_scale.unsqueeze(-1)
    quant_x = torch.clamp(torch.round(x / quant_scale * bnt), -bnt - 1, bnt)
    return quant_x


def compute_scales(x, method="abs_max", group_size=-1):
    if method == "abs_max":
        quant_scale = float(torch.max(torch.abs(x.flatten())))
        quant_scale = 1e-8 if quant_scale == 0.0 else quant_scale
    elif method == "avg":
        quant_scale, _ = x.view(x.shape[0], -1).abs().max(dim=1)
        quant_scale = torch.mean(quant_scale)
    elif method == "abs_max_channel_wise":
        # abs_max_values, _ = x.abs().max(dim=-1, keepdim=True)
        # quant_scale = abs_max_values.squeeze()
        if len(x.shape) > 2:
            x = x.flatten(1)
        quant_scale, _ = x.abs().max(dim=-1)
    elif method == "groupwise":
        input_processed = x.view([x.shape[0], x.shape[1] // group_size, group_size])
        abs_max_values, _ = input_processed.abs().max(dim=-1, keepdim=True)
        quant_scale = abs_max_values.squeeze()
    elif method == "dynamic_per_token":
        init_shape = x.shape
        reshaped_x = x.reshape((-1, x.shape[-1]))
        tmp = torch.zeros(reshaped_x.shape[0], device=x.device)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp)
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp)
        xmax = torch.maximum(torch.abs(xmin), xmax)
        tmp = xmax == 0
        quant_scale = xmax.unsqueeze(1).repeat(1, reshaped_x.shape[-1])
        quant_scale[tmp] = 1
        quant_scale = quant_scale.reshape(init_shape)
    return quant_scale


def fake_quant_dequant(x, method="abs_max", bits=8, group_size=-1):
    bnt = (1 << (bits - 1)) - 1
    quant_scale = compute_scales(x, method=method, group_size=group_size)
    if method == "groupwise":
        quant_scale = torch.repeat_interleave(quant_scale, group_size, dim=-1)
        quant_scale = quant_scale.reshape(x.shape)
    for _ in range(len(x.shape) - len(quant_scale.shape)):
        quant_scale = quant_scale.unsqueeze(-1)
    quant_value = torch.clamp(torch.round(x / quant_scale * bnt), -bnt - 1, bnt)
    quant_dequant_value = quant_value / bnt * quant_scale
    return quant_dequant_value


def compute_scales_with_zero(x, bits=8, sym=False, perchannel=True):
    maxq = torch.tensor(2**bits - 1)
    shape = x.shape

    if perchannel:
        x = x.flatten(1)
    else:
        x = x.flatten().unsqueeze(0)
    tmp = torch.zeros(x.shape[0], device=x.device)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)

    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        tmp = xmin < 0
        if torch.any(tmp):
            xmin[tmp] = -xmax[tmp]
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1

    if maxq < 0:
        scale = xmax
        zero = xmin
    else:
        scale = (xmax - xmin) / maxq
        if sym:
            zero = torch.full_like(scale, (maxq + 1) / 2)
        else:
            zero = torch.round(-xmin / scale)
    shape = [-1] + [1] * (len(shape) - 1)
    scale = scale.reshape(shape)
    zero = zero.reshape(shape)
    return scale, zero


def get_fp_maxval(bits=8, mantissa_bit=3, sign_bits=1):
    _bits = torch.tensor(bits)
    _mantissa_bit = torch.tensor(mantissa_bit)
    _sign_bits = torch.tensor(sign_bits)
    m = torch.clamp(torch.round(_mantissa_bit), 1, _bits - _sign_bits)
    e = _bits - _sign_bits - m
    bias = 2 ** (e - 1) - 1
    mantissa = 1
    for i in range(mantissa_bit - 1):
        mantissa += 1 / (2 ** (i + 1))
    maxval = mantissa * 2 ** (2**e - 1 - bias)
    return maxval


def get_fp_search_maxval(x, bits=8, mantissa_bit=3, sign_bits=1):
    best_scale = None
    calibration_loss = float("inf")
    # for scale in search_scales:
    for scale in range(1, 1000, 1):
        scale /= 100
        new_x = x / scale
        quant_dequant_x = quantize_to_fp8(
            new_x, bits=bits, mantissa_bit=mantissa_bit, sign_bits=sign_bits
        )
        quant_dequant_x *= scale
        cur_loss = mse_loss(x, quant_dequant_x)
        if cur_loss <= calibration_loss:
            calibration_loss = cur_loss
            best_scale = scale
    return best_scale


def quantize_to_fp8(x, bits=8, mantissa_bit=3, sign_bits=1):
    """
    Default is E4M3.
    """
    bits = torch.tensor(bits)
    mantissa_bit = torch.tensor(mantissa_bit)
    sign_bits = torch.tensor(sign_bits)
    m = torch.clamp(torch.round(mantissa_bit), 1, bits - sign_bits)
    e = bits - sign_bits - m
    bias = 2 ** (e - 1) - 1
    mantissa = 1
    for i in range(mantissa_bit - 1):
        mantissa += 1 / (2 ** (i + 1))
    maxval = mantissa * 2 ** (2**e - 1 - bias)
    minval = -maxval
    minval = -maxval if sign_bits == 1 else torch.zeros_like(maxval)
    input_clamp = torch.min(torch.max(x, minval), maxval)
    log_scales = torch.clamp(
        (torch.floor(torch.log2(torch.abs(input_clamp)) + bias)).detach(), 1.0
    )
    log_scales = 2.0 ** (log_scales - m - bias.float())
    # dequant
    qdq_out = torch.round(input_clamp / log_scales) * log_scales
    qdq_out = qdq_out.type(x.dtype)
    return qdq_out


def tensor_quant_dequant_fp8(x, scale, bits=8, mantissa_bit=3, sign_bits=1):
    for _ in range(len(x.shape) - 1):
        scale = scale.unsqueeze(-1)
    new_x = x / scale
    quant_dequant_x = quantize_to_fp8(
        new_x, bits=bits, mantissa_bit=mantissa_bit, sign_bits=sign_bits
    )
    quant_dequant_x *= scale
    return quant_dequant_x


# This function is copied from DeepSeek-V3 (MIT License):
# Copyright (c) 2023 DeepSeek-AI
# Original source: https://github.com/deepseek-ai/DeepSeek-V3
@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


# This function is copied from DeepSeek-V3 (MIT License):
# Copyright (c) 2023 DeepSeek-AI
# Original source: https://github.com/deepseek-ai/DeepSeek-V3
def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128. # noqa: E501

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2. # noqa: E501
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# This function is copied from DeepSeek-V3 (MIT License):
# Copyright (c) 2023 DeepSeek-AI
# Original source: https://github.com/deepseek-ai/DeepSeek-V3
@triton.jit
def weight_quant(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """Quantizes FP32 weights to FP8 format using block-wise quantization."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    max_val = tl.max(tl.abs(x))
    scale = max_val / 448.0
    scale = tl.where(max_val == 0.0, 1.0, scale)
    y = x / scale
    y = y.to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n + pid_n, scale)


def per_block_weight_quant(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantizes FP32 weight tensor to FP8 format using block-wise quantization."""
    assert x.is_contiguous()
    assert x.dim() == 2

    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    m_blocks = triton.cdiv(M, block_size)
    n_blocks = triton.cdiv(N, block_size)
    s = torch.empty((m_blocks, n_blocks), dtype=torch.float32, device=x.device)

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )

    weight_quant[grid](x, y, s, M, N, BLOCK_SIZE=block_size)

    return y, s


def reduce_block_padding(input: torch.Tensor, block_sizes: dict, pad_value: float = 0):
    """Padding the input using block-based reduction for each dimension.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        block_sizes (dict): A dictionary specifying the block size for
            padding each dimension. Example: `{-1: 128, -2: 128}` pads
            the input over 2D blocks.
    """
    with torch.no_grad():
        padded_tensor = input
        num_dims = padded_tensor.dim()
        # Process each specified dimension independently
        for dim, block in block_sizes.items():
            # Convert negative dimension to positive index
            pos_dim = dim if dim >= 0 else num_dims + dim

            # Calculate how many elements are missing along that dimension
            current_size = padded_tensor.size(pos_dim)
            remainder = current_size % block
            pad_amt = 0 if remainder == 0 else block - remainder

            if pad_amt > 0:
                # F.pad expects a pad tuple of length 2*num_dims.
                pad = [0] * (2 * num_dims)
                # For dimension pos_dim, the right padding is at index:
                # (num_dims - 1 - pos_dim)*2 + 1.
                pad_index = (num_dims - 1 - pos_dim) * 2
                pad[pad_index + 1] = (
                    pad_amt  # Set padding on the right side of the target dimension
                )

                padded_tensor = F.pad(padded_tensor, pad, value=pad_value)

        return padded_tensor
