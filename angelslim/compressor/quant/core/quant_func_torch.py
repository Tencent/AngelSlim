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

"""
Pure PyTorch implementations of weight quantization functions.

This module provides CPU/Windows-compatible implementations that mirror
the Triton kernels in quant_func.py.
"""

from typing import Tuple

import torch

# FP8 E4M3 max value
FP8_MAX = 448.0


def weight_dequant_torch(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Pure PyTorch implementation of block-wise weight dequantization.

    Args:
        x: Quantized weight tensor of shape (M, N)
        s: Scale tensor of shape (m_blocks, n_blocks)
        block_size: Block size used for quantization

    Returns:
        Dequantized weight tensor of shape (M, N)
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"

    M, N = x.size()
    device = x.device

    # Output tensor
    y = torch.empty((M, N), dtype=torch.get_default_dtype(), device=device)

    # Calculate number of blocks
    m_blocks = (M + block_size - 1) // block_size
    n_blocks = (N + block_size - 1) // block_size

    # Convert to float32 for computation
    x_float = x.to(torch.float32)

    # Process each block
    for mb in range(m_blocks):
        m_start = mb * block_size
        m_end = min(m_start + block_size, M)
        for nb in range(n_blocks):
            n_start = nb * block_size
            n_end = min(n_start + block_size, N)

            scale = s[mb, nb]
            y[m_start:m_end, n_start:n_end] = x_float[m_start:m_end, n_start:n_end] * scale

    return y


def per_block_weight_quant_torch(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch implementation of block-wise FP8 weight quantization.

    Args:
        x: Input weight tensor of shape (M, N) in float type
        block_size: Block size for quantization

    Returns:
        Tuple of (quantized_tensor, scale_tensor):
            - y: Quantized FP8 tensor of shape (M, N)
            - s: Scale tensor of shape (m_blocks, n_blocks)
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must have 2 dimensions"

    M, N = x.size()
    device = x.device

    # Output tensors
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    m_blocks = (M + block_size - 1) // block_size
    n_blocks = (N + block_size - 1) // block_size
    s = torch.empty((m_blocks, n_blocks), dtype=torch.float32, device=device)

    # Convert to float32 for computation
    x_float = x.to(torch.float32)

    # Process each block
    for mb in range(m_blocks):
        m_start = mb * block_size
        m_end = min(m_start + block_size, M)
        for nb in range(n_blocks):
            n_start = nb * block_size
            n_end = min(n_start + block_size, N)

            block = x_float[m_start:m_end, n_start:n_end]
            max_val = block.abs().amax()

            # Compute scale (guard against zero)
            scale = max_val / FP8_MAX
            if scale.item() == 0.0:
                scale = torch.tensor(1.0, dtype=torch.float32, device=device)

            # Quantize block
            y_block = (block / scale).to(torch.float8_e4m3fn)

            # Store results
            y[m_start:m_end, n_start:n_end] = y_block
            s[mb, nb] = scale

    return y, s


def per_block_weight_quant_torch_fast(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized PyTorch implementation using tensor operations.

    This is faster than the loop-based version for large tensors.

    Args:
        x: Input weight tensor of shape (M, N) in float type
        block_size: Block size for quantization

    Returns:
        Tuple of (quantized_tensor, scale_tensor)
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must have 2 dimensions"

    M, N = x.size()

    # Pad tensor to be divisible by block_size
    pad_m = (block_size - M % block_size) % block_size
    pad_n = (block_size - N % block_size) % block_size

    if pad_m > 0 or pad_n > 0:
        x_padded = torch.nn.functional.pad(x, (0, pad_n, 0, pad_m), value=0.0)
    else:
        x_padded = x

    M_padded, N_padded = x_padded.size()
    m_blocks = M_padded // block_size
    n_blocks = N_padded // block_size

    # Reshape to blocks: [m_blocks, block_size, n_blocks, block_size]
    x_blocks = x_padded.view(m_blocks, block_size, n_blocks, block_size)
    x_blocks = x_blocks.permute(
        0, 2, 1, 3
    ).contiguous()  # [m_blocks, n_blocks, block_size, block_size]

    # Compute max absolute value per block
    x_float = x_blocks.to(torch.float32)
    max_vals = x_float.abs().amax(dim=(2, 3))  # [m_blocks, n_blocks]

    # Compute scales
    s = max_vals / FP8_MAX
    s = torch.where(s == 0.0, torch.ones_like(s), s)

    # Quantize
    s_expanded = s[:, :, None, None]  # [m_blocks, n_blocks, 1, 1]
    y_blocks = (x_float / s_expanded).to(torch.float8_e4m3fn)

    # Reshape back: [m_blocks, n_blocks, block_size, block_size] -> [M_padded, N_padded]
    y_blocks = y_blocks.permute(
        0, 2, 1, 3
    ).contiguous()  # [m_blocks, block_size, n_blocks, block_size]
    y_padded = y_blocks.view(M_padded, N_padded)

    # Remove padding
    y = y_padded[:M, :N].contiguous()

    # Adjust scale tensor size if needed
    s = s[: (M + block_size - 1) // block_size, : (N + block_size - 1) // block_size]

    return y, s


def weight_dequant_torch_fast(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    Optimized PyTorch implementation of weight dequantization using tensor operations.

    Args:
        x: Quantized weight tensor of shape (M, N)
        s: Scale tensor of shape (m_blocks, n_blocks)
        block_size: Block size used for quantization

    Returns:
        Dequantized weight tensor of shape (M, N)
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"

    M, N = x.size()

    # Pad tensor to be divisible by block_size
    pad_m = (block_size - M % block_size) % block_size
    pad_n = (block_size - N % block_size) % block_size

    if pad_m > 0 or pad_n > 0:
        x_padded = torch.nn.functional.pad(x.to(torch.float32), (0, pad_n, 0, pad_m), value=0.0)
        # Also pad scale tensor
        s_m, s_n = s.size()
        target_s_m = (M + pad_m) // block_size
        target_s_n = (N + pad_n) // block_size
        if target_s_m > s_m or target_s_n > s_n:
            s_padded = torch.nn.functional.pad(
                s, (0, target_s_n - s_n, 0, target_s_m - s_m), value=1.0
            )
        else:
            s_padded = s
    else:
        x_padded = x.to(torch.float32)
        s_padded = s

    M_padded, N_padded = x_padded.size()
    m_blocks = M_padded // block_size
    n_blocks = N_padded // block_size

    # Reshape to blocks
    x_blocks = x_padded.view(m_blocks, block_size, n_blocks, block_size)
    x_blocks = x_blocks.permute(0, 2, 1, 3).contiguous()

    # Apply scales
    s_expanded = s_padded[:m_blocks, :n_blocks, None, None]
    y_blocks = x_blocks * s_expanded

    # Reshape back
    y_blocks = y_blocks.permute(0, 2, 1, 3).contiguous()
    y_padded = y_blocks.view(M_padded, N_padded)

    # Remove padding and convert to default dtype
    y = y_padded[:M, :N].to(torch.get_default_dtype()).contiguous()

    return y
