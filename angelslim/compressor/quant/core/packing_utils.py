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

AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

    # unpacking columnwise
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    iweights = iweights.view(iweights.shape[0], -1)

    # unpacking columnwise
    if qzeros is not None:
        izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
            torch.int8  # smallest dtype available
        )
        izeros = izeros.view(izeros.shape[0], -1)
    else:
        izeros = qzeros

    return iweights, izeros


def reverse_awq_order(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    reverse_order_tensor = torch.arange(
        iweights.shape[-1],
        dtype=torch.int32,
        device=izeros.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    if izeros is not None:
        izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]

    return iweights, izeros


def pack_exllama(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits, device=iweights.device)

    # packing rowwise
    iweights = iweights.view(iweights.shape[0] // (32 // bits), 32 // bits, -1)
    qweight = torch.bitwise_left_shift(iweights, shifts[None, :, None]).sum(dim=1).to(torch.int32)

    # packing columnwise
    izeros = izeros.view(-1, izeros.shape[1] // (32 // bits), 32 // bits)
    qzeros = torch.bitwise_left_shift(izeros, shifts[None, None, :]).sum(dim=-1).to(torch.int32)

    return qweight, qzeros


def unpack_reorder_pack(qweight, qzeros, bits):
    # Unpack the qweight and qzeros tensors
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    # Reverse the order of the iweight and izeros tensors
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)

    # overflow checks
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)

    # Subtract 1 from the izeros tensor (exllama adds 1 during inference)
    # We can remove it if we remove the +1 in the exllama code
    izeros = izeros - 1
    # Pack the qweight and qzeros tensors
    qweight, qzeros = pack_exllama(iweight, izeros, bits)

    return qweight, qzeros


def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    # Unpack the qweight and qzeros tensors
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    # Reverse the order of the iweight and izeros tensors
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)

    # overflow checks
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)

    # fp16 weights
    scales = scales.repeat_interleave(group_size, dim=0)
    izeros = izeros.repeat_interleave(group_size, dim=0)
    iweight = (iweight - izeros) * scales

    return iweight


def pack_weight_to_int8(weight):
    """Pack two INT4 values into one INT8 byte (CPU, numpy-based).

    Original implementation using Python loops for packing.
    Kept for debugging and fallback.
    For GPU-accelerated packing, use pack_weight_to_int8_gpu.

    Args:
        weight: Tensor of shape (out_features, in_features) with values in [-8, 7].

    Returns:
        Packed INT8 tensor of shape (out_features, in_features // 2) on CPU.
    """
    weight = weight.t().contiguous().cpu()
    weight = weight.to(torch.float32).detach().numpy().astype(np.int8)

    i = 0
    row = 0
    packed_weight = np.zeros((weight.shape[0] // 2, weight.shape[1]), dtype=np.int8)
    while row < packed_weight.shape[0]:
        for j in range(i, i + (8 // 4)):
            packed_weight[row] |= (weight[j] & 0x0F) << (4 * (j - i))
        i += 8 // 4
        row += 1

    packed_weight = packed_weight.astype(np.int8)
    packed_weight = torch.from_numpy(packed_weight).t().contiguous()
    return packed_weight


def pack_weight_to_int8_gpu(weight):
    """Pack two INT4 values into one INT8 byte using pure PyTorch (GPU-accelerated).

    Supports both CPU and GPU tensors — no numpy dependency, so packing
    can be done directly on GPU without device transfer overhead.

    Input layout (after transpose): rows are paired (row 0,1 -> packed row 0, etc.)
    Low nibble = even row, high nibble = odd row.

    Args:
        weight: Tensor of shape (out_features, in_features) with values in [-8, 7].
            Can be on any device (CPU or CUDA).

    Returns:
        Packed INT8 tensor of shape (out_features, in_features // 2),
        on the same device as input.
    """
    # Transpose to (in_features, out_features) for row-pair packing
    weight = weight.t().contiguous().to(torch.int8)

    # Vectorized packing: pair adjacent rows and combine low/high nibbles
    # Even rows -> low nibble, odd rows -> high nibble
    even_rows = weight[0::2]  # shape: (rows//2, cols)
    odd_rows = weight[1::2]  # shape: (rows//2, cols)
    packed_weight = (even_rows & 0x0F) | ((odd_rows & 0x0F) << 4)

    # Transpose back to (out_features, in_features // 2)
    packed_weight = packed_weight.t().contiguous()
    return packed_weight
