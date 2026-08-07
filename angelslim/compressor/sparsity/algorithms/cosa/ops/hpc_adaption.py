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

"""Layout adapters: dense ``(B, H, L, D)`` <-> hpc varlen + paged KV."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "convert_to_varlen_q",
    "convert_to_paged_kv",
    "convert_varlen_out_to_bshd",
]


def convert_to_varlen_q(query_states: torch.Tensor, dtype: torch.dtype = torch.bfloat16):
    """``(B, H, S, D)`` -> ``(B*S, H, D)`` plus ``cu_seqlens``."""
    if query_states.dim() != 4:
        raise ValueError(
            f"query_states must be 4-D (B, H, S, D), got {tuple(query_states.shape)}."
        )
    B, H_q, q_len, D = query_states.shape
    if D != 128:
        raise ValueError(f"hpc kernel requires head_dim == 128, got {D}.")

    q = query_states.permute(0, 2, 1, 3).reshape(B * q_len, H_q, D).to(dtype).contiguous()
    cu_seqlens_q = torch.arange(0, B + 1, device=query_states.device, dtype=torch.int32) * int(
        q_len
    )
    return q, cu_seqlens_q, int(q_len)


def convert_to_paged_kv(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    page_size: int = 64,
    logical_block_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
):
    """``(B, H, S, D)`` -> paged ``(B*pages, page_size, H, D)`` + page table."""
    if page_size not in (32, 64):
        raise ValueError(f"page_size must be 32 or 64 (hpc paged block size), got {page_size}.")
    if logical_block_size % page_size != 0:
        raise ValueError(
            f"logical_block_size ({logical_block_size}) must be a multiple of "
            f"page_size ({page_size})."
        )
    if key_states.shape != value_states.shape:
        raise ValueError(
            f"key/value must share shape, got {tuple(key_states.shape)} vs "
            f"{tuple(value_states.shape)}."
        )
    if key_states.dim() != 4:
        raise ValueError(f"key/value must be 4-D (B, H, S, D), got {tuple(key_states.shape)}.")

    B, H_kv, k_len, D = key_states.shape
    if D != 128:
        raise ValueError(f"hpc kernel requires head_dim == 128, got {D}.")

    k_block_num = (k_len + logical_block_size - 1) // logical_block_size
    k_pad = k_block_num * logical_block_size
    pages_per_batch = k_pad // page_size

    def _to_paged(x: torch.Tensor) -> torch.Tensor:
        if k_pad != k_len:
            x = F.pad(x, (0, 0, 0, k_pad - k_len))
        x = x.permute(0, 2, 1, 3).reshape(B * pages_per_batch, page_size, H_kv, D)
        return x.to(dtype).contiguous()

    kcache = _to_paged(key_states)
    vcache = _to_paged(value_states)
    block_ids = torch.arange(
        B * pages_per_batch, device=key_states.device, dtype=torch.int32
    ).reshape(B, pages_per_batch)
    seqlens_kvcache = torch.full((B,), int(k_len), device=key_states.device, dtype=torch.int32)
    return kcache, vcache, block_ids, seqlens_kvcache


def convert_varlen_out_to_bshd(out: torch.Tensor, batch_size: int, q_len: int) -> torch.Tensor:
    """``(B*S, H, D)`` -> ``(B, H, S, D)``."""
    total_seq, H_q, D = out.shape
    if total_seq != batch_size * q_len:
        raise ValueError(
            f"out total_seq ({total_seq}) != batch_size*q_len ({batch_size}*{q_len})."
        )
    return out.reshape(batch_size, q_len, H_q, D).permute(0, 2, 1, 3).contiguous()
