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

"""CoSA prefill: Triton selection proxy + ``hpc`` block-sparse attention.

The proxy emits priority-ordered block ids; the ``hpc`` kernel walks that order
and threshold-skips the tail. Both ``triton`` and ``hpc`` are hard dependencies.
Inputs are post-repeat ``(B, H, L, D)``.
"""

from __future__ import annotations

import contextlib

import torch

from .ops.hpc_adaption import (
    convert_to_paged_kv,
    convert_to_varlen_q,
    convert_varlen_out_to_bshd,
)
from .ops.proxy import cosa_estimate

_HPC_PAGE_SIZE = 64


def default_chunk_size(k_len: int) -> int:
    """Pick a proxy chunk size that keeps peak memory roughly constant."""
    pow2 = 1 << (k_len - 1).bit_length()
    return int(max(min(max(2048, pow2), 128 * 1024 * 2048 // pow2), 2048))


def cosa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    stride: int,
    topp: float,
    block_size: int,
    chunk_size,
    skipsoftmax_threshold: float,
) -> torch.Tensor:
    """Prefill. ``q,k,v`` and the return are post-repeat ``(B, H, L, D)``."""
    import hpc

    batch_size, _, k_len, _ = k.shape
    q_len = q.shape[2]

    assert block_size == 128, f"the hpc kernel is dim128/block128 only, got {block_size}"
    assert batch_size == 1, f"CoSA prefill is validated at batch_size 1, got {batch_size}"

    if chunk_size is None:
        chunk_size = default_chunk_size(k_len)

    # Triton launches on the *current* device; pin to q's GPU under device_map.
    pin = torch.cuda.device(q.device) if q.is_cuda else contextlib.nullcontext()
    with pin:
        attn_sums, ordered_blockmask = cosa_estimate(
            q,
            k,
            block_size=block_size,
            stride=stride,
            topp=topp,
            chunk_size=chunk_size,
        )
        del attn_sums

        q_thd, cu_seqlens_q, max_seqlens_q = convert_to_varlen_q(q)
        kcache, vcache, block_ids, seqlens_kvcache = convert_to_paged_kv(
            k, v, page_size=_HPC_PAGE_SIZE, logical_block_size=block_size
        )
        hpc_out = hpc.attention_with_kvcache_blocksparse_anyorderskip_prefill_bf16(
            q_thd,
            kcache,
            vcache,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            max_seqlens_q,
            ordered_blockmask,
            float(skipsoftmax_threshold),
        )

    return convert_varlen_out_to_bshd(hpc_out, batch_size, q_len)
