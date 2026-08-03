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

"""Pure-torch FlashPrefill reference (clean-room; fallback path + oracle).

Stage 3 of the FlashPrefill method (arXiv:2603.06199): given the keep mask from
``blockmask.build_block_keep_mask`` (stages 1+2), expand it to a token mask,
combine with the causal mask, and run masked softmax attention. This is the
fallback when the real ``block_sparse_attn`` kernel cannot run (head_dim 256 on
Qwen3.5) and the numerical oracle the tests compare against. It is NOT a perf
path (materializes the full L×L scores). ``alpha=0`` keeps every causal block,
so the reference recovers dense attention exactly — which the tests assert.

Clean-room: implemented from the paper's method only; qhfan/FlashPrefill source
was not consulted (see this package's NOTICE).

Inputs are post-repeat ``(B, H, L, D)`` (K/V already at num_attention_heads),
matching the framework's prefill_fn contract.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .blockmask import build_block_keep_mask


def flashprefill_reference_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    alpha: float,
    block_size: int = 128,
    sink: int = 256,
    window: int = 512,
    last_n_block_full: int = 2,
) -> torch.Tensor:
    """Max-based-thresholded block-sparse attention, ``(B,H,L,D)`` -> same.

    ``alpha=0`` keeps every causal block (== dense); larger ``alpha`` is sparser.
    """
    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)
    keep_block = build_block_keep_mask(
        q,
        k,
        alpha=alpha,
        block_size=block_size,
        sink=sink,
        window=window,
        last_n_block_full=last_n_block_full,
    )

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale  # (B,H,L,L)
    causal = torch.ones(L, L, device=q.device, dtype=torch.bool).triu(1)  # True=masked
    # Expand block keep mask -> token keep mask.
    tok = keep_block.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)[
        :, :, :L, :L
    ]  # (B,H,L,L) True=keep
    full_mask = (~tok) | causal[None, None]  # True=masked
    scores = scores.masked_fill(full_mask, float("-inf"))
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
    return torch.matmul(probs, v)
