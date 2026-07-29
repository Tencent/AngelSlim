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

"""Pure-torch XAttention reference (pseudo-sparse fallback path + oracle).

This mirrors the *shape* of upstream XAttention's selection — per query-block,
keep the smallest set of (causal) key-blocks whose pooled, softmaxed attention
mass covers ``threshold`` — at boolean-mask granularity, then runs masked
softmax attention. It is the fallback when the real ``block_sparse_attn`` path
cannot run (head_dim 256 on Qwen3.5) and the numerical oracle the tests compare
against. It is NOT a perf path (materializes the full L×L scores).

Difference from the real kernel: upstream estimates block importance from an
*antidiagonal-strided* sub-sample of q/k (``stride``); this reference uses the
full (un-strided) block scores, which is the conservative (more-accurate, never
drops MORE) choice — so ``threshold -> 1`` recovers dense attention exactly,
which the correctness tests assert. ``stride``/``norm`` are not modelled here.

Inputs are post-repeat ``(B, H, L, D)`` (K/V already at num_attention_heads),
matching the framework's prefill_fn contract.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def xattention_reference_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    threshold: float,
    block_size: int = 128,
) -> torch.Tensor:
    """Block-coverage sparse attention, ``(B, H, L, D)`` -> ``(B, H, L, D)``.

    Per (head, query-block): pool the causal key dimension into key-blocks, take
    the max query-row score per key-block, softmax across key-blocks, then keep
    the top key-blocks whose cumulative mass first reaches ``threshold``. The
    chosen key-blocks form a block mask, expanded to a full token mask and
    combined with the causal mask for the final softmax. threshold>=1 keeps every
    causal block (== dense).
    """
    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale  # (B,H,L,L)

    # Causal mask at token granularity.
    causal = torch.ones(L, L, device=q.device, dtype=torch.bool).triu(1)  # True=masked
    scores = scores.masked_fill(causal[None, None], float("-inf"))

    nqb = (L + block_size - 1) // block_size
    nkb = (L + block_size - 1) // block_size

    # Build a per-(query-block, key-block) keep mask via threshold coverage.
    keep_block = torch.zeros(B, H, nqb, nkb, device=q.device, dtype=torch.bool)
    for qb in range(nqb):
        q0, q1 = qb * block_size, min((qb + 1) * block_size, L)
        blk = scores[:, :, q0:q1, :]  # (B,H,bq,L)
        # pool key tokens into key-blocks (max), respecting causality (a fully
        # masked key-block stays -inf and gets ~0 weight).
        kb_scores = []
        for kb in range(nkb):
            k0, k1 = kb * block_size, min((kb + 1) * block_size, L)
            kb_scores.append(blk[:, :, :, k0:k1].amax(dim=-1))  # (B,H,bq)
        kb_scores = torch.stack(kb_scores, dim=-1)  # (B,H,bq,nkb)
        # reduce the query rows in this block to one score per key-block (max).
        kb_score = kb_scores.amax(dim=2)  # (B,H,nkb)
        # softmax over key-blocks -> mass; -inf (non-causal) -> 0.
        mass = torch.softmax(kb_score, dim=-1)  # (B,H,nkb)
        # sort descending, take the prefix whose cumsum first reaches threshold.
        sorted_mass, order = torch.sort(mass, dim=-1, descending=True)
        csum = torch.cumsum(sorted_mass, dim=-1)
        reached = (csum >= threshold).float()
        first = torch.argmax(reached, dim=-1)  # (B,H)
        none_reached = reached.sum(dim=-1) == 0
        n_keep = torch.where(none_reached, torch.full_like(first, nkb), first + 1)
        ar = torch.arange(nkb, device=q.device).view(1, 1, nkb)
        keep_sorted = ar < n_keep.unsqueeze(-1)  # (B,H,nkb) over sorted idx
        keep = torch.zeros_like(keep_sorted)
        keep.scatter_(-1, order, keep_sorted)
        # never keep a key-block strictly in the future of this query-block.
        future = ar > qb
        keep = keep & ~future
        keep_block[:, :, qb, :] = keep

    # Expand block mask -> token mask.
    tok = keep_block.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)[
        :, :, :L, :L
    ]  # (B,H,L,L) True=keep
    full_mask = (~tok) | causal[None, None]  # True=masked
    scores = scores.masked_fill(full_mask, float("-inf"))
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
    return torch.matmul(probs, v)
