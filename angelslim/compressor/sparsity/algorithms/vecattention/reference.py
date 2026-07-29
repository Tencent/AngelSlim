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

"""Pure-torch VecAttention reference (pseudo-sparse path + oracle).

Reproduces VecAttention's MinP column-selection *method* at masked-softmax
granularity. The real fused Triton kernel
(``ops/vecattention_kernel.fuse_qk_softmax_minp_wo_causal``) does, per
query-block:

  1. **average-pool** the query rows of the block into one mean query vector
     (``average_vector``);
  2. score that mean query against all keys, and keep every key column ``j`` whose
     logit is within ``gap`` of the per-block running max — ``qk[j] + gap >=
     max_j qk`` — where ``gap = -log(threshold)`` (MinP thresholding: larger
     ``threshold`` => smaller ``gap`` => fewer columns kept; ``threshold -> 0`` =>
     ``gap -> inf`` => keep everything == dense);
  3. additionally always keep the **initial** key block (attention sink) and a
     **local** band of ``block_size_k`` key blocks around the query, plus
     causality.

This reference mirrors that selection on the full (un-strided, un-chunked) scores
— the conservative choice (never drops MORE than the kernel) — then runs masked
softmax attention over the kept columns. It is BOTH the pseudo-sparse fallback when
``vllm_flash_attn`` is unbuilt AND the numerical oracle the tests compare against;
``threshold -> 0`` recovers dense exactly (asserted).

FIDELITY CAVEAT (read before trusting this as a bit-level oracle): this is a
*conservative superset* of the kernel's selection, NOT a bit-faithful mirror, in
two deliberate ways:

  * **Global vs running max.** The keep test here uses the GLOBAL per-row max
    (``qk.amax`` over all keys); the fused kernel uses the online-softmax RUNNING
    max accumulated block-by-block. A column kept early against a smaller running
    max can fall below the final global max, so the kernel may keep a (slightly)
    different column set.
  * **Local band width.** This keeps a ``block_size_k * block_size_q``-token local
    band; the kernel's always-keep set is the sink block + a narrower
    ``2n`` local sub-block band.

Both make the reference keep *at least as many* columns as the kernel, so it is a
safe pseudo-sparse fallback (it never under-attends relative to the kernel) and
``threshold->0 == dense`` holds exactly. But because it is a superset, on diffuse
(non-peaked) inputs it scores close to dense, so a kernel↔reference agreement
check is only discriminating on PEAKED inputs where the dropped columns carry ~0
mass (which is exactly the regime the integration test uses). Do not read a high
kernel↔reference agreement on random inputs as evidence the kernel selection is
correct.

Inputs are post-repeat ``(B, H, L, D)`` (K/V already at num_attention_heads),
matching the framework's prefill_fn contract.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def build_vecattention_keep_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    threshold: float,
    block_size_q: int = 64,
    block_size_k: int = 16,
) -> torch.Tensor:
    """Return the per-(query-block, key-token) keep mask, bool ``(B, H, nqb, L)``.

    For each query-block: mean-pool the queries, score against all keys, keep key
    ``j`` iff ``qk[j] + gap >= max_j qk`` (``gap = -log(threshold)``), union the
    always-kept initial (sink) block + local band, intersect with causality.
    ``threshold <= 0`` => keep every causal key (== dense). True == keep.
    """
    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)
    # gap = -log(threshold): larger threshold => smaller gap => fewer columns.
    # threshold <= 0 is the keep-all sentinel (gap = +inf) — this must hold
    # REGARDLESS of logit scale (a finite gap like -log(1e-9)≈20.7 would NOT keep
    # every column when logits span > that), so dense recovery is exact.
    gap = float("inf") if threshold <= 0.0 else -math.log(threshold)

    nqb = (L + block_size_q - 1) // block_size_q
    qf = q.float()
    kf = k.float()
    keep_tok = torch.zeros(B, H, nqb, L, device=q.device, dtype=torch.bool)
    k_local_tokens = block_size_k * block_size_q  # local band width in tokens
    tok_idx = torch.arange(L, device=q.device)
    for qb in range(nqb):
        q0, q1 = qb * block_size_q, min((qb + 1) * block_size_q, L)
        # (1) mean query vector for this block.
        mean_q = qf[:, :, q0:q1, :].mean(dim=2)  # (B,H,D)
        # (2) score against all keys; mask future keys (causal at block level:
        #     keys beyond the block's last query position cannot be attended).
        qk = torch.matmul(mean_q.unsqueeze(2), kf.transpose(-1, -2)).squeeze(2) * scale  # (B,H,L)
        last_q = q1 - 1
        future = tok_idx.view(1, 1, L) > last_q
        qk = qk.masked_fill(future, float("-inf"))
        # MinP: keep key j iff qk[j] + gap >= max_j qk.
        rowmax = qk.amax(dim=-1, keepdim=True)  # (B,H,1)
        keep = (qk + gap) >= rowmax  # (B,H,L)
        # (3) always-keep: initial sink block + local band before the query block.
        keep = keep | (tok_idx.view(1, 1, L) < block_size_q)  # sink (first q-block)
        local_lo = max(0, q0 - k_local_tokens)
        local_band = (tok_idx.view(1, 1, L) >= local_lo) & (tok_idx.view(1, 1, L) <= last_q)
        keep = keep | local_band
        # never keep a future key (re-assert after the always-keep unions).
        keep = keep & ~future
        keep_tok[:, :, qb, :] = keep
    return keep_tok


def vecattention_reference_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    threshold: float,
    block_size_q: int = 64,
    block_size_k: int = 16,
    group_k_block: int = 16,  # accepted for signature parity; reference is exact
) -> torch.Tensor:
    """MinP column-selected sparse attention, ``(B, H, L, D)`` -> ``(B, H, L, D)``.

    ``threshold`` is the MinP factor (``gap = -log(threshold)``): a key column is
    kept for a query-block iff its mean-query logit is within ``gap`` of that
    block's max logit. ``block_size_q`` mean-pools the queries; the first key
    block (sink) and a local band of ``block_size_k`` blocks before the query are
    always kept; causality is enforced last. ``threshold -> 0`` keeps every causal
    column (== dense).
    """
    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)
    qf = q.float()
    kf = k.float()

    # Per-token causal mask for the final attention.
    causal_tok = torch.ones(L, L, device=q.device, dtype=torch.bool).triu(1)  # True=masked

    keep_tok = build_vecattention_keep_mask(
        q, k, threshold=threshold, block_size_q=block_size_q, block_size_k=block_size_k
    )

    # Expand block-row keep mask -> per-query-token, combine with token causal.
    tok = keep_tok.repeat_interleave(block_size_q, dim=2)[:, :, :L, :]  # (B,H,L,L) True=keep
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale  # (B,H,L,L)
    full_mask = (~tok) | causal_tok[None, None]  # True=masked
    scores = scores.masked_fill(full_mask, float("-inf"))
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
    return torch.matmul(probs, v)
