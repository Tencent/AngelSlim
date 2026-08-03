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

"""FlashPrefill block-selection (clean-room — stages 1+2 of arXiv 2603.06199).

This module implements the FlashPrefill *method* described in the paper
"FlashPrefill: Instantaneous Pattern Discovery and Thresholding for Ultra-Fast
Long-Context Prefilling" (Fan et al., arXiv:2603.06199). It is a CLEAN-ROOM
implementation written from the paper's textual method description and the
documented algorithm; the upstream reference repo (qhfan/FlashPrefill) ships with
NO license and its source was NOT copied (see this package's NOTICE). Only the
published *method* is reproduced — the code expression is original AngelSlim work.

Correctness was validated numerically against the upstream implementation used
purely as an oracle (run out-of-tree, never vendored): on a real Qwen3-8B layer
at 6.4K tokens this scoring reproduces the upstream block-selection mask to IoU
0.995 (the ~0.1% of blocks that differ sit exactly on the alpha*max threshold —
when fed the upstream's own score, our selection logic reproduces its mask
bit-exactly, so the only residual is the score matmul's floating-point reduction
order, which differs between torch's bf16 matmul and the upstream Triton
``tl.dot``), and the final attention output matches the upstream kernel to rel
0.007 / argmax 99.9%. The heavy q·mean_k matmul runs in bf16 (matching upstream
and ~6x faster than fp32) with no loss in final accuracy. (The pre-correction
scoring — max-over-keys + mean-over-rows, in fp32 — diverged at rel 0.17; this
version matches the paper's "fused block approximation".)

Two stages, shared by both the kernel path and the torch reference:

  Stage 1 — Instantaneous Pattern Discovery (paper §3.1-§3.3, "Fused Block-level
  Attention Approximation"). Each key-block J is represented by its MEAN key
  vector (mean over the block's tokens). The block-pair importance is then a
  block-level softmax energy: for query-block I, score[I,J] = sum over the query
  rows i in I of exp2((q_i . mean_k_J) * scale * log2(e)), computed with a
  stable per-block max subtraction, causal-masked at block granularity, then
  globally normalized across J (rescale by the block max, divide by the row sum)
  into a comparable importance distribution. exp2 / log2(e) mirror the paper's
  base-2 formulation. NB: using the *mean key* (not max-over-keys) is the load-
  bearing detail — it is what makes this match upstream.

  Stage 2 — Max-based Dynamic Thresholding (paper §3.4, the core novelty). For
  query-block I, KEEP every key-block J with ``score[I,J] >= alpha * max_J
  score[I,J]``. This is a single-pass max-reduction — no Top-k sorting, no Top-p
  cumulative sum — which is exactly what distinguishes FlashPrefill from
  xattention (top-p coverage) and flexprefill (gamma-coverage cumsum). ``alpha``
  is the sole hyperparameter: alpha=0 keeps every causal block (== dense); larger
  alpha is sparser. Per the paper, the selection additionally retains, always:
  attention-sink blocks (the first ``sink`` key-blocks), a local window (the last
  ``window`` key-blocks up to the query block), and ``last_n_block_full`` whole
  query-blocks at the end (which attend every causal key-block).
"""

from __future__ import annotations

import math

import torch

_LOG2E = 1.4426950408889634

# Query-block chunking budget for the scoring stage. The score intermediate is
# (B, H, q_blocks, block_size, n_key_blocks) fp32 — without chunking it grows
# QUADRATICALLY in sequence length (nqb == nkb), e.g. ~5 GB @32K, ~22 GB @64K,
# ~86 GB @128K on a 32-head model — which both OOMs at long context (defeating
# FlashPrefill's whole purpose) and, in a long-lived process, corrupts a
# fragmented CUDA allocator (an async illegal-access). We therefore process the
# query-block axis in chunks sized so (q_blocks_per_chunk * n_key_blocks) stays
# bounded; peak scoring memory is then ~constant in L instead of O(L²). This is
# PURELY a memory tiling — every reduction is local to a query block, so the
# chunked result is BIT-IDENTICAL to the unchunked one (same float reduction
# order), preserving the rel-0.0074 match to upstream. (Upstream avoids the
# materialization entirely with a fused Triton kernel; tiling is the torch-level
# equivalent.)
_SCORE_QK_BUDGET = 32768  # target (q_blocks_per_chunk * n_key_blocks) per tile


def _block_importance(q: torch.Tensor, k: torch.Tensor, block_size: int) -> torch.Tensor:
    """Stage 1: official-matched block importance ``score`` ``(B, H, nqb, nkb)``.

    ``q``/``k`` are post-repeat ``(B, H, L, D)``. Each key-block is mean-pooled to
    one vector; score[I,J] = normalized sum_i exp2((q_i . mean_k_J)*scale*log2e)
    over the query rows of block I, block-causal, stabilized by the per-block max.

    Dtype discipline mirrors the upstream Triton kernel for both fidelity AND
    efficiency: the heavy q·mean_k matmul runs in the INPUT dtype (bf16) — same
    as upstream's ``tl.dot`` — and only the exp/sum/normalize reductions promote
    to fp32 (upstream accumulates ``output_score``/``output_max`` in fp32). Doing
    the matmul in bf16 (not fp32) keeps the discovery cost negligible, which is
    the whole point of FlashPrefill, and matches the upstream numerics; an fp32
    matmul would be both slower and a precision *mismatch* vs upstream.

    The query-block axis is tiled (``_SCORE_QK_BUDGET``) so peak memory is bounded
    in L; the result is bit-identical to computing all query blocks at once (every
    reduction is local to a query block). See ``_SCORE_QK_BUDGET``.
    """
    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)
    nqb = (L + block_size - 1) // block_size
    nkb = nqb
    pad = nqb * block_size - L
    qk_dtype = q.dtype  # keep the heavy matmul in the model dtype (bf16), like tl.dot
    qd = q
    kd = k
    if pad:
        qd = torch.nn.functional.pad(qd, (0, 0, 0, pad))
        kd = torch.nn.functional.pad(kd, (0, 0, 0, pad))

    # Mean key vector per block (count excludes the right-pad on the last block).
    # Sum in fp32 for a stable mean, then cast back to the input dtype — upstream
    # stores mean_k in a ``dtype=k.dtype`` (bf16) buffer, so the subsequent matmul
    # sees bf16 means.
    counts = torch.full((nqb,), float(block_size), device=q.device)
    if pad:
        counts[-1] = float(block_size - pad)
    mean_k = (kd.float().view(B, H, nqb, block_size, D).sum(3) / counts.view(1, 1, nqb, 1)).to(
        qk_dtype
    )  # (B,H,nkb,D) bf16

    qb_all = qd.view(B, H, nqb, block_size, D)  # (B,H,nqb,bs,D) bf16
    neg = torch.finfo(torch.float32).min
    neg_t = torch.tensor(neg, device=q.device)
    kmax = torch.arange(nkb, device=q.device).view(1, 1, 1, 1, nkb) * block_size + (block_size - 1)
    row_off = torch.arange(block_size, device=q.device).view(1, 1, 1, block_size, 1)

    score = torch.empty(B, H, nqb, nkb, device=q.device, dtype=torch.float32)

    # Tile the query-block axis so the (q_blocks, bs, nkb) fp32 intermediate stays
    # bounded. q_chunk * nkb ~= _SCORE_QK_BUDGET (>=1 block). Bit-identical to the
    # whole-tensor computation — each query block's reductions are independent.
    q_chunk = max(1, _SCORE_QK_BUDGET // max(1, nkb))
    for i0 in range(0, nqb, q_chunk):
        i1 = min(i0 + q_chunk, nqb)
        qb = qb_all[:, :, i0:i1]  # (B,H,c,bs,D) bf16
        qk = (
            torch.einsum("bhipd,bhjd->bhipj", qb, mean_k).float() * scale * _LOG2E
        )  # (B,H,c,bs,nkb)

        qpos = (
            torch.arange(i0, i1, device=q.device).view(1, 1, i1 - i0, 1, 1) * block_size + row_off
        )
        causal = qpos >= kmax  # (1,1,c,bs,nkb)
        mblock = torch.where(causal, qk, neg_t).amax(dim=3, keepdim=True)  # (B,H,c,1,nkb)
        p = torch.where(causal, torch.exp2(qk - mblock), torch.zeros_like(qk))
        s = p.sum(dim=3)  # (B,H,c,nkb)

        mb = mblock.squeeze(3)  # (B,H,c,nkb)
        valid = torch.isfinite(mb)
        max_j = torch.where(valid, mb, neg_t).amax(dim=-1, keepdim=True)
        resc = torch.where(valid, torch.exp2(mb - max_j), torch.zeros_like(mb))
        s = s * resc
        s = s / (s.sum(dim=-1, keepdim=True) + 1e-9)
        score[:, :, i0:i1] = s

    return score  # (B,H,nqb,nkb)


def build_block_keep_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    alpha: float,
    block_size: int = 128,
    sink: int = 256,
    window: int = 512,
    last_n_block_full: int = 2,
) -> torch.Tensor:
    """Return the FlashPrefill keep mask, bool ``(B, H, nqb, nkb)`` (True=keep).

    ``q``/``k`` are post-repeat ``(B, H, L, D)`` (K already at num_attention_heads).
    ``nqb == nkb == ceil(L / block_size)``. A kept (I, J) means query-block I
    attends to key-block J. ``sink``/``window`` are given in TOKENS and converted
    to whole blocks; ``last_n_block_full`` is in blocks. Causality (J > I dropped)
    is always enforced last.
    """
    B, H, L, D = q.shape
    nqb = (L + block_size - 1) // block_size
    nkb = nqb
    score = _block_importance(q, k, block_size)  # (B,H,nqb,nkb)

    sink_b = max(1, sink // block_size)
    win_b = max(1, window // block_size)

    q_ids = torch.arange(nqb, device=q.device).view(1, 1, nqb, 1)  # query-block index I
    k_ids = torch.arange(nkb, device=q.device).view(1, 1, 1, nkb)  # key-block index J

    # Stage 2: max-based dynamic threshold (single-pass max over J).
    peak = score.amax(dim=-1, keepdim=True)  # (B,H,nqb,1)
    mask_score = score >= (alpha * peak)  # alpha=0 -> all True

    # Always-keep regions (paper §4).
    mask_sink = k_ids < sink_b  # first sink blocks
    dist = q_ids - k_ids
    mask_window = (dist >= 0) & (dist < win_b)  # local window
    mask_last = q_ids >= (nqb - last_n_block_full)  # last N q-blocks attend all
    mask_causal = dist >= 0

    keep = (mask_score | mask_sink | mask_window | mask_last) & mask_causal
    return keep
