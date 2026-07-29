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

"""Pure-torch reference implementations of the MInference prefill patterns.

The ``minference`` family covers three user-facing variants that all
map to one algorithm class:

  * ``a_shape``  — A-shape / StreamingLLM: attention sink (first ``n_init``
    keys) + sliding window (last ``n_local`` keys).
  * ``tri_shape`` — Tri-shape: ``a_shape`` on all but the last ``n_last``
    queries, plus full causal attention for the last ``n_last`` queries.
  * ``minference`` (vertical_and_slash) — per-head estimate of the important
    vertical columns + slash diagonals from the last-64 queries' attention.

These mirror MInference upstream
(``minference/modules/minference_forward.py`` and
``minference/ops/streaming_kernel.py`` at revision ``a4eb395``) at **boolean-mask
granularity** — i.e. this is the *pseudo-sparse* reference path. The production
fused Triton/CUDA kernels
(``pit_sparse_flash_attention``, ``TritonMultiStageDotProductionAttention``) are
vendored at kernel-build time; until then a missing kernel
hard-fails unless ``allow_pseudo_sparse=true`` selects these references.

Invariants shared with upstream (verified against the source):
  * All three consume **post-``repeat_kv``** K/V at ``num_attention_heads``
    granularity (``requires_unrepeated_kv=False``).
  * ``q_len == 1`` (decode) never reaches here; the forward template early-exits
    to the configured dense attention implementation.
  * Each pattern **degrades to exact dense causal attention** when its budget
    covers the whole sequence (``n_init+n_local >= k_len``; vertical/slash
    sizes ``>= q_len``). This is the property the real-weights parity tests use.
"""

from __future__ import annotations

import math  # noqa: F401

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dense reference (the degrade-to target)
# ---------------------------------------------------------------------------
def dense_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Plain causal SDPA. ``q,k,v`` are post-repeat ``(B, H, L, D)``."""
    scaling = q.shape[-1] ** -0.5
    attn = torch.matmul(q, k.transpose(2, 3)) * scaling
    Lq, Lk = q.shape[2], k.shape[2]
    causal = torch.ones((Lq, Lk), device=q.device, dtype=torch.bool).triu(1 + (Lk - Lq))
    attn = attn.masked_fill(causal[None, None], float("-inf"))
    probs = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, v)


def _masked_softmax_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keep: torch.Tensor
) -> torch.Tensor:
    """SDPA where ``keep`` (broadcastable bool ``[*, Lq, Lk]``) marks allowed
    key positions. Disallowed positions get ``-inf`` before softmax.

    ``keep`` MUST already be causal (upper triangle False); we do not re-impose
    causality so callers can express patterns that are a subset of the causal
    mask exactly as upstream does.
    """
    scaling = q.shape[-1] ** -0.5
    attn = torch.matmul(q, k.transpose(2, 3)) * scaling
    attn = attn.masked_fill(~keep, float("-inf"))
    probs = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, v)


def _causal_mask(Lq: int, Lk: int, device) -> torch.Tensor:
    """Lower-triangular keep mask ``(Lq, Lk)`` aligned to the bottom-right
    (query position ``i`` attends keys ``0..i`` with the standard offset)."""
    offset = Lk - Lq
    qi = torch.arange(Lq, device=device).view(Lq, 1)
    ki = torch.arange(Lk, device=device).view(1, Lk)
    return ki <= (qi + offset)


# ---------------------------------------------------------------------------
# a_shape (StreamingLLM): sink + sliding window
# ---------------------------------------------------------------------------
def a_shape_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_init: int = 128,
    n_local: int = 3968,
) -> torch.Tensor:
    """Attention sink (first ``n_init`` keys) + sliding window (last ``n_local``).

    Mirrors ``streaming_kernel.a_shape_kernel`` at mask granularity. Degrades to
    dense when ``n_init + n_local >= k_len``.
    """
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    device = q.device
    causal = _causal_mask(Lq, Lk, device)  # (Lq, Lk)

    ki = torch.arange(Lk, device=device).view(1, Lk)
    qi = torch.arange(Lq, device=device).view(Lq, 1)
    offset = Lk - Lq

    sink = ki < n_init
    # Sliding window: key within the last n_local positions relative to query.
    window = ki > (qi + offset - n_local)
    keep = (sink | window) & causal
    # Guarantee every query keeps at least its own (diagonal) key. With a
    # pathological config (n_init=0 and n_local=0, or a query past the sink with
    # an empty window) a row could otherwise be fully masked -> softmax over all
    # -inf -> NaN. vertical_and_slash already has this guard; mirror it here.
    self_key = (ki == (qi + offset)) & causal
    keep = keep | self_key
    return _masked_softmax_attention(q, k, v, keep[None, None])


# ---------------------------------------------------------------------------
# tri_shape: a_shape on q[:-n_last] + full causal on the last n_last queries
# ---------------------------------------------------------------------------
def tri_shape_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_init: int = 128,
    n_local: int = 3968,
    n_last: int = 100,
) -> torch.Tensor:
    """Tri-shape. Mirrors ``streaming_kernel.tri_shape_kernel``.

    The last ``min(n_last, q_len-1)`` queries attend fully (causal); the rest
    use ``a_shape``. Degrades to dense when ``a_shape`` does and ``n_last``
    covers the tail.
    """
    Lq = q.shape[2]
    n_last = min(n_last, max(Lq - 1, 0))
    if n_last <= 0:
        return a_shape_attention(q, k, v, n_init, n_local)

    q1 = q[:, :, :-n_last]
    y1 = a_shape_attention(q1, k[:, :, :-n_last], v[:, :, :-n_last], n_init, n_local)

    q2 = q[:, :, -n_last:]
    # Full causal attention for the last n_last queries over ALL keys.
    Lk = k.shape[2]
    device = q.device
    qi = torch.arange(Lq - n_last, Lq, device=device).view(n_last, 1)
    ki = torch.arange(Lk, device=device).view(1, Lk)
    keep = ki <= qi  # causal over the full key range
    y2 = _masked_softmax_attention(q2, k, v, keep[None, None])
    return torch.cat([y1, y2], dim=2)


# ---------------------------------------------------------------------------
# vertical_and_slash (the "minference" variant)
# ---------------------------------------------------------------------------
def _sum_all_diagonals(mat: torch.Tensor) -> torch.Tensor:
    """Sum each diagonal of ``mat`` (B,H,n,m). Returns (B,H, n+m-1).

    Diagonal-aligned reproduction of upstream ``sum_all_diagonal_matrix`` using
    a stride trick. ``out[..., d]`` is the sum over the d-th diagonal (d=0 is the
    top-right-most, increasing toward bottom-left), matching upstream indexing
    where ``slash`` index ``q_len-1`` is the main diagonal.
    """
    B, H, n, m = mat.shape
    zero = torch.zeros((B, H, n, n), device=mat.device, dtype=mat.dtype)
    padded = torch.cat((zero, mat, zero), dim=-1)  # (B,H,n, m+2n)
    strided = padded.as_strided(
        (B, H, n, n + m), (H * n * (2 * n + m), n * (2 * n + m), 2 * n + m + 1, 1)
    )
    sum_diags = strided.sum(dim=2)  # (B,H, n+m)
    return sum_diags[:, :, 1:]  # (B,H, n+m-1)


def vertical_and_slash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    vertical_size: int = 1000,
    slash_size: int = 6096,
    last_q: int = 64,
) -> torch.Tensor:
    """Per-head vertical-column + slash-diagonal sparse attention.

    Mirrors ``minference_forward.vertical_and_slash_kernel`` at mask
    granularity:
      1. Estimate attention from the last ``last_q`` queries over all keys.
      2. ``vertical`` = column importance (sum over the last-q rows); keep the
         top ``vertical_size`` columns. First 30 columns always kept (sinks).
      3. ``slash`` = diagonal importance; keep the top ``slash_size`` diagonals.
         Last 100 diagonals (the local band) always kept.
      4. Union of (kept columns ∪ kept diagonals), intersected with causal.

    Degrades to dense when ``vertical_size >= q_len`` and ``slash_size >= q_len``
    (every column and diagonal is kept).
    """
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    device = q.device
    scaling = D**-0.5

    vertical_size = min(Lq, max(int(vertical_size), 30))
    slash_size = min(Lq, max(int(slash_size), 50))
    lq = min(last_q, Lq)

    # --- 1. estimation over the last lq queries ---------------------------
    q_est = q[:, :, -lq:, :]
    qk = torch.matmul(q_est, k.transpose(2, 3)) * scaling  # (B,H,lq,Lk)
    # Causal within the estimation block (its keys align to the tail).
    qi = torch.arange(Lk - lq, Lk, device=device).view(lq, 1)
    ki = torch.arange(Lk, device=device).view(1, Lk)
    qk = qk.masked_fill(ki > qi, float("-inf"))
    qk = F.softmax(qk, dim=-1, dtype=torch.float32)  # (B,H,lq,Lk)

    # --- 2. vertical: column importance -----------------------------------
    vertical = qk.sum(dim=2)  # (B,H,Lk)
    if vertical.shape[-1] > 30:
        vertical[..., :30] = torch.inf  # always keep the first 30 (sinks)
    vsize = min(vertical_size, vertical.shape[-1])
    vtop = torch.topk(vertical, vsize, dim=-1).indices  # (B,H,vsize)
    keep_cols = torch.zeros((B, H, Lk), dtype=torch.bool, device=device)
    keep_cols.scatter_(-1, vtop, True)  # (B,H,Lk)

    # --- 3. slash: diagonal importance ------------------------------------
    # _sum_all_diagonals returns sums indexed by d = (key - query_est) + (lq-1)
    # in estimation-block coords. Because estimation row i is full-query position
    # (Lq-lq+i), that index is IDENTICALLY the full-matrix diagonal id
    # d = (key - query) + (Lq-1) the consumer uses below. The causal diagonals
    # are d in [0, Lk-1] (d = Lk-1 is the main diagonal). So we take the FIRST
    # Lk sums — NOT the last Lk. (The previous `slash[..., -Lk:]` dropped the
    # leading lq-1 entries, shifting every kept diagonal id by lq-1 — e.g. with
    # lq=64 the dominant main diagonal was read as id 64 instead of 127. Proven
    # by test_vertical_and_slash_diagonal_alignment.)
    slash = _sum_all_diagonals(qk)  # (B,H, lq+Lk-1), index d == full diag_id
    slash = slash[..., :Lk]  # keep causal diagonals d in [0, Lk-1]
    if slash.shape[-1] > 100:
        slash[..., -100:] = torch.inf  # always keep the local band (near main diag)
    ssize = min(slash_size, slash.shape[-1])
    stop = torch.topk(slash, ssize, dim=-1).indices  # (B,H,ssize) diag ids
    # diag id d corresponds to key offset (d - (Lk-1)) from the query diagonal.
    # Build a (B,H,Lq,Lk) keep-mask for the chosen diagonals.
    qpos = torch.arange(Lq, device=device).view(1, 1, Lq, 1)
    kpos = torch.arange(Lk, device=device).view(1, 1, 1, Lk)
    offset = Lk - Lq
    # diagonal id for entry (qi,ki) is (ki - (qi+offset)) + (Lk-1)
    diag_id = (kpos - (qpos + offset)) + (Lk - 1)  # (1,1,Lq,Lk)
    keep_diag = _scatter_diagonals(diag_id, stop, B, H, Lq, Lk)

    # --- 4. union, intersect causal ---------------------------------------
    keep = keep_cols.view(B, H, 1, Lk) | keep_diag
    causal = _causal_mask(Lq, Lk, device).view(1, 1, Lq, Lk)
    keep = keep & causal
    # guarantee every query keeps at least its own (diagonal) key
    self_key = (diag_id == (Lk - 1)) & causal
    keep = keep | self_key
    return _masked_softmax_attention(q, k, v, keep)


def _scatter_diagonals(diag_id, stop, B, H, Lq, Lk):
    """Build a (B,H,Lq,Lk) bool mask: True where ``diag_id`` is one of the
    per-head chosen diagonal ids in ``stop`` (B,H,ssize)."""
    device = diag_id.device
    ssize = stop.shape[-1]
    # one-hot over diagonal ids then gather: cheaper to compare via broadcasting.
    # diag_id: (1,1,Lq,Lk) ; stop: (B,H,ssize)
    keep = torch.zeros((B, H, Lq, Lk), dtype=torch.bool, device=device)
    diag_flat = diag_id.expand(B, H, Lq, Lk)
    for s in range(ssize):
        sel = stop[:, :, s].view(B, H, 1, 1)
        keep |= diag_flat == sel
    return keep


# ---------------------------------------------------------------------------
# Variant dispatch
# ---------------------------------------------------------------------------
def minference_reference_prefill(
    variant: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    streaming_kwargs: dict | None = None,
    best_pattern: dict | None = None,
    layer_idx: int = 0,
) -> torch.Tensor:
    """Dispatch ``q,k,v`` (post-repeat ``(B,H,L,D)``) to the chosen variant.

    ``a_shape`` / ``tri_shape`` use ``streaming_kwargs`` (n_init/n_local/n_last).
    ``minference`` uses ``best_pattern[layer_idx][head]`` if provided, else the
    dense fallback ``("vertical_and_slash", 1000, 6096)`` per head.
    """
    sk = streaming_kwargs or {}
    if variant == "a_shape":
        return a_shape_attention(q, k, v, sk.get("n_init", 128), sk.get("n_local", 3968))
    if variant == "tri_shape":
        return tri_shape_attention(
            q,
            k,
            v,
            sk.get("n_init", 128),
            sk.get("n_local", 3968),
            sk.get("n_last", 100),
        )
    if variant == "minference":
        return _minference_per_head(q, k, v, best_pattern, layer_idx)
    raise ValueError(f"unknown minference variant {variant!r}")


def _minference_per_head(q, k, v, best_pattern, layer_idx):
    """Per-head vertical_and_slash dispatch (the head-heterogeneous path).

    Upstream picks ``(ty, vertical, slash)`` per head from ``best_pattern`` with
    the dense fallback ``("vertical_and_slash", 1000, 6096, 1)``. We honour the
    per-head split: heads sharing a (vertical, slash) budget batch together.
    """
    B, H, Lq, D = q.shape
    layer_pat = (best_pattern or {}).get(layer_idx, {}) if best_pattern else {}
    outs = []
    for h in range(H):
        entry = layer_pat.get(h) or layer_pat.get(str(h))
        if entry is None:
            ty, vsize, ssize = "vertical_and_slash", 1000, 6096
        else:
            ty, vsize, ssize = entry[0], int(entry[1]), int(entry[2])
        qh, kh, vh = q[:, h : h + 1], k[:, h : h + 1], v[:, h : h + 1]
        if ty == "stream_llm":
            outs.append(a_shape_attention(qh, kh, vh, vsize, ssize))
        else:  # vertical_and_slash (+ block_sparse falls back to v_and_s here)
            outs.append(vertical_and_slash_attention(qh, kh, vh, vsize, ssize))
    return torch.cat(outs, dim=1)
