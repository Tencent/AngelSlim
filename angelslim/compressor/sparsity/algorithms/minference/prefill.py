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

"""MInference prefill dispatch: real vendored kernels with pseudo-sparse fallback.

Single entry point :func:`minference_prefill` chooses, per the variant and the
kernel/fallback policy:

  * real vendored kernel when available (Triton for a_shape/tri_shape; Triton +
    the CUDA index ext for minference); else
  * the pure-torch reference (``reference.py``) when ``allow_pseudo_sparse``; else
  * hard-fail with :class:`KernelMissingError`.

The per-head vertical/slash index *estimation* (last-64-query attention → top-k
columns + diagonals) is ported faithfully from MInference upstream
(``minference_forward.gather_last_q_vertical_slash_topk_v4.vertical_and_slash_kernel``,
revision a4eb395) so the indices fed to the real kernel match upstream.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..._kernel_check import KernelMissingError, warn_pseudo_sparse_fallback
from . import kernels as _k
from .reference import minference_reference_prefill


# ---------------------------------------------------------------------------
# Vertical/slash index estimation (ported from upstream, a4eb395)
# ---------------------------------------------------------------------------
def _sum_all_diagonal_matrix(mat: torch.Tensor) -> torch.Tensor:
    """Sum each diagonal of ``mat`` (B,H,n,m). Upstream stride-trick version."""
    b, h, n, m = mat.shape
    zero = torch.zeros((b, h, n, n), device=mat.device, dtype=mat.dtype)
    padded = torch.cat((zero, mat, zero), dim=-1)  # (b,h,n, m+2n)
    strided = padded.as_strided(
        (b, h, n, n + m),
        (h * n * (2 * n + m), n * (2 * n + m), 2 * n + m + 1, 1),
    )
    sum_diags = torch.sum(strided, 2)  # (b,h, n+m)
    return sum_diags[:, :, 1:]


def estimate_vertical_slash_indexes(
    q: torch.Tensor,
    k: torch.Tensor,
    vertical_size: int,
    slash_size: int,
    head_dim: int,
):
    """Estimate per-head vertical-column + slash-diagonal indices.

    Faithful port of upstream ``vertical_and_slash_kernel``'s estimation block.
    ``q,k`` are post-repeat ``(B, H, L, D)``. Returns ``(v_idx, s_idx)`` for
    :func:`vertical_slash_sparse_attention`.
    """
    q_len = q.shape[2]
    vertical_size = min(q_len, max(int(vertical_size), 30))
    slash_size = min(q_len, max(int(slash_size), 50))
    last_q = min(64, q_len)

    arange = torch.arange(last_q, device=q.device)
    last_q_mask = arange[None, None, :, None] >= arange[None, None, None, :]

    qk = torch.einsum("bhmk,bhnk->bhmn", q[:, :, -last_q:, :], k) / math.sqrt(head_dim)
    qk[:, :, :, -last_q:] = torch.where(
        last_q_mask[..., -last_q:, -last_q:].to(q.device),
        qk[:, :, :, -last_q:],
        -torch.inf,
    )
    qk = F.softmax(qk, dim=-1, dtype=torch.float32)

    vertical = qk.sum(-2, keepdim=True)
    vertical[..., :30] = torch.inf
    vertical_topk = torch.topk(vertical, vertical_size, -1).indices

    slash = _sum_all_diagonal_matrix(qk)[..., : -last_q + 1]
    slash[..., -100:] = torch.inf
    slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

    return vertical_topk, slash


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def _kernel_prefill(variant, q, k, v, streaming_kwargs, best_pattern, layer_idx, head_dim):
    """Run the real vendored kernel for ``variant``. ``q,k,v`` post-repeat."""
    if variant == "a_shape":
        sf = _k.get_streaming_forward()
        return sf(q, k, v, streaming_kwargs["n_init"], streaming_kwargs["n_local"])

    if variant == "tri_shape":
        sf = _k.get_streaming_forward()
        n_last = min(streaming_kwargs.get("n_last", 100), q.shape[2] - 1)
        if n_last <= 0:
            return sf(q, k, v, streaming_kwargs["n_init"], streaming_kwargs["n_local"])
        q1 = q[:, :, :-n_last]
        y1 = sf(
            q1,
            k[:, :, :-n_last],
            v[:, :, :-n_last],
            streaming_kwargs["n_init"],
            streaming_kwargs["n_local"],
        )
        # full-causal tail for the last n_last queries (upstream tri_shape_kernel)
        q2 = q[:, :, -n_last:]
        Lk = k.shape[2]
        qi = torch.arange(q.shape[2] - n_last, q.shape[2], device=q.device).view(n_last, 1)
        ki = torch.arange(Lk, device=q.device).view(1, Lk)
        keep = (ki <= qi)[None, None]
        attn = torch.matmul(q2, k.transpose(2, 3)) / math.sqrt(head_dim)
        attn = attn.masked_fill(~keep, float("-inf"))
        probs = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        y2 = torch.matmul(probs, v)
        return torch.cat([y1, y2], dim=2)

    if variant == "minference":
        vsa = _k.get_vertical_slash_attention()
        sf = _k.get_streaming_forward()
        B, H, Lq, D = q.shape
        layer_pat = (best_pattern or {}).get(layer_idx, {}) if best_pattern else {}
        outs = []
        for h in range(H):
            entry = layer_pat.get(h) or layer_pat.get(str(h))
            # Dispatch on the searched pattern TYPE (entry[0]), mirroring the
            # reference (reference.py): a `stream_llm` head runs the streaming
            # (a_shape) kernel with (n_init, n_local) = (entry[1], entry[2]); any
            # other type (incl. the dense fallback) runs vertical-and-slash. The
            # earlier kernel path ignored entry[0] and always ran v-and-s, which
            # silently mis-ran a stream_llm head and fed n_init/n_local in as
            # vertical/slash budgets. (Today's dense fallback is
            # vertical_and_slash, where both paths agree; this aligns the kernel
            # with the reference for the day a searched pattern JSON ships.)
            if entry is None:
                ptype, a, b = "vertical_and_slash", 1000, 6096
            else:
                ptype, a, b = entry[0], int(entry[1]), int(entry[2])
            qh, kh, vh = q[:, h : h + 1], k[:, h : h + 1], v[:, h : h + 1]
            if ptype == "stream_llm":
                outs.append(sf(qh, kh, vh, a, b))  # a=n_init, b=n_local
            else:
                v_idx, s_idx = estimate_vertical_slash_indexes(qh, kh, a, b, D)
                outs.append(vsa(qh, kh, vh, v_idx, s_idx))
        return torch.cat(outs, dim=1)

    raise ValueError(f"unknown minference variant {variant!r}")


def minference_prefill(
    variant: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    streaming_kwargs: dict,
    best_pattern: dict | None,
    layer_idx: int,
    head_dim: int,
    allow_pseudo_sparse: bool,
):
    """Prefill dispatch with the kernel/fallback policy.

    ``q,k,v`` are post-repeat ``(B, H, L, D)``.
    """
    if _k.kernels_available(variant, head_dim):
        # Pin the active CUDA device to the input's device. The vertical_slash
        # CUDA ext + Triton kernel launch on the *current* device's stream, not
        # the tensor's device; under accelerate ``device_map`` layer sharding a
        # cuda:1 layer would otherwise launch on cuda:0 and corrupt the result
        # (same bug class as the Stem dispatcher fix; G1). The a_shape/tri_shape
        # streaming kernel already guards internally, but wrapping the whole
        # dispatch is uniform and a no-op on a single GPU.
        if q.is_cuda:
            with torch.cuda.device(q.device):
                return _kernel_prefill(
                    variant, q, k, v, streaming_kwargs, best_pattern, layer_idx, head_dim
                )
        return _kernel_prefill(
            variant, q, k, v, streaming_kwargs, best_pattern, layer_idx, head_dim
        )

    # Kernel unavailable (or unsupported head_dim) -> torch reference.
    if allow_pseudo_sparse:
        warn_pseudo_sparse_fallback(variant, f"head_dim={head_dim}")
        return minference_reference_prefill(
            variant,
            q,
            k,
            v,
            streaming_kwargs=streaming_kwargs,
            best_pattern=best_pattern,
            layer_idx=layer_idx,
        )

    raise KernelMissingError(
        f"[sparse:{variant}] the real MInference kernel is unavailable for this "
        f"model in this environment (needs triton + CUDA"
        + (
            " + the convert_vertical_slash_indexes CUDA extension, and the "
            "vertical_slash kernel supports head_dim in {16,32,64,128} only "
            f"(got {head_dim})"
            if variant == "minference"
            else ""
        )
        + ").\nTo run anyway with the slow pure-torch reference, set "  # noqa: F541
        f"compression.sparsity.allow_pseudo_sparse=true in your YAML."
    )
