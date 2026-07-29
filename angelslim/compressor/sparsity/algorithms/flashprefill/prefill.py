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

"""FlashPrefill prefill dispatch (clean-room; kernel/reference policy).

FlashPrefill (arXiv:2603.06199) keeps, per query-block, the key-blocks whose
block-approximate energy clears a max-based dynamic threshold (``alpha`` * the
per-query-block peak), plus sink + local window. The block-selection
(``blockmask.build_block_keep_mask``) is original AngelSlim torch; the
selected-block attention (stage 3) runs on ``block_sparse_attn`` — the SAME
BSD-3-Clause-licensed kernel Stem / xattention use — fed our keep mask. No upstream
FlashPrefill code is vendored (the upstream repo is unlicensed); see the package
NOTICE.

head_dim must be in {16,32,64,128} (block_sparse_attn miscomputes at 256), so
Qwen3.5 (head_dim 256) routes to the torch reference.
"""

from __future__ import annotations

import torch

from .kernels_check import kernels_available


def _flash_dense(q, k, v, *, causal=True):
    """Flash-attention dense prefill on ``(B, H, L, D)`` — the alpha<=0 keep-all
    path (nothing pruned == dense; flash dense matches the model's fa2 baseline)."""
    from flash_attn import flash_attn_func

    out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal)
    return out.transpose(1, 2)


def _kernel_prefill(q, k, v, *, alpha, block_size, sink, window, last_n_block_full, causal=True):
    """Stage 3 via BSD-3-Clause ``block_sparse_attn_func`` with our clean-room keep mask.

    ``q,k,v`` are post-repeat ``(B, H, L, D)``. Uses the same varlen calling
    convention as the in-repo (MIT) xattention kernel: flatten batch=1 to
    ``(L, H, D)``, cu_seqlens ``[0, L]``, all-sparse ``head_mask_type=1``.

    ``alpha <= 0`` means "keep every (causal) block", i.e. dense attention, so we
    route straight to flash dense. This is a deliberate optimization, not a
    workaround: when nothing is pruned there is no reason to pay for a block-
    sparse kernel, and flash dense is also numerically *better* here —
    ``block_sparse_attn`` over a full causal mask is not bit-identical to flash
    dense and that per-layer gap compounds across the stack (≈0.11 rel over 36
    Qwen3-8B layers, though top-1 is unchanged), whereas flash dense matches the
    model's own fa2 baseline exactly. For ``alpha > 0`` (the real sparse case)
    the block_sparse_attn path is used as intended.
    """
    if alpha <= 0.0:
        return _flash_dense(q, k, v, causal=causal)

    from block_sparse_attn import block_sparse_attn_func

    from .blockmask import build_block_keep_mask

    B, H, L, D = q.shape
    assert B == 1, "FlashPrefill kernel path requires batch_size == 1 (block_sparse_attn)"
    keep = build_block_keep_mask(
        q,
        k,
        alpha=alpha,
        block_size=block_size,
        sink=sink,
        window=window,
        last_n_block_full=last_n_block_full,
    )
    nqb = (L + block_size - 1) // block_size
    nkb = nqb
    qf = q.transpose(1, 2).reshape(L, H, D)
    kf = k.transpose(1, 2).reshape(L, H, D)
    vf = v.transpose(1, 2).reshape(L, H, D)
    q_cu = torch.tensor([0, L], dtype=torch.int32, device=q.device)
    k_cu = torch.tensor([0, L], dtype=torch.int32, device=q.device)
    head_mask_type = torch.ones(H, dtype=torch.int32, device=q.device)
    out = block_sparse_attn_func(
        qf,
        kf,
        vf,
        q_cu,
        k_cu,
        head_mask_type,
        None,  # streaming_info
        keep[:, :, :nqb, :nkb].contiguous(),  # base_blockmask (B,H,nqb,nkb)
        L,
        L,
        p_dropout=0.0,
        deterministic=True,
        is_causal=causal,
    )
    return out.view(B, L, H, D).transpose(1, 2)  # back to (B,H,L,D)


def flashprefill_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    alpha: float,
    block_size: int,
    sink: int,
    window: int,
    last_n_block_full: int,
    head_dim: int,
    allow_pseudo_sparse: bool,
):
    """Prefill dispatch. ``q,k,v`` are post-repeat ``(B, H, L, D)``."""
    from ..._kernel_check import (
        KernelMissingError,
        kernel_available,
        warn_pseudo_sparse_fallback,
    )

    # The alpha<=0 "keep everything == dense" path runs flash-attention dense
    # (_kernel_prefill -> _flash_dense -> `from flash_attn import flash_attn_func`),
    # which needs flash_attn — a dependency the kernel-availability probe
    # (triton + block_sparse_attn) does NOT cover. So a user with block_sparse_attn
    # but no flash_attn, running the documented alpha=0 dense config, would hit a
    # bare ImportError from deep in _flash_dense, bypassing the allow_pseudo_sparse
    # policy. Pre-validate HERE so that case takes the policy (the torch reference
    # recovers dense exactly at alpha<=0) or an actionable hard-fail instead.
    # alpha>0 uses block_sparse_attn and does NOT need flash_attn, so only the
    # dense sub-path is gated.
    if kernels_available(head_dim) and alpha <= 0.0 and not kernel_available("flash_attn"):
        if allow_pseudo_sparse:
            from .reference import flashprefill_reference_prefill

            warn_pseudo_sparse_fallback("flashprefill", "alpha<=0 dense path needs flash_attn")
            return flashprefill_reference_prefill(
                q,
                k,
                v,
                alpha=alpha,
                block_size=block_size,
                sink=sink,
                window=window,
                last_n_block_full=last_n_block_full,
            )
        raise KernelMissingError(
            "[sparse:flashprefill] the alpha<=0 (dense) path runs flash-attention "
            "dense, which requires flash_attn; it is not installed. Install "
            "flash_attn, use alpha>0 (the block_sparse_attn sparse path, which does "
            "not need it), or set compression.sparsity.allow_pseudo_sparse=true to "
            "use the pure-torch reference."
        )

    # The block_sparse_attn kernel path hard-asserts batch_size == 1
    # (_kernel_prefill: `assert B == 1`). Pre-validate HERE so batch>1 gets an
    # actionable message (or the reference) instead of a bare AssertionError from
    # deep in the kernel.
    batch_size = q.shape[0]
    if kernels_available(head_dim) and batch_size != 1:
        if allow_pseudo_sparse:
            from .reference import flashprefill_reference_prefill

            warn_pseudo_sparse_fallback("flashprefill", f"batch_size={batch_size}")
            return flashprefill_reference_prefill(
                q,
                k,
                v,
                alpha=alpha,
                block_size=block_size,
                sink=sink,
                window=window,
                last_n_block_full=last_n_block_full,
            )
        raise KernelMissingError(
            f"[sparse:flashprefill] the real block_sparse_attn kernel requires "
            f"batch_size == 1; got batch_size={batch_size}. Set "
            f"compression.sparsity.allow_pseudo_sparse=true to fall back to the "
            f"pure-torch reference, or run with batch size 1."
        )

    if kernels_available(head_dim):
        # Pin the CUDA device to the input's device — the block_sparse_attn
        # CUDA kernel launches on the *current* device's stream, which under
        # accelerate device_map sharding would otherwise be the wrong GPU.
        if q.is_cuda:
            with torch.cuda.device(q.device):
                return _kernel_prefill(
                    q,
                    k,
                    v,
                    alpha=alpha,
                    block_size=block_size,
                    sink=sink,
                    window=window,
                    last_n_block_full=last_n_block_full,
                )
        return _kernel_prefill(
            q,
            k,
            v,
            alpha=alpha,
            block_size=block_size,
            sink=sink,
            window=window,
            last_n_block_full=last_n_block_full,
        )

    # Kernel unavailable (e.g. head_dim 256 on Qwen3.5) -> torch reference.
    if allow_pseudo_sparse:
        from .reference import flashprefill_reference_prefill

        warn_pseudo_sparse_fallback("flashprefill", f"head_dim={head_dim}")
        return flashprefill_reference_prefill(
            q,
            k,
            v,
            alpha=alpha,
            block_size=block_size,
            sink=sink,
            window=window,
            last_n_block_full=last_n_block_full,
        )

    raise KernelMissingError(
        "[sparse:flashprefill] the real FlashPrefill kernel path is unavailable "
        f"for this model (needs triton + block_sparse_attn, and head_dim in "
        f"{{16,32,64,128}}; got head_dim={head_dim} — block_sparse_attn "
        f"miscomputes at head_dim 256). To run anyway with the slow pure-torch "
        f"reference, set compression.sparsity.allow_pseudo_sparse=true in your "
        f"YAML."
    )
