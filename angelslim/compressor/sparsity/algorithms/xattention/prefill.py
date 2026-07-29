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

"""XAttention prefill dispatch (kernel / reference policy).

XAttention (MInference family) selects, per head, the smallest set of key-blocks
whose cumulative (antidiagonal-strided) attention mass covers ``threshold``
(default 0.9), then runs the selected-block attention through
``block_sparse_attn``. The real path needs ``triton`` + ``block_sparse_attn``,
with head_dim in {16,32,64,128} (block_sparse_attn miscomputes at head_dim 256),
so Qwen3.5 (head_dim 256) routes to the torch reference.

The vendored entry already takes ``(B, H, L, D)`` (our native layout) — no
transpose, unlike flexprefill. K/V are already repeated to num_attention_heads
by the caller (the template's prefill_fn), matching the post-repeat K/V contract.
"""

from __future__ import annotations

import torch

from . import kernels as _k


def _kernel_prefill(q, k, v, *, stride, norm, threshold, block_size, chunk_size):
    """Real fused XAttention on post-repeat ``(B, H, L, D)`` K/V.

    The vendored ``Xattention_prefill`` consumes ``(B, H, L, D)`` directly and
    asserts ``block_size == 128`` and ``batch == 1`` upstream.
    """
    return _k.xattention_prefill(
        q,
        k,
        v,
        stride=stride,
        norm=norm,
        threshold=threshold,
        block_size=block_size,
        chunk_size=chunk_size,
    )


def xattention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    stride: int,
    norm: int,
    threshold: float,
    block_size: int,
    chunk_size: int,
    head_dim: int,
    allow_pseudo_sparse: bool,
):
    """Prefill dispatch. ``q,k,v`` are post-repeat ``(B, H, L, D)``."""
    from ..._kernel_check import KernelMissingError, warn_pseudo_sparse_fallback

    # The vendored kernel hard-asserts block_size == 128 and batch_size == 1
    # (xattention.py:547-548). Pre-validate HERE so an unsupported config/shape
    # gets an actionable message (or the reference) instead of a bare
    # AssertionError from deep in the kernel.
    batch_size = q.shape[0]
    _kernel_ok = _k.kernels_available(head_dim)
    _cfg_ok = (block_size == 128) and (batch_size == 1)

    if _kernel_ok and not _cfg_ok:
        reason = (
            f"block_size {block_size} != 128 (kernel only supports 128)"
            if block_size != 128
            else f"batch_size {batch_size} != 1 (kernel requires batch 1)"
        )
        if allow_pseudo_sparse:
            from .reference import xattention_reference_prefill

            warn_pseudo_sparse_fallback("xattention", reason)
            return xattention_reference_prefill(
                q,
                k,
                v,
                threshold=threshold,
                block_size=block_size,
            )
        raise KernelMissingError(
            f"[sparse:xattention] the real XAttention kernel cannot run this input: "
            f"{reason}. Set compression.sparsity.allow_pseudo_sparse=true to fall "
            f"back to the pure-torch reference, or use a supported config "
            f"(block_size=128, batch_size=1)."
        )

    if _kernel_ok:
        # Pin the CUDA device to the input's device — the block_sparse_attn
        # CUDA kernel and the Triton estimate launch on the *current* device's
        # stream, which under accelerate device_map sharding would otherwise be
        # the wrong GPU.
        if q.is_cuda:
            with torch.cuda.device(q.device):
                return _kernel_prefill(
                    q,
                    k,
                    v,
                    stride=stride,
                    norm=norm,
                    threshold=threshold,
                    block_size=block_size,
                    chunk_size=chunk_size,
                )
        return _kernel_prefill(
            q,
            k,
            v,
            stride=stride,
            norm=norm,
            threshold=threshold,
            block_size=block_size,
            chunk_size=chunk_size,
        )

    # Kernel unavailable (e.g. head_dim 256 on Qwen3.5) -> torch reference.
    if allow_pseudo_sparse:
        from .reference import xattention_reference_prefill

        warn_pseudo_sparse_fallback("xattention", f"head_dim={head_dim}")
        return xattention_reference_prefill(
            q,
            k,
            v,
            threshold=threshold,
            block_size=block_size,
        )

    raise KernelMissingError(
        "[sparse:xattention] the real XAttention path is unavailable for this "
        f"model (needs triton + block_sparse_attn, and head_dim in "
        f"{{16,32,64,128}}; got head_dim={head_dim} — block_sparse_attn "
        f"miscomputes at head_dim 256). To run anyway with the slow pure-torch "
        f"reference, set compression.sparsity.allow_pseudo_sparse=true in your "
        f"YAML."
    )
