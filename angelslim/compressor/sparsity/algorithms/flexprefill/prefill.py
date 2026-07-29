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

"""FlexPrefill prefill dispatch (kernel / reference policy).

FlexPrefill (MInference family) selects, per head, the smallest set of
key-blocks whose cumulative attention mass covers ``gamma`` (default 0.9), using
the last query block to estimate vertical + slash importance; ``tau`` decides
when a head degrades to block-sparse. The real path is a pure-Triton kernel
(no CUDA ext, unlike minference's vertical_slash); head_dim must be in
{16,32,64,128}, so Qwen3.5 (head_dim 256) routes to the torch reference.
"""

from __future__ import annotations

import torch

from . import kernels as _k


def _kernel_prefill(q, k, v, *, gamma, tau, min_budget, max_budget, block_size):
    """Real fused FlexPrefill on post-repeat ``(B, H, L, D)`` K/V.

    The vendored entry wants ``(B, L, H, D)``; transpose in/out. K/V are already
    repeated to num_attention_heads by the caller (the template's prefill_fn),
    matching the post-repeat K/V contract used across the suite.
    """
    qb = q.transpose(1, 2).contiguous()  # (B, L, H, D)
    kb = k.transpose(1, 2).contiguous()
    vb = v.transpose(1, 2).contiguous()
    out = _k.flex_prefill_attention(
        qb,
        kb,
        vb,
        gamma=gamma,
        tau=tau,
        min_budget=min_budget,
        max_budget=max_budget,
        block_size=block_size,
    )
    return out.transpose(1, 2)  # back to (B, H, L, D)


def flexprefill_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    gamma: float,
    tau: float,
    min_budget,
    max_budget,
    block_size: int,
    head_dim: int,
    allow_pseudo_sparse: bool,
):
    """Prefill dispatch. ``q,k,v`` are post-repeat ``(B, H, L, D)``."""
    from ..._kernel_check import KernelMissingError, warn_pseudo_sparse_fallback

    # The vendored Triton prefill kernel hard-asserts q_len == k_len (full
    # prefill, not a chunked / prefix-cache / multi-turn step where q_len < k_len),
    # q.dtype == bfloat16, and block_size in {32,64,128} (the prefill kernel's
    # supported tiling — the decode sibling accepts 16 but is unused on the
    # prefill-only sparse path, and the kernel's own assert message wrongly listed
    # 16). Pre-validate HERE so an unsupported shape/dtype/block_size gets an
    # actionable message (or the reference, which is block-size-agnostic) instead
    # of a bare AssertionError from deep in the kernel.
    q_len, k_len = q.shape[2], k.shape[2]
    _kernel_ok = _k.kernels_available(head_dim)
    _KERNEL_BLOCK_SIZES = (32, 64, 128)
    if q_len != k_len:
        reason = (
            "q_len != k_len (chunked/prefix-cache prefill unsupported by the "
            f"kernel: q_len={q_len}, k_len={k_len})"
        )
    elif q.dtype != torch.bfloat16:
        reason = f"dtype {q.dtype} != bfloat16"
    elif block_size not in _KERNEL_BLOCK_SIZES:
        reason = (
            f"block_size={block_size} unsupported by the prefill kernel "
            f"(supported: {{32, 64, 128}})"
        )
    else:
        reason = None

    if _kernel_ok and reason is not None:
        if allow_pseudo_sparse:
            from .reference import flexprefill_reference_prefill

            warn_pseudo_sparse_fallback("flexprefill", reason)
            return flexprefill_reference_prefill(
                q,
                k,
                v,
                gamma=gamma,
                tau=tau,
                block_size=block_size,
            )
        raise KernelMissingError(
            f"[sparse:flexprefill] the real Triton kernel cannot run this input: "
            f"{reason}. The kernel requires a full bf16 prefill (q_len == k_len) "
            f"with block_size in {{32, 64, 128}}. Set "
            f"compression.sparsity.allow_pseudo_sparse=true to fall back to the "
            f"pure-torch reference."
        )

    if _kernel_ok:
        # Pin the CUDA device to the input's device — the Triton kernel
        # launches on the *current* device's stream, which under accelerate
        # device_map sharding would otherwise be the wrong GPU.
        if q.is_cuda:
            with torch.cuda.device(q.device):
                return _kernel_prefill(
                    q,
                    k,
                    v,
                    gamma=gamma,
                    tau=tau,
                    min_budget=min_budget,
                    max_budget=max_budget,
                    block_size=block_size,
                )
        return _kernel_prefill(
            q,
            k,
            v,
            gamma=gamma,
            tau=tau,
            min_budget=min_budget,
            max_budget=max_budget,
            block_size=block_size,
        )

    # Kernel unavailable (e.g. head_dim 256 on Qwen3.5) -> torch reference.
    if allow_pseudo_sparse:
        from .reference import flexprefill_reference_prefill

        warn_pseudo_sparse_fallback("flexprefill", f"head_dim={head_dim}")
        return flexprefill_reference_prefill(
            q,
            k,
            v,
            gamma=gamma,
            tau=tau,
            block_size=block_size,
        )

    raise KernelMissingError(
        "[sparse:flexprefill] the real FlexPrefill Triton kernel is unavailable "
        f"for this model (needs triton + flash_attn, and head_dim in "
        f"{{16,32,64,128}}; got head_dim={head_dim}). To run anyway with the slow "
        f"pure-torch reference, set compression.sparsity.allow_pseudo_sparse=true "
        f"in your YAML."
    )
