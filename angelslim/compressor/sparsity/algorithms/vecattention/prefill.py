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

"""VecAttention prefill dispatch (kernel / reference).

VecAttention keeps, per query-block, the key columns whose mean-query logit
clears a per-head **MinP threshold** (``gap = -log(threshold)``), plus an
attention-sink (initial) block and a local band, then runs the selected-column
attention. The real path uses ``vllm_flash_attn.sparse_attn_func`` (an EXTERNAL
optional compiled vLLM-flash-attention fork, ``pip install``-ed — not vendored
in this tree); when it is not installed — or head_dim is 256 (Qwen3.5), which is
not a kernel size — the framework routes to the pure-torch reference,
exactly like minference / flexprefill / xattention / flashprefill.

Inputs are post-repeat ``(B, H, L, D)`` (K/V already at num_attention_heads).
"""

from __future__ import annotations

import torch

from .kernels_check import kernels_available


def _kernel_prefill(q, k, v, *, threshold, block_size_q, block_size_k, group_k_block, chunk_size):
    """Real VecAttention prefill via the vendored ``vllm_flash_attn`` kernel.

    Lifts the original VecAttention core (mean-query MinP column selection +
    ``sparse_attn_func``) from the legacy subsystem. ``q,k,v`` are post-repeat
    ``(B, H, L, D)``. Only reached when ``kernels_available(head_dim)`` is True.
    """
    # The vendored core now lives in this package (algorithms/vecattention/).
    # Import lazily so a missing/unbuilt kernel surfaces as
    # kernels_available()==False (handled in the dispatcher), not an import
    # error at module load.
    from .modules.forward import vecattention_forward

    return vecattention_forward(
        q,
        k,
        v,
        threshold=threshold,
        q_pooling_size=block_size_q,
        k_local_size=block_size_k,
        group_k_block=group_k_block,
        causal=True,
        chunk_size=chunk_size,
    )


def vecattention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    threshold: float,
    block_size_q: int,
    block_size_k: int,
    group_k_block: int,
    chunk_size: int,
    head_dim: int,
    allow_pseudo_sparse: bool,
):
    """Prefill dispatch. ``q,k,v`` are post-repeat ``(B, H, L, D)``."""
    from ..._kernel_check import KernelMissingError, warn_pseudo_sparse_fallback

    # The kernel forward hard-asserts q_pooling_size (== block_size_q) in {64,128}
    # and chunk_size % block_size_q == 0 (modules/forward.py:176-177). These are
    # pure CONFIG constraints; pre-validate HERE so a bad config gets an actionable
    # message (or the reference) instead of a bare AssertionError from deep in the
    # kernel.
    _kernel_ok = kernels_available(head_dim)
    _cfg_ok = (block_size_q in (64, 128)) and (chunk_size % block_size_q == 0)
    if _kernel_ok and not _cfg_ok:
        reason = (
            f"block_size_q {block_size_q} not in {{64,128}}"
            if block_size_q not in (64, 128)
            else f"chunk_size {chunk_size} not a multiple of block_size_q " f"{block_size_q}"
        )
        if allow_pseudo_sparse:
            from .reference import vecattention_reference_prefill

            warn_pseudo_sparse_fallback("vecattention", reason)
            return vecattention_reference_prefill(
                q,
                k,
                v,
                threshold=threshold,
                block_size_q=block_size_q,
                block_size_k=block_size_k,
                group_k_block=group_k_block,
            )
        raise KernelMissingError(
            f"[sparse:vecattention] the real vllm_flash_attn kernel cannot run this "
            f"config: {reason}. Set compression.sparsity.allow_pseudo_sparse=true to "
            f"fall back to the pure-torch reference, or use a supported config "
            f"(block_size_q in {{64,128}}, chunk_size a multiple of block_size_q)."
        )

    if kernels_available(head_dim):
        # Pin the CUDA device to the input's device — the kernel launches on
        # the *current* device's stream, which under accelerate device_map
        # sharding would otherwise be the wrong GPU.
        if q.is_cuda:
            with torch.cuda.device(q.device):
                return _kernel_prefill(
                    q,
                    k,
                    v,
                    threshold=threshold,
                    block_size_q=block_size_q,
                    block_size_k=block_size_k,
                    group_k_block=group_k_block,
                    chunk_size=chunk_size,
                )
        return _kernel_prefill(
            q,
            k,
            v,
            threshold=threshold,
            block_size_q=block_size_q,
            block_size_k=block_size_k,
            group_k_block=group_k_block,
            chunk_size=chunk_size,
        )

    # Kernel unavailable (vllm_flash_attn unbuilt, or head_dim 256) -> torch reference.
    if allow_pseudo_sparse:
        from .reference import vecattention_reference_prefill

        warn_pseudo_sparse_fallback("vecattention", f"head_dim={head_dim}")
        return vecattention_reference_prefill(
            q,
            k,
            v,
            threshold=threshold,
            block_size_q=block_size_q,
            block_size_k=block_size_k,
            group_k_block=group_k_block,
        )

    raise KernelMissingError(
        "[sparse:vecattention] the real VecAttention kernel path is unavailable "
        f"(needs vllm_flash_attn.sparse_attn_func built, and head_dim in "
        f"{{64,128}}; got head_dim={head_dim}). Install the external kernel:\n"
        "  pip install vllm_flash_attn   # external compiled vLLM-FA fork\n"
        "Or run with the slow pure-torch reference by setting "
        "compression.sparsity.allow_pseudo_sparse=true in your YAML."
    )
