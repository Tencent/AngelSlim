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

"""Vendored XAttention kernel (MInference @ a4eb395, MIT) + availability gate.

``kernels_available(head_dim)`` reports whether the real fused XAttention prefill
can run for a given head_dim. XAttention's estimate is pure-Triton, but the
selected-block attention is executed by ``block_sparse_attn`` (the SAME kernel
Stem uses). ``block_sparse_attn`` nominally advertises ``head_dim <= 256``, yet
at head_dim 256 it returns NUMERICALLY WRONG results (verified: rel 1.56 / argmax
~0 vs a torch SDPA oracle, while flash-vs-SDPA at 256 is rel ~0.001 — so the
fault is the kernel, not the reference). We therefore restrict the real path to
``head_dim in {16,32,64,128}`` and route head_dim 256 (Qwen3.5) to the torch
reference, exactly like minference's vertical_slash and
flexprefill's Triton kernel.

``block_sparse_attn`` links libtorch, so ``import torch`` must run first (the
availability probe imports under that assumption); ``triton`` is also required.
Accessors are lazy so importing this package never triggers a Triton build.
"""

from __future__ import annotations

# head_dim 256 (Qwen3.5) is EXCLUDED on purpose — block_sparse_attn miscomputes
# at 256 (see module docstring), so Qwen3.5 routes to the torch reference.
_SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)


def kernels_available(head_dim: int | None = None) -> bool:
    """True if the real XAttention prefill can run here for ``head_dim``.

    Requires triton + block_sparse_attn importable AND head_dim in {16,32,64,128}.
    head_dim None => only the import check (a coarse availability probe).
    """
    try:
        import block_sparse_attn  # noqa: F401
        import torch  # noqa: F401  (block_sparse_attn links libtorch; load first)
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001  ImportError, OSError (libc10.so), etc.
        return False
    if head_dim is None:
        return True
    return head_dim in _SUPPORTED_HEAD_DIMS


def xattention_prefill(*args, **kwargs):
    """Lazy proxy to the vendored kernel's public entry (defer the Triton import
    until first real use, so merely importing the algorithm package is cheap)."""
    from .xattention import Xattention_prefill as _impl

    return _impl(*args, **kwargs)
