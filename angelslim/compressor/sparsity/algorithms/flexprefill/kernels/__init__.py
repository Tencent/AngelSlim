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

"""Vendored FlexPrefill Triton kernel (MInference @ a4eb395, MIT) + availability.

``kernels_available(head_dim)`` reports whether the real fused FlexPrefill kernel
can run for a given head_dim. The upstream Triton block-wise attention asserts
``head_dim in {16, 32, 64, 128}`` (it builds tl.make_block_ptr tiles of that
width), so head_dim 256 (Qwen3.5) is NOT supported and the FlexPrefill algorithm
routes it to the torch reference, exactly like minference's
vertical_slash kernel. flash_attn (the short-seq path) + triton are hard deps;
``import``-ing this module raises a clear error if they are missing.
"""

from __future__ import annotations

_SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)


def kernels_available(head_dim: int | None = None) -> bool:
    """True if the real FlexPrefill Triton kernel can run here for ``head_dim``.

    Requires triton + flash_attn importable AND head_dim in {16,32,64,128}.
    head_dim None => only the import check (used for a coarse availability probe).
    """
    try:
        import flash_attn  # noqa: F401
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    if head_dim is None:
        return True
    return head_dim in _SUPPORTED_HEAD_DIMS


def flex_prefill_attention(*args, **kwargs):
    """Lazy proxy to the vendored kernel's public entry (defer the Triton import
    until first real use, so merely importing the algorithm package is cheap)."""
    from .flex_prefill import flex_prefill_attention as _impl

    return _impl(*args, **kwargs)
