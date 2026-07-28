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

"""FlashPrefill kernel-availability gate (clean-room; no vendored kernel file).

Unlike minference / flexprefill / xattention, FlashPrefill vendors NO upstream
kernel file — the upstream repo (qhfan/FlashPrefill) is unlicensed, so we
implement the method clean-room (``blockmask.py`` + ``reference.py``) and run the
*selected-block* attention (stage 3) through ``block_sparse_attn``, the SAME
BSD-3-Clause-licensed kernel Stem and xattention already use. There is therefore no
``kernels/`` subpackage; this module just reports whether that real path can run.

The block-selection (stages 1+2) is plain torch and always available; the real
*kernel* path additionally needs ``triton`` + ``block_sparse_attn`` and a
head_dim in {16,32,64,128} (block_sparse_attn miscomputes at head_dim 256 — the
same fault verified for xattention — so Qwen3.5 routes to the torch reference).
"""

from __future__ import annotations

# head_dim 256 (Qwen3.5) is EXCLUDED — block_sparse_attn miscomputes at 256
# (verified vs a torch SDPA oracle in the xattention bring-up), so Qwen3.5 routes
# to the torch reference.
_SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)


def kernels_available(head_dim: int | None = None) -> bool:
    """True if the real FlashPrefill kernel path can run here for ``head_dim``.

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
