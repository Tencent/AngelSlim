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

"""VecAttention kernel-availability gate.

VecAttention's real selected-column attention runs on
``vllm_flash_attn.sparse_attn_func`` — an EXTERNAL optional compiled
vLLM-flash-attention fork that is ``pip install``-ed into the environment, NOT
vendored in this tree (see ``algorithms/vecattention/NOTICE``). When it is not
installed, ``import vllm_flash_attn`` fails and this returns False, so the
framework routes to the torch reference when
``allow_pseudo_sparse=True`` or hard-fails otherwise.

The fused MinP kernel pools queries into blocks of ``q_pooling_size`` ∈ {64,128}
and the sparse attention kernel supports head_dim ∈ {64,128}; head_dim 256
(Qwen3.5) is not a kernel size, so it routes to the reference — exactly like
minference / flexprefill / xattention / flashprefill at head_dim 256.
"""

from __future__ import annotations

# head_dim the vllm_flash_attn sparse kernel supports. head_dim 256 (Qwen3.5) is
# EXCLUDED → torch reference (the same routing the other block-sparse algos use).
_SUPPORTED_HEAD_DIMS = (64, 128)


def kernels_available(head_dim: int | None = None) -> bool:
    """True if the real VecAttention kernel path can run here for ``head_dim``.

    Requires ``vllm_flash_attn.sparse_attn_func`` importable AND head_dim in
    {64,128}. head_dim None => only the import check (a coarse availability probe).
    """
    try:
        import torch  # noqa: F401  (vllm_flash_attn links libtorch; load first)
        from vllm_flash_attn import sparse_attn_func  # noqa: F401
    except Exception:  # noqa: BLE001  ImportError, OSError (missing .so), etc.
        return False
    if head_dim is None:
        return True
    return head_dim in _SUPPORTED_HEAD_DIMS
