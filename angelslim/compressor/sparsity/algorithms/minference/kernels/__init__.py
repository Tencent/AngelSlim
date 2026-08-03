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

"""Vendored MInference kernels + availability gate.

Two kernel families, with different build requirements:
  * ``a_shape`` / ``tri_shape`` -> ``streaming_kernel.streaming_forward``:
    pure Triton, needs only ``triton`` + CUDA at runtime.
  * ``minference`` (vertical_and_slash) ->
    ``pit_sparse_flash_attention.vertical_slash_sparse_attention``: Triton PLUS
    the ``convert_vertical_slash_indexes`` CUDA extension (JIT-built on first
    use via ``_cuda_ext``).

``kernels_available(variant)`` reports whether the real kernel for a variant can
run here. The caller (forward dispatch) uses it to decide kernel vs the
pseudo-sparse / hard-fail policy. Accessors are lazy so importing
this package never triggers a Triton/nvcc build.
"""

from __future__ import annotations


def _triton_available() -> bool:
    try:
        import torch
        import triton  # noqa: F401

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def kernels_available(variant: str, head_dim: int | None = None) -> bool:
    """Can the real kernel for ``variant`` run in this environment / shape?

    a_shape / tri_shape: Triton + CUDA; the streaming kernel pads head_dim up to
    the nearest power of two in {16,32,64,128,256} and its inner kernel asserts
    membership in that set — so head_dim <= 256 works, but 257..512 pads to 512
    (which the inner kernel does NOT accept) and would AssertionError. Gate on
    <= 256 so an unsupported large head_dim falls back to the reference instead
    of crashing.
    minference: additionally needs the CUDA index extension AND the
    vertical_slash Triton kernel, which only supports head_dim in {16,32,64,128}
    (it asserts ``Lk in {16,32,64,128}``). head_dim 256 (e.g. Qwen3.5) is NOT
    supported by the real kernel -> reports False so the caller falls back to the
    pure-torch reference.
    """
    if not _triton_available():
        return False
    if variant in ("a_shape", "tri_shape"):
        return head_dim is None or head_dim <= 256
    if variant == "minference":
        # head_dim gate first (cheap): the vertical_slash kernel supports only
        # {16,32,64,128}; head_dim 256 (Qwen3.5) is declined regardless of the
        # ext so the caller falls back to the reference.
        if head_dim is not None and head_dim not in (16, 32, 64, 128):
            return False
        # Then realize the build (cached) rather than the optimistic buildable()
        # probe: CUDA_HOME being set does NOT guarantee nvcc/arch actually
        # compile. If the JIT build fails here, report False so the caller routes
        # through the allow_pseudo_sparse policy — otherwise the kernel would
        # later raise a bare RuntimeError that bypasses that fallback. The build
        # is a process singleton (first call ~1 min, then cached), so this stays
        # cheap on repeat.
        from ._cuda_ext import get_cuda_ext

        return get_cuda_ext() is not None
    return False


def get_streaming_forward():
    """Lazy accessor for the a_shape/tri_shape Triton kernel."""
    from .streaming_kernel import streaming_forward

    return streaming_forward


def get_vertical_slash_attention():
    """Lazy accessor for the vertical_and_slash kernel (triggers CUDA JIT build
    on first real call, not on import)."""
    from .pit_sparse_flash_attention import vertical_slash_sparse_attention

    return vertical_slash_sparse_attention
