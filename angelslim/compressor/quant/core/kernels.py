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

"""Triton kernels for Delta-Aware Quantization (DAQ)."""

from dataclasses import dataclass
from typing import Protocol, Tuple, runtime_checkable

import torch
import triton
import triton.language as tl

from angelslim.compressor.quant.core.quant_func import weight_dequant  # noqa: F401

# ---------------------------------------------------------------------------
# QuantScheme: describes static properties of a quantization format
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantScheme:
    """Describes the static properties of a quantization format.

    This is a pure-data, pickle-safe descriptor. It answers *what* format is
    used but not *how* to run the kernel — that is ``QuantKernel``'s job.
    """

    name: str
    dtype: torch.dtype
    max_val: float = None  # type: ignore[assignment]
    symmetric: bool = True

    def __post_init__(self):
        if self.max_val is None:
            if self.dtype.is_floating_point:
                val = torch.finfo(self.dtype).max
            else:
                val = torch.iinfo(self.dtype).max
            object.__setattr__(self, "max_val", float(val))


FP8_E4M3_SCHEME = QuantScheme(name="fp8_e4m3", dtype=torch.float8_e4m3fn)

FP8_E4M3_MAX = FP8_E4M3_SCHEME.max_val


# ---------------------------------------------------------------------------
# QuantKernel: protocol for kernel-level quantization operations
# ---------------------------------------------------------------------------


@runtime_checkable
class QuantKernel(Protocol):
    """Protocol that abstracts kernel-level quantization operations.

    Different quantization formats (FP8 E4M3, INT8, INT4 …) require
    fundamentally different kernel logic (implicit vs. explicit round,
    symmetric vs. asymmetric clamp, etc.).  Each format provides its own
    ``QuantKernel`` implementation so that upper layers (scale search, DAQ
    main loop) stay format-agnostic.
    """

    @property
    def scheme(self) -> QuantScheme: ...

    def compute_scale(self, x: torch.Tensor, block_size: int) -> torch.Tensor: ...

    def quantize(self, x: torch.Tensor, scale: torch.Tensor, block_size: int) -> torch.Tensor: ...

    def quantize_with_scale_compute(
        self, x: torch.Tensor, quantization_method: str, block_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def simulate_quant_dequant(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Triton kernels (parameterized via QMAX)
# ---------------------------------------------------------------------------


@triton.jit
def compute_scale_kernel(x_ptr, s_ptr, M, N, QMAX, BLOCK_SIZE: tl.constexpr):
    """Compute scale factors for each block: scale = max(|x|) / QMAX."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n_blocks = tl.cdiv(N, BLOCK_SIZE)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.max(tl.abs(x)) / QMAX
    scale = tl.maximum(scale, 1e-10)

    tl.store(s_ptr + pid_m * n_blocks + pid_n, scale)


@triton.jit
def standard_weight_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, QMAX, BLOCK_SIZE: tl.constexpr):
    """Standard block-wise quantization. Uses 1e-10 as scale lower bound."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n_blocks = tl.cdiv(N, BLOCK_SIZE)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.max(tl.abs(x)) / QMAX
    scale = tl.maximum(scale, 1e-10)

    y = x / scale
    y = y.to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n_blocks + pid_n, scale)


@triton.jit
def weight_quant_with_custom_scale_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """Block-wise quantization with pre-computed scale factors."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n_blocks = tl.cdiv(N, BLOCK_SIZE)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(s_ptr + pid_m * n_blocks + pid_n)
    y = x / scale
    y = y.to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + offs, y, mask=mask)


# ---------------------------------------------------------------------------
# FP8E4M3Kernel: QuantKernel implementation for FP8 E4M3
# ---------------------------------------------------------------------------


class FP8E4M3Kernel:
    """``QuantKernel`` implementation for the FP8 E4M3 format."""

    def __init__(self, scheme: QuantScheme = FP8_E4M3_SCHEME):
        self._scheme = scheme

    @property
    def scheme(self) -> QuantScheme:
        return self._scheme

    # -- scale computation --------------------------------------------------

    def compute_scale(self, x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
        assert x.is_contiguous() and x.dim() == 2
        M, N = x.size()
        sM = int(torch.tensor(1.0 * M / block_size).ceil().item())
        sN = int(torch.tensor(1.0 * N / block_size).ceil().item())
        scale = x.new_empty(sM, sN, dtype=torch.float32)

        def grid(meta):
            return (
                triton.cdiv(M, meta["BLOCK_SIZE"]),
                triton.cdiv(N, meta["BLOCK_SIZE"]),
            )

        compute_scale_kernel[grid](x, scale, M, N, self._scheme.max_val, BLOCK_SIZE=block_size)
        return scale

    # -- quantize with pre-computed scale -----------------------------------

    def quantize(
        self, x: torch.Tensor, scale: torch.Tensor, block_size: int = 128
    ) -> torch.Tensor:
        assert x.is_contiguous() and scale.is_contiguous() and x.dim() == 2
        M, N = x.size()
        y = torch.empty_like(x, dtype=self._scheme.dtype)

        def grid(meta):
            return (
                triton.cdiv(M, meta["BLOCK_SIZE"]),
                triton.cdiv(N, meta["BLOCK_SIZE"]),
            )

        weight_quant_with_custom_scale_kernel[grid](x, y, scale, M, N, BLOCK_SIZE=block_size)
        return y

    # -- quantize with scale computation ------------------------------------

    def quantize_with_scale_compute(
        self,
        x: torch.Tensor,
        quantization_method: str = "blockwise",
        block_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.is_contiguous() and x.dim() == 2
        assert quantization_method in (
            "blockwise",
            "per_channel",
        ), f"Unknown quantization method: {quantization_method}"

        if quantization_method == "blockwise":
            M, N = x.size()
            y = torch.empty_like(x, dtype=self._scheme.dtype)
            sM = int(torch.tensor(1.0 * M / block_size).ceil().item())
            sN = int(torch.tensor(1.0 * N / block_size).ceil().item())
            s = x.new_empty(sM, sN, dtype=torch.float32)

            def grid(meta):
                return (
                    triton.cdiv(M, meta["BLOCK_SIZE"]),
                    triton.cdiv(N, meta["BLOCK_SIZE"]),
                )

            standard_weight_quant_kernel[grid](
                x, y, s, M, N, self._scheme.max_val, BLOCK_SIZE=block_size
            )
            return y, s
        else:
            qmax = self._scheme.max_val
            abs_max = torch.abs(x).max(dim=1, keepdim=True)[0].clamp(min=1e-12)
            scale = abs_max / qmax
            quantized = x / scale
            quantized = torch.clamp(quantized, -qmax, qmax)
            return quantized.to(self._scheme.dtype), scale.to(torch.float32)

    # -- simulate quant → dequant (for scale search metrics) ----------------

    def simulate_quant_dequant(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        w_scaled = x / scale
        w_quantized = (
            torch.clamp(torch.round(w_scaled), -self._scheme.max_val, self._scheme.max_val)
            .to(self._scheme.dtype)
            .to(torch.float32)
        )
        return w_quantized * scale


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_KERNEL_REGISTRY = {
    "fp8_e4m3": FP8E4M3Kernel,
}


def create_quant_kernel(scheme: QuantScheme) -> QuantKernel:
    """Create a ``QuantKernel`` instance from a ``QuantScheme``."""
    cls = _KERNEL_REGISTRY.get(scheme.name)
    if cls is None:
        raise ValueError(
            f"No QuantKernel registered for scheme '{scheme.name}'. "
            f"Available: {list(_KERNEL_REGISTRY.keys())}"
        )
    return cls(scheme)
