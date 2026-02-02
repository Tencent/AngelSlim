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

"""
Platform detection and backend selection for AngelSlim.

This module provides utilities for detecting the runtime environment
and selecting appropriate backends (Triton vs PyTorch) based on
platform capabilities.

Environment Variables:
    ANGELSLIM_BACKEND: Force backend selection ("triton" or "pytorch")
    ANGELSLIM_TORCH_COMPILE: Enable/disable torch.compile ("0" or "1")
"""

import os
import sys
from enum import Enum
from functools import lru_cache
from typing import Optional

import torch


class Platform(Enum):
    """Supported platforms."""

    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"
    UNKNOWN = "unknown"


class Backend(Enum):
    """Available computation backends."""

    TRITON = "triton"
    PYTORCH = "pytorch"


@lru_cache(maxsize=1)
def get_platform() -> Platform:
    """Detect the current platform."""
    if sys.platform.startswith("linux"):
        return Platform.LINUX
    elif sys.platform == "win32":
        return Platform.WINDOWS
    elif sys.platform == "darwin":
        return Platform.MACOS
    return Platform.UNKNOWN


@lru_cache(maxsize=1)
def is_triton_available() -> bool:
    """
    Check if Triton is available and functional.

    Returns:
        bool: True if Triton can be used, False otherwise.
    """
    # Check environment variable override
    env_backend = os.environ.get("ANGELSLIM_BACKEND", "").lower()
    if env_backend == "pytorch":
        return False
    if env_backend == "triton":
        # User explicitly requested Triton, try to use it
        try:
            import triton

            if not torch.cuda.is_available():
                raise RuntimeError("ANGELSLIM_BACKEND=triton but CUDA is not available")
            return True
        except ImportError:
            raise RuntimeError("ANGELSLIM_BACKEND=triton but triton is not installed")

    # Auto-detection: check CUDA availability first
    if not torch.cuda.is_available():
        return False

    # Try to import triton
    try:
        import triton

        # Test if JIT compilation works
        return _test_triton_jit()
    except ImportError:
        return False
    except Exception:
        return False


def _test_triton_jit() -> bool:
    """
    Test if Triton JIT compilation actually works.

    This is needed because triton-windows may import but fail at JIT time.
    """
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _test_kernel(x_ptr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + offs)
            tl.store(x_ptr + offs, x + 1.0)

        # Try to compile and run the kernel
        x = torch.zeros(128, device="cuda", dtype=torch.float32)
        _test_kernel[(1,)](x, BLOCK=128)
        torch.cuda.synchronize()

        # Verify the kernel ran correctly
        return torch.allclose(x, torch.ones(128, device="cuda", dtype=torch.float32))
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_default_backend() -> Backend:
    """
    Get the default computation backend for the current environment.

    Priority:
    1. ANGELSLIM_BACKEND environment variable
    2. Triton if available and functional
    3. PyTorch fallback

    Returns:
        Backend: The selected backend.
    """
    if is_triton_available():
        return Backend.TRITON
    return Backend.PYTORCH


@lru_cache(maxsize=1)
def is_torch_compile_supported() -> bool:
    """
    Check if torch.compile is supported and should be enabled.

    Returns:
        bool: True if torch.compile should be used.
    """
    # Check environment variable override
    env_compile = os.environ.get("ANGELSLIM_TORCH_COMPILE", "").lower()
    if env_compile == "0" or env_compile == "false":
        return False
    if env_compile == "1" or env_compile == "true":
        return True

    # Windows: torch.compile has issues with dynamo
    if get_platform() == Platform.WINDOWS:
        return False

    # Check PyTorch version (torch.compile requires 2.0+)
    try:
        version_parts = torch.__version__.split(".")[:2]
        major = int(version_parts[0])
        if major < 2:
            return False
    except Exception:
        return False

    return True


def use_triton() -> bool:
    """Check if Triton backend should be used."""
    return get_default_backend() == Backend.TRITON


def use_pytorch() -> bool:
    """Check if PyTorch fallback should be used."""
    return get_default_backend() == Backend.PYTORCH


def get_backend_info() -> dict:
    """
    Get detailed information about the current backend configuration.

    Returns:
        dict: Backend information including platform, backend, and capabilities.
    """
    return {
        "platform": get_platform().value,
        "backend": get_default_backend().value,
        "triton_available": is_triton_available(),
        "torch_compile_supported": is_torch_compile_supported(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "env_backend": os.environ.get("ANGELSLIM_BACKEND", "auto"),
        "env_torch_compile": os.environ.get("ANGELSLIM_TORCH_COMPILE", "auto"),
    }
