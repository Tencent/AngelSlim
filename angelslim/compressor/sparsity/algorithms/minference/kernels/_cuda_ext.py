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

"""JIT loader for the vendored MInference ``convert_vertical_slash_indexes``
CUDA extension (``csrc/kernels.cpp`` + ``csrc/vertical_slash_index.cu``, MIT).

Upstream ships this as an AOT-compiled ``minference.cuda`` extension built at
``pip install`` time. AngelSlim has no compiled artifact in the package, so we
JIT-compile on first use via ``torch.utils.cpp_extension.load`` and cache the
module as a process singleton. The first call costs ~1 min (nvcc compile);
subsequent calls (and subsequent processes) hit Torch's on-disk extension cache
(``~/.cache/torch_extensions``) and load instantly.

This is the *runtime* build path. AOT packaging (a ``setup.py``
CUDAExtension + CI build lane + wheels) is a future step; when that lands, this
loader prefers the installed ``minference.cuda`` / ``angelslim._minference_cuda``
module and falls back to JIT only if absent.
"""

from __future__ import annotations

import os
import threading

_LOCK = threading.Lock()
_EXT = None  # cached compiled module (process singleton)
_BUILD_ERROR = None  # remember a failed build so we don't retry every call

_CSRC_DIR = os.path.join(os.path.dirname(__file__), "csrc")
_SOURCES = [
    os.path.join(_CSRC_DIR, "kernels.cpp"),
    os.path.join(_CSRC_DIR, "vertical_slash_index.cu"),
]


def cuda_ext_buildable() -> bool:
    """Best-effort: can we plausibly build/load the extension here?

    Requires CUDA available + a CUDA_HOME (nvcc). Does NOT trigger a build —
    use :func:`get_cuda_ext` for that. Cheap enough to call from a gate.
    """
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        return bool(torch.cuda.is_available()) and CUDA_HOME is not None
    except Exception:
        return False


def get_cuda_ext():
    """Return the compiled extension module, building it on first call.

    Returns ``None`` if the build is impossible/failed (the caller then routes
    through the pseudo-sparse / hard-fail policy). The failure is
    cached so we don't pay a doomed nvcc invocation on every prefill.
    """
    global _EXT, _BUILD_ERROR
    if _EXT is not None:
        return _EXT
    if _BUILD_ERROR is not None:
        return None
    with _LOCK:
        if _EXT is not None:
            return _EXT
        if _BUILD_ERROR is not None:
            return None
        # Prefer an AOT-installed extension if one has been shipped.
        for mod_name in ("angelslim._minference_cuda", "minference.cuda"):
            try:
                import importlib

                _EXT = importlib.import_module(mod_name)
                return _EXT
            except Exception:
                pass
        # JIT build.
        try:
            from torch.utils.cpp_extension import load

            _EXT = load(
                name="angelslim_minference_cuda",
                sources=_SOURCES,
                verbose=False,
            )
            return _EXT
        except Exception as e:  # nvcc missing, arch mismatch, etc.
            _BUILD_ERROR = e
            return None


def cuda_ext_build_error():
    """Return the cached build exception (or None)."""
    return _BUILD_ERROR
