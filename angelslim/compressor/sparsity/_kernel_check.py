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

"""Kernel-availability checks with hard-fail + opt-in pseudo-sparse.

A missing kernel hard-fails by default with an
actionable error that names ``allow_pseudo_sparse=true``. Users opt into the
slow pseudo-sparse fallback explicitly.
"""

from __future__ import annotations

import importlib
import warnings


class KernelMissingError(RuntimeError):
    """A required sparse kernel is not installed and pseudo-sparse was not opted in."""


def kernel_available(import_name: str) -> bool:
    """Return True if ``import_name`` can be imported (torch must be loaded).

    Several kernels (e.g. ``block_sparse_attn``) link against libtorch, so the
    caller must ensure ``import torch`` has run first.
    """
    try:
        importlib.import_module(import_name)
        return True
    except Exception:  # ImportError, OSError (libc10.so), etc.
        return False


def warn_pseudo_sparse_fallback(algo_name: str, reason: str = "") -> None:
    """Emit the single canonical 'falling back to pseudo-sparse' warning.

    The per-algorithm prefill dispatchers inline the kernel-availability check
    (their checks differ in shape — triton+ext vs an importable kernel), so they
    cannot all route through :func:`require_kernel`. They call THIS instead right
    before taking the pseudo-sparse reference path, so the opt-in slow fallback is
    never silent — a user who set ``allow_pseudo_sparse=true`` and is unknowingly
    running the ~100x-slower pure-torch path gets one clear warning.
    """
    tail = f" ({reason})" if reason else ""
    warnings.warn(
        f"[sparse:{algo_name}] real kernel unavailable{tail}; falling back to "
        f"the pure-torch pseudo-sparse reference (CORRECT but much slower — for "
        f"accuracy debugging, not production). Set allow_pseudo_sparse=false to "
        f"hard-fail instead.",
        UserWarning,
        stacklevel=2,
    )


def require_kernel(
    import_name: str,
    install_hint: str,
    *,
    algo_name: str,
    allow_pseudo_sparse: bool,
) -> bool:
    """Gate a kernel import behind the hard-fail / pseudo-sparse policy.

    This is the canonical single statement of the kernel-missing →
    hard-fail unless ``allow_pseudo_sparse`` policy. The per-algorithm prefill
    dispatchers INLINE the availability check (their checks differ in shape —
    triton+CUDA-ext vs a single importable kernel), but they emit the same
    fallback warning via :func:`warn_pseudo_sparse_fallback`, so the slow path is
    never silent. New algorithm backends with a simple importable-kernel check
    should call ``require_kernel`` at setup instead of re-implementing the policy.

    Returns
    -------
    bool
        True  -> the real kernel is available; use it.
        False -> kernel missing but ``allow_pseudo_sparse`` is set; a single
                 ``UserWarning`` was emitted and the caller should use the
                 pseudo-sparse path.

    Raises
    ------
    KernelMissingError
        Kernel missing and ``allow_pseudo_sparse`` is False (the default).
    """
    if kernel_available(import_name):
        return True

    if allow_pseudo_sparse:
        warnings.warn(
            f"[sparse:{algo_name}] kernel {import_name!r} not available; "
            f"falling back to pseudo-sparse (slow; for accuracy debugging "
            f"only). Pass allow_pseudo_sparse=false to hard-fail instead.",
            UserWarning,
            stacklevel=2,
        )
        return False

    raise KernelMissingError(
        f"[sparse:{algo_name}] required kernel {import_name!r} is not "
        f"installed.\nInstall hint:\n  {install_hint}\n"
        f"To run anyway with the slow pseudo-sparse fallback, set "
        f"compression.sparsity.allow_pseudo_sparse=true in your YAML "
        f"(or pass allow_pseudo_sparse=True to the algorithm)."
    )
