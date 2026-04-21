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


"""Configuration class for the Stem sparse attention module."""

from __future__ import annotations

# Supported option values.
SUPPORTED_BACKENDS = {"torch", "hpc"}
SUPPORTED_HPC_DTYPES = {"bf16", "fp8"}


class StemConfig:
    """Immutable configuration container for Stem attention.

    Args:
        attn_kwargs: Dictionary of keyword arguments forwarded to the attention
            backend. Recognised keys include:

            - ``backend``: ``"torch"`` (default) or ``"hpc"``.
            - ``hpc_dtype``: ``"bf16"`` (default) or ``"fp8"`` (only used when
              ``backend="hpc"``).
            - ``stem_alpha``, ``block_size``, ``stride``, ``chunk_size``, etc.

    Raises:
        ValueError: If *backend* or *hpc_dtype* is not in the supported set.
    """

    def __init__(self, attn_kwargs: dict | None = None) -> None:
        self.attn_kwargs: dict = dict(attn_kwargs or {})
        self.attn_kwargs.setdefault("backend", "torch")
        self.attn_kwargs.setdefault("hpc_dtype", "bf16")

        backend = self.attn_kwargs["backend"]
        hpc_dtype = self.attn_kwargs["hpc_dtype"]

        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported stem backend: {backend!r}. "
                f"Choose from {sorted(SUPPORTED_BACKENDS)}."
            )
        if hpc_dtype not in SUPPORTED_HPC_DTYPES:
            raise ValueError(
                f"Unsupported hpc_dtype: {hpc_dtype!r}. "
                f"Choose from {sorted(SUPPORTED_HPC_DTYPES)}."
            )

    def __repr__(self) -> str:
        return f"StemConfig(attn_kwargs={self.attn_kwargs!r})"
