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

"""Configuration class for VecAttention sparse prefill."""

from __future__ import annotations


class VecAttentionConfig:
    """Configuration container for VecAttention sparse attention.

    Args:
        attn_kwargs: Dictionary of keyword arguments. Recognised keys:

            - ``threshold`` (float): MinP threshold for key column selection.
              Higher = more aggressive sparsity. Default 0.9.
            - ``block_size_q`` (int): Query pooling block size. Must be 64 or 128.
              Default 64.
            - ``block_size_k`` (int): Key local block size. Default 16.
            - ``group_k_block`` (int): Number of k-blocks processed together.
              Default 1.
            - ``chunk_size`` (int): Prefill chunk size (tokens). Must be a
              multiple of block_size_q. Default 65536.

    Raises:
        ValueError: If block_size_q is not 64 or 128, or chunk_size is not
            a multiple of block_size_q.
    """

    def __init__(self, attn_kwargs: dict | None = None) -> None:
        self.attn_kwargs: dict = dict(attn_kwargs or {})
        # Set defaults
        self.attn_kwargs.setdefault("threshold", 0.1)
        self.attn_kwargs.setdefault("block_size_q", 64)
        self.attn_kwargs.setdefault("block_size_k", 16)
        self.attn_kwargs.setdefault("group_k_block", 16)
        self.attn_kwargs.setdefault("chunk_size", 64 * 1024)

        # Validate
        block_size_q = self.attn_kwargs["block_size_q"]
        chunk_size = self.attn_kwargs["chunk_size"]

        if block_size_q not in (64, 128):
            raise ValueError(f"block_size_q must be 64 or 128, got {block_size_q}")
        if chunk_size % block_size_q != 0:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be a multiple of "
                f"block_size_q ({block_size_q})"
            )

    @property
    def threshold(self):
        return self.attn_kwargs["threshold"]

    @property
    def block_size_q(self):
        return self.attn_kwargs["block_size_q"]

    @property
    def block_size_k(self):
        return self.attn_kwargs["block_size_k"]

    @property
    def group_k_block(self):
        return self.attn_kwargs["group_k_block"]

    @property
    def chunk_size(self):
        return self.attn_kwargs["chunk_size"]

    def __repr__(self) -> str:
        return f"VecAttentionConfig(attn_kwargs={self.attn_kwargs!r})"
