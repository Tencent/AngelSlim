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


"""Backend dispatcher: routes Stem prefill to the correct implementation."""

from __future__ import annotations

import torch

from .torch_impl import stem_forward_torch


def stem_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    prefill_kwargs: dict,
) -> torch.Tensor:
    """Dispatch a Stem prefill call to the appropriate backend.

    Args:
        query_states: Query tensor of shape ``(B, H_q, L_q, D)``.
        key_states: Key tensor of shape ``(B, H_kv, L_kv, D)``.
        value_states: Value tensor of shape ``(B, H_kv, L_kv, D)``.
        prefill_kwargs: Must contain ``"attn_forward_config"`` (with a
            ``"backend"`` key) and ``"layer_idx"``.

    Returns:
        Attention output tensor of shape ``(B, H_q, L_q, D)``.

    Raises:
        ValueError: If the requested backend is not ``"torch"`` or ``"hpc"``.
    """
    config = prefill_kwargs["attn_forward_config"]
    backend = config.get("backend", "torch")

    if backend == "torch":
        return stem_forward_torch(query_states, key_states, value_states, prefill_kwargs)

    if backend == "hpc":
        # Lazy import to avoid hard dependency on the ``hpc`` C++ extension
        # when only the pure-torch path is needed.
        from .hpc_impl import stem_forward_hpc

        return stem_forward_hpc(query_states, key_states, value_states, prefill_kwargs)

    raise ValueError(f"Unknown stem backend: {backend!r}")
