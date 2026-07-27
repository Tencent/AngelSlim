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

"""Qwen3-MoE HF -> mcore weight conversion."""

from __future__ import annotations

from typing import Dict

from torch import Tensor

from angelslim.compressor.mcore_qad.models.base import (
    attn_to_mcore,
    experts_to_mcore,
    globals_to_mcore,
)


def qwen3_to_mcore(hf: Dict[str, Tensor], cfg, meta) -> Dict[str, Tensor]:
    out: Dict[str, Tensor] = {}
    globals_to_mcore(hf, meta, out)
    for i in range(cfg.num_layers):
        p, m = f"model.layers.{i}", f"decoder.layers.{i}"
        attn_to_mcore(hf, p, m, cfg, out)
        out[f"{m}.mlp.router.weight"] = hf[f"{p}.mlp.gate.weight"]
        experts_to_mcore(hf, p, m, cfg.num_moe_experts, out)
    return out
