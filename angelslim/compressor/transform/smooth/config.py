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

"""Configuration dataclasses shared by the vLLM (online) and convert
(offline) smooth pipelines.

These are *pure data containers* — keep the module free of any heavy
imports so it can be loaded from CLI scripts without pulling in torch
distributed / vLLM machinery.
"""

from dataclasses import dataclass

__all__ = [
    "SmoothAlphaSearchConfig",
]


@dataclass
class SmoothAlphaSearchConfig:
    """Configuration for smooth alpha grid search."""

    alpha_min: float = 0.3
    alpha_max: float = 1.0
    alpha_steps: int = 8  # [0.3, 0.4, ..., 1.0]
    act_quant_method: str = "per_token"  # "per_tensor" | "per_token"
    act_quant_type: str = "int8"  # "int8" | "fp8"
    weight_quant_method: str = (
        "per_channel"  # "per_tensor" | "per_channel" | "per_group" | "per_block"
    )
    weight_quant_type: str = "int8"  # "int8" | "int4" | "fp8"
    weight_quant_bits: int = 8
    weight_group_size: int = 128  # per_group, -1 = per_channel
    block_size: int = 128  # per_block fp8
    use_ema_for_absmax: bool = False
    smooth_search_mode: str = "default"  # "default" | "per-tensor-act-first"
    act_mul_min: float = 0.1  # per-tensor-act-first: multiplier range min
    act_mul_max: float = 1.0  # per-tensor-act-first: multiplier range max
    smooth_min: float = 1e-6  # per-tensor-act-first: smooth clamp lower bound
    smooth_max: float = 1e6  # per-tensor-act-first: smooth clamp upper bound
