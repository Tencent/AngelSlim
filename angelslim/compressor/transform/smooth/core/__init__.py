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

"""Backend-agnostic smooth core: pure tensor algorithms shared by the
vLLM online calibration path and the HuggingFace offline conversion path.
"""

from .stats_search import (
    SmoothStats,
    load_alpha_search_results,
    load_smooth_stats,
    save_alpha_search_results,
    save_smooth_stats,
    smooth_alpha_search_layer,
)
from .tensor_math import (
    inplace_div_fp32,
    inplace_mul_fp32,
    per_channel_absmax,
    smooth_default,
    smooth_per_tensor_act_first,
    smooth_qdq_act,
    smooth_qdq_weight,
)

__all__ = [
    # formulas
    "smooth_default",
    "smooth_per_tensor_act_first",
    # qdq
    "smooth_qdq_act",
    "smooth_qdq_weight",
    # searcher
    "smooth_alpha_search_layer",
    # smooth_stats
    "SmoothStats",
    "load_smooth_stats",
    "save_smooth_stats",
    "load_alpha_search_results",
    "save_alpha_search_results",
    # tensor_ops
    "per_channel_absmax",
    "inplace_mul_fp32",
    "inplace_div_fp32",
]
