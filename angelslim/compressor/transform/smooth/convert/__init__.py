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

"""HuggingFace-side smooth pipeline (offline weight conversion).

Re-exports the public API consumed by
``tools/smooth/convert_smooth_weights.py`` (Phase 2 driver).
"""

from .apply_funcs import (
    apply_down_proj_smooth,
    apply_down_proj_smooth_from_search,
    apply_qk_smooth,
    apply_vo_smooth,
)
from .utils import (
    DEFAULT_KEY_MAP,
    HY_V3_KEY_MAP,
    LLAMA_KEY_MAP,
    MIXTRAL_KEY_MAP,
    PREDEFINED_KEY_MAPS,
    QWEN3_MOE_KEY_MAP,
    attn_key_to_hf_prefix,
    find_first_attn_module,
    get_submodule_safe,
    maybe_materialize,
    snapshot_attn_output_before,
    snapshot_mlp_outputs_before,
    verify_attn_output_diff,
    verify_mlp_output_diff,
)

__all__ = [
    # apply
    "apply_qk_smooth",
    "apply_vo_smooth",
    "apply_down_proj_smooth",
    "apply_down_proj_smooth_from_search",
    # key maps
    "DEFAULT_KEY_MAP",
    "HY_V3_KEY_MAP",
    "LLAMA_KEY_MAP",
    "MIXTRAL_KEY_MAP",
    "QWEN3_MOE_KEY_MAP",
    "PREDEFINED_KEY_MAPS",
    # helpers
    "get_submodule_safe",
    "maybe_materialize",
    "attn_key_to_hf_prefix",
    # snapshot / verify
    "find_first_attn_module",
    "snapshot_attn_output_before",
    "snapshot_mlp_outputs_before",
    "verify_attn_output_diff",
    "verify_mlp_output_diff",
]
