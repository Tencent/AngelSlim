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

"""vLLM-side smooth pipeline (online calibration).

Re-exports the public API used by:
* ``tools/smooth/run_vllm_smooth.py`` (Phase 1 driver)
* the patched ``vllm/tools/smooth_moe_inject.py`` (FusedMoE kernel hook)
"""

from .hooks import (
    SmoothAlphaValueHook,
    SmoothAttnHook,
    SmoothDownProjInputHook,
    get_smooth_stats,
    print_smooth_stats,
    remove_smooth_alpha_search_hooks,
    setup_smooth_alpha_search_hooks,
    setup_smooth_hooks,
)
from .moe_inject import (
    collect_fused_moe_alpha_search_values,
    collect_fused_moe_smooth_stats,
)
from .searcher_dist import SmoothAlphaSearcher

__all__ = [
    # Hook classes
    "SmoothAttnHook",
    "SmoothDownProjInputHook",
    "SmoothAlphaValueHook",
    # Setup / teardown
    "setup_smooth_hooks",
    "setup_smooth_alpha_search_hooks",
    "remove_smooth_alpha_search_hooks",
    # Stats retrieval
    "get_smooth_stats",
    "print_smooth_stats",
    # Searcher
    "SmoothAlphaSearcher",
    # FusedMoE kernel injection entry points
    "collect_fused_moe_smooth_stats",
    "collect_fused_moe_alpha_search_values",
]
