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

"""SmoothQuant transform module.

Three sub-packages share a common :mod:`.core` algorithm layer:

* :mod:`.core`     — backend-agnostic tensor primitives (formulas, QDQ,
  RoPE-aware pairing, GQA expansion, alpha-search inner loop, smooth-stats
  serialisation).  Imported by both the vLLM and convert pipelines.
* :mod:`.vllm`     — online stat collection on a live vLLM model: hook
  classes, ``setup_smooth_hooks`` / ``get_smooth_stats``, the TP-aware
  ``SmoothAlphaSearcher``, and FusedMoE kernel-injection entry points.
* :mod:`.convert`  — offline weight conversion on a HuggingFace model:
  ``apply_qk_smooth`` / ``apply_vo_smooth`` / ``apply_down_proj_smooth``
  (+ alpha-search variant), plus snapshot/verify utilities.

Top-level :mod:`.config` holds the dataclasses that travel with both
pipelines (:class:`SmoothAlphaSearchConfig`).
"""

from .config import SmoothAlphaSearchConfig

__all__ = [
    "SmoothAlphaSearchConfig",
]
