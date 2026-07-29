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

"""VecAttention sparse-attention forward methods.

``vecattention_forward`` is the forward used by the VecAttention algorithm in
the sparse framework (imported by ``prefill.py``). ``qwen_vl_attn_forward`` is a
legacy Qwen2.5-VL vision-language patched forward retained for the VLM path.
"""

from .forward import qwen_vl_attn_forward, vecattention_forward

__all__ = ["vecattention_forward", "qwen_vl_attn_forward"]
