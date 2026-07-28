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

"""Scheme registry + concrete schemes."""

from angelslim.compressor.mcore_qad.quant.schemes import (  # noqa: F401  (register side-effects)
    per_channel,
    per_expert,
    per_group,
    per_tensor,
    per_token,
    two_level_block,
)
from angelslim.compressor.mcore_qad.quant.schemes.base import (
    SCHEME_REGISTRY,
    HostInfo,
    ScaleScheme,
)

__all__ = ["SCHEME_REGISTRY", "HostInfo", "ScaleScheme"]
