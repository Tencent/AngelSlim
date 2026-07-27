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

"""Per-model adapters. Importing registers each model_type into the adapter registry."""

from angelslim.compressor.mcore_qad.models import (  # noqa: F401  (registration side-effects)
    hy_v3,
    qwen3,
)
from angelslim.compressor.mcore_qad.models.base import (
    ModelMeta,
    auto_config,
    get_adapter,
    load_hf_config,
    load_hf_into_mcore,
    register,
)

__all__ = [
    "ModelMeta",
    "load_hf_config",
    "auto_config",
    "load_hf_into_mcore",
    "get_adapter",
    "register",
]
