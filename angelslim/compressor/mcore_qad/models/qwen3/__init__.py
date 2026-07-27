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

from angelslim.compressor.mcore_qad.models.base import register
from angelslim.compressor.mcore_qad.models.qwen3.config import qwen3_moe_config
from angelslim.compressor.mcore_qad.models.qwen3.convert import qwen3_to_mcore

register("qwen3_moe", qwen3_moe_config, qwen3_to_mcore)
