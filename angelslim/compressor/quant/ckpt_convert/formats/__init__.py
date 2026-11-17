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

from .fp8_block import FP8Block
from .int4 import INT4Group
from .int8 import INT8Channel
from .mxfp4 import MXFP4
from .mxfp8 import MXFP8
from .nvfp4 import NVFP4

FORMATS = {f.name: f for f in (FP8Block, NVFP4, MXFP8, MXFP4, INT8Channel, INT4Group)}

__all__ = ["FORMATS", "FP8Block", "INT4Group", "INT8Channel", "MXFP4", "MXFP8", "NVFP4"]
