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

"""Backend-agnostic quantization core (no mcore deps below this package)."""

from angelslim.compressor.mcore_qad.quant.formats import FORMAT_REGISTRY
from angelslim.compressor.mcore_qad.quant.policy import build_quantizer
from angelslim.compressor.mcore_qad.quant.quantizer import IdentityQuantizer, Quantizer
from angelslim.compressor.mcore_qad.quant.schemes import SCHEME_REGISTRY
from angelslim.compressor.mcore_qad.quant.sources import SOURCE_REGISTRY
from angelslim.compressor.mcore_qad.quant.spec import QuantSpec

__all__ = [
    "FORMAT_REGISTRY",
    "SCHEME_REGISTRY",
    "SOURCE_REGISTRY",
    "Quantizer",
    "IdentityQuantizer",
    "QuantSpec",
    "build_quantizer",
]
