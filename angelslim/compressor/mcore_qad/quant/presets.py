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

"""Backend-agnostic quantization format presets: name -> (weight_spec, act_spec).

Single source of truth for the supported formats; reused by the mcore backend, the
model builder, and tests. Add a new format here once and it's available everywhere.
Weights use a learnable scale (the trainable lever; weights frozen); activations are
dynamic (or identity for weight-only). All names mirror the compressed-tensors /
vLLM preset they export to, so a format name maps 1:1 to a deployable layout:

    nvfp4   = NVFP4    (W4A4)    | nvfp4a16 = NVFP4A16 (W4A16)
    w4a16   = W4A16    (int4)    | w8a8     = W8A8     (int8)
    fp8     = FP8      (W8A8)    | w4afp8   = W4AFP8   (int4 weight + fp8 act, Hopper-native)
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from angelslim.compressor.mcore_qad.quant.spec import QuantSpec

WeightActSpec = Tuple[QuantSpec, QuantSpec]


def _nvfp4_weight() -> QuantSpec:
    return QuantSpec(
        fmt="e2m1",
        scheme="two_level_block",
        group_size=16,
        block_scale_fmt="e4m3",
        source="learnable",
    )


def _int4_group_weight() -> QuantSpec:
    return QuantSpec(fmt="int4", scheme="per_group", group_size=128, source="learnable")


FORMATS: Dict[str, Callable[[], WeightActSpec]] = {
    "nvfp4": lambda: (
        _nvfp4_weight(),
        QuantSpec(
            fmt="e2m1",
            scheme="two_level_block",
            group_size=16,
            block_scale_fmt="e4m3",
            source="dynamic",
        ),
    ),
    "nvfp4a16": lambda: (_nvfp4_weight(), QuantSpec.identity()),
    "w4a16": lambda: (_int4_group_weight(), QuantSpec.identity()),
    "w8a8": lambda: (
        QuantSpec(fmt="int8", scheme="per_channel", source="learnable"),
        QuantSpec(fmt="int8", scheme="per_token", source="dynamic"),
    ),
    "fp8": lambda: (
        QuantSpec(fmt="e4m3", scheme="per_channel", source="learnable"),
        QuantSpec(fmt="e4m3", scheme="per_token", source="dynamic"),
    ),
    "w4afp8": lambda: (
        _int4_group_weight(),
        QuantSpec(fmt="e4m3", scheme="per_token", source="dynamic"),
    ),
}


def get_format(name: str) -> WeightActSpec:
    if name not in FORMATS:
        raise KeyError(f"unknown format {name!r}; available: {sorted(FORMATS)}")
    return FORMATS[name]()
