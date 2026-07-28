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

"""Keep mcore QAD fake quantization aligned with vLLM-loadable formats.

We train in fake-quant, but the learned scales must map cleanly onto what vLLM
actually supports, or the artifact is un-deployable. This module is the single
source of truth for "what granularity/dtype is allowed for which format", plus a
validator that build_quantizer calls to reject unsupported specs early.

------------------------------------------------------------------------------
vLLM-supported formats (verified against compressed-tensors / modelopt / quark):

NVFP4  (W4A4):
  weight:
    - element  : E2M1 (4-bit), packed uint8 [out, in//2]
    - block    : group_size = 16 ALONG INPUT(K), scale dtype E4M3
                 -> weight_scale [out, in//16]
    - global   : per-tensor FP32 (weight_scale_2)        # per-EXPERT for MoE
  activation:
    - block    : group_size = 16 along K, E4M3, DYNAMIC (per-token, runtime)
    - global   : per-tensor FP32 (input_scale), static/calibrated  # per-EXPERT for MoE
    - NO per-channel activation scale slot at runtime.

MXFP4 (experimental in vLLM):
    - block = 32, scale = E8M0 (power-of-two), single level. (Different from NVFP4!)

INT8 (W8A8, compressed-tensors):
    - weight : per-channel (out) symmetric, INT8
    - act    : per-tensor OR per-token dynamic, symmetric

INT4 (W4A16, GPTQ/AWQ-style, weight-only):
    - weight : group_size 128 (or 64) along in, symmetric/asymmetric, INT4
    - act    : high precision (no activation quant)

------------------------------------------------------------------------------
NOT supported by vLLM (reject or warn):
  * NVFP4 with group_size != 16, or block scale dtype != E4M3.
  * NVFP4 weight as per-tensor / per-out-channel only (must be block-16).
  * NVFP4 activation with a static per-channel scale (no runtime slot).
  * Asymmetric NVFP4 / learnable zero-points (NVFP4 is symmetric; we train scale only).
  * INT4 W4A4 (use NVFP4 for 4-bit activations instead).
  * Arbitrary group sizes for INT4 other than {64,128}.
"""

from __future__ import annotations

from angelslim.compressor.mcore_qad.quant.spec import QuantSpec

#: fixed NVFP4 constants.
NVFP4_GROUP_SIZE = 16
NVFP4_BLOCK_SCALE_FMT = "e4m3"


class UnsupportedQuantFormat(ValueError):
    """Raised when a QuantSpec cannot be represented in a vLLM-loadable checkpoint."""


def validate_spec(spec: QuantSpec, role: str) -> None:
    """Reject specs that vLLM cannot load. Called from build_quantizer.

    Implemented rules (extend as more formats are supported):
      * NVFP4 (fmt e2m1): scheme must be two_level_block (optionally per_expert);
        group_size == 16; block_scale_fmt == "e4m3".
      * MXFP4 not yet wired here.
      * INT8 / INT4: light checks; other combos pass through for now.
    """
    if spec.is_identity():
        return
    if spec.fmt == "e2m1":
        if spec.scheme != "two_level_block":
            raise UnsupportedQuantFormat(
                f"[{role}] NVFP4 (e2m1) requires scheme 'two_level_block' "
                f"(got {spec.scheme!r}); per-tensor/per-channel NVFP4 is not vLLM-loadable."
            )
        gs = spec.group_size or NVFP4_GROUP_SIZE
        if gs != NVFP4_GROUP_SIZE:
            raise UnsupportedQuantFormat(
                f"[{role}] NVFP4 group_size must be {NVFP4_GROUP_SIZE} (got {gs})."
            )
        bsf = spec.block_scale_fmt or NVFP4_BLOCK_SCALE_FMT
        if bsf != NVFP4_BLOCK_SCALE_FMT:
            raise UnsupportedQuantFormat(
                f"[{role}] NVFP4 block scale dtype must be {NVFP4_BLOCK_SCALE_FMT} (got {bsf})."
            )
        return
    if spec.fmt == "int4" and spec.scheme == "per_group":
        if (spec.group_size or 128) not in (64, 128):
            raise UnsupportedQuantFormat(
                f"[{role}] INT4 group_size must be 64 or 128 (got {spec.group_size})."
            )
    # int8 and other combinations: allowed for now.
    return
