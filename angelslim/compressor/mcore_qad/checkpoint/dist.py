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

"""Reshardable (layout-agnostic) FP-weight checkpointing via mcore dist-checkpointing.

A checkpoint saved under one (TP,PP,EP) layout loads correctly under any other -- the
basis for converting HF once and training under arbitrary parallelism. Order:
  build model -> load_dist_checkpoint (FP weights) -> quantize_mcore_model -> train.
Quantization must come AFTER loading (it adds weight parametrizations).
"""

from __future__ import annotations

import os

from megatron.core import dist_checkpointing as dc


def save_dist_checkpoint(model, ckpt_dir: str) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    dc.save(model.sharded_state_dict(), ckpt_dir)


def load_dist_checkpoint(model, ckpt_dir: str, strict: bool = True):
    sharded = model.sharded_state_dict()
    model.load_state_dict(dc.load(sharded, ckpt_dir), strict=strict)
    return model
