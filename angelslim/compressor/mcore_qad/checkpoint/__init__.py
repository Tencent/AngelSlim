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

"""Checkpointing: reshardable FP weights (dist) + trainable quantizer scales."""

from angelslim.compressor.mcore_qad.checkpoint.scales import (
    load_initial_scales,
    save_scales,
)

__all__ = [
    "save_dist_checkpoint",
    "load_dist_checkpoint",
    "save_scales",
    "load_initial_scales",
]


def __getattr__(name):
    """Keep Megatron-Core optional until distributed checkpoint I/O is used."""
    if name in ("save_dist_checkpoint", "load_dist_checkpoint"):
        from angelslim.compressor.mcore_qad.checkpoint.dist import (
            load_dist_checkpoint,
            save_dist_checkpoint,
        )

        return {
            "save_dist_checkpoint": save_dist_checkpoint,
            "load_dist_checkpoint": load_dist_checkpoint,
        }[name]
    raise AttributeError(name)
