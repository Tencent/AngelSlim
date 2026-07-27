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

"""Minimal mcore patches for the non-TE (local) + sequence-parallel + MoE path.

1. WrappedTorchNorm: the local spec returns a plain torch.nn.RMSNorm (per-token, so
   correct on an SP-scattered sequence), but its __new__ hard-asserts against
   sequence_parallel. Our norms are frozen, so we drop that assert and tag the norm
   weight as sequence-parallel.

2. Activation recompute: mcore's full recompute uses a RE-ENTRANT checkpoint
   (CheckpointFunction). Re-entrant autograd only backprops to the *inputs* passed in,
   so (a) it errors when no input requires grad and (b) it never reaches parameters
   created inside the function. With a FROZEN backbone where only the quantizer scales
   (inside each layer) are trainable, both break it. We swap it for the NON-reentrant
   torch checkpoint (saved-tensor-hooks), which recomputes under grad and correctly
   backprops to the internal scale params -- no grad-requiring input needed. Safe here
   because we always run dropout=0 (deterministic recompute).
"""

from __future__ import annotations

import megatron.core.transformer.torch_norm as _tn
import torch

_PATCHED = False


def _nonreentrant_checkpoint(function, distribute_saved_activations, *args):
    # signature matches mcore tensor_parallel.checkpoint(function, distribute, *args).
    # distribute_saved_activations (a TP memory optimization) is dropped; correctness first.
    import torch.utils.checkpoint as tuc

    return tuc.checkpoint(function, *args, use_reentrant=False, preserve_rng_state=False)


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    def _new(
        cls,
        config,
        hidden_size,
        eps=1e-5,
        persist_layer_norm=False,
        zero_centered_gamma=False,
        normalization="LayerNorm",
    ):
        assert not config.layernorm_zero_centered_gamma
        if config.normalization == "LayerNorm":
            norm_cls = torch.nn.LayerNorm
        elif config.normalization == "RMSNorm":
            norm_cls = torch.nn.RMSNorm
        else:
            raise Exception(f"unsupported normalization {config.normalization}")
        m = norm_cls(normalized_shape=hidden_size, eps=eps)
        if getattr(config, "sequence_parallel", False):
            # per-token norm is SP-safe; tag weight so mcore SP grad handling is happy.
            m.weight.sequence_parallel = True
        return m

    _tn.WrappedTorchNorm.__new__ = staticmethod(_new)

    import os

    if os.environ.get("MCORE_QAD_NONREENTRANT", "0") == "1":
        # Optional non-reentrant checkpoint. NOTE: with a frozen backbone + the
        # input-require-grad entry point (mcore.model.enable_input_require_grads), mcore's
        # DEFAULT reentrant checkpoint trains all scales correctly AND frees recompute
        # per-layer (non-reentrant was observed to hold ~all layers' recompute live in
        # backward -> OOM). Reentrant is the default; set MCORE_QAD_NONREENTRANT=1 to override.
        import megatron.core.tensor_parallel as _tp
        import megatron.core.tensor_parallel.random as _rnd

        _rnd.checkpoint = _nonreentrant_checkpoint
        _tp.checkpoint = _nonreentrant_checkpoint

    _PATCHED = True
