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
