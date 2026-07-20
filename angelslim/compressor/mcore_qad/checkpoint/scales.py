"""Quantizer scales -- the only trainable state (weights are frozen).

The trained scales are the ONLY thing worth persisting (weights stay the frozen FP
dist-checkpoint), so we never snapshot full model/optimizer state -- just the tiny scale
tensors, optionally every N steps. We don't support resume-from-scales (no checkpoint
training by design); to seed scales use load_initial_scales below.

* save_scales        : per-rank file of the trainable scales (optionally step-tagged).
* load_initial_scales: seed scales from an external/PTQ file, matched by name
  (the design decision is to read precomputed scales rather than run PTQ in-framework).

External scale file: a torch-saved dict {qualified_name: tensor}, where qualified_name
matches a model buffer/parameter, e.g.
  "...weight_q.scheme.block_store.ref"   (NVFP4 weight reference scale)
  "...act_q.scheme.store.scale"          (calibrated activation scale)
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def save_scales(model: nn.Module, out_dir: str, tag: str | None = None) -> str:
    """Save the trainable scales for this rank. ``tag`` (e.g. 'step100') suffixes the
    filename for periodic snapshots; default is the single final 'scales_rankR.pt'."""
    os.makedirs(out_dir, exist_ok=True)
    scales = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    suffix = f"_{tag}" if tag else ""
    path = os.path.join(out_dir, f"scales_rank{_rank()}{suffix}.pt")
    torch.save(scales, path)
    return path


@torch.no_grad()
def load_initial_scales(model: nn.Module, scales) -> int:
    """Seed quantizer scales from an external dict/path. Returns #tensors loaded.

    Matches by qualified name against model buffers and parameters (so it seeds
    CalibratedScale.scale, LearnableScale.ref/alpha, etc.). Missing keys are ignored.
    """
    if isinstance(scales, str):
        scales = torch.load(scales, map_location="cpu")
    own = dict(model.named_buffers())
    own.update(dict(model.named_parameters()))
    n = 0
    for k, v in scales.items():
        if k in own and own[k].shape == v.shape:
            own[k].data.copy_(v.to(own[k].device, own[k].dtype))
            n += 1
    return n
