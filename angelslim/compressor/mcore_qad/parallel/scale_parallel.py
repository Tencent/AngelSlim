"""Attach mcore parallel attributes / grad-sync to scale parameters.

Given a Quantizer and the ParallelSpec its scheme produced:
  * SHARDED   -> tag scale params as tensor-model-parallel (so dist-checkpoint
                 reshards them and mcore treats them as partitioned).
  * REPLICATED-> register a backward hook that all-reduces the scale gradient over
                 the listed process groups (TP and/or CP), so replicas stay in sync
                 (otherwise per-tensor / replicated-axis scales silently diverge).
  * EXPERT    -> handled by EP module placement; TP sharding (if any) still applies
                 via the column/row spec, so nothing special here.
"""

from __future__ import annotations

import torch

from angelslim.compressor.mcore_qad.parallel.parallel_spec import (
    ParallelSpec,
    ScalePlacement,
)


def _resolve_dim(dim: int, ndim: int) -> int:
    return dim % ndim if ndim > 0 else 0


def _register_grad_allreduce(param: torch.nn.Parameter, group_names) -> None:
    from megatron.core import parallel_state as ps

    def _groups():
        out = []
        for g in group_names:
            if g == "tp":
                out.append(ps.get_tensor_model_parallel_group())
            elif g == "cp":
                out.append(ps.get_context_parallel_group())
        return out

    def hook(grad):
        for grp in _groups():
            if grp is not None and torch.distributed.get_world_size(grp) > 1:
                torch.distributed.all_reduce(grad, group=grp)
        return grad

    param.register_hook(hook)


def configure_scale_parallelism(quantizer: torch.nn.Module, pspec: ParallelSpec) -> None:
    """Tag/sync every trainable scale Parameter inside ``quantizer`` per ``pspec``."""
    from megatron.core.tensor_parallel.layers import (
        set_tensor_model_parallel_attributes,
    )

    for p in quantizer.parameters():
        if not p.requires_grad:
            continue
        if pspec.placement is ScalePlacement.SHARDED:
            dim = _resolve_dim(pspec.partition_dim or 0, p.dim())
            set_tensor_model_parallel_attributes(p, True, dim, pspec.partition_stride)
        elif pspec.placement is ScalePlacement.REPLICATED:
            if pspec.grad_reduce_groups:
                _register_grad_allreduce(p, pspec.grad_reduce_groups)
        # EXPERT: TP sharding handled via the column/row spec already.
