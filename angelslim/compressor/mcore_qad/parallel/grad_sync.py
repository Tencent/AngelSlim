"""Average scale gradients across the data-like dimensions (DP and CP).

DP and CP both partition the token workload (DP over batches, CP over the sequence),
so a scale's gradient is partial on each and must be averaged over the combined DP+CP
group. Dense scales use that group; MoE expert scales use the expert-DP group when
EP>1 (experts live on the EP axis), else the same DP+CP group. This is orthogonal to
the TP replicated-scale sync in scale_parallel.py (TP replica consistency).
"""

from __future__ import annotations

import torch
from megatron.core import parallel_state as ps


def _size(group) -> int:
    return torch.distributed.get_world_size(group) if group is not None else 1


def all_reduce_data_parallel_grads(named_params) -> None:
    """In-place average each trainable scale grad over its data-parallel (DP+CP) group."""
    dp = ps.get_data_parallel_group(with_context_parallel=True)  # DP + CP
    edp = (
        ps.get_expert_data_parallel_group()
        if ps.get_expert_model_parallel_world_size() > 1
        else dp
    )
    dp_n, edp_n = _size(dp), _size(edp)
    if dp_n <= 1 and edp_n <= 1:
        return
    for name, p in named_params:
        if p.grad is None:
            continue
        group, n = (edp, edp_n) if "experts" in name else (dp, dp_n)
        if n > 1:
            torch.distributed.all_reduce(p.grad, group=group)
            p.grad /= n
