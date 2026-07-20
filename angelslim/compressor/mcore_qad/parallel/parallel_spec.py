"""How a scale tensor must live under mcore parallelism.

A ScaleScheme, knowing its own axis/granularity AND the host module's parallel
layout, emits a ``ParallelSpec`` describing whether the scale param is:
  * SHARDED   : aligned with a TP/EP shard axis -> each rank holds a slice,
                gradients are local, no sync needed (cheapest + cleanest).
  * REPLICATED: a per-tensor / replicated-axis scale -> every rank holds a full
                copy; gradients are PARTIAL and MUST be all-reduced over the
                listed groups, or ranks silently diverge.
See the design discussion (TP/EP/CP scale correctness matrix) for which case
each (granularity x host) combination falls into.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ScalePlacement(Enum):
    SHARDED = "sharded"  # mcore "tensor-model-parallel parameter"
    REPLICATED = "replicated"  # mcore "duplicated" param (maybe + grad allreduce)
    EXPERT = "expert"  # mcore "expert parameter" (expert-DP grad reduction)


@dataclass
class ParallelSpec:
    placement: ScalePlacement
    #: for SHARDED: which dim of the scale tensor is partitioned across TP.
    partition_dim: Optional[int] = None
    partition_stride: int = 1
    #: for REPLICATED: process groups whose grads must be summed to stay in sync.
    #: entries are symbolic names resolved against mcore parallel state,
    #: e.g. {"tp", "cp"}.
    grad_reduce_groups: List[str] = field(default_factory=list)

    @classmethod
    def sharded(cls, partition_dim: int, stride: int = 1) -> "ParallelSpec":
        return cls(ScalePlacement.SHARDED, partition_dim=partition_dim, partition_stride=stride)

    @classmethod
    def replicated(cls, grad_reduce_groups: Optional[List[str]] = None) -> "ParallelSpec":
        return cls(ScalePlacement.REPLICATED, grad_reduce_groups=grad_reduce_groups or [])

    @classmethod
    def expert(cls) -> "ParallelSpec":
        return cls(ScalePlacement.EXPERT)
