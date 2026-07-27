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

"""Distributed / model-parallel initialization helpers for the mcore backend.

Works both for single-process (TP=PP=EP=CP=1) smoke tests and for torchrun-launched
multi-rank runs that set RANK/WORLD_SIZE/LOCAL_RANK.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed


def init_distributed() -> tuple[int, int, int]:
    """Init torch.distributed from env (or a single-rank default). Returns (rank, world, local)."""
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29577")
        torch.cuda.set_device(local)
        dist.init_process_group(backend="nccl", world_size=world, rank=rank)
    return rank, world, local


def init_model_parallel(
    tp: int = 1, pp: int = 1, ep: int = 1, cp: int = 1, etp: int | None = None, seed: int = 1234
) -> tuple[int, int, int]:
    """Init torch.distributed + mcore parallel state with the given layout.

    ``etp`` (expert tensor parallel) defaults to ``tp``; set etp=1 to keep experts
    un-sharded along TP (sharded by EP only) -- used by the grouped-expert path.
    """
    rank, world, local = init_distributed()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
    )
    model_parallel_cuda_manual_seed(seed)
    return rank, world, local


def teardown() -> None:
    if ps.model_parallel_is_initialized():
        ps.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
