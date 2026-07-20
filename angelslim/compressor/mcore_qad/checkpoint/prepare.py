"""Coordinate automatic HF-to-mcore checkpoint conversion before training."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def checkpoint_is_ready(path: str | Path) -> bool:
    """Return whether mcore wrote its final, valid checkpoint metadata."""
    metadata_path = Path(path) / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        with metadata_path.open() as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError):
        return False
    required_fields = (
        "sharded_backend",
        "sharded_backend_version",
        "common_backend",
        "common_backend_version",
    )
    return isinstance(metadata, dict) and all(
        metadata.get(field) is not None for field in required_fields
    )


def _single_process_environment() -> dict[str, str]:
    """Build an isolated distributed environment for the converter subprocess."""
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("TORCHELASTIC_"):
            environment.pop(key)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]

    project_root = str(Path(__file__).resolve().parents[4])
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else project_root
    )
    environment.update(
        {
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
            "GROUP_RANK": "0",
            "ROLE_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    return environment


def _conversion_group() -> tuple[int, int, int, bool]:
    """Initialize the torchrun world so only global rank zero performs conversion."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size == 1:
        return 0, 1, local_rank, False
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed MCoreQAD checkpoint conversion requires CUDA.")

    torch.cuda.set_device(local_rank)
    created = False
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(hours=12))
        created = True
    return dist.get_rank(), dist.get_world_size(), local_rank, created


def _destroy_group_if_owned(created: bool) -> None:
    if created and dist.is_initialized():
        dist.destroy_process_group()


def _distributed_readiness(
    checkpoint_path: str,
    world_size: int,
    local_rank: int,
) -> tuple[bool, bool]:
    """Return (all_ranks_ready, any_rank_ready) in identical collective order."""
    local_ready = int(checkpoint_is_ready(checkpoint_path))
    if world_size == 1:
        ready = bool(local_ready)
        return ready, ready

    device = torch.device("cuda", local_rank)
    all_ready = torch.tensor([local_ready], dtype=torch.int32, device=device)
    any_ready = all_ready.clone()
    dist.all_reduce(all_ready, op=dist.ReduceOp.MIN)
    dist.all_reduce(any_ready, op=dist.ReduceOp.MAX)
    return bool(all_ready.item()), bool(any_ready.item())


def _converter_command(config: Any) -> list[str]:
    mcore_config = config.compression_config.MCoreQAD
    command = [
        sys.executable,
        "-m",
        "angelslim.compressor.mcore_qad.tools.hf_to_megatron",
        "--hf",
        config.model_config.model_path,
        "--out",
        mcore_config.checkpoint_path,
    ]
    if mcore_config.checkpoint_conversion_cpu:
        command.append("--cpu")
    return command


def ensure_mcore_checkpoint(config: Any) -> bool:
    """Create a missing mcore checkpoint once, coordinated across torchrun ranks.

    Returns ``True`` when this launch performed conversion and ``False`` when an
    existing checkpoint was reused.
    """
    checkpoint_path = config.compression_config.MCoreQAD.checkpoint_path
    configured_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if configured_world_size == 1 and checkpoint_is_ready(checkpoint_path):
        return False

    rank, world_size, local_rank, created_group = _conversion_group()
    all_ready, any_ready = _distributed_readiness(
        checkpoint_path,
        world_size,
        local_rank,
    )
    if all_ready:
        return False
    if any_ready:
        _destroy_group_if_owned(created_group)
        raise RuntimeError(
            "MCoreQAD checkpoint is visible on only some ranks. Use a shared "
            "checkpoint path for multi-node runs."
        )

    conversion_error = None
    if rank == 0:
        command = _converter_command(config)
        print(
            f"[MCoreQAD] checkpoint missing; converting "
            f"{config.model_config.model_path} -> {checkpoint_path}",
            flush=True,
        )
        try:
            subprocess.run(
                command,
                check=True,
                env=_single_process_environment(),
            )
        except Exception as error:  # propagated to every torchrun rank below
            conversion_error = error

    if world_size > 1:
        status = torch.tensor(
            [0 if conversion_error is None else 1],
            dtype=torch.int32,
            device=torch.device("cuda", local_rank),
        )
        dist.broadcast(status, src=0)
        if status.item():
            _destroy_group_if_owned(created_group)
            raise RuntimeError(
                "MCoreQAD checkpoint conversion failed on rank 0; see its output above."
            ) from conversion_error
    elif conversion_error is not None:
        raise RuntimeError("MCoreQAD checkpoint conversion failed.") from conversion_error

    all_ready, _ = _distributed_readiness(checkpoint_path, world_size, local_rank)
    if not all_ready:
        _destroy_group_if_owned(created_group)
        raise RuntimeError(
            "MCoreQAD checkpoint conversion completed, but the checkpoint is not "
            "visible on every rank. Use a shared checkpoint path for multi-node runs."
        )
    return True
