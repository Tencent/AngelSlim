"""Common low-level helpers shared across all calibration sub-modules.

Nothing in here imports any sibling sub-module – it is the dependency root.
"""

import torch

__all__ = [
    # Public-ish helpers (kept un-prefixed for backwards compat with any
    # external callers that may have reached into the old flat module).
    # Internal helpers (leading underscore) are *not* exported here.
]


def _find_layers(module, layers=None, name=""):
    """Find all linear layers to monitor."""
    from vllm.model_executor.layers.linear import LinearBase

    if not layers:
        layers = [torch.nn.Linear, LinearBase]
    if isinstance(module, tuple(layers)):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            _find_layers(
                child,
                layers=layers,
                name=name + "." + name1 if name != "" else name1,
            )
        )
    return res


def _get_stat_value(stats, key):
    """Helper function to extract scalar value from stats, handling inf values."""
    val = stats[key]
    if isinstance(val, torch.Tensor):
        val = val.item()
    if key == "min" and val == float("inf"):
        return "N/A"
    if key == "max" and val == float("-inf"):
        return "N/A"
    return val


def _get_dist_info():
    """
    Get distributed training information (rank and world_size).

    Returns:
        tuple: (rank, world_size) - Returns (0, 1) if not in distributed mode
    """
    import torch.distributed as dist

    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


# ---------------------------------------------------------------------------
# Per-head KV role assignment (K/V workload split across replicated TP ranks)
# ---------------------------------------------------------------------------
# When vLLM's tensor parallelism replicates KV heads (``tp_size > num_kv_heads``)
# every KV head is held by ``replication = tp_size // num_kv_heads`` consecutive
# ranks.  The default calibration path makes *every* rank compute statistics
# for both K and V, which wastes CPU memory and search compute whenever
# ``replication >= 2``.
#
# We therefore split the workload: within each replication group, odd-indexed
# ranks (rank % 2 == 1) compute **K only** and even-indexed ranks
# (rank % 2 == 0) compute **V only**.  When ``replication < 2`` (or single-GPU)
# we fall back to the original behaviour: every rank computes both.
#
# A "role" is one of ``"k"``, ``"v"`` or ``"both"``.
# ---------------------------------------------------------------------------


def _get_kv_role(rank: int, world_size: int, num_kv_heads_total: int | None) -> str:
    """
    Return which kv-cache side (``"k"``, ``"v"`` or ``"both"``) the current
    rank is responsible for.

    Args:
        rank: global rank of this worker
        world_size: total number of TP workers
        num_kv_heads_total: total number of KV heads in the model (before TP
            replication).  If ``None`` or not yet known, we assume
            replication=1 and return ``"both"``.

    Rules:
        replication = world_size // num_kv_heads_total
        - replication <  2  → "both"   (no replication, every rank does both)
        - replication >= 2  → odd rank  → "k",
                              even rank → "v"
    """
    if world_size <= 1 or num_kv_heads_total is None or num_kv_heads_total <= 0:
        return "both"
    replication = world_size // num_kv_heads_total
    if replication < 2 or world_size % num_kv_heads_total != 0:
        return "both"
    return "k" if (rank % 2 == 1) else "v"


def _compute_perhead_layout(rank: int, world_size: int, num_kv_heads_total: int | None):
    """
    Compute per-rank KV-head layout info under the K/V-split scheme.

    Returns a tuple ``(role, heads_per_rank, global_head_offset, replication)``
    where:
        role              : "k" | "v" | "both"
        heads_per_rank    : number of *unique* KV heads handled by this rank
        global_head_offset: starting global KV-head index owned by this rank
        replication       : tp replication factor (max(1, world_size // H))

    Notes:
        * When replication == 1, every rank still owns ``H // world_size``
          heads (or 1 if world_size == 1) and role is "both".
        * When replication >= 2, odd/even ranks inside the same replication
          group share the same ``global_head_offset`` but handle K/V separately.
    """
    if num_kv_heads_total is None or num_kv_heads_total <= 0 or world_size <= 1:
        return "both", num_kv_heads_total or 0, 0, 1

    if world_size % num_kv_heads_total == 0 and world_size >= num_kv_heads_total:
        replication = world_size // num_kv_heads_total
        # Each replication group owns exactly one KV head.
        group_id = rank // replication
        global_head_offset = group_id
        heads_per_rank = 1
    elif num_kv_heads_total % world_size == 0:
        # No replication: each rank owns several distinct KV heads.
        replication = 1
        heads_per_rank = num_kv_heads_total // world_size
        global_head_offset = rank * heads_per_rank
    else:
        # Irregular – bail out to "both" (be safe)
        replication = 1
        heads_per_rank = max(1, num_kv_heads_total // max(1, world_size))
        global_head_offset = rank * heads_per_rank

    role = _get_kv_role(rank, world_size, num_kv_heads_total)
    return role, heads_per_rank, global_head_offset, replication


def _all_reduce_stats(stats_dict, stats_type="statistics", verbose=False):
    """
    Internal function to perform all-reduce on statistics across all workers.
    Handles uncalibrated layers/experts by setting default values.

    Args:
        stats_dict: Dictionary of activation/MoE statistics with 'min'/'max' keys
        stats_type: Type of statistics for logging (e.g., "activation", "MoE")
        verbose: If True, print detailed debug information

    Returns:
        tuple: (rank, world_size) or (0, 1) if not distributed
    """
    import torch.distributed as dist
    from torch.distributed import ReduceOp

    rank, world_size = _get_dist_info()

    if world_size <= 1:
        return rank, world_size

    if rank == 0:
        print(f"Performing {stats_type} all-reduce across {world_size} workers...")

    for name, stats in stats_dict.items():
        # Check if min/max are still inf/-inf (layer/expert not calibrated)
        min_val = stats["min"].item() if isinstance(stats["min"], torch.Tensor) else stats["min"]
        max_val = stats["max"].item() if isinstance(stats["max"], torch.Tensor) else stats["max"]

        if min_val == float("inf") or max_val == float("-inf"):
            if rank == 0:
                print(
                    f"[WARNING] '{name}' was not calibrated (min={min_val}, "
                    f"max={max_val}), setting to default value 1"
                )
            stats["min"] = torch.tensor(1.0)
            stats["max"] = torch.tensor(1.0)

        # All-reduce min (use MIN operation)
        min_tensor = (
            stats["min"].clone().cuda()
            if stats["min"].device.type == "cpu"
            else stats["min"].clone()
        )
        if verbose:
            print(f"Rank {rank}: layer {name} Min tensor before all-reduce: {min_tensor}")
        dist.all_reduce(min_tensor, op=ReduceOp.MIN)
        if verbose:
            print(f"Rank {rank}: layer {name} Min tensor after all-reduce: {min_tensor}")
        stats["min"] = min_tensor.cpu()
        del min_tensor  # Immediately free GPU memory
        torch.cuda.empty_cache()

        # All-reduce max (use MAX operation)
        max_tensor = (
            stats["max"].clone().cuda()
            if stats["max"].device.type == "cpu"
            else stats["max"].clone()
        )
        if verbose:
            print(f"Rank {rank}: layer {name} Max tensor before all-reduce: {max_tensor}")
        dist.all_reduce(max_tensor, op=ReduceOp.MAX)
        if verbose:
            print(f"Rank {rank}: layer {name} Max tensor after all-reduce: {max_tensor}")
        stats["max"] = max_tensor.cpu()
        del max_tensor  # Immediately free GPU memory
        torch.cuda.empty_cache()

    # Synchronize all ranks before continuing
    dist.barrier()

    if rank == 0:
        print(f"{stats_type.capitalize()} all-reduce completed.")

    return rank, world_size


def _print_stats_table(stats_dict, title):
    """
    Helper function to print statistics in a formatted table.

    Args:
        stats_dict: Dictionary of statistics with 'min'/'max' keys
        title: Title for the statistics table
    """
    print("\n" + "=" * 80)
    print(f"{title} (Min/Max)")
    print("=" * 80)
    for name, stats in stats_dict.items():
        min_val = _get_stat_value(stats, "min")
        max_val = _get_stat_value(stats, "max")
        call_count = stats.get("call_count", 0)
        print(f"{name:60s} | Min: {min_val:>12} | Max: {max_val:>12} | Calls: {call_count:4d}")
    print("=" * 80 + "\n")
