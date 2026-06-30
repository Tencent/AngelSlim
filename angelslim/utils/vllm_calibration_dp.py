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

"""Offline vLLM calibration DP helpers.

The calibration entrypoint uses this module to implement launcher-level data
parallelism with explicit Ray actors:

- the driver process connects to an existing Ray cluster once;
- the driver creates one placement group and one long-lived actor per DP replica;
- each actor creates exactly one vLLM instance through the existing worker_fn;
- every vLLM instance still uses its own TP workers via
  ``distributed_executor_backend=ray``;
- vLLM reuses the actor's placement group and captures TP child tasks there;
- the driver merges all partial calibration JSON payloads back into the
  standard stage-1 filenames.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import socket
from collections.abc import Callable
from typing import Any

# Native vLLM DP environment variable names that must NOT be set
_NATIVE_VLLM_DP_ENV_NAMES = (
    "VLLM_DP_SIZE",
    "VLLM_DP_SIZE_LOCAL",
    "VLLM_DP_RANK",
    "VLLM_DP_RANK_LOCAL",
    "VLLM_DP_MASTER_IP",
    "VLLM_DP_MASTER_PORT",
)


def validate_vllm_calibration_dp_args(parser, args):
    """Validate Ray actor-managed calibration DP arguments after YAML overrides."""
    if args.tp_size < 1:
        parser.error("--tp-size must be >= 1")
    if args.dp_size < 1:
        parser.error("--dp-size must be >= 1")
    if args.dp_timeout < 1:
        parser.error("--dp-timeout must be >= 1")

    if args.dp_size > 1 and args.distributed_executor_backend != "ray":
        parser.error(
            "Ray actor-managed calibration DP requires --distributed-executor-backend=ray"
        )

    # Stage-1 calibration prompts also need to be shardable across DP ranks.
    if args.dp_size > 1 and args.num_samples < args.dp_size:
        parser.error(
            "DP calibration requires --num-samples to be >= --dp-size "
            "so every DP rank receives at least one calibration prompt"
        )

    if (
        args.dp_size > 1
        and args.search_kv_scale
        and args.kv_granularity != "none"
        and args.search_kv_num_samples < args.dp_size
    ):
        parser.error(
            "DP KV scale search requires --search-kv-num-samples to be >= --dp-size "
            "so every DP rank receives at least one search prompt"
        )

    return args


def run_vllm_calibration_with_dp(args, worker_fn: Callable[[Any], Any]) -> None:
    """Run calibration directly or launch Ray actor-managed DP replicas.

    When ``args.dp_size == 1`` the worker is invoked in-process without any Ray
    actors. When ``args.dp_size > 1`` the function:

    1. connects to (or starts) a Ray cluster;
    2. creates one placement group + long-lived actor per DP rank;
    3. fans out the calibration ``worker_fn`` and collects the per-rank payloads;
    4. merges payloads on the driver and (optionally) launches a second
       KV-cache scale search stage on the same actors, broadcasting the merged
       activation stats via Ray's object store (so it works across nodes
       without requiring a shared filesystem).
    """
    args.dp_rank = 0

    # Make sure stale native vLLM DP environment variables in the launching
    # shell never leak into the worker process; they are incompatible with this
    # launcher-level DP scheme.
    for name in _NATIVE_VLLM_DP_ENV_NAMES:
        os.environ.pop(name, None)

    if args.dp_size == 1:
        # Single replica - no extra actors needed
        worker_fn(args)
        return

    print("\n" + "=" * 80)
    print("Launching Ray actor-managed calibration DP")
    print(f"Ray address        : {args.ray_address or os.environ.get('RAY_ADDRESS') or 'auto'}")
    print(f"DP size            : {args.dp_size}")
    print(f"TP size            : {args.tp_size}")
    print(f"Required GPUs      : {args.dp_size * args.tp_size}")
    print(f"Placement strategy : {args.placement_strategy}")
    print("Driver creates one placement group and one long-lived actor per DP rank;")
    print("vLLM Ray executor manages TP workers inside each rank actor.")
    print("=" * 80)

    try:
        _run_ray_actor_calibration_dp(args, worker_fn)
    except Exception as e:
        print(f"\nDP calibration failed: {e}")
        raise


def _run_ray_actor_calibration_dp(args, worker_fn: Callable[[Any], Any]) -> list[dict[str, Any]]:
    """Launch one explicit Ray actor per calibration DP replica."""
    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    ray_address = args.ray_address or os.environ.get("RAY_ADDRESS") or "auto"
    owns_ray = not ray.is_initialized()

    # Collect environment variables to pass to all Ray workers
    # This ensures consistent vLLM configuration across all DP ranks
    common_runtime_env = {"env_vars": {}}

    # List of vLLM environment variables that should be consistent across all DP ranks
    vllm_env_vars = [
        "VLLM_ALLOW_INSECURE_SERIALIZATION",
        "VLLM_MOE_COLLECT_STATS",
        "VLLM_MOE_COLLECT_STATS_VERBOSE",
        "VLLM_MOE_COLLECT_PER_EXPERT_STATS",
        "VLLM_ENABLE_CHUNKED_PREFILL",
        "VLLM_ATTENTION_BACKEND",
        "ASYNC_SCHEDULING",
        "VLLM_ENABLE_PREFIX_CACHING",
        "PRECISIONMODE",
        "RAY_DEDUP_LOGS",
        "PYTHONDONTWRITEBYTECODE",
        # Add any other environment variables that might be needed
    ]

    for env_var in vllm_env_vars:
        if env_var in os.environ:
            common_runtime_env["env_vars"][env_var] = os.environ[env_var]
            print(f"[DP] Will pass {env_var}={os.environ[env_var]} via runtime_env")

    if owns_ray:
        ray.init(
            address=ray_address,
            runtime_env=common_runtime_env,
            ignore_reinit_error=True,
            log_to_driver=True,
        )

    placement_groups = []
    actors = []

    try:
        # Validate cluster resources
        validate_cluster_resources(args)

        # Create placement groups for each DP rank
        for dp_rank in range(args.dp_size):
            bundles = create_replica_bundles(args.tp_size)
            pg = ray.util.placement_group(
                bundles=bundles,
                strategy=args.placement_strategy,
                name=f"calibration_dp_{dp_rank}",
            )
            placement_groups.append(pg)
            print(
                f"[DP] Created placement group for rank {dp_rank}: "
                f"{args.tp_size} GPU bundle(s)"
            )

        # Wait for all placement groups to be ready
        ready_refs = [pg.ready() for pg in placement_groups]
        ready, pending = ray.wait(
            ready_refs,
            num_returns=len(ready_refs),
            timeout=args.dp_timeout,
        )
        if pending:
            raise TimeoutError(
                "Timed out waiting for calibration DP placement groups: "
                f"ready={len(ready)}, pending={len(pending)}, "
                f"timeout={args.dp_timeout}s. "
                "Check whether enough free GPUs are available for "
                "dp_size * tp_size."
            )

        # Create actors for each DP rank
        ReplicaActor = ray.remote(
            num_cpus=1,
            num_gpus=0,
            max_restarts=0,
            max_task_retries=0,
        )(CalibrationReplica)

        for dp_rank, pg in enumerate(placement_groups):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True,
            )
            actor = ReplicaActor.options(
                scheduling_strategy=scheduling_strategy,
                name=f"calibration-replica-{dp_rank}",
            ).remote(worker_fn)
            actors.append(actor)
            print(f"[DP] Submitted calibration actor for rank {dp_rank}")

        # Submit tasks to actors
        args_dict = vars(copy.deepcopy(args))
        result_refs = [
            actor.run.remote(args_dict, dp_rank) for dp_rank, actor in enumerate(actors)
        ]

        # Wait for results with timeout
        ready, pending = ray.wait(
            result_refs,
            num_returns=len(result_refs),
            timeout=args.dp_timeout,
        )
        if pending:
            raise TimeoutError(
                "Timed out waiting for calibration DP actors: "
                f"finished={len(ready)}, pending={len(pending)}, "
                f"timeout={args.dp_timeout}s."
            )

        results = ray.get(result_refs)
        merged_activation_stats = _merge_dp_payloads(args.output_dir, results)

        if args.search_kv_scale and args.kv_granularity != "none":
            if not merged_activation_stats:
                raise RuntimeError(
                    "Merged activation stats are empty; cannot launch DP KV scale search. "
                    "Check the stage-1 actor logs."
                )

            print("\n" + "=" * 80)
            print("Launching DP KV-cache scale search with merged activation stats...")
            print("=" * 80)

            # Broadcast the merged stats through Ray's object store instead of
            # relying on a shared filesystem. This is mandatory for multi-node
            # clusters where ``args.output_dir`` only exists on the driver.
            activation_stats_ref = ray.put(merged_activation_stats)

            args_dict = vars(copy.deepcopy(args))
            kv_refs = [
                actor.run_kv_search.remote(args_dict, dp_rank, activation_stats_ref)
                for dp_rank, actor in enumerate(actors)
            ]
            ready, pending = ray.wait(
                kv_refs,
                num_returns=len(kv_refs),
                timeout=args.dp_timeout,
            )
            if pending:
                raise TimeoutError(
                    "Timed out waiting for DP KV search actors: "
                    f"finished={len(ready)}, pending={len(pending)}, "
                    f"timeout={args.dp_timeout}s."
                )
            kv_results = ray.get(kv_refs)
            _merge_dp_kv_search_payloads(
                args.output_dir,
                kv_results,
                args.kv_granularity,
                activation_stats=merged_activation_stats,
            )

        return results

    finally:
        # Clean up actors and placement groups
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass

        for pg in placement_groups:
            try:
                ray.util.remove_placement_group(pg)
            except Exception:
                pass

        if owns_ray:
            ray.shutdown()


def _safe_rmtree(path: str) -> None:
    """Delete a directory tree, refusing obviously-dangerous targets.

    The DP launcher creates per-rank subdirectories under the user-provided
    ``output_dir`` and clears them between runs. This helper guards against
    accidentally wiping system or root paths if a misconfiguration ever causes
    ``output_dir`` to resolve to something pathological.
    """
    if not path:
        return
    abs_path = os.path.abspath(path)
    norm = os.path.normpath(abs_path)
    # Reject root, very shallow paths, and well-known system mount points.
    forbidden_prefixes = (
        "/",
        "/root",
        "/home",
        "/etc",
        "/var",
        "/usr",
        "/bin",
        "/sbin",
        "/lib",
        "/boot",
        "/dev",
        "/proc",
        "/sys",
    )
    if norm in forbidden_prefixes or norm == os.path.expanduser("~"):
        raise RuntimeError(f"Refusing to rmtree suspicious path: {norm!r}")
    if len(norm) < 10 or norm.count(os.sep) < 2:
        raise RuntimeError(f"Refusing to rmtree shallow path: {norm!r}")
    if not os.path.isdir(norm):
        return
    shutil.rmtree(norm)


class CalibrationReplica:
    """Ray actor that runs one calibration DP replica."""

    def __init__(self, worker_fn: Callable[[Any], Any]) -> None:
        self._worker_fn = worker_fn
        self._has_calibrated = False
        self._dp_rank: int | None = None
        self._llm: Any | None = None

    def _prepare_args(
        self,
        args_dict: dict[str, Any],
        dp_rank: int,
        output_suffix: str,
    ):
        args = argparse.Namespace(**args_dict)
        args.dp_rank = dp_rank
        args.dp_size = args_dict.get("dp_size", 1)

        for name in _NATIVE_VLLM_DP_ENV_NAMES:
            os.environ.pop(name, None)

        args.output_dir = os.path.join(args.output_dir, output_suffix)
        _safe_rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)
        return args

    def run(
        self,
        args_dict: dict[str, Any],
        dp_rank: int,
    ) -> dict[str, Any]:
        if self._has_calibrated:
            raise RuntimeError(
                "CalibrationReplica calibration stage can only run once. "
                f"existing_dp_rank={self._dp_rank}, "
                f"new_dp_rank={dp_rank}"
            )

        self._has_calibrated = True
        self._dp_rank = dp_rank

        args = self._prepare_args(args_dict, dp_rank, f"dp_rank_{dp_rank}")
        # In DP mode stage-1 only collects activation/MoE statistics. The
        # KV-cache scale search must wait until the driver has merged stats
        # across all DP ranks; it then runs in ``run_kv_search`` below.
        args.search_kv_scale = False

        dp_log(args, "actor_start", f"Actor started on host={socket.gethostname()}")
        result = self._worker_fn(args, return_llm=True)
        if isinstance(result, tuple) and len(result) == 2:
            payload, self._llm = result
        else:
            payload = result
            self._llm = None
        dp_log(args, "actor_done", "Calibration completed")

        return {
            "dp_rank": dp_rank,
            "output_dir": args.output_dir,
            "payload": payload,
        }

    def run_kv_search(
        self,
        args_dict: dict[str, Any],
        dp_rank: int,
        activation_stats: dict,
    ) -> dict[str, Any]:
        """Run the second-stage KV-cache scale search on this actor.

        ``activation_stats`` is the merged stats dict produced by the driver.
        It is passed in directly (typically materialised from a Ray object
        store reference) so the actor never has to read it from disk; this is
        what makes the launcher work on multi-node clusters where
        ``args.output_dir`` is not on a shared filesystem.
        """
        args = self._prepare_args(args_dict, dp_rank, f"dp_rank_{dp_rank}_kv_search")
        args.kv_search_only = True
        args.kv_search_activation_stats = activation_stats
        args.search_kv_scale = True

        if self._llm is None:
            raise RuntimeError(
                "Cannot run DP KV search because the first-stage vLLM instance "
                "was not retained in the actor."
            )

        dp_log(args, "kv_search_start", f"KV search actor started on host={socket.gethostname()}")
        payload = self._worker_fn(args, llm=self._llm)
        dp_log(args, "kv_search_done", "KV search completed")

        return {
            "dp_rank": dp_rank,
            "output_dir": args.output_dir,
            "payload": payload,
        }


def create_replica_bundles(tp_size: int) -> list[dict[str, float]]:
    """Create bundles for a single DP replica."""
    bundles = [
        {
            "CPU": 1,
            "GPU": 1,
        }
    ]

    bundles.extend(
        {
            "GPU": 1,
        }
        for _ in range(tp_size - 1)
    )

    return bundles


def validate_cluster_resources(args) -> None:
    """Validate that the Ray cluster has enough resources for the DP/TP configuration."""
    import ray

    required_gpus = args.dp_size * args.tp_size
    cluster_resources = ray.cluster_resources()
    available_gpus = int(cluster_resources.get("GPU", 0))

    if available_gpus < required_gpus:
        raise RuntimeError(
            f"Need {required_gpus} GPUs for DP={args.dp_size}, TP={args.tp_size}, "
            f"but Ray reports {available_gpus} available GPUs."
        )

    # Validate STRICT_PACK capacity if needed
    if args.placement_strategy == "STRICT_PACK":
        validate_strict_pack_capacity(args)


def validate_strict_pack_capacity(args) -> None:
    """Validate that each DP replica can fit entirely on a single node with STRICT_PACK."""
    import ray

    capacity = 0
    node_infos = []

    for node in ray.nodes():
        if not node.get("Alive", False):
            continue

        resources = node.get("Resources", {})
        gpus = int(resources.get("GPU", 0))
        cpus = int(resources.get("CPU", 0))

        # Each replica only needs 1 CPU bundle, so capacity is GPU-bound.
        # We still keep the cpu>=1 check to avoid pathological CPU-less nodes.
        if cpus < 1:
            replicas_on_node = 0
        else:
            replicas_on_node = gpus // args.tp_size

        capacity += replicas_on_node

        node_infos.append(
            {
                "address": node.get("NodeManagerAddress"),
                "gpus": gpus,
                "cpus": cpus,
                "replica_capacity": replicas_on_node,
            }
        )

    if capacity < args.dp_size:
        raise RuntimeError(
            "STRICT_PACK placement is infeasible. "
            f"Need {args.dp_size} replicas with "
            f"TP={args.tp_size}, but capacity={capacity}. "
            "Consider switching --placement-strategy to PACK if a single replica "
            "may span multiple nodes. "
            f"nodes={node_infos}"
        )


def dp_log(
    args,
    stage: str,
    message: str,
) -> None:
    """Log a message with DP rank and host information."""
    print(
        f"[DP {getattr(args, 'dp_rank', 0)}/{getattr(args, 'dp_size', 1)}] "
        f"[host={socket.gethostname()}] "
        f"[pid={os.getpid()}] "
        f"[stage={stage}] "
        f"{message}",
        flush=True,
    )


def _merge_dp_payloads(output_dir: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge calibration statistics from all DP ranks.

    Returns the merged ``activation_stats.json`` content (possibly empty) so
    the caller can feed it directly into the second-stage KV scale search
    without having to re-read it from disk.
    """
    print("\n" + "=" * 80, flush=True)
    print("Merging DP calibration statistics...", flush=True)
    print("=" * 80, flush=True)

    _MINMAX_FILES = (
        "activation_stats.json",
        "moe_expert_stats.json",
        "mtp_activation_stats.json",
        "mtp_moe_expert_stats.json",
    )

    # Collect payloads from all ranks
    payloads = [result["payload"] for result in results]
    dp_size = len(payloads)

    # Initialize merged statistics
    merged_stats = {}
    for filename in _MINMAX_FILES:
        merged_stats[filename] = {}

    # Merge min/max statistics from all ranks
    for filename in _MINMAX_FILES:
        print(f"\nMerging {filename}...")

        # Collect all entries for this file from all ranks
        all_entries = {}
        for rank, payload in enumerate(payloads):
            if filename not in payload:
                print(f"  Rank {rank}: File {filename} not found")
                continue

            rank_stats = payload[filename]
            for key, stats in rank_stats.items():
                if key not in all_entries:
                    all_entries[key] = []
                all_entries[key].append({"rank": rank, "stats": stats})

        # Merge each key
        for key, rank_data in all_entries.items():
            if len(rank_data) < dp_size:
                print(f"  Key {key}: Only found in {len(rank_data)}/{dp_size} ranks")

            # Validate data types
            first_stats = rank_data[0]["stats"]
            if (
                not isinstance(first_stats, dict)
                or "min" not in first_stats
                or "max" not in first_stats
            ):
                print(f"  Key {key}: Invalid stats format, skipping")
                continue

            # Check if min/max are scalars or lists
            first_min = first_stats["min"]
            is_list = isinstance(first_min, list)

            # Validate consistency across ranks
            skip_key = False
            for entry in rank_data[1:]:
                stats = entry["stats"]
                if isinstance(stats["min"], list) != is_list:
                    print(f"  Key {key}: Mixed scalar/list types across ranks, skipping")
                    skip_key = True
                    break
                if is_list and len(stats["min"]) != len(first_min):
                    print(f"  Key {key}: List length mismatch, skipping")
                    skip_key = True
                    break
            if skip_key:
                continue

            # Merge
            if is_list:
                # List min/max
                list_len = len(first_min)
                merged_min = [float("inf")] * list_len
                merged_max = [float("-inf")] * list_len

                for entry in rank_data:
                    stats = entry["stats"]
                    for i in range(list_len):
                        merged_min[i] = min(merged_min[i], stats["min"][i])
                        merged_max[i] = max(merged_max[i], stats["max"][i])

                merged_stats[filename][key] = {
                    "min": merged_min,
                    "max": merged_max,
                }
            else:
                # Scalar min/max
                merged_min = float("inf")
                merged_max = float("-inf")

                for entry in rank_data:
                    stats = entry["stats"]
                    merged_min = min(merged_min, stats["min"])
                    merged_max = max(merged_max, stats["max"])

                merged_stats[filename][key] = {
                    "min": merged_min,
                    "max": merged_max,
                }

        print(f"  Merged {len(merged_stats[filename])} keys")

    # Save merged statistics
    print("\n" + "=" * 80)
    print("Saving merged statistics...")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    for filename, stats in merged_stats.items():
        if not stats:
            continue

        output_path = os.path.join(output_dir, filename)

        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved merged {filename} to {output_path}")

    print("\n" + "=" * 80, flush=True)
    print("DP calibration merge completed!", flush=True)
    print("=" * 80, flush=True)

    return merged_stats.get("activation_stats.json", {})


def _same_multiplier_grid(lhs: list[float], rhs: list[float]) -> bool:
    if len(lhs) != len(rhs):
        return False
    return all(abs(float(a) - float(b)) <= 1e-12 for a, b in zip(lhs, rhs))


def _select_best_multiplier_from_profiles(
    profile_entries: list[dict[str, Any]]
) -> tuple[float, float]:
    multipliers = profile_entries[0]["multipliers"]
    base_scale = profile_entries[0].get("base_scale")
    total_sse = [0.0 for _ in multipliers]
    total_numel = 0

    for profile in profile_entries:
        if not _same_multiplier_grid(multipliers, profile["multipliers"]):
            raise ValueError("KV search multiplier grids differ across DP ranks")
        if base_scale is not None and profile.get("base_scale") is not None:
            if abs(float(base_scale) - float(profile["base_scale"])) > 1e-12:
                raise ValueError("KV search base_scale differs across DP ranks")
        if len(profile["sse"]) != len(multipliers):
            raise ValueError("KV search SSE length does not match multiplier grid")
        for i, sse in enumerate(profile["sse"]):
            total_sse[i] += float(sse)
        total_numel += int(profile.get("numel", 0))

    if total_numel <= 0:
        return 1.0, float("inf")

    global_mse = [sse / total_numel for sse in total_sse]
    best_idx = min(range(len(global_mse)), key=lambda i: global_mse[i])
    return float(multipliers[best_idx]), float(global_mse[best_idx])


def _merge_dp_kv_search_payloads(
    output_dir: str,
    results: list[dict[str, Any]],
    kv_granularity: str,
    activation_stats: dict | None = None,
) -> None:
    """Merge DP-local KV MSE profiles and save final scale files.

    ``activation_stats`` may be provided directly (e.g. from
    ``_merge_dp_payloads``) to avoid re-reading ``activation_stats.json`` from
    disk. This is required for multi-node deployments where ``output_dir`` is
    not on a shared filesystem.
    """
    print("\n" + "=" * 80, flush=True)
    print("Merging DP KV-cache scale search profiles...", flush=True)
    print("=" * 80, flush=True)

    if activation_stats is None:
        activation_stats_path = os.path.join(output_dir, "activation_stats.json")
        with open(activation_stats_path, "r", encoding="utf8") as f:
            activation_stats = json.load(f)

    payloads = [result["payload"] for result in results]
    fp8_max = 448.0

    if kv_granularity == "per-tensor":
        profile_key = "kv_scale_mse_profiles.json"
        per_tensor_profiles: dict[str, list[dict[str, Any]]] = {}
        for rank, payload in enumerate(payloads):
            profiles = payload.get(profile_key, {})
            if not profiles:
                print(f"  Rank {rank}: no {profile_key} found")
                continue
            for stats_key, profile in profiles.items():
                per_tensor_profiles.setdefault(stats_key, []).append(profile)

        kv_multipliers = {}
        for stats_key, entries in per_tensor_profiles.items():
            best_multiplier, best_mse = _select_best_multiplier_from_profiles(entries)
            kv_multipliers[stats_key] = best_multiplier
            print(
                f"  {stats_key}: best_multiplier={best_multiplier:.6f}, "
                f"global_mse={best_mse:.6e}, dp_parts={len(entries)}"
            )

        multipliers_path = os.path.join(output_dir, "kv_scale_multipliers.json")
        with open(multipliers_path, "w", encoding="utf8") as f:
            json.dump(kv_multipliers, f, indent=2)

        tuned_kv_scales = {}
        for stats_key, multiplier in kv_multipliers.items():
            stats = activation_stats[stats_key]
            abs_max = max(abs(stats["min"]), abs(stats["max"]))
            base_scale = abs_max / fp8_max * 2.0 if abs_max != 0 else 1e-8
            tuned_scale = base_scale * multiplier
            save_key = f"{stats_key.replace('attn.attn', 'attn')}.scale"
            tuned_kv_scales[save_key] = tuned_scale
        tuned_scales_path = os.path.join(output_dir, "kv_cache_tuned_scales.json")
        with open(tuned_scales_path, "w", encoding="utf8") as f:
            json.dump(tuned_kv_scales, f, indent=2)

        print(f"Saved DP KV multipliers to {multipliers_path}")
        print(f"Saved DP KV tuned scales to {tuned_scales_path}")

    elif kv_granularity == "per-head":
        profile_key = "kv_scale_mse_profiles_per_head.json"
        per_head_profiles: dict[str, list[list[dict[str, Any]]]] = {}
        for rank, payload in enumerate(payloads):
            profiles = payload.get(profile_key, {})
            if not profiles:
                print(f"  Rank {rank}: no {profile_key} found")
                continue
            for stats_key, head_profiles in profiles.items():
                per_head_profiles.setdefault(stats_key, []).append(head_profiles)

        kv_multipliers_perhead = {}
        for stats_key, rank_head_profiles in per_head_profiles.items():
            stats = activation_stats.get(stats_key)
            if not stats or not isinstance(stats.get("min"), list):
                print(f"  {stats_key}: missing per-head activation stats, skipping")
                continue
            num_heads = len(stats["min"])
            multipliers = []
            for head_idx in range(num_heads):
                entries = [
                    head_profiles[head_idx]
                    for head_profiles in rank_head_profiles
                    if head_idx < len(head_profiles) and head_profiles[head_idx]
                ]
                if not entries:
                    multipliers.append(1.0)
                    continue
                best_multiplier, _ = _select_best_multiplier_from_profiles(entries)
                multipliers.append(best_multiplier)
            kv_multipliers_perhead[stats_key] = multipliers
            print(
                f"  {stats_key}: multipliers min={min(multipliers):.6f} "
                f"max={max(multipliers):.6f} over {len(multipliers)} heads"
            )

        multipliers_path = os.path.join(output_dir, "kv_scale_multipliers_per_head.json")
        with open(multipliers_path, "w", encoding="utf8") as f:
            json.dump(kv_multipliers_perhead, f, indent=2)

        tuned_kv_scales_perhead = {}
        for stats_key, multipliers in kv_multipliers_perhead.items():
            stats = activation_stats[stats_key]
            min_vals = stats["min"]
            max_vals = stats["max"]
            for head_idx, multiplier in enumerate(multipliers):
                abs_max = max(abs(min_vals[head_idx]), abs(max_vals[head_idx]))
                base_scale = abs_max / fp8_max * 2.0 if abs_max != 0 else 1e-8
                tuned_scale = base_scale * multiplier
                base_key = stats_key.replace("attn.attn", "attn")
                save_key = f"{base_key}.head_{head_idx}.scale"
                tuned_kv_scales_perhead[save_key] = tuned_scale
        tuned_scales_path = os.path.join(output_dir, "kv_cache_tuned_scales_per_head.json")
        with open(tuned_scales_path, "w", encoding="utf8") as f:
            json.dump(tuned_kv_scales_perhead, f, indent=2)

        print(f"Saved DP per-head KV multipliers to {multipliers_path}")
        print(f"Saved DP per-head KV tuned scales to {tuned_scales_path}")

    else:
        print(f"Skipping KV search merge for kv_granularity={kv_granularity}")

    print("\n" + "=" * 80)
    print("DP KV-cache scale search merge completed!")
    print("=" * 80)
