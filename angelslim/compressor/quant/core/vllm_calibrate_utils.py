import os

import torch

__all__ = [
    "setup_activation_hooks",
    "get_activation_stats",
    "print_activation_stats",
    "print_moe_stats",
    "get_moe_stats",
    "setup_mtp_activation_hooks",
    "get_mtp_activation_stats",
    "print_mtp_activation_stats",
    "get_mtp_moe_stats",
    "print_mtp_moe_stats",
    "setup_kvcache_value_hooks",
    "KVScaleSearcher",
    "get_kv_scale_search_results",
    "remove_kv_scale_search_hooks",
    # KV-only calibration (no weight/activation hooks)
    "setup_kvcache_only_hooks",
    "get_kvcache_only_stats",
    "print_kvcache_only_stats",
    # Per-head KV calibration
    "setup_kvcache_perhead_hooks",
    "get_kvcache_perhead_stats",
    "print_kvcache_perhead_stats",
    "remove_kvcache_perhead_hooks",
    "setup_kvcache_perhead_value_hooks",
    "remove_kvcache_perhead_value_hooks",
    "KVScaleSearcherPerHead",
    "get_kv_scale_search_results_perhead",
    # P-matrix scale search (per Q-head)
    "setup_p_matrix_scale_hooks",
    "get_p_matrix_scale_stats",
    "remove_p_matrix_scale_hooks",
    "PMatrixScaleSearcher",
    # P-matrix first-N-columns analysis (independent mode wrapper)
    "setup_p_first_cols_hooks",
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


class ActivationHook:
    """Hook class for collecting activation statistics (pickle-safe)."""

    def __init__(self, layer_name, activation_stats):
        self.layer_name = layer_name
        self.activation_stats = activation_stats
        self.call_count = 0  # Track how many times this hook is called

    def __call__(self, module, input, output):
        self.call_count += 1

        # Get the input activation
        if isinstance(input, tuple):
            act = input[0]
        else:
            act = input

        if isinstance(act, torch.Tensor):
            # if act.numel() == 0:
            #     #print("Empty tensor", module)
            #     act = torch.tensor([0.0])
            # Use tensor operations to avoid graph breaks
            with torch.no_grad():
                act_min = act.min().detach().cpu()
                act_max = act.max().detach().cpu()

                # Update global min/max using tensor operations
                stats = self.activation_stats[self.layer_name]
                stats["min"] = torch.minimum(stats["min"], act_min)
                stats["max"] = torch.maximum(stats["max"], act_max)
                stats["call_count"] = self.call_count  # Store call count


class KVCacheHook:
    """Hook class for collecting kv cache statistics (pickle-safe)."""

    def __init__(self, layer_name, kcache_stats, vcache_stats):
        self.layer_name = layer_name
        self.kcache_stats = kcache_stats
        self.vcache_stats = vcache_stats
        self.call_count = 0  # Track how many times this hook is called

    def __call__(self, module, input, output):
        self.call_count += 1

        # Get the input activation
        _, k, v = input[0], input[1], input[2]

        if isinstance(k, torch.Tensor):
            # Use tensor operations to avoid graph breaks
            with torch.no_grad():
                k_act_min = k.min().detach().cpu()
                k_act_max = k.max().detach().cpu()
                v_act_min = v.min().detach().cpu()
                v_act_max = v.max().detach().cpu()

                # Update global min/max using tensor operations
                k_stats = self.kcache_stats[self.layer_name]
                k_stats["min"] = torch.minimum(k_stats["min"], k_act_min)
                k_stats["max"] = torch.maximum(k_stats["max"], k_act_max)
                v_stats = self.vcache_stats[self.layer_name]
                v_stats["min"] = torch.minimum(v_stats["min"], v_act_min)
                v_stats["max"] = torch.maximum(v_stats["max"], v_act_max)
                k_stats["call_count"] = self.call_count  # Store call count
                v_stats["call_count"] = self.call_count  # Store call count


def setup_activation_hooks(model, kv_granularity="per-tensor"):
    """
    Setup activation hooks on the model to collect min/max statistics.
    This function is applied to each worker's model instance.

    Args:
        kv_granularity: Controls KV-cache hook registration.
            'none'       – skip KV hooks entirely (only Linear/MoE hooks registered).
            'per-tensor' – register per-layer (per-tensor) KV min/max hooks (default).
            'per-head'   – register per-head KV min/max hooks (calls
                           setup_kvcache_perhead_hooks internally).
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase

    try:
        # vLLM ≥ 0.20 (Tencent custom): Attention moved under model_executor.
        from vllm.model_executor.layers.attention import Attention
    except ImportError:
        # Older vLLM layout.
        from vllm.attention.layer import Attention

    # Find all linear layers to monitor
    layers_to_monitor = _find_layers(model, layers=[torch.nn.Linear, LinearBase])
    print(f"---------Found {len(layers_to_monitor)} layers to monitor---------")
    for name in list(layers_to_monitor.keys())[:5]:  # Print first 5
        print(f"  {name}")
    if len(layers_to_monitor) > 5:
        print(f"  ... and {len(layers_to_monitor) - 5} more activation layers")

    # Initialize activation statistics storage
    if not hasattr(model, "_activation_stats"):
        model._activation_stats = {}
        for name in layers_to_monitor.keys():
            model._activation_stats[name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }

    # Register hooks for all linear layers
    if not hasattr(model, "_activation_hooks"):
        model._activation_hooks = []
        for name, layer in layers_to_monitor.items():
            hook = ActivationHook(name, model._activation_stats)
            hook_handle = layer.register_forward_hook(hook)
            model._activation_hooks.append(hook_handle)

    # KV-cache hooks: behaviour controlled by kv_granularity
    if kv_granularity == "none":
        print("---------KV-cache hooks skipped (kv_granularity=none)---------")

    elif kv_granularity == "per-tensor":
        kvcache_layers = _find_layers(model, layers=[Attention])
        print(
            f"---------Found {len(kvcache_layers)} kv cache layers to monitor (per-tensor)---------"  # noqa: E501
        )
        for name in list(kvcache_layers.keys())[:5]:
            print(f"  {name}")
        if len(kvcache_layers) > 5:
            print(f"  ... and {len(kvcache_layers) - 5} more kv cache layers")

        if not hasattr(model, "_kcache_stats"):
            model._kcache_stats = {}
            model._vcache_stats = {}
            for name in kvcache_layers.keys():
                model._kcache_stats[name] = {
                    "min": torch.tensor(float("inf")),
                    "max": torch.tensor(float("-inf")),
                }
                model._vcache_stats[name] = {
                    "min": torch.tensor(float("inf")),
                    "max": torch.tensor(float("-inf")),
                }

        if not hasattr(model, "_kvcache_hooks"):
            model._kvcache_hooks = []
            for name, layer in kvcache_layers.items():
                hook = KVCacheHook(name, model._kcache_stats, model._vcache_stats)
                hook_handle = layer.register_forward_hook(hook)
                model._kvcache_hooks.append(hook_handle)

    elif kv_granularity == "per-head":
        # Delegate to the dedicated per-head hook setup
        setup_kvcache_perhead_hooks(model)
        print("---------Per-head KV-cache hooks registered via setup_activation_hooks---------")

    # Register MoE statistics storage and hooks
    moe_layers = _find_layers(model, layers=[FusedMoE])
    if moe_layers:
        print(f"---------Found {len(moe_layers)} MoE layers to monitor---------")
        for name in list(moe_layers.keys())[:5]:  # Print first 5
            print(f"  {name}")
        if len(moe_layers) > 5:
            print(f"  ... and {len(moe_layers) - 5} more")

        # Check if per-expert stats collection is enabled
        per_expert = os.getenv("VLLM_MOE_COLLECT_PER_EXPERT_STATS", "0") == "1"
        print(
            f"---------Per-expert stats collection: {'ENABLED' if per_expert else 'DISABLED'}---------"  # noqa: E501
        )

        # Initialize MoE activation statistics storage
        if not hasattr(model, "_moe_activation_stats"):
            model._moe_activation_stats = {}
            for name, layer in moe_layers.items():
                # Get the number of experts from the FusedMoE layer
                num_experts = getattr(layer, "global_num_experts", None)
                if num_experts is None:
                    num_experts = getattr(layer, "num_experts", 256)
                    print(
                        f"[WARNING] Could not find global_num_experts "
                        f"for {name}, using {num_experts}"
                    )

                for stage in ["gate_up_proj", "down_proj"]:
                    # Layer-level stats (overall)
                    model._moe_activation_stats[f"{name}.{stage}"] = {
                        "min": torch.tensor(float("inf")),
                        "max": torch.tensor(float("-inf")),
                    }
                    # Per-expert stats (only when enabled)
                    if per_expert:
                        for expert_id in range(num_experts):
                            model._moe_activation_stats[f"{name}.{expert_id}.{stage}"] = {
                                "min": torch.tensor(float("inf")),
                                "max": torch.tensor(float("-inf")),
                            }

                # Set layer name attribute on weights for statistics collection
                if hasattr(layer, "w13_weight") and layer.w13_weight is not None:
                    layer.w13_weight._vllm_layer_name = name
                    layer.w13_weight._moe_activation_stats_of_model = model._moe_activation_stats
                    print(
                        f"[DEBUG] Set w13_weight._vllm_layer_name = {name}, "
                        f"type={type(layer.w13_weight)}"
                    )
                else:
                    print(
                        f"[DEBUG] Cannot set w13_weight._vllm_layer_name: "
                        f"hasattr={hasattr(layer, 'w13_weight')}, "
                        f"is_none={getattr(layer, 'w13_weight', None) is None}"
                    )

    print("---------Activation hooks registered---------")
    return f"Registered {len(model._activation_hooks)} hooks"


def get_activation_stats(model):
    """
    Retrieve activation statistics from the model.
    Performs all-reduce across all workers to get global min/max.
    """
    if not hasattr(model, "_activation_stats"):
        return None

    # Perform all-reduce to get global min/max across all workers
    try:
        _all_reduce_stats(model._activation_stats, stats_type="activation")
        if hasattr(model, "_kcache_stats"):
            _all_reduce_stats(model._kcache_stats, stats_type="kcache")
            _all_reduce_stats(model._vcache_stats, stats_type="vcache")
    except Exception as e:
        print(f"Warning: Could not perform all-reduce: {e}")

    # Convert tensors to Python scalars for easier use
    stats_dict = {}
    for name, stats in model._activation_stats.items():
        stats_dict[name] = {
            "min": stats["min"].item() if isinstance(stats["min"], torch.Tensor) else stats["min"],
            "max": stats["max"].item() if isinstance(stats["max"], torch.Tensor) else stats["max"],
        }
    if hasattr(model, "_kcache_stats"):
        kcache_stats_dict = {}
        for name, stats in model._kcache_stats.items():
            kcache_stats_dict[name + ".k_cache"] = {
                "min": (
                    stats["min"].item() if isinstance(stats["min"], torch.Tensor) else stats["min"]
                ),
                "max": (
                    stats["max"].item() if isinstance(stats["max"], torch.Tensor) else stats["max"]
                ),
            }
        stats_dict.update(kcache_stats_dict)
    if hasattr(model, "_vcache_stats"):
        vcache_stats_dict = {}
        for name, stats in model._vcache_stats.items():
            vcache_stats_dict[name + ".v_cache"] = {
                "min": (
                    stats["min"].item() if isinstance(stats["min"], torch.Tensor) else stats["min"]
                ),
                "max": (
                    stats["max"].item() if isinstance(stats["max"], torch.Tensor) else stats["max"]
                ),
            }
        stats_dict.update(vcache_stats_dict)
    return stats_dict


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


def print_activation_stats(model):
    """
    Print activation statistics in a readable format.
    Performs all-reduce to get global statistics across all workers.
    """
    if not hasattr(model, "_activation_stats"):
        print("No activation statistics available")
        return

    # Perform all-reduce to get global min/max
    try:
        rank, world_size = _all_reduce_stats(model._activation_stats, stats_type="activation")
        if hasattr(model, "_kcache_stats"):
            _all_reduce_stats(model._kcache_stats, stats_type="kcache")
            _all_reduce_stats(model._vcache_stats, stats_type="vcache")
    except Exception as e:
        print(f"Warning: Could not perform all-reduce: {e}")
        rank, world_size = 0, 1

    # Only rank 0 prints the statistics (or single process)
    if rank != 0:
        return

    # Print statistics
    if world_size > 1:
        print(f"\n[Global statistics across {world_size} workers]")
    _print_stats_table(model._activation_stats, "Activation Statistics")
    if hasattr(model, "_kcache_stats"):
        _print_stats_table(model._kcache_stats, "K-cache Statistics")
        _print_stats_table(model._vcache_stats, "V-cache Statistics")


def collect_fused_moe_internal_stats(
    stage,
    hidden_states,
    topk_ids,
    global_num_experts,
    layer_name=None,
    global_moe_activation_stats=None,
):
    """
    Collect FusedMoE internal activation statistics and accumulate in global dictionary.
    Only collects stats during actual generation (skips CUDA graph capture phase).

    Args:
        stage: "gate_up_proj" or "down_proj"
        hidden_states: Input tensor [num_tokens, hidden_size] or [num_tokens*top_k, hidden_size]
        topk_ids: Expert IDs [num_tokens, top_k]
        global_num_experts: Total number of experts
        layer_name: Layer name for identification (if None, will try to get from context)
        global_moe_activation_stats: Global dictionary to store statistics

    Environment Variables:
        VLLM_MOE_COLLECT_STATS: Set to "1" to enable statistics collection
        VLLM_MOE_COLLECT_STATS_VERBOSE: Set to "1" to enable verbose debug output
    """
    # Use os.getenv directly instead of vllm.envs to avoid caching issues in Ray workers
    # Check if MoE stats collection is enabled
    if os.getenv("VLLM_MOE_COLLECT_STATS", "0") != "1":
        return

    # Check verbose flag (default off to avoid hang in distributed setting)
    verbose = os.getenv("VLLM_MOE_COLLECT_STATS_VERBOSE", "0") == "1"

    #
    if global_moe_activation_stats is None:
        return

    # Skip if layer_name is not provided (weight not properly initialized yet)
    if layer_name is None:
        return

    # Only collect stats for MoE layers (should contain "experts" in the name)
    if "experts" not in layer_name.lower():
        return

    # Get rank information
    rank, world_size = _get_dist_info()

    # Collect statistics
    key = f"{layer_name}.{stage}"
    with torch.no_grad():
        # --- Layer-level (overall) stats ---
        if key in global_moe_activation_stats:
            stats = global_moe_activation_stats[key]
            act_min = hidden_states.min().detach().cpu()
            act_max = hidden_states.max().detach().cpu()
            if verbose:
                print(
                    f"[VERBOSE] Rank {rank}/{world_size}: Collected MoE stats "
                    f"for {key}, min: {act_min.item()}, max: {act_max.item()}"
                )
            stats["min"] = torch.minimum(stats["min"], act_min)
            stats["max"] = torch.maximum(stats["max"], act_max)
            if verbose:
                print(
                    f"[VERBOSE] Rank {rank}/{world_size}: "
                    f"Updated MoE stats for {key}, min: {stats['min'].item()}, "
                    f"max: {stats['max'].item()}"
                )
            stats["call_count"] = stats.get("call_count", 0) + 1

        # --- Per-expert stats (only when enabled) ---
        if os.getenv("VLLM_MOE_COLLECT_PER_EXPERT_STATS", "0") != "1":
            return

        # topk_ids shape: [num_tokens, top_k], hidden_states shape: [num_tokens, hidden_size]
        # For down_proj stage, hidden_states may be [num_tokens * top_k, hidden_size]
        num_tokens_hs = hidden_states.shape[0]
        num_tokens_topk = topk_ids.shape[0]
        top_k = topk_ids.shape[1]

        if num_tokens_hs == num_tokens_topk:
            # gate_up_proj: hidden_states is [num_tokens, hidden_size]
            # Each token may be assigned to multiple experts, use the same hidden_state for each
            flat_expert_ids = topk_ids.reshape(-1)  # [num_tokens * top_k]
            flat_hidden = (
                hidden_states.unsqueeze(1)
                .expand(-1, top_k, -1)
                .reshape(-1, hidden_states.shape[-1])
            )  # [num_tokens * top_k, hidden_size]
        elif num_tokens_hs == num_tokens_topk * top_k:
            # down_proj: hidden_states is [num_tokens * top_k, hidden_size]
            flat_expert_ids = topk_ids.reshape(-1)  # [num_tokens * top_k]
            flat_hidden = hidden_states  # already [num_tokens * top_k, hidden_size]
        else:
            # Fallback: skip per-expert stats if shape doesn't match
            if verbose:
                print(
                    f"[VERBOSE] Rank {rank}/{world_size}: Skipping per-expert "
                    f"stats for {key}, shape mismatch: "
                    f"hidden_states={hidden_states.shape}, topk_ids={topk_ids.shape}"
                )
            return

        # Iterate over each unique expert in the current batch
        unique_experts = flat_expert_ids.unique()
        for expert_id_tensor in unique_experts:
            expert_id = expert_id_tensor.item()
            if expert_id < 0:
                continue  # Skip invalid expert ids (e.g., -1 padding)
            expert_key = f"{layer_name}.{expert_id}.{stage}"
            if expert_key not in global_moe_activation_stats:
                # Dynamically create entry if not pre-allocated
                global_moe_activation_stats[expert_key] = {
                    "min": torch.tensor(float("inf")),
                    "max": torch.tensor(float("-inf")),
                }
            expert_stats = global_moe_activation_stats[expert_key]
            mask = flat_expert_ids == expert_id_tensor
            expert_hidden = flat_hidden[mask]
            if expert_hidden.numel() == 0:
                continue
            e_min = expert_hidden.min().detach().cpu()
            e_max = expert_hidden.max().detach().cpu()
            expert_stats["min"] = torch.minimum(expert_stats["min"], e_min)
            expert_stats["max"] = torch.maximum(expert_stats["max"], e_max)
            expert_stats["call_count"] = expert_stats.get("call_count", 0) + 1
            if verbose:
                print(
                    f"[VERBOSE] Rank {rank}/{world_size}: Expert {expert_id} "
                    f"stats for {key}, min: {e_min.item()}, max: {e_max.item()}"
                )


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


def get_moe_stats(model):
    """
    Retrieve moe statistics from the model.
    Performs all-reduce across all workers to get global min/max.
    """
    if not hasattr(model, "_moe_activation_stats"):
        return None

    # Perform all-reduce to get global min/max across all workers
    try:
        _all_reduce_stats(model._moe_activation_stats, stats_type="MoE")
    except Exception as e:
        print(f"Warning: Could not perform all-reduce: {e}")

    # Convert tensors to Python scalars for easier use
    stats_dict = {}
    for name, stats in model._moe_activation_stats.items():
        stats_dict[name] = {
            "min": stats["min"].item() if isinstance(stats["min"], torch.Tensor) else stats["min"],
            "max": stats["max"].item() if isinstance(stats["max"], torch.Tensor) else stats["max"],
        }
    return stats_dict


def print_moe_stats(model, verbose=False):
    """
    Print MoE activation statistics in a readable format.
    Performs all-reduce to get global statistics across all workers.

    Args:
        model: The model containing MoE activation statistics
        verbose: If True, print detailed debug information during all-reduce
    """
    if not hasattr(model, "_moe_activation_stats"):
        print("No MoE activation statistics available")
        return

    # Perform all-reduce to get global min/max
    try:
        rank, world_size = _all_reduce_stats(
            model._moe_activation_stats, stats_type="MoE", verbose=verbose
        )
    except Exception as e:
        print(f"Warning: Could not perform all-reduce: {e}")
        rank, world_size = 0, 1

    # Only rank 0 prints the statistics (or single process)
    if rank != 0:
        return

    # Print statistics
    if world_size > 1:
        print(f"\n[Global statistics across {world_size} workers]")
    _print_stats_table(model._moe_activation_stats, "MoE gate_up and down Statistics")


# =============================================================================
# MTP (Multi-Token Prediction) draft model activation hooks
# =============================================================================


def setup_mtp_activation_hooks(draft_model):
    """
    Setup activation hooks on the MTP draft model to collect min/max statistics.
    This function should be applied via llm.apply_draft_model(setup_mtp_activation_hooks).

    The MTP draft model (e.g., HYV3MTP) is a separate nn.Module from the main model,
    containing its own Linear, Attention, and FusedMoE layers that need independent
    hook registration.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase

    try:
        # vLLM ≥ 0.20 (Tencent custom): Attention moved under model_executor.
        from vllm.model_executor.layers.attention import Attention
    except ImportError:
        # Older vLLM layout.
        from vllm.attention.layer import Attention

    # Find all linear layers in the MTP model
    layers_to_monitor = _find_layers(draft_model, layers=[torch.nn.Linear, LinearBase])
    kvcache_layers = _find_layers(draft_model, layers=[Attention])
    print(f"---------[MTP] Found {len(layers_to_monitor)} layers to monitor---------")
    print(f"---------[MTP] Found {len(kvcache_layers)} kv cache layers to monitor---------")
    for name in list(layers_to_monitor.keys())[:5]:
        print(f"  [MTP] {name}")
    if len(layers_to_monitor) > 5:
        print(f"  ... and {len(layers_to_monitor) - 5} more MTP activation layers")

    for name in list(kvcache_layers.keys())[:5]:
        print(f"  [MTP] {name}")
    if len(kvcache_layers) > 5:
        print(f"  ... and {len(kvcache_layers) - 5} more MTP kv cache layers")

    # Initialize activation statistics storage (prefix with "mtp." to distinguish)
    if not hasattr(draft_model, "_activation_stats"):
        draft_model._activation_stats = {}
        for name in layers_to_monitor.keys():
            draft_model._activation_stats[name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }

    if not hasattr(draft_model, "_kcache_stats"):
        draft_model._kcache_stats = {}
        draft_model._vcache_stats = {}
        for name in kvcache_layers.keys():
            draft_model._kcache_stats[name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }
            draft_model._vcache_stats[name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }

    # Register hooks for all linear layers
    if not hasattr(draft_model, "_activation_hooks"):
        draft_model._activation_hooks = []
        for name, layer in layers_to_monitor.items():
            hook = ActivationHook(name, draft_model._activation_stats)
            hook_handle = layer.register_forward_hook(hook)
            draft_model._activation_hooks.append(hook_handle)

    if not hasattr(draft_model, "_kvcache_hooks"):
        draft_model._kvcache_hooks = []
        for name, layer in kvcache_layers.items():
            hook = KVCacheHook(name, draft_model._kcache_stats, draft_model._vcache_stats)
            hook_handle = layer.register_forward_hook(hook)
            draft_model._kvcache_hooks.append(hook_handle)

    # Register MoE statistics storage and hooks
    moe_layers = _find_layers(draft_model, layers=[FusedMoE])
    if moe_layers:
        print(f"---------[MTP] Found {len(moe_layers)} MoE layers to monitor---------")
        for name in list(moe_layers.keys())[:5]:
            print(f"  [MTP] {name}")
        if len(moe_layers) > 5:
            print(f"  ... and {len(moe_layers) - 5} more")

        per_expert = os.getenv("VLLM_MOE_COLLECT_PER_EXPERT_STATS", "0") == "1"

        if not hasattr(draft_model, "_moe_activation_stats"):
            draft_model._moe_activation_stats = {}
            for name, layer in moe_layers.items():
                num_experts = getattr(layer, "global_num_experts", None)
                if num_experts is None:
                    num_experts = getattr(layer, "num_experts", 256)

                for stage in ["gate_up_proj", "down_proj"]:
                    draft_model._moe_activation_stats[f"{name}.{stage}"] = {
                        "min": torch.tensor(float("inf")),
                        "max": torch.tensor(float("-inf")),
                    }
                    if per_expert:
                        for expert_id in range(num_experts):
                            draft_model._moe_activation_stats[f"{name}.{expert_id}.{stage}"] = {
                                "min": torch.tensor(float("inf")),
                                "max": torch.tensor(float("-inf")),
                            }

                if hasattr(layer, "w13_weight") and layer.w13_weight is not None:
                    layer.w13_weight._vllm_layer_name = name
                    layer.w13_weight._moe_activation_stats_of_model = (
                        draft_model._moe_activation_stats
                    )

    print("---------[MTP] Activation hooks registered---------")
    return f"[MTP] Registered {len(draft_model._activation_hooks)} hooks"


def get_mtp_activation_stats(draft_model):
    """
    Retrieve activation statistics from the MTP draft model.
    Performs all-reduce across all workers to get global min/max.
    """
    return get_activation_stats(draft_model)


def print_mtp_activation_stats(draft_model):
    """Print MTP draft model activation statistics."""
    if not hasattr(draft_model, "_activation_stats"):
        print("[MTP] No activation statistics available")
        return
    print_activation_stats(draft_model)


def get_mtp_moe_stats(draft_model):
    """Retrieve MoE statistics from the MTP draft model."""
    return get_moe_stats(draft_model)


def print_mtp_moe_stats(draft_model, verbose=False):
    """Print MTP draft model MoE statistics."""
    if not hasattr(draft_model, "_moe_activation_stats"):
        print("[MTP] No MoE activation statistics available")
        return
    print_moe_stats(draft_model, verbose=verbose)


# KV Cache Scale Search
# =============================================================================


class KVCacheValueHook:
    """
    Hook that captures the raw (BF16) k/v tensors entering vllm Attention,
    so we can compute MSE between the original and FP8-quantized kv cache.
    Stores a *list* of tensors – one entry per calibration forward pass.
    """

    def __init__(self, layer_name: str, kvcache_values: dict):
        self.layer_name = layer_name
        self.kvcache_values = kvcache_values

    def __call__(self, module, input, output):
        # input to vllm Attention: (q, k, v, ...)
        _, k, v = input[0], input[1], input[2]
        with torch.no_grad():
            self.kvcache_values[self.layer_name]["k"].append(k.detach().cpu())
            self.kvcache_values[self.layer_name]["v"].append(v.detach().cpu())


def _setup_kvcache_value_hooks(model):
    """
    Register hooks that collect raw k/v tensors for scale-search calibration.
    Called inside a worker via llm.apply_model().
    Returns the number of hooks registered.
    """
    try:
        # vLLM ≥ 0.20 (Tencent custom): Attention moved under model_executor.
        from vllm.model_executor.layers.attention import Attention
    except ImportError:
        # Older vLLM layout.
        from vllm.attention.layer import Attention

    kvcache_layers = _find_layers(model, layers=[Attention])

    if not hasattr(model, "_kvcache_search_values"):
        model._kvcache_search_values = {}
        for name in kvcache_layers:
            model._kvcache_search_values[name] = {"k": [], "v": []}

    if not hasattr(model, "_kvcache_search_hooks"):
        model._kvcache_search_hooks = []
        for name, layer in kvcache_layers.items():
            hook = KVCacheValueHook(name, model._kvcache_search_values)
            handle = layer.register_forward_hook(hook)
            model._kvcache_search_hooks.append(handle)

    return f"Registered {len(model._kvcache_search_hooks)} kv-search hooks"


def _get_kv_search_values(model):
    """
    Retrieve collected raw k/v tensors from a worker.
    Returns {layer_name: {"k": [tensor, ...], "v": [tensor, ...]}}
    """
    if not hasattr(model, "_kvcache_search_values"):
        return {}
    return model._kvcache_search_values


def _remove_kvcache_value_hooks(model):
    """Remove kv-value hooks from the model (clean up after search)."""
    if hasattr(model, "_kvcache_search_hooks"):
        for h in model._kvcache_search_hooks:
            h.remove()
        del model._kvcache_search_hooks
    if hasattr(model, "_kvcache_search_values"):
        del model._kvcache_search_values


def _fp8_quantize_dequant(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Simulate FP8 KV-cache quantization/dequantization on a BF16 tensor.

    The vllm convention for kv_cache_scales is:
        scale  =  abs_max / fp8_max
    so quantization is:
        q = clamp(tensor / scale, fp8_min, fp8_max).to(fp8)
        dequant = q.to(bf16) * scale

    Args:
        tensor: BF16 (or float32) tensor.
        scale:  per-tensor scale (positive float).

    Returns:
        Dequantized tensor, same dtype as input.
    """
    orig_dtype = tensor.dtype
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    fp8_min = torch.finfo(torch.float8_e4m3fn).min  # -448.0

    t = tensor.float()
    q = (t / scale).clamp(fp8_min, fp8_max).to(torch.float8_e4m3fn)
    dq = q.to(torch.float32) * scale
    return dq.to(orig_dtype)


def _mse_fp8_kv(kv_tensors: list, scale: float) -> float:
    """
    Compute average MSE between original and FP8-quantised kv tensors
    for a given scale value.

    Args:
        kv_tensors: list of raw (BF16) tensors.
        scale:      candidate per-tensor scale.

    Returns:
        Mean MSE (scalar float).
    """
    total_mse = 0.0
    total_numel = 0
    for t in kv_tensors:
        t_dq = _fp8_quantize_dequant(t, scale)
        mse = ((t.float() - t_dq.float()) ** 2).mean().item()
        total_mse += mse * t.numel()
        total_numel += t.numel()
    if total_numel == 0:
        return float("inf")
    return total_mse / total_numel


def _search_best_multiplier_flat(
    flat: torch.Tensor,
    base_scale: float,
    min_multiplier: float = 0.8,
    max_multiplier: float = 16.0,
    num_steps: int = 100,
) -> float:
    """
    Grid search for the multiplier `m` that minimises FP8 quantisation MSE.

    Accepts a pre-concatenated flat float32 tensor.  When ``flat`` lives on a
    CUDA device the search is executed entirely on GPU:

    * All FP8 clamp / cast / MSE operations run as CUDA kernels.
    * MSE values are accumulated into a GPU tensor; only a single
      ``argmin().item()`` sync is issued at the very end (vs. one sync per
      step in the CPU path).

    On CPU the original sequential loop is used (unchanged behaviour).

    Args:
        flat:            float32 tensor of shape (N,).  May be on CPU or CUDA.
        base_scale:      the scale derived from calibration.
        min_multiplier:  lower bound of search range.
        max_multiplier:  upper bound of search range.
        num_steps:       number of grid points.

    Returns:
        Best multiplier (float).
    """
    import math

    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    fp8_min = torch.finfo(torch.float8_e4m3fn).min  # -448.0

    log_min = math.log(min_multiplier)
    log_max = math.log(max_multiplier)

    if flat.is_cuda:
        # GPU path: accumulate MSE values without per-step CPU sync.
        # A single argmin().item() at the end gives one CUDA synchronisation
        # point, which is far cheaper than num_steps synchronisations.
        mse_vals = torch.empty(num_steps, dtype=torch.float32, device=flat.device)
        for i in range(num_steps):
            m = math.exp(log_min + (log_max - log_min) * i / (num_steps - 1))
            scale = base_scale * m
            q_fp8 = (
                (flat / scale).clamp(fp8_min, fp8_max).to(torch.float8_e4m3fn).to(torch.float32)
            )
            mse_vals[i] = ((flat - q_fp8 * scale) ** 2).mean()
        best_idx = int(mse_vals.argmin().item())  # single sync
        return math.exp(log_min + (log_max - log_min) * best_idx / (num_steps - 1))
    else:
        # CPU path (original behaviour – kept for fallback / compatibility).
        best_m = 1.0
        best_mse = float("inf")
        for i in range(num_steps):
            m = math.exp(log_min + (log_max - log_min) * i / (num_steps - 1))
            scale = base_scale * m
            q_fp8 = (
                (flat / scale).clamp(fp8_min, fp8_max).to(torch.float8_e4m3fn).to(torch.float32)
            )
            mse = ((flat - q_fp8 * scale) ** 2).mean().item()
            if mse < best_mse:
                best_mse = mse
                best_m = m
        return best_m


def _search_best_multiplier(
    kv_tensors: list,
    base_scale: float,
    min_multiplier: float = 0.8,
    max_multiplier: float = 16.0,
    num_steps: int = 100,
) -> float:
    """Convenience wrapper – concatenates tensors then delegates to the flat variant."""
    if not kv_tensors:
        return 1.0
    flat = torch.cat([t.reshape(-1).float() for t in kv_tensors])
    return _search_best_multiplier_flat(
        flat, base_scale, min_multiplier, max_multiplier, num_steps
    )


class KVScaleSearcher:
    """
    Callable (for use with ``llm.apply_model``) that runs the full per-layer
    KV-cache scale search **inside each vLLM worker**.

    Typical usage (on the driver):

    .. code-block:: python

        searcher = KVScaleSearcher(
            activation_stats=stats_dict,       # from get_activation_stats()
            min_multiplier=0.8,
            max_multiplier=16.0,
            num_steps=100,
        )
        results_list = llm.apply_model(searcher)
        multipliers = results_list[0]          # rank-0 worker result

    The ``activation_stats`` dict is expected to contain entries like::

        "model.layers.0.self_attn.attn.k_cache": {"min": ..., "max": ...}
        "model.layers.0.self_attn.attn.v_cache": {"min": ..., "max": ...}

    which follow the format produced by ``get_activation_stats()``.
    """

    def __init__(
        self,
        activation_stats: dict,
        min_multiplier: float = 0.8,
        max_multiplier: float = 16.0,
        num_steps: int = 100,
    ):
        self.activation_stats = activation_stats
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.num_steps = num_steps

    def __call__(self, model):
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0

        # Collect raw kv tensors stored by the value hook
        kv_values = _get_kv_search_values(model)
        if not kv_values:
            print(
                "[KVScaleSearcher] WARNING: No kv values collected. "
                "Did you call setup_kvcache_value_hooks before inference?"
            )
            return {}

        # Decide whether to run the grid search on GPU.
        # Strategy: concatenate on CPU (avoids any GPU OOM during cat), then
        # move the flat tensor to GPU for the actual compute.  GPU memory for
        # one flat tensor is small (e.g. 64 samples × 4096 tokens × 128 dims
        # × 4 bytes ≈ 128 MB) and is freed immediately after each layer's
        # search, so only one flat tensor lives on GPU at a time.
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            # Pick the GPU that this worker owns (rank 0 of local group).
            # torch.cuda.current_device() is correct inside apply_model workers.
            search_device = torch.device("cuda", torch.cuda.current_device())
        else:
            search_device = torch.device("cpu")

        # Build a flat list of (stats_key, flat_tensor, base_scale) work items.
        # IMPORTANT: torch.cat is called here in the main thread, NOT inside the
        # worker threads.  Concurrent large tensor allocations in multiple threads
        # trigger page-fault storms (threads pile up in D state on __do_page_fault
        # / rwsem_down_write_slowpath), causing multi-minute stalls every few layers.
        # Pre-allocating all flat tensors serially before spawning threads avoids this.
        work_items = []
        for layer_name, tensors_dict in kv_values.items():
            for kv_slot, tensors in tensors_dict.items():
                stats_key = f"{layer_name}.{kv_slot}_cache"
                if stats_key not in self.activation_stats:
                    print(
                        f"[KVScaleSearcher] WARNING: {stats_key} not found in "
                        f"activation_stats, skipping."
                    )
                    continue
                if not tensors:
                    print(f"[KVScaleSearcher] WARNING: No tensors for {stats_key}, skipping.")
                    continue
                stats = self.activation_stats[stats_key]
                abs_max = max(abs(stats["min"]), abs(stats["max"]))
                base_scale = abs_max / fp8_max * 2.0
                # Pre-concatenate on CPU (serial, no page-fault contention).
                flat_cpu = torch.cat([t.reshape(-1).float() for t in tensors])
                work_items.append((stats_key, flat_cpu, base_scale))

        if use_gpu:
            print(
                f"[KVScaleSearcher] Running grid search on {search_device} "
                f"({len(work_items)} work items, {self.num_steps} steps each)"
            )
        else:
            print(
                f"[KVScaleSearcher] Running grid search on CPU " f"({len(work_items)} work items)"
            )

        multipliers = {}

        def _search_one(stats_key, flat_cpu, base_scale):
            if use_gpu:
                # Move to GPU just for the compute; free immediately after.
                flat = flat_cpu.to(search_device, non_blocking=True)
            else:
                flat = flat_cpu
            best_m = _search_best_multiplier_flat(
                flat=flat,
                base_scale=base_scale,
                min_multiplier=self.min_multiplier,
                max_multiplier=self.max_multiplier,
                num_steps=self.num_steps,
            )
            if use_gpu:
                del flat  # release GPU memory before the next layer is scheduled
            return stats_key, best_m, base_scale

        # GPU compute is already parallelised inside CUDA, so running layers
        # sequentially on a single GPU avoids memory pressure while still
        # benefiting from GPU throughput.  On CPU we keep the thread pool so
        # multiple cores are used.
        if use_gpu:
            for key, flat_cpu, base_scale in work_items:
                stats_key, best_m, bs = _search_one(key, flat_cpu, base_scale)
                multipliers[stats_key] = best_m
                print(
                    f"[KVScaleSearcher] {stats_key}: best_multiplier={best_m:.4f} "
                    f"(base_scale={bs:.6f})"
                )
        else:
            # CPU path: use threads so multiple cores are utilised.
            num_workers = min(len(work_items), os.cpu_count() or 4, 8)
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = {
                    pool.submit(_search_one, key, tensors, scale): key
                    for key, tensors, scale in work_items
                }
                for fut in as_completed(futures):
                    stats_key, best_m, base_scale = fut.result()
                    multipliers[stats_key] = best_m
                    print(
                        f"[KVScaleSearcher] {stats_key}: best_multiplier={best_m:.4f} "
                        f"(base_scale={base_scale:.6f})"
                    )

        return multipliers


def get_kv_scale_search_results(results_list: list) -> dict:
    """
    Extract the multiplier dict from the list returned by ``llm.apply_model``.
    Takes the result from rank-0 worker.
    """
    if not results_list:
        return {}
    first = results_list[0]
    if first is None:
        return {}
    return first


def remove_kv_scale_search_hooks(model):
    """
    Clean up kv-value hooks after search.  Pass to ``llm.apply_model``.
    """
    _remove_kvcache_value_hooks(model)
    return "KV-search hooks removed"


# Public alias so callers can do:
#   from angelslim.compressor.quant import setup_kvcache_value_hooks
setup_kvcache_value_hooks = _setup_kvcache_value_hooks


# =============================================================================
# KV-Cache Only Calibration
# (no weight / activation / MoE hooks – faster startup and lower memory)
# =============================================================================


def setup_kvcache_only_hooks(model):
    """
    Register **only** kv-cache min/max statistic hooks on vLLM Attention layers.

    This is a lightweight alternative to ``setup_activation_hooks`` for the
    use-case where you only need kv-cache scales (e.g. fp8 kv-cache without
    weight quantisation).  No linear-layer or MoE hooks are registered, so
    the memory footprint and hook overhead are minimal.

    Designed to be passed directly to ``llm.apply_model()``.

    Returns:
        A human-readable status string (e.g. "Registered 80 kv-only hooks").
    """
    try:
        # vLLM ≥ 0.20 (Tencent custom): Attention moved under model_executor.
        from vllm.model_executor.layers.attention import Attention
    except ImportError:
        # Older vLLM layout.
        from vllm.attention.layer import Attention

    kvcache_layers = _find_layers(model, layers=[Attention])
    print(f"[KVOnly] Found {len(kvcache_layers)} Attention layers to monitor")

    if not hasattr(model, "_kvcache_only_stats"):
        model._kvcache_only_stats = {"k": {}, "v": {}}
        for name in kvcache_layers:
            model._kvcache_only_stats["k"][name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }
            model._kvcache_only_stats["v"][name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }

    if not hasattr(model, "_kvcache_only_hooks"):
        model._kvcache_only_hooks = []
        for name, layer in kvcache_layers.items():
            hook = KVCacheHook(
                name,
                model._kvcache_only_stats["k"],
                model._kvcache_only_stats["v"],
            )
            handle = layer.register_forward_hook(hook)
            model._kvcache_only_hooks.append(handle)

    return f"Registered {len(model._kvcache_only_hooks)} kv-only hooks"


def get_kvcache_only_stats(model):
    """
    Retrieve kv-cache min/max statistics collected by ``setup_kvcache_only_hooks``.

    Performs an all-reduce across workers so every rank holds the global
    min/max before rank-0 returns the final dict.

    Returns:
        dict with keys like ``"model.layers.0.self_attn.attn.k_cache"`` and
        values ``{"min": float, "max": float}``, matching the format produced
        by ``get_activation_stats()`` so that ``KVScaleSearcher`` can consume
        it directly.  Returns ``None`` if no stats are available.
    """
    if not hasattr(model, "_kvcache_only_stats"):
        return None

    try:
        _all_reduce_stats(model._kvcache_only_stats["k"], stats_type="kv-only-k")
        _all_reduce_stats(model._kvcache_only_stats["v"], stats_type="kv-only-v")
    except Exception as e:
        print(f"[KVOnly] Warning: Could not perform all-reduce: {e}")

    stats_dict = {}
    for name, stats in model._kvcache_only_stats["k"].items():
        stats_dict[f"{name}.k_cache"] = {
            "min": stats["min"].item() if isinstance(stats["min"], torch.Tensor) else stats["min"],
            "max": stats["max"].item() if isinstance(stats["max"], torch.Tensor) else stats["max"],
        }
    for name, stats in model._kvcache_only_stats["v"].items():
        stats_dict[f"{name}.v_cache"] = {
            "min": stats["min"].item() if isinstance(stats["min"], torch.Tensor) else stats["min"],
            "max": stats["max"].item() if isinstance(stats["max"], torch.Tensor) else stats["max"],
        }
    return stats_dict


def print_kvcache_only_stats(model):
    """
    Print kv-cache-only statistics.  Only rank-0 prints.
    Designed to be passed to ``llm.apply_model()``.
    """
    if not hasattr(model, "_kvcache_only_stats"):
        print("[KVOnly] No kv-only statistics available.")
        return

    try:
        rank, world_size = _all_reduce_stats(
            model._kvcache_only_stats["k"], stats_type="kv-only-k"
        )
        _all_reduce_stats(model._kvcache_only_stats["v"], stats_type="kv-only-v")
    except Exception as e:
        print(f"[KVOnly] Warning: Could not perform all-reduce: {e}")
        rank, world_size = 0, 1

    if rank != 0:
        return

    if world_size > 1:
        print(f"\n[KVOnly] Global statistics across {world_size} workers")
    _print_stats_table(model._kvcache_only_stats["k"], "KV-Only K-cache Statistics")
    _print_stats_table(model._kvcache_only_stats["v"], "KV-Only V-cache Statistics")


def remove_kvcache_only_hooks(model):
    """
    Remove hooks registered by ``setup_kvcache_only_hooks``.
    Designed to be passed to ``llm.apply_model()``.
    """
    if hasattr(model, "_kvcache_only_hooks"):
        for h in model._kvcache_only_hooks:
            h.remove()
        del model._kvcache_only_hooks
    if hasattr(model, "_kvcache_only_stats"):
        del model._kvcache_only_stats
    return "KV-only hooks removed"


# =============================================================================
# Per-Head KV-Cache Calibration
# (one scale per attention head per layer instead of one per layer)
# =============================================================================


def _get_num_heads_from_tensor(k: torch.Tensor, module) -> int:
    """
    Infer the number of KV heads from the tensor and (optionally) the module.

    vLLM passes k with shape  (total_tokens, num_kv_heads * head_dim)  for the
    PagedAttention path, or sometimes (batch, seq, num_kv_heads, head_dim).
    We try the module attribute first, then fall back to shape inspection.
    """
    # 1) Try to read directly from the Attention implementation object.
    for attr in ("num_kv_heads", "num_heads"):
        if hasattr(module, attr):
            val = getattr(module, attr)
            if isinstance(val, int) and val > 0:
                return val
        impl = getattr(module, "impl", None) or getattr(module, "attn", None)
        if impl is not None and hasattr(impl, attr):
            val = getattr(impl, attr)
            if isinstance(val, int) and val > 0:
                return val

    # 2) Fall back: if k is 4-D the head dimension is dim[-2].
    if k.ndim == 4:
        return k.shape[-2]

    # 3) Cannot determine – treat as single head (per-tensor).
    return 1


def _infer_num_kv_heads_total(model) -> int | None:
    """
    Try hard to discover the model's **global** (pre-TP-replication) number
    of KV heads by probing common config locations.  Returns ``None`` when
    the value cannot be determined – callers should then fall back to
    ``H_local * world_size`` which is correct when there is no TP
    replication.

    Probed locations (first hit wins):
        1. ``model.config.num_key_value_heads`` (HuggingFace)
        2. ``model.model_config.num_key_value_heads``
        3. ``model.config.num_kv_heads``
        4. ``model.hf_config.num_key_value_heads``
        5. ``model.config.text_config.num_key_value_heads`` (VLM with nested)
    """
    candidates = [
        ("config", "num_key_value_heads"),
        ("config", "num_kv_heads"),
        ("model_config", "num_key_value_heads"),
        ("model_config", "num_kv_heads"),
        ("hf_config", "num_key_value_heads"),
        ("hf_config", "num_kv_heads"),
    ]
    for cfg_attr, val_attr in candidates:
        cfg = getattr(model, cfg_attr, None)
        if cfg is None:
            continue
        if hasattr(cfg, val_attr):
            v = getattr(cfg, val_attr)
            if isinstance(v, int) and v > 0:
                return v
        # Try nested text_config (used by some VLMs).
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, val_attr):
            v = getattr(text_cfg, val_attr)
            if isinstance(v, int) and v > 0:
                return v
    return None


class KVCachePerHeadHook:
    """
    Forward hook on vLLM ``Attention`` layers that tracks per-head min/max
    statistics for both K and V tensors.

    The hook reshapes k/v from ``(T, num_heads * head_dim)`` (or any layout
    that contains a head axis) to ``(T, num_heads, head_dim)``, then reduces
    over the token and head-dim axes so the result has shape ``(num_heads,)``.

    Stats are stored as 1-D tensors so the existing ``_all_reduce_stats``
    (which calls ``dist.all_reduce`` on arbitrary-shape tensors) works
    transparently.
    """

    def __init__(
        self,
        layer_name: str,
        kcache_stats: dict,
        vcache_stats: dict,
        num_kv_heads_total: int | None = None,
    ):
        self.layer_name = layer_name
        self.kcache_stats = kcache_stats
        self.vcache_stats = vcache_stats
        self._num_heads: int | None = None  # resolved on first call
        # Total KV-head count across all TP ranks (before replication).
        # Passed in from ``setup_kvcache_perhead_hooks``; used to detect
        # the replicated-TP case correctly (see _get_kv_role).
        self._num_kv_heads_total: int | None = num_kv_heads_total
        # Role determines whether this rank tracks K, V or both.  Resolved
        # lazily on first call once we know the real head count, so we can
        # detect TP-replication (world_size > num_kv_heads_total).
        self._role: str | None = None

    def __call__(self, module, input, output):
        _, k, v = input[0], input[1], input[2]
        if not isinstance(k, torch.Tensor):
            return

        with torch.no_grad():
            # Resolve head count on first call and cache it.
            if self._num_heads is None:
                self._num_heads = _get_num_heads_from_tensor(k, module)

            # Resolve K/V role once.  Prefer the externally-provided
            # ``num_kv_heads_total`` (known accurately from model config);
            # fall back to ``H_local * world_size`` only if unset.
            if self._role is None:
                rank, world_size = _get_dist_info()
                if self._num_kv_heads_total is not None and self._num_kv_heads_total > 0:
                    num_kv_heads_total = self._num_kv_heads_total
                else:
                    num_kv_heads_total = self._num_heads * world_size
                self._role = _get_kv_role(rank, world_size, num_kv_heads_total)

            H = self._num_heads

            def _per_head_minmax(t: torch.Tensor):
                """Return (min_vec, max_vec) of shape (num_heads,)."""
                # t shape: (..., H * head_dim)  OR  (..., H, head_dim)
                if t.ndim >= 3 and t.shape[-2] == H:
                    # Already in (..., H, head_dim) layout
                    t_heads = t.reshape(-1, H, t.shape[-1])  # (T, H, D)
                    actual_H = H
                else:
                    # Assume last dim = H * head_dim
                    last = t.shape[-1]
                    if H > 0 and last % H == 0:
                        head_dim = last // H
                        t_heads = t.reshape(-1, H, head_dim)  # (T, H, D)
                        actual_H = H
                    else:
                        # Cannot reshape cleanly; treat as single pseudo-head
                        t_heads = t.reshape(1, -1, 1)  # (1, N, 1)
                        actual_H = 1

                # Reduce over (T, D) → shape (actual_H,)
                t_flat = t_heads.reshape(t_heads.shape[0], actual_H, -1).float()  # (T, H, D)
                h_min = t_flat.min(dim=0).values.min(dim=-1).values  # (H,)
                h_max = t_flat.max(dim=0).values.max(dim=-1).values  # (H,)
                return h_min.cpu(), h_max.cpu()

            # Role-based selective computation:
            # - "both": compute K and V (default / no-replication / single-GPU)
            # - "k"   : compute K only, skip V
            # - "v"   : compute V only, skip K
            if self._role in ("both", "k"):
                k_min, k_max = _per_head_minmax(k)
                k_stats = self.kcache_stats[self.layer_name]
                if k_stats["min"].shape != k_min.shape:
                    # First call – initialise running stats to match head count.
                    k_stats["min"] = k_min.clone()
                    k_stats["max"] = k_max.clone()
                else:
                    k_stats["min"] = torch.minimum(k_stats["min"], k_min)
                    k_stats["max"] = torch.maximum(k_stats["max"], k_max)

            if self._role in ("both", "v"):
                v_min, v_max = _per_head_minmax(v)
                v_stats = self.vcache_stats[self.layer_name]
                if v_stats["min"].shape != v_min.shape:
                    v_stats["min"] = v_min.clone()
                    v_stats["max"] = v_max.clone()
                else:
                    v_stats["min"] = torch.minimum(v_stats["min"], v_min)
                    v_stats["max"] = torch.maximum(v_stats["max"], v_max)


def _all_gather_stats_perhead(
    stats_dict, stats_type="statistics", num_kv_heads_total: int | None = None
):
    """
    Collect per-head KV stats from all TP workers into a full global
    ``(num_kv_heads_total,)`` vector on every rank.

    Two modes are supported transparently:

    1. **No K/V split** (``replication < 2``): each rank owns ``H_local``
       distinct KV heads and holds valid min/max for them.  We scatter
       each rank's slice into its owned head range and do a global
       min/max all_reduce.

    2. **K/V split** (``replication >= 2``, the efficient path for GQA
       with ``tp_size > num_kv_heads``): within each replication group
       only one rank holds valid K stats and another holds valid V stats.
       ``stats_type`` must end with ``"-k"`` or ``"-v"`` so we know which
       side this dict represents.  Ranks that don't own this side
       contribute neutral values (``+inf`` for min, ``-inf`` for max) and
       the opposite-side rank's valid slice wins through the reduce.

    Args:
        stats_dict: per-layer dict of ``{"min": tensor, "max": tensor}``
        stats_type: string label; suffix ``"-k"`` or ``"-v"`` selects role
        num_kv_heads_total: the model's global KV head count.  If ``None``
            we fall back to ``H_local * world_size`` (correct only when
            there is no replication).

    Falls back gracefully to no-op when ``world_size == 1``.

    **Idempotent**: uses a ``_gathered`` flag on ``stats_dict`` to ensure
    the reduce is performed at most once.
    """
    import torch.distributed as dist

    rank, world_size = _get_dist_info()
    if world_size <= 1:
        return rank, world_size

    # Idempotency guard.
    _gathered_key = f"__gathered_{stats_type}__"
    if stats_dict.get(_gathered_key, False):
        if rank == 0:
            print(f"Per-head {stats_type} all-reduce already done, skipping.")
        return rank, world_size

    # Determine which side this dict represents from stats_type suffix.
    side = None
    if stats_type.endswith("-k"):
        side = "k"
    elif stats_type.endswith("-v"):
        side = "v"

    # Probe H_local from this rank's first non-sentinel layer tensor.
    local_H_local = 0
    for name, stats in stats_dict.items():
        if name.startswith("__gathered_"):
            continue
        t = stats["min"]
        if (
            isinstance(t, torch.Tensor)
            and t.ndim == 1
            and t.numel() >= 1
            and not (t.numel() == 1 and torch.isinf(t).all())
        ):
            local_H_local = t.numel()
            break

    # Agree on H_local across all ranks (MAX handles ranks that didn't
    # collect this side under the K/V-split scheme).
    h_local_tensor = torch.tensor([local_H_local], dtype=torch.long, device="cuda")
    dist.all_reduce(h_local_tensor, op=dist.ReduceOp.MAX)
    H_local = int(h_local_tensor.item())
    if H_local <= 0:
        stats_dict[_gathered_key] = True
        return rank, world_size

    # Derive num_kv_heads_total & replication.
    if num_kv_heads_total is None or num_kv_heads_total <= 0:
        # No info: assume no replication.
        num_kv_heads_total = H_local * world_size
        replication = 1
    else:
        replication = (
            max(1, world_size // num_kv_heads_total)
            if num_kv_heads_total > 0 and world_size % num_kv_heads_total == 0
            else 1
        )

    # Decide role & layout for this rank.
    role, heads_per_rank, global_head_offset, _ = _compute_perhead_layout(
        rank, world_size, num_kv_heads_total
    )

    have_data = (role == "both") or (side is None) or (role == side)

    if rank == 0:
        print(
            f"Performing per-head {stats_type} all-reduce across {world_size} workers "
            f"(H_local={H_local}, H_global={num_kv_heads_total}, "
            f"replication={replication}, side={side}, role={role})..."
        )

    for name, stats in stats_dict.items():
        if name.startswith("__gathered_"):  # skip sentinel keys
            continue
        for key in ["min", "max"]:
            neutral = float("inf") if key == "min" else float("-inf")
            full = torch.full((num_kv_heads_total,), neutral, dtype=torch.float32, device="cuda")
            if have_data:
                t = stats[key]
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t, dtype=torch.float32)
                t_gpu = t.to(device="cuda", dtype=torch.float32)
                if t_gpu.numel() == H_local:
                    end = global_head_offset + H_local
                    if end <= num_kv_heads_total:
                        full[global_head_offset:end] = t_gpu
                del t_gpu

            op = dist.ReduceOp.MIN if key == "min" else dist.ReduceOp.MAX
            dist.all_reduce(full, op=op)

            stats[key] = full.cpu()
        torch.cuda.empty_cache()

    dist.barrier()
    stats_dict[_gathered_key] = True
    if rank == 0:
        print(f"Per-head {stats_type} all-reduce completed.")
    return rank, world_size


def setup_kvcache_perhead_hooks(model):
    """
    Register per-head kv-cache min/max hooks on all vLLM Attention layers.

    Like ``setup_kvcache_only_hooks`` but collects one min/max per KV head
    instead of one per layer.  The stats tensors start as scalar ``inf`` /
    ``-inf`` and are replaced with ``(num_heads,)`` vectors on the first
    forward call (when the actual head count is known).

    Designed to be passed directly to ``llm.apply_model()``.
    """
    try:
        # vLLM ≥ 0.20 (Tencent custom): Attention moved under model_executor.
        from vllm.model_executor.layers.attention import Attention
    except ImportError:
        # Older vLLM layout.
        from vllm.attention.layer import Attention

    kvcache_layers = _find_layers(model, layers=[Attention])
    print(f"[KVPerHead] Found {len(kvcache_layers)} Attention layers to monitor")

    # Try to discover the model's total KV-head count so hooks can make an
    # informed K/V-split decision.  This is needed because inside the hook
    # we only see the *local* (per-TP-rank) head count, which under
    # replication (tp_size > num_kv_heads_total) is smaller than the true
    # global value.
    num_kv_heads_total = _infer_num_kv_heads_total(model)
    model._kvcache_num_kv_heads_total = num_kv_heads_total
    print(f"[KVPerHead] Inferred num_kv_heads_total = {num_kv_heads_total}")

    if not hasattr(model, "_kvcache_perhead_stats"):
        model._kvcache_perhead_stats = {"k": {}, "v": {}}
        for name in kvcache_layers:
            # Scalar sentinels – will be replaced by (H,) vectors on first hook call.
            model._kvcache_perhead_stats["k"][name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }
            model._kvcache_perhead_stats["v"][name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }

    if not hasattr(model, "_kvcache_perhead_hooks"):
        model._kvcache_perhead_hooks = []
        for name, layer in kvcache_layers.items():
            hook = KVCachePerHeadHook(
                name,
                model._kvcache_perhead_stats["k"],
                model._kvcache_perhead_stats["v"],
                num_kv_heads_total=num_kv_heads_total,
            )
            handle = layer.register_forward_hook(hook)
            model._kvcache_perhead_hooks.append(handle)

    return f"Registered {len(model._kvcache_perhead_hooks)} kv-perhead hooks"


def get_kvcache_perhead_stats(model):
    """
    Retrieve per-head kv-cache statistics.

    Returns a dict with keys like::

        "model.layers.0.self_attn.attn.k_cache"

    and values::

        {"min": [float, ...],   # list of length num_kv_heads
         "max": [float, ...]}

    This is intentionally different from the per-tensor format so that
    downstream tools can distinguish the two.  Returns ``None`` if no
    stats are available.
    """
    if not hasattr(model, "_kvcache_perhead_stats"):
        return None

    num_kv_heads_total = getattr(model, "_kvcache_num_kv_heads_total", None)
    try:
        _all_gather_stats_perhead(
            model._kvcache_perhead_stats["k"],
            stats_type="kv-perhead-k",
            num_kv_heads_total=num_kv_heads_total,
        )
        _all_gather_stats_perhead(
            model._kvcache_perhead_stats["v"],
            stats_type="kv-perhead-v",
            num_kv_heads_total=num_kv_heads_total,
        )
    except Exception as e:
        print(f"[KVPerHead] Warning: Could not perform all-gather: {e}")

    def _to_list(t):
        if isinstance(t, torch.Tensor):
            return t.tolist()
        return [float(t)]

    stats_dict = {}
    for name, stats in model._kvcache_perhead_stats["k"].items():
        if name.startswith("__gathered_"):  # skip sentinel keys
            continue
        stats_dict[f"{name}.k_cache"] = {
            "min": _to_list(stats["min"]),
            "max": _to_list(stats["max"]),
        }
    for name, stats in model._kvcache_perhead_stats["v"].items():
        if name.startswith("__gathered_"):  # skip sentinel keys
            continue
        stats_dict[f"{name}.v_cache"] = {
            "min": _to_list(stats["min"]),
            "max": _to_list(stats["max"]),
        }
    return stats_dict


def print_kvcache_perhead_stats(model):
    """
    Print per-head kv-cache statistics.  Only rank-0 prints.
    Designed to be passed to ``llm.apply_model()``.
    """
    if not hasattr(model, "_kvcache_perhead_stats"):
        print("[KVPerHead] No per-head kv statistics available.")
        return

    num_kv_heads_total = getattr(model, "_kvcache_num_kv_heads_total", None)
    try:
        rank, world_size = _all_gather_stats_perhead(
            model._kvcache_perhead_stats["k"],
            stats_type="kv-perhead-k",
            num_kv_heads_total=num_kv_heads_total,
        )
        _all_gather_stats_perhead(
            model._kvcache_perhead_stats["v"],
            stats_type="kv-perhead-v",
            num_kv_heads_total=num_kv_heads_total,
        )
    except Exception as e:
        print(f"[KVPerHead] Warning: Could not perform all-gather: {e}")
        rank, world_size = 0, 1

    if rank != 0:
        return

    if world_size > 1:
        print(f"\n[KVPerHead] Global statistics across {world_size} workers")

    def _fmt(stats, label):
        print(f"\n  === {label} ===")
        for name, s in stats.items():
            if name.startswith("__gathered_"):  # skip sentinel keys
                continue
            mn = s["min"]
            mx = s["max"]
            if isinstance(mn, torch.Tensor):
                mn_str = f"[{mn.min().item():.4f} .. {mn.max().item():.4f}] ({mn.numel()} heads)"
                mx_str = f"[{mx.min().item():.4f} .. {mx.max().item():.4f}]"
            else:
                mn_str = str(mn)
                mx_str = str(mx)
            print(f"    {name}: min={mn_str}  max={mx_str}")

    _fmt(model._kvcache_perhead_stats["k"], "Per-Head K-cache Statistics")
    _fmt(model._kvcache_perhead_stats["v"], "Per-Head V-cache Statistics")


def remove_kvcache_perhead_hooks(model):
    """
    Remove hooks registered by ``setup_kvcache_perhead_hooks``.
    Designed to be passed to ``llm.apply_model()``.
    """
    if hasattr(model, "_kvcache_perhead_hooks"):
        for h in model._kvcache_perhead_hooks:
            h.remove()
        del model._kvcache_perhead_hooks
    if hasattr(model, "_kvcache_perhead_stats"):
        del model._kvcache_perhead_stats
    return "KV-perhead hooks removed"


# ---------------------------------------------------------------------------
# Per-head KV value capture (for MSE-based scale search)
# ---------------------------------------------------------------------------


class KVCachePerHeadValueHook:
    """
    Like ``KVCacheValueHook`` but stores tensors in head-separated form.

    Stored shape per batch element: ``(num_heads, seq_len_local, head_dim)``
    so the scale searcher can work per-head without extra reshape overhead.
    """

    def __init__(
        self, layer_name: str, kvcache_values: dict, num_kv_heads_total: int | None = None
    ):
        self.layer_name = layer_name
        self.kvcache_values = kvcache_values
        self._num_heads: int | None = None
        self._num_kv_heads_total: int | None = num_kv_heads_total
        # Role determines whether this rank captures K tensors, V tensors
        # or both.  Resolved lazily on first call (see KVCachePerHeadHook).
        self._role: str | None = None

    def __call__(self, module, input, output):
        _, k, v = input[0], input[1], input[2]
        if not isinstance(k, torch.Tensor):
            return

        with torch.no_grad():
            if self._num_heads is None:
                self._num_heads = _get_num_heads_from_tensor(k, module)
            if self._role is None:
                rank, world_size = _get_dist_info()
                if self._num_kv_heads_total is not None and self._num_kv_heads_total > 0:
                    num_kv_heads_total = self._num_kv_heads_total
                else:
                    num_kv_heads_total = self._num_heads * world_size
                self._role = _get_kv_role(rank, world_size, num_kv_heads_total)
            H = self._num_heads

            def _to_head_layout(t: torch.Tensor):
                """Return tensor of shape (H, T, head_dim)."""
                last = t.shape[-1]
                if t.ndim >= 3 and t.shape[-2] == H:
                    # (..., H, D) → (H, T, D)
                    t_h = t.reshape(-1, H, t.shape[-1])  # (T, H, D)
                    return t_h.permute(1, 0, 2).contiguous()  # (H, T, D)
                elif H > 0 and last % H == 0:
                    head_dim = last // H
                    t_h = t.reshape(-1, H, head_dim)  # (T, H, D)
                    return t_h.permute(1, 0, 2).contiguous()  # (H, T, D)
                else:
                    # Fallback: single pseudo-head
                    return t.reshape(1, -1, 1)

            # Role-based selective capture: only store the side this rank
            # is responsible for (saves ~50% CPU memory on replicated TP).
            if self._role in ("both", "k"):
                k_heads = _to_head_layout(k).detach().cpu()  # (H, T, D)
                self.kvcache_values[self.layer_name]["k"].append(k_heads)
            if self._role in ("both", "v"):
                v_heads = _to_head_layout(v).detach().cpu()  # (H, T, D)
                self.kvcache_values[self.layer_name]["v"].append(v_heads)


def _setup_kvcache_perhead_value_hooks(model):
    """
    Register per-head value-capture hooks for scale-search calibration.
    Called inside a worker via llm.apply_model().
    """
    try:
        # vLLM ≥ 0.20 (Tencent custom): Attention moved under model_executor.
        from vllm.model_executor.layers.attention import Attention
    except ImportError:
        # Older vLLM layout.
        from vllm.attention.layer import Attention

    kvcache_layers = _find_layers(model, layers=[Attention])

    # Prefer the total KV head count saved by ``setup_kvcache_perhead_hooks``;
    # fall back to inferring it from the model config again.
    num_kv_heads_total = getattr(model, "_kvcache_num_kv_heads_total", None)
    if num_kv_heads_total is None:
        num_kv_heads_total = _infer_num_kv_heads_total(model)
        model._kvcache_num_kv_heads_total = num_kv_heads_total

    if not hasattr(model, "_kvcache_perhead_search_values"):
        model._kvcache_perhead_search_values = {}
        for name in kvcache_layers:
            model._kvcache_perhead_search_values[name] = {"k": [], "v": []}

    if not hasattr(model, "_kvcache_perhead_search_hooks"):
        model._kvcache_perhead_search_hooks = []
        for name, layer in kvcache_layers.items():
            hook = KVCachePerHeadValueHook(
                name,
                model._kvcache_perhead_search_values,
                num_kv_heads_total=num_kv_heads_total,
            )
            handle = layer.register_forward_hook(hook)
            model._kvcache_perhead_search_hooks.append(handle)

    return f"Registered {len(model._kvcache_perhead_search_hooks)} kv-perhead-search hooks"


def _get_kv_perhead_search_values(model):
    """Return collected per-head k/v tensors stored by value hooks."""
    if not hasattr(model, "_kvcache_perhead_search_values"):
        return {}
    return model._kvcache_perhead_search_values


def _remove_kvcache_perhead_value_hooks(model):
    """Remove per-head value-capture hooks from the model."""
    if hasattr(model, "_kvcache_perhead_search_hooks"):
        for h in model._kvcache_perhead_search_hooks:
            h.remove()
        del model._kvcache_perhead_search_hooks
    if hasattr(model, "_kvcache_perhead_search_values"):
        del model._kvcache_perhead_search_values


# Public aliases
setup_kvcache_perhead_value_hooks = _setup_kvcache_perhead_value_hooks
remove_kvcache_perhead_value_hooks = _remove_kvcache_perhead_value_hooks


# ---------------------------------------------------------------------------
# Per-head scale search
# ---------------------------------------------------------------------------


class KVScaleSearcherPerHead:
    """
    Callable (for use with ``llm.apply_model``) that runs the per-head KV
    scale search **inside each vLLM worker**.

    The ``activation_stats`` dict is expected to contain entries like::

        "model.layers.0.self_attn.attn.k_cache": {
            "min": [float, ...],   # length == num_kv_heads
            "max": [float, ...]
        }

    which follow the format produced by ``get_kvcache_perhead_stats()``.

    Returns a dict with the same keys and values being lists of per-head
    best multipliers (one float per KV head).
    """

    def __init__(
        self,
        activation_stats: dict,
        min_multiplier: float = 0.8,
        max_multiplier: float = 16.0,
        num_steps: int = 100,
    ):
        self.activation_stats = activation_stats
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.num_steps = num_steps

    def __call__(self, model):
        fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0

        kv_values = _get_kv_perhead_search_values(model)
        if not kv_values:
            print(
                "[KVScaleSearcherPerHead] WARNING: No per-head kv values collected. "
                "Did you call setup_kvcache_perhead_value_hooks before inference?"
            )
            return {}

        use_gpu = torch.cuda.is_available()
        search_device = (
            torch.device("cuda", torch.cuda.current_device()) if use_gpu else torch.device("cpu")
        )

        # Under the K/V-split scheme (replication >= 2), each rank captures
        # tensors only for its assigned side (``role``), so the opposite
        # side's list is empty.  The activation_stats dict already contains
        # the full global (num_kv_heads_total,) vectors thanks to the
        # reduce step in ``_all_gather_stats_perhead``.
        #
        # To figure out where this rank's local slice sits in the global
        # head array we use the actual global head count from activation_stats.
        rank, world_size = _get_dist_info()

        # Infer num_kv_heads_total from activation_stats (length of min/max
        # list of any entry).
        num_kv_heads_total = None
        for _sk, _sv in self.activation_stats.items():
            mn = _sv.get("min")
            if isinstance(mn, list):
                num_kv_heads_total = len(mn)
                break
            if isinstance(mn, torch.Tensor):
                num_kv_heads_total = mn.numel()
                break
        if num_kv_heads_total is None or num_kv_heads_total <= 0:
            num_kv_heads_total = 1

        # Determine this rank's role & head layout.
        role, heads_per_rank, global_head_offset, replication = _compute_perhead_layout(
            rank, world_size, num_kv_heads_total
        )

        # Build work items: one per (layer, kv_slot, local_head_idx)
        # Each item: (stats_key, global_head_idx, flat_cpu_tensor, base_scale)
        work_items = []
        for layer_name, tensors_dict in kv_values.items():
            for kv_slot, tensors in tensors_dict.items():
                # Role filter: skip the opposite side entirely.
                if role != "both" and kv_slot != role:
                    continue

                stats_key = f"{layer_name}.{kv_slot}_cache"
                if stats_key not in self.activation_stats:
                    print(
                        f"[KVScaleSearcherPerHead] WARNING: {stats_key} not found in "
                        f"activation_stats, skipping."
                    )
                    continue
                if not tensors:
                    # Expected on the opposite role – silently skip to avoid
                    # polluting the log under the split scheme.
                    continue

                stats = self.activation_stats[stats_key]
                min_vals = stats["min"]  # list[float] of length num_kv_heads_total
                max_vals = stats["max"]  # list[float] of length num_kv_heads_total

                # tensors[i] has shape (H_local, T_i, D) – cat along T dimension
                stacked = torch.cat(tensors, dim=1)  # (H_local, total_T, D)
                H_local = stacked.shape[0]

                for local_h in range(H_local):
                    global_h = global_head_offset + local_h
                    if global_h >= num_kv_heads_total:
                        # Safety guard – shouldn't happen under the
                        # documented layouts.
                        continue
                    abs_max = max(abs(min_vals[global_h]), abs(max_vals[global_h]))
                    if abs_max == 0:
                        base_scale = 1e-8
                    else:
                        base_scale = abs_max / fp8_max * 2.0
                    # Extract head slice: shape (total_T, D) → flatten → (N,)
                    flat_cpu = stacked[local_h].reshape(-1).float()
                    work_items.append((stats_key, global_h, flat_cpu, base_scale))

        print(
            f"[KVScaleSearcherPerHead] rank={rank}/{world_size}, role={role}, "
            f"head_offset={global_head_offset}, H_total={num_kv_heads_total}, "
            f"replication={replication}, {len(work_items)} head-level work items "
            f"({'GPU' if use_gpu else 'CPU'}, {self.num_steps} steps each)"
        )

        # Results: stats_key → {global_head_idx: multiplier}
        # Using a dict keyed by global head index so get_kv_scale_search_results_perhead
        # can correctly merge partial results from all TP workers.
        multipliers: dict[str, dict] = {}

        for stats_key, global_head_idx, flat_cpu, base_scale in work_items:
            if use_gpu:
                flat = flat_cpu.to(search_device, non_blocking=True)
            else:
                flat = flat_cpu
            best_m = _search_best_multiplier_flat(
                flat=flat,
                base_scale=base_scale,
                min_multiplier=self.min_multiplier,
                max_multiplier=self.max_multiplier,
                num_steps=self.num_steps,
            )
            if use_gpu:
                del flat
            if stats_key not in multipliers:
                multipliers[stats_key] = {}
            multipliers[stats_key][global_head_idx] = best_m

        # Print summary (one line per stats_key)
        for key, head_dict in multipliers.items():
            mults = list(head_dict.values())
            mn = min(mults)
            mx = max(mults)
            print(
                f"[KVScaleSearcherPerHead] {key}: "
                f"multipliers min={mn:.4f} max={mx:.4f} over {len(mults)} local heads"
            )

        return multipliers


def get_kv_scale_search_results_perhead(results_list: list) -> dict:
    """
    Merge per-head multiplier dicts from all TP workers.

    Each worker's result is ``{stats_key: {global_head_idx: multiplier}}``.
    This function merges all workers' dicts and converts to the final format
    ``{stats_key: [multiplier_head0, multiplier_head1, ...]}``, sorted by
    global head index.

    Under tensor parallelism rank r owns heads [r*H_local : (r+1)*H_local],
    so concatenating in rank order gives the full head list.
    """
    if not results_list:
        return {}

    # Merge {stats_key: {global_head_idx: multiplier}} across all workers.
    merged: dict = {}
    for worker_result in results_list:
        if not worker_result:
            continue
        for stats_key, head_dict in worker_result.items():
            if stats_key not in merged:
                merged[stats_key] = {}
            merged[stats_key].update(head_dict)

    # Convert to sorted list: {stats_key: [m0, m1, ...]}
    final: dict = {}
    for stats_key, head_dict in merged.items():
        sorted_indices = sorted(head_dict.keys())
        final[stats_key] = [head_dict[i] for i in sorted_indices]

    return final


# =============================================================================
# P-Matrix Scale Search  (per Q-head, FP8 fake-quant MSE, prefill-only)
# =============================================================================
#
# Overview
# --------
# The attention probability matrix P = softmax(Q K^T / sqrt(d)) is large
# (T x T per head) and cannot be captured directly from the vLLM FlashInfer
# backend.  Instead we re-compute P on-the-fly inside a forward hook that
# intercepts the (q, k, v) inputs to vllm.attention.layer.Attention.
#
# To keep peak GPU memory bounded we process Q in blocks of ``q_block_size``
# tokens and accumulate only the per-head MSE sum (a tiny (H_q_local, N)
# tensor) rather than storing the full P matrix.
#
# Granularity: per Q-head.  Under TP=16 each rank holds
#   H_q_local = total_num_q_heads / tp_size
# Q-heads, so the workload is naturally partitioned with no redundancy.
# After the forward pass we all-gather the per-rank (H_q_local, N) sums
# into a global (H_q_total, N) tensor on rank-0, then pick argmin per head.
#
# Decode steps (T==1) are skipped automatically.
# =============================================================================


class PMatrixScaleHook:
    """
    Forward hook on vLLM ``Attention`` layers that re-computes the attention
    probability matrix P in blocks and accumulates per-head FP8 quantisation
    MSE for a user-supplied list of candidate scales.

    The hook is **prefill-only**: decode steps (where the Q sequence length
    is 1) are skipped.

    Args:
        layer_name:    Identifier string for this layer.
        scale_list:    List of candidate FP8 scales to evaluate.
        q_block_size:  Number of Q tokens processed per block (controls peak
                       GPU memory).  Larger values are faster but use more
                       memory.
        p_stats:       Shared dict ``{layer_name: {"sum_sq": tensor,
                       "numel": int}}`` written by this hook.
    """

    # Fixed histogram bin edges for P-value distribution analysis.
    # P is in [0, 1].  Edges are chosen to highlight the long-tail region
    # near zero that dominates FP8 quantisation error.
    # Bins: [0, 1e-6), [1e-6, 1e-5), [1e-5, 1e-4), [1e-4, 1e-3), [1e-3, 1e-2),
    #       [1e-2, 0.05), [0.05, 0.1), [0.1, 0.2), [0.2, 0.4), [0.4, 0.8),
    #       [0.8, 1.0]
    HIST_EDGES = [
        0.0,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        0.05,
        0.1,
        0.2,
        0.4,
        0.8,
        1.0 + 1e-9,
    ]

    # Class-level flag so the (potentially) per-call warning about
    # T_k != T_q is only printed once per process.
    _tk_mismatch_warned = False

    def __init__(
        self,
        layer_name: str,
        scale_list: list,
        q_block_size: int,
        p_stats: dict,
        collect_dist: bool = True,
        analyze_first_cols: int = 0,
        mse_skip_first_cols: int = 0,
    ):
        self.layer_name = layer_name
        self.scale_list = scale_list
        self.q_block_size = q_block_size
        self.p_stats = p_stats
        self.collect_dist = collect_dist
        # When >0, also accumulate per-Q-head, per-column statistics on the
        # first ``analyze_first_cols`` columns of P (per row, restricted to
        # rows where the column is not causally masked).  This is used to
        # answer questions like "is column 0 (the BOS / leftmost key) much
        # larger than the rest?".  Memory cost is O(H_q * N) accumulators.
        self.analyze_first_cols = int(max(0, analyze_first_cols))
        # When >0, *exclude* the first ``mse_skip_first_cols`` columns of P
        # (sample-local key positions 0 .. K-1) from the FP8-quantisation
        # NMSE objective.  Use this to ignore attention-sink columns that
        # would otherwise dominate the error signal but are typically kept
        # in higher precision in production kernels.
        self.mse_skip_first_cols = int(max(0, mse_skip_first_cols))
        self._num_q_heads: int | None = None
        self._num_kv_heads: int | None = None
        self._head_dim: int | None = None
        self._scaling: float | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_head_info(self, module, q: torch.Tensor, k: torch.Tensor):
        """Resolve head counts and head_dim from module attrs or tensor shapes."""
        if self._num_q_heads is not None:
            return

        # Try module attributes first (most reliable).
        for attr in ("num_heads", "num_q_heads"):
            v = getattr(module, attr, None)
            if isinstance(v, int) and v > 0:
                self._num_q_heads = v
                break
        for attr in ("num_kv_heads",):
            v = getattr(module, attr, None)
            if isinstance(v, int) and v > 0:
                self._num_kv_heads = v
                break

        # head_dim
        for attr in ("head_dim",):
            v = getattr(module, attr, None)
            if isinstance(v, int) and v > 0:
                self._head_dim = v
                break

        # scaling factor (1/sqrt(d))
        for attr in ("scaling", "scale"):
            v = getattr(module, attr, None)
            if isinstance(v, float) and v > 0:
                self._scaling = v
                break

        # Fall back to shape inference.
        if self._head_dim is None:
            # q: (T, H_q * D)  or  (T, H_q, D)
            if q.ndim == 3:
                self._head_dim = q.shape[-1]
                if self._num_q_heads is None:
                    self._num_q_heads = q.shape[-2]
            elif q.ndim == 2 and self._num_q_heads is not None:
                self._head_dim = q.shape[-1] // self._num_q_heads
            else:
                # Cannot determine – skip this hook call.
                self._head_dim = 0

        if self._num_q_heads is None or self._num_q_heads <= 0:
            self._num_q_heads = 1
        if self._num_kv_heads is None or self._num_kv_heads <= 0:
            self._num_kv_heads = self._num_q_heads
        if self._scaling is None:
            self._scaling = (self._head_dim**-0.5) if self._head_dim > 0 else 1.0

    @staticmethod
    def _reshape_to_heads(t: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        """
        Reshape ``t`` from ``(T, num_heads * head_dim)`` or ``(T, num_heads, head_dim)``
        to ``(T, num_heads, head_dim)``.
        """
        if t.ndim == 3 and t.shape[-2] == num_heads:
            return t  # already (T, H, D)
        T = t.shape[0]
        return t.reshape(T, num_heads, head_dim)

    # ------------------------------------------------------------------
    # Main hook
    # ------------------------------------------------------------------

    @staticmethod
    def _get_per_sample_ranges(total_tokens: int):
        """
        Retrieve per-sample (start, end) token ranges from the current
        forward context, so we can compute a **separate** full P matrix
        per calibration sample (instead of treating the concatenated batch
        as a single sequence, which would let later samples see KVs from
        earlier samples via the causal mask).

        Falls back to a single range [0, total_tokens) if metadata is
        unavailable.
        """
        try:
            from vllm.forward_context import get_forward_context

            ctx = get_forward_context()
            attn_meta = getattr(ctx, "attn_metadata", None)
        except Exception:
            attn_meta = None

        if attn_meta is None:
            return [(0, total_tokens)]

        # attn_metadata can be a dict {layer_name: meta} or a single meta.
        meta_obj = attn_meta
        if isinstance(attn_meta, dict):
            # Any entry will do – seq_lens is identical across layers.
            meta_obj = next(iter(attn_meta.values()), None)
        if meta_obj is None:
            return [(0, total_tokens)]

        # Try query_start_loc / query_start_loc_cpu first (most reliable
        # for distinguishing per-sample token ranges in chunked prefill).
        qsl = None
        for attr in ("query_start_loc_cpu", "query_start_loc"):
            v = getattr(meta_obj, attr, None)
            if v is not None:
                qsl = v.detach().cpu().tolist() if hasattr(v, "detach") else list(v)
                break

        if qsl is not None and len(qsl) >= 2:
            ranges = []
            for i in range(len(qsl) - 1):
                s, e = int(qsl[i]), int(qsl[i + 1])
                if e > s:
                    ranges.append((s, e))
            if ranges:
                # Sanity check: last end should equal total_tokens.
                if ranges[-1][1] == total_tokens:
                    return ranges
                # If mismatch (e.g. padding), still return what we have.
                return ranges

        # Fallback: use seq_lens assuming pure prefill (query_len == seq_len).
        sl = None
        for attr in ("seq_lens_cpu", "seq_lens"):
            v = getattr(meta_obj, attr, None)
            if v is not None:
                sl = v.detach().cpu().tolist() if hasattr(v, "detach") else list(v)
                break

        if sl is not None:
            ranges = []
            cur = 0
            for L in sl:
                L = int(L)
                if L <= 0:
                    continue
                e = min(cur + L, total_tokens)
                if e > cur:
                    ranges.append((cur, e))
                cur = e
                if cur >= total_tokens:
                    break
            if ranges:
                return ranges

        # Last resort: treat the whole batch as one sequence.
        return [(0, total_tokens)]

    def __call__(self, module, input, output):
        # input to vllm Attention: (q, k, v, ...)
        q, k = input[0], input[1]
        if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
            return

        # Skip decode steps (T_q == 1).
        T_q = q.shape[0]
        if T_q <= 1:
            return

        with torch.no_grad():
            self._resolve_head_info(module, q, k)
            H_q = self._num_q_heads
            H_kv = self._num_kv_heads
            D = self._head_dim
            if D <= 0 or H_q <= 0:
                return

            N_scales = len(self.scale_list)
            device = q.device
            N_first = self.analyze_first_cols
            K_skip = self.mse_skip_first_cols

            # Initialise accumulators on first call.
            stats = self.p_stats[self.layer_name]
            if N_scales > 0 and stats["sum_sq"] is None:
                stats["sum_sq"] = torch.zeros(H_q, N_scales, dtype=torch.float32, device=device)
                # Per-head sum of P^2 (same scope as sum_sq), used as the
                # denominator when computing NMSE.
                stats["p_norm_sq"] = torch.zeros(H_q, dtype=torch.float32, device=device)
            # Distribution accumulators (per Q-head).  These are tiny
            # (O(H_q × num_bins) floats), so memory overhead is negligible.
            # Use fp32 instead of fp64: dist_count is at most ~1e10 and all
            # accumulated values are in [0, 1], so fp32 precision is ample
            # and avoids the expensive fp32->fp64 cast of p_flat per block.
            # Only initialise if distribution collection is enabled – when
            # disabled we skip the whole branch in the inner loop below.
            if self.collect_dist and stats.get("dist_hist") is None:
                num_bins = len(self.HIST_EDGES) - 1
                stats["dist_hist"] = torch.zeros(H_q, num_bins, dtype=torch.float32, device=device)
                stats["dist_sum"] = torch.zeros(H_q, dtype=torch.float32, device=device)
                stats["dist_sum_sq"] = torch.zeros(H_q, dtype=torch.float32, device=device)
                stats["dist_max"] = torch.zeros(H_q, dtype=torch.float32, device=device)
                stats["dist_count"] = 0  # total number of P elements accumulated
                stats["_dist_block_counter"] = 0  # used to subsample histogram

            # First-N-columns analysis accumulators (per Q-head, per column).
            # Statistics are computed only over rows where the column is
            # **not** causally masked, i.e. for column j we use rows i with
            # local position (within the sample) >= j.
            if N_first > 0 and stats.get("first_cols_sum") is None:
                stats["first_cols_sum"] = torch.zeros(
                    H_q, N_first, dtype=torch.float32, device=device
                )
                stats["first_cols_sum_sq"] = torch.zeros(
                    H_q, N_first, dtype=torch.float32, device=device
                )
                stats["first_cols_max"] = torch.zeros(
                    H_q, N_first, dtype=torch.float32, device=device
                )
                # Per-column count is identical across heads, but we keep it
                # column-shaped for convenience.  Stored as float32 to allow
                # cheap all-reduce later.
                stats["first_cols_count"] = torch.zeros(
                    N_first, dtype=torch.float32, device=device
                )

            # Reshape to (T, H, D) and upcast to fp32 for numerical parity
            # with the previous implementation (user explicitly requested
            # fp32 QK matmul).
            q_3d = self._reshape_to_heads(q, H_q, D).float()  # (T_q, H_q, D)
            k_3d = self._reshape_to_heads(k, H_kv, D).float()  # (T_k, H_kv, D)
            T_k = k_3d.shape[0]

            # Assumption for prefill-only calibration: query_len == key_len
            # for each sample, so T_k == T_q.  If they differ (e.g. chunked
            # prefill with paged KV cache), we can't reconstruct K from the
            # input tensor alone – fall back to treating the batch as one.
            sample_ranges_from_meta = T_k == T_q
            if not sample_ranges_from_meta and not PMatrixScaleHook._tk_mismatch_warned:
                print(
                    f"[PMatrix][WARN] T_k ({T_k}) != T_q ({T_q}) for layer "
                    f"'{self.layer_name}'. Falling back to treating the whole "
                    f"batch as one sequence for P-matrix MSE accumulation. "
                    f"This can happen with chunked prefill + paged KV cache. "
                    f"The computed MSE will still be approximately correct "
                    f"because cross-sample QK scores are typically small, "
                    f"but not exact. Consider disabling chunked prefill "
                    f"(export VLLM_ENABLE_CHUNKED_PREFILL=0) for strict "
                    f"per-sample semantics."
                )
                PMatrixScaleHook._tk_mismatch_warned = True

            # GQA: expand K to match Q heads.
            n_rep = H_q // H_kv
            if n_rep > 1:
                # (T_k, H_kv, D) → (T_k, H_q, D)
                k_3d = k_3d.unsqueeze(2).expand(-1, -1, n_rep, -1).reshape(T_k, H_q, D)

            # Transpose for batched matmul: (H_q, T, D)
            q_hd = q_3d.permute(1, 0, 2)  # (H_q, T_q, D)
            k_hd = k_3d.permute(1, 0, 2)  # (H_q, T_k, D)

            scaling = self._scaling
            Bq = self.q_block_size

            # -----------------------------------------------------------------
            # Split into per-sample ranges so each sample's P matrix is
            # computed over its own K range only (no cross-sample attention).
            #
            # Equivalence: summing (P - Q(P))^2 over all (sample, i, j)
            #              pairs and dividing by the total count is exactly
            #              the "global MSE over 32 × 32k samples" that we
            #              want to minimise.  Q-blocking *within* a sample
            #              does NOT change the softmax (softmax is per-row
            #              over the full K range of that sample) – it only
            #              streams the computation to cap peak memory.
            # -----------------------------------------------------------------
            if sample_ranges_from_meta:
                sample_ranges = self._get_per_sample_ranges(T_q)
            else:
                sample_ranges = [(0, T_q)]

            for s_start, s_end in sample_ranges:
                S = s_end - s_start
                if S <= 1:
                    continue

                # Per-sample Q / K slices – K is restricted to this sample.
                q_s = q_hd[:, s_start:s_end, :]  # (H_q, S, D)
                k_s = k_hd[:, s_start:s_end, :]  # (H_q, S, D)

                for q_start in range(0, S, Bq):
                    q_end = min(q_start + Bq, S)
                    Bq_actual = q_end - q_start
                    q_blk = q_s[:, q_start:q_end, :]  # (H_q, Bq, D)

                    # scores: (H_q, Bq, S)  – K is the *full* sample range,
                    #                         so softmax is identical to
                    #                         computing the whole sample's
                    #                         P matrix at once.
                    # Keep the QK matmul in fp32 (q_blk / k_s are already fp32).
                    scores = torch.bmm(q_blk, k_s.transpose(1, 2))
                    scores.mul_(scaling)

                    # Causal mask within this sample:
                    #   q_local position = q_start + i   (0 .. S-1)
                    #   k_local position = j             (0 .. S-1)
                    q_pos = torch.arange(q_start, q_end, device=device).unsqueeze(1)  # (Bq, 1)
                    k_pos = torch.arange(S, device=device).unsqueeze(0)  # (1, S)
                    causal_mask = k_pos > q_pos  # True = masked
                    scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

                    # P_block: (H_q, Bq, S) in float32
                    P_block = torch.softmax(scores, dim=-1)
                    del scores

                    # Accumulate NMSE numerator (sum diff²) and denominator
                    # (sum P²) for each candidate scale, optionally
                    # excluding the first ``K_skip`` sample-local columns
                    # (attention-sink columns).  Use in-place ops to avoid
                    # materialising extra (H_q, Bq, S) tensors.
                    if N_scales > 0:
                        fp8_max = torch.finfo(torch.float8_e4m3fn).max
                        # Build a (1, 1, S) skip mask once per Q-block.
                        # mask[..., j] = 0 for j in [0, K_skip), else 1.
                        # The mask is a fp32 multiplier so we can fuse it
                        # into the in-place square op.
                        skip_mask = None
                        if K_skip > 0 and K_skip < S:
                            skip_mask = torch.ones(1, 1, S, dtype=torch.float32, device=device)
                            skip_mask[:, :, :K_skip] = 0.0
                        elif K_skip >= S:
                            # Every key column is skipped; nothing to
                            # accumulate from this block.
                            skip_mask = torch.zeros(1, 1, S, dtype=torch.float32, device=device)

                        for i, s in enumerate(self.scale_list):
                            # diff = P_block - fake_quant(P_block, s)
                            #      = P_block - (clamp(P_block/s)_fp8 * s)
                            P_q = (
                                (P_block / s)
                                .clamp_(-fp8_max, fp8_max)
                                .to(torch.float8_e4m3fn)
                                .to(torch.float32)
                                .mul_(s)
                            )
                            P_q.sub_(P_block).square_()  # diff² in-place on P_q
                            if skip_mask is not None:
                                P_q.mul_(skip_mask)  # zero out skipped cols
                            stats["sum_sq"][:, i] += P_q.sum(dim=(1, 2))
                            del P_q

                        # NMSE denominator: sum of P^2 over the same scope.
                        if skip_mask is not None:
                            P_sq_masked = (P_block * P_block) * skip_mask
                            stats["p_norm_sq"] += P_sq_masked.sum(dim=(1, 2))
                            del P_sq_masked
                        else:
                            stats["p_norm_sq"] += (P_block * P_block).sum(dim=(1, 2))

                        # Per-block element count (for diagnostics; the NMSE
                        # itself does not need this number, but we keep it
                        # so old MSE = sum_sq / numel can still be derived).
                        # When K_skip > 0 we subtract the (analytically known)
                        # number of *valid* (non-causally-masked) elements
                        # whose key-position falls in [0, K_skip).
                        block_numel = Bq_actual * S
                        if K_skip > 0:
                            # For each skipped column j in [0, K_skip):
                            #   valid rows in this Q-block =
                            #     max(0, q_end - max(j, q_start))
                            ks = min(K_skip, S)
                            j_idx = torch.arange(ks, device=device, dtype=torch.float32)
                            skipped_rows = (
                                (float(q_end) - torch.clamp(j_idx, min=float(q_start)))
                                .clamp_(min=0.0)
                                .sum()
                                .item()
                            )
                            block_numel -= int(skipped_rows)
                            del j_idx
                        stats["numel"] += int(block_numel)
                        if skip_mask is not None:
                            del skip_mask

                    # -----------------------------------------------------
                    # First-N-columns analysis (per Q-head, per column).
                    # We slice the *first* N_first columns of P_block, which
                    # correspond to keys at sample-local positions
                    # 0 .. N_first-1.
                    #
                    # Causal-mask handling:
                    #   For column j, query rows i < j are causally masked
                    #   and have P[i,j] = 0 (softmax of -∞).  Including
                    #   these zeros in sum / sum_sq gives the same result as
                    #   excluding them, so we sum over the entire Bq dim and
                    #   compute the mean using a *valid-row* count derived
                    #   analytically below.  This means the reported mean /
                    #   std are conditional on rows that can actually attend
                    #   to column j, exactly what we want for sink analysis.
                    #   The max is also unaffected because 0 ≤ any P value.
                    # -----------------------------------------------------
                    if N_first > 0:
                        ncols = min(N_first, P_block.shape[-1])
                        first_cols = P_block[:, :, :ncols]  # (H_q, Bq, ncols)
                        stats["first_cols_sum"][:, :ncols] += first_cols.sum(dim=1)
                        stats["first_cols_sum_sq"][:, :ncols] += (first_cols * first_cols).sum(
                            dim=1
                        )
                        block_max_cols = first_cols.amax(dim=1)  # (H_q, ncols)
                        stats["first_cols_max"][:, :ncols] = torch.maximum(
                            stats["first_cols_max"][:, :ncols], block_max_cols
                        )
                        # Per-column count of *valid* (non-masked) rows in
                        # this Q-block:
                        #   valid(j) = max(0, q_end - max(j, q_start))
                        # i.e. rows with local position in [q_start, q_end)
                        # AND >= j.
                        col_idx = torch.arange(ncols, device=device, dtype=torch.float32)
                        valid = (float(q_end) - torch.clamp(col_idx, min=float(q_start))).clamp_(
                            min=0.0
                        )
                        stats["first_cols_count"][:ncols] += valid
                        del first_cols, block_max_cols, col_idx, valid

                    # -----------------------------------------------------
                    # Accumulate P-value distribution statistics (per head).
                    # Histogram is O((Bq·S)·num_bins) in work and is by far
                    # the most expensive part of the distribution branch, so
                    # we **subsample**: compute the histogram only on every
                    # HIST_SUBSAMPLE-th Q-block (scale/mean/std/max are still
                    # accumulated every block – they are very cheap).
                    # Masked positions are 0 (softmax after -inf score) –
                    # they fall into the first bin and contribute 0 to
                    # sum / sum_sq / max.
                    # Skip the whole branch when distribution collection is
                    # disabled (saves memory + compute, especially for
                    # long-context calibration).
                    # -----------------------------------------------------
                    if not self.collect_dist:
                        del q_blk, P_block
                        continue

                    p_flat = P_block.reshape(H_q, -1)  # (H_q, Bq*S) fp32
                    stats["dist_sum"] += p_flat.sum(dim=1)
                    stats["dist_sum_sq"] += (p_flat * p_flat).sum(dim=1)
                    block_max = p_flat.max(dim=1).values  # (H_q,) fp32
                    stats["dist_max"] = torch.maximum(stats["dist_max"], block_max)
                    stats["dist_count"] += p_flat.shape[1]

                    HIST_SUBSAMPLE = 8
                    do_hist = (stats["_dist_block_counter"] % HIST_SUBSAMPLE) == 0
                    stats["_dist_block_counter"] += 1
                    if do_hist:
                        # Per-head histogram: use torch.bucketize which is
                        # vectorised and fast.
                        edges = torch.tensor(self.HIST_EDGES, dtype=torch.float32, device=device)
                        bucket_idx = torch.bucketize(p_flat, edges, right=False)
                        bucket_idx = (bucket_idx - 1).clamp_(min=0, max=len(self.HIST_EDGES) - 2)
                        # Scale histogram up to compensate for subsampling
                        # so totals stay proportional to the real counts.
                        stats["dist_hist"].scatter_add_(
                            dim=1,
                            index=bucket_idx.long(),
                            src=torch.full_like(
                                bucket_idx,
                                fill_value=float(HIST_SUBSAMPLE),
                                dtype=torch.float32,
                            ),
                        )
                        del bucket_idx

                    # Free block tensors immediately to keep peak memory low.
                    # NOTE: do NOT call torch.cuda.empty_cache() here – it
                    # forces a full allocator purge + stream sync and was the
                    # single biggest contributor to the per-step slowdown.
                    del q_blk, P_block, p_flat


def setup_p_matrix_scale_hooks(
    model,
    scale_list: list,
    q_block_size: int = 1024,
    collect_dist: bool = True,
    analyze_first_cols: int = 0,
    mse_skip_first_cols: int = 0,
):
    """
    Register P-matrix scale search hooks on all vLLM Attention layers.

    Args:
        model:              The vLLM worker model (passed by ``llm.apply_model``).
        scale_list:         List of candidate FP8 scales to evaluate.  Pass an
                            empty list to skip the FP8-scale NMSE search
                            entirely (useful for pure analysis runs).
        q_block_size:       Q-token block size for memory-bounded P computation.
        collect_dist:       When True (default), also accumulate per-Q-head
                            P-value distribution statistics (histogram / mean /
                            std / max).  Set to False to skip the distribution
                            branch entirely.
        analyze_first_cols: When >0, additionally accumulate per-Q-head,
                            per-column statistics on the first N columns of P
                            (mean / std / max / count).  Useful for
                            characterising attention sinks (e.g. is the
                            leftmost key dominant?).
        mse_skip_first_cols:When >0, *exclude* the first N sample-local key
                            columns of P from the FP8 NMSE objective.  Used
                            to ignore attention-sink columns that would
                            otherwise dominate the error signal.

    Returns:
        Human-readable status string.
    """
    try:
        # vLLM ≥ 0.20 (Tencent custom): Attention moved under model_executor.
        from vllm.model_executor.layers.attention import Attention
    except ImportError:
        # Older vLLM layout.
        from vllm.attention.layer import Attention

    attn_layers = _find_layers(model, layers=[Attention])
    print(f"[PMatrix] Found {len(attn_layers)} Attention layers to hook")

    if not hasattr(model, "_p_matrix_stats"):
        model._p_matrix_stats = {}
        for name in attn_layers:
            model._p_matrix_stats[name] = {
                "sum_sq": None,
                "p_norm_sq": None,
                "numel": 0,
                "dist_hist": None,
                "dist_sum": None,
                "dist_sum_sq": None,
                "dist_max": None,
                "dist_count": 0,
                "first_cols_sum": None,
                "first_cols_sum_sq": None,
                "first_cols_max": None,
                "first_cols_count": None,
            }

    model._p_matrix_scale_list = list(scale_list)
    model._p_matrix_q_block_size = q_block_size
    model._p_matrix_collect_dist = bool(collect_dist)
    model._p_matrix_analyze_first_cols = int(max(0, analyze_first_cols))
    model._p_matrix_mse_skip_first_cols = int(max(0, mse_skip_first_cols))

    if not hasattr(model, "_p_matrix_hooks"):
        model._p_matrix_hooks = []
        for name, layer in attn_layers.items():
            hook = PMatrixScaleHook(
                layer_name=name,
                scale_list=scale_list,
                q_block_size=q_block_size,
                p_stats=model._p_matrix_stats,
                collect_dist=bool(collect_dist),
                analyze_first_cols=int(max(0, analyze_first_cols)),
                mse_skip_first_cols=int(max(0, mse_skip_first_cols)),
            )
            handle = layer.register_forward_hook(hook)
            model._p_matrix_hooks.append(handle)

    return (
        f"Registered {len(model._p_matrix_hooks)} P-matrix hooks "
        f"({len(scale_list)} candidate scales, q_block_size={q_block_size}, "
        f"collect_dist={bool(collect_dist)}, "
        f"analyze_first_cols={int(max(0, analyze_first_cols))}, "
        f"mse_skip_first_cols={int(max(0, mse_skip_first_cols))})"
    )


def get_p_matrix_scale_stats(model):
    """
    Collect per-head P-matrix MSE statistics from all TP workers and return
    the best scale per Q-head per layer.

    Under TP=N each rank holds ``H_q_local = H_q_total / N`` Q-heads.
    We all-gather the ``(H_q_local, N_scales)`` sum_sq tensors from every
    rank into a global ``(H_q_total, N_scales)`` tensor on rank-0, then
    pick ``argmin`` per head.

    Returns (on rank-0):
        dict with keys like ``"model.layers.0.self_attn.attn"`` and values::

            {
                "scale_list":        [s0, s1, ...],
                "best_scale_per_head": [s_h0, s_h1, ...],   # length H_q_total
                "mse_per_head":      [[mse_s0, mse_s1, ...], ...],  # H_q_total rows
            }

    Returns ``None`` on non-zero ranks (they have already contributed via
    all-gather).
    """
    import torch.distributed as dist

    if not hasattr(model, "_p_matrix_stats"):
        return None

    rank, world_size = _get_dist_info()
    scale_list = getattr(model, "_p_matrix_scale_list", [])
    N_scales = len(scale_list)
    analyze_first_cols = int(getattr(model, "_p_matrix_analyze_first_cols", 0))

    results = {}

    # All-gather helper: (H_q_local, K)  →  (H_q_total, K)
    def _allgather_heads(local: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            gpu = local.cuda()
            gathered = [torch.zeros_like(gpu) for _ in range(world_size)]
            dist.all_gather(gathered, gpu)
            del gpu
            out = torch.cat(gathered, dim=0).cpu()
            del gathered
            return out
        return local

    for layer_name, stats in model._p_matrix_stats.items():
        sum_sq = stats["sum_sq"]  # (H_q_local, N_scales) or None
        numel = stats["numel"]  # int
        p_norm_sq = stats.get("p_norm_sq")  # (H_q_local,) or None
        first_cols_sum = stats.get("first_cols_sum")  # (H_q_local, N) or None

        # Skip layers that were never called at all (e.g. decode-only).
        if sum_sq is None and first_cols_sum is None:
            continue

        # ---------------- NMSE / best-scale branch ----------------
        has_mse = sum_sq is not None and N_scales > 0

        if has_mse:
            sum_sq_cpu = sum_sq.cpu().float()  # (H_q_local, N_scales)
            p_norm_sq_cpu = (
                p_norm_sq.cpu().float()
                if p_norm_sq is not None
                else torch.zeros(sum_sq_cpu.shape[0], dtype=torch.float32)
            )  # (H_q_local,)

            # Synchronise numel across ranks (sum).
            numel_tensor = torch.tensor([numel], dtype=torch.float64)
            if world_size > 1:
                numel_gpu = numel_tensor.cuda()
                dist.all_reduce(numel_gpu, op=dist.ReduceOp.SUM)
                total_numel = int(numel_gpu.item())
                del numel_gpu
            else:
                total_numel = numel

            if total_numel == 0:
                has_mse = False

        if has_mse:
            global_sum_sq = _allgather_heads(sum_sq_cpu)
            # p_norm_sq is (H_q_local,) ; broadcast through allgather helper
            # by adding a dummy K=1 dim.
            global_p_norm_sq = _allgather_heads(p_norm_sq_cpu.unsqueeze(1)).squeeze(1)
            # All-reduce p_norm_sq across ranks (it is *not* head-sharded;
            # each rank computed the sum over its local heads, so allgather
            # is the correct op above).

        # ---------------- Distribution branch ----------------
        collect_dist = bool(getattr(model, "_p_matrix_collect_dist", True))
        has_dist = collect_dist and (stats.get("dist_hist") is not None)

        if has_dist:
            dist_hist_cpu = stats["dist_hist"].cpu()  # (H_q_local, num_bins)
            dist_sum_cpu = stats["dist_sum"].cpu()  # (H_q_local,)
            dist_sum_sq_cpu = stats["dist_sum_sq"].cpu()  # (H_q_local,)
            dist_max_cpu = stats["dist_max"].cpu()  # (H_q_local,)
            dist_count_local = int(stats["dist_count"])

            global_hist = _allgather_heads(dist_hist_cpu)  # (H_q_total, num_bins)
            global_dist_sum = _allgather_heads(dist_sum_cpu.unsqueeze(1)).squeeze(
                1
            )  # (H_q_total,)
            global_dist_sum_sq = _allgather_heads(dist_sum_sq_cpu.unsqueeze(1)).squeeze(1)
            global_dist_max = _allgather_heads(dist_max_cpu.unsqueeze(1)).squeeze(1)

            # dist_count is the same on every rank per head (Q-block iterations
            # are driven by identical attn_metadata), so just keep the local
            # value.  But to be safe we all-reduce the *max* across ranks.
            if world_size > 1:
                dc_gpu = torch.tensor([dist_count_local], dtype=torch.float64).cuda()
                dist.all_reduce(dc_gpu, op=dist.ReduceOp.MAX)
                dist_count_global = int(dc_gpu.item())
                del dc_gpu
            else:
                dist_count_global = dist_count_local

        # ---------------- First-N-columns branch ----------------
        has_first_cols = first_cols_sum is not None and analyze_first_cols > 0
        if has_first_cols:
            fc_sum_cpu = stats["first_cols_sum"].cpu().float()  # (H_q_local, N)
            fc_sum_sq_cpu = stats["first_cols_sum_sq"].cpu().float()
            fc_max_cpu = stats["first_cols_max"].cpu().float()
            # first_cols_count is the same across heads & ranks
            # (Q-block iteration is driven by identical attn_metadata),
            # but we all-reduce SUM across ranks anyway to be safe – every
            # rank participated in the same prefill, so SUM/world_size is
            # the right answer.  Use MAX instead of SUM to avoid double
            # counting because every rank already accumulated the full count.
            fc_count_local = stats["first_cols_count"].cpu().float()  # (N,)

            global_fc_sum = _allgather_heads(fc_sum_cpu)  # (H_q_total, N)
            global_fc_sum_sq = _allgather_heads(fc_sum_sq_cpu)
            global_fc_max = _allgather_heads(fc_max_cpu)

            if world_size > 1:
                fc_cnt_gpu = fc_count_local.cuda()
                dist.all_reduce(fc_cnt_gpu, op=dist.ReduceOp.MAX)
                global_fc_count = fc_cnt_gpu.cpu()
                del fc_cnt_gpu
            else:
                global_fc_count = fc_count_local

        # Only rank-0 builds the final result dict.
        if rank != 0:
            continue

        layer_result = {}

        if has_mse:
            # NMSE per head per scale:
            #   numerator   = sum_diff_sq[h, s]   (over masked elements)
            #   denominator = sum_P_sq[h]         (over the same elements)
            #   NMSE[h, s]  = numerator / denominator
            # When the user did not request to skip any sink columns the
            # denominator collapses to the standard sum(P²) and NMSE is
            # exactly equivalent to MSE / mean(P²).
            denom = global_p_norm_sq.clamp(min=1e-30).unsqueeze(1)  # (H_q_total, 1)
            nmse = global_sum_sq / denom  # (H_q_total, N_scales)
            # Plain MSE is still cheap to derive; emit it for diagnostics.
            mse = global_sum_sq / max(int(total_numel), 1)

            # Best scale index per head: (H_q_total,) -- argmin on NMSE.
            best_idx = nmse.argmin(dim=1)
            best_scales = [scale_list[int(i)] for i in best_idx.tolist()]
            mse_skip = int(getattr(model, "_p_matrix_mse_skip_first_cols", 0))
            layer_result["scale_list"] = scale_list
            layer_result["best_scale_per_head"] = best_scales
            layer_result["nmse_per_head"] = nmse.tolist()
            layer_result["mse_per_head"] = mse.tolist()
            layer_result["p_norm_sq_per_head"] = global_p_norm_sq.tolist()
            layer_result["mse_skip_first_cols"] = mse_skip
            layer_result["numel_used"] = int(total_numel)

        if has_dist:
            # Distribution summary per head.
            if dist_count_global > 0:
                means = (global_dist_sum / dist_count_global).tolist()
                # Var = E[x^2] - E[x]^2
                var = (global_dist_sum_sq / dist_count_global) - (
                    global_dist_sum / dist_count_global
                ) ** 2
                var = var.clamp(min=0.0)
                stds = var.sqrt().tolist()
                # Normalise histogram to probabilities per head.
                head_totals = global_hist.sum(dim=1, keepdim=True).clamp(min=1.0)
                hist_prob = (global_hist / head_totals).tolist()
                hist_counts = global_hist.to(torch.int64).tolist()
            else:
                means = [0.0] * global_dist_sum.shape[0]
                stds = [0.0] * global_dist_sum.shape[0]
                hist_prob = []
                hist_counts = []

            layer_result["p_dist"] = {
                "hist_edges": list(PMatrixScaleHook.HIST_EDGES),
                "hist_prob_per_head": hist_prob,  # normalised (sum=1 per head)
                "hist_counts_per_head": hist_counts,  # raw element counts
                "mean_per_head": means,
                "std_per_head": stds,
                "max_per_head": global_dist_max.tolist(),
                "total_count_per_head": dist_count_global,
            }

        if has_first_cols:
            # Per-head, per-column mean / std / max.
            # global_fc_count is (N,) and shared across heads, so we
            # broadcast it for the divisions.
            cnt = global_fc_count.clamp(min=1.0).unsqueeze(0)  # (1, N)
            mean_c = global_fc_sum / cnt  # (H_q_total, N)
            var_c = (global_fc_sum_sq / cnt) - mean_c * mean_c
            var_c = var_c.clamp(min=0.0)
            std_c = var_c.sqrt()

            # Cross-head aggregates (mean over heads, per column).
            mean_per_col_layer = mean_c.mean(dim=0)  # (N,)
            max_per_col_layer = global_fc_max.amax(dim=0)  # (N,)
            N_local = int(mean_c.shape[1])

            # ---------------------------------------------------------
            # Sink length detection.
            # Attention sinks frequently span more than just the very
            # first key (e.g. BOS + a few system tokens).  Try several
            # candidate prefix lengths K and pick the one that maximises
            # the contrast between the prefix mean and the rest:
            #     ratio_K = mean[0..K-1].mean() / mean[K..N-1].mean()
            # We also expose the full ratio table so downstream tools
            # can apply their own threshold.
            # ---------------------------------------------------------
            cand_Ks = [k for k in (1, 2, 4, 8, 16, 32) if k < N_local]
            sink_ratios = {}
            best_K = 1
            best_ratio = -1.0
            for K in cand_Ks:
                head_part = float(mean_per_col_layer[:K].mean().item())
                tail_part = float(mean_per_col_layer[K:].mean().item())
                r = head_part / tail_part if tail_part > 0 else float("inf")
                sink_ratios[str(K)] = r
                if r > best_ratio:
                    best_ratio = r
                    best_K = K

            # Backwards compatibility: top_col_ratio_layer == ratio_K=1.
            top_col_ratio_layer = sink_ratios.get("1", 1.0)

            # Per-column relative gain (column mean divided by the mean
            # of the *non-sink* tail, with sink length = best_K).  This
            # gives a clean decay curve, e.g.
            #   col_gain[0]=120x, col_gain[1]=80x, col_gain[2]=15x, ...
            tail_mean = (
                float(mean_per_col_layer[best_K:].mean().item()) if best_K < N_local else 0.0
            )
            if tail_mean > 0:
                col_gain_layer = (mean_per_col_layer / tail_mean).tolist()
            else:
                col_gain_layer = [float("inf")] * N_local

            p_first_cols_dict = {
                "num_cols": N_local,
                "count_per_col": global_fc_count.to(torch.int64).tolist(),
                # Cross-head aggregated views (cheap; always emitted).
                "mean_per_col_layer": mean_per_col_layer.tolist(),  # (N,)
                "max_per_col_layer": max_per_col_layer.tolist(),  # (N,)
                "col_gain_layer": col_gain_layer,  # (N,)
                "sink_ratios": sink_ratios,  # {K -> ratio}
                "sink_K_best": best_K,
                "sink_ratio_best": float(best_ratio),
                # Backwards-compat field (= ratio at K=1).
                "top_col_ratio_layer": float(top_col_ratio_layer),
            }

            # Per-head detail (large): only emit when the user asked for it.
            keep_per_head = bool(getattr(model, "_p_matrix_first_cols_per_head_detail", True))
            if keep_per_head:
                p_first_cols_dict["mean_per_head_per_col"] = mean_c.tolist()
                p_first_cols_dict["std_per_head_per_col"] = std_c.tolist()
                p_first_cols_dict["max_per_head_per_col"] = global_fc_max.tolist()

            layer_result["p_first_cols"] = p_first_cols_dict

        if layer_result:
            results[layer_name] = layer_result

    if rank != 0:
        return None

    return results


def remove_p_matrix_scale_hooks(model):
    """
    Remove P-matrix hooks and free accumulated statistics.
    Designed to be passed to ``llm.apply_model()``.
    """
    if hasattr(model, "_p_matrix_hooks"):
        for h in model._p_matrix_hooks:
            h.remove()
        del model._p_matrix_hooks
    if hasattr(model, "_p_matrix_stats"):
        del model._p_matrix_stats
    for attr in (
        "_p_matrix_scale_list",
        "_p_matrix_q_block_size",
        "_p_matrix_collect_dist",
        "_p_matrix_analyze_first_cols",
        "_p_matrix_mse_skip_first_cols",
        "_p_matrix_first_cols_per_head_detail",
    ):
        if hasattr(model, attr):
            delattr(model, attr)
    return "P-matrix hooks removed"


class PMatrixScaleSearcher:
    """
    Lightweight callable (for ``llm.apply_model``) that **only** triggers
    ``get_p_matrix_scale_stats`` inside each worker.

    The actual MSE accumulation happens inside ``PMatrixScaleHook`` during
    the forward pass; this class just collects and returns the results.

    Usage::

        # 1. Register hooks
        llm.apply_model(lambda m: setup_p_matrix_scale_hooks(m, scale_list, q_block_size))
        # 2. One prefill forward pass
        llm.generate(prompts, SamplingParams(max_tokens=1))
        # 3. Collect results
        searcher = PMatrixScaleSearcher()
        results_list = llm.apply_model(searcher)
        # results_list[0] is the rank-0 result dict (or None for other ranks)
        # 4. Clean up
        llm.apply_model(remove_p_matrix_scale_hooks)
    """

    def __call__(self, model):
        return get_p_matrix_scale_stats(model)


def setup_p_first_cols_hooks(
    model,
    num_cols: int = 64,
    q_block_size: int = 1024,
    per_head_detail: bool = True,
):
    """
    Convenience wrapper around :func:`setup_p_matrix_scale_hooks` for the
    "first-N-columns analysis only" mode.

    Args:
        model:           The vLLM worker model (passed by ``llm.apply_model``).
        num_cols:        Number of leading columns of P to analyse (default: 64).
        q_block_size:    Q-token block size for memory-bounded P computation.
        per_head_detail: When True (default), emit per-head per-column mean /
                         std / max in the output JSON (size scales with
                         H_q_total × N × num_layers).  Set to False to keep
                         only cross-head aggregated views (much smaller).
    """
    msg = setup_p_matrix_scale_hooks(
        model,
        scale_list=[],
        q_block_size=q_block_size,
        collect_dist=False,
        analyze_first_cols=int(num_cols),
    )
    # Stash the per-head-detail flag so get_p_matrix_scale_stats can read it.
    model._p_matrix_first_cols_per_head_detail = bool(per_head_detail)
    return msg + f" per_head_detail={bool(per_head_detail)}"
