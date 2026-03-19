import os

import torch

__all__ = [
    "setup_activation_hooks",
    "get_activation_stats",
    "print_activation_stats",
    "print_moe_stats",
    "get_moe_stats",
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
            # Use tensor operations to avoid graph breaks
            with torch.no_grad():
                act_min = act.min().detach().cpu()
                act_max = act.max().detach().cpu()

                # Update global min/max using tensor operations
                stats = self.activation_stats[self.layer_name]
                stats["min"] = torch.minimum(stats["min"], act_min)
                stats["max"] = torch.maximum(stats["max"], act_max)
                stats["call_count"] = self.call_count  # Store call count


def setup_activation_hooks(model):
    """
    Setup activation hooks on the model to collect min/max statistics.
    This function is applied to each worker's model instance.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase

    # Find all linear layers to monitor
    layers_to_monitor = _find_layers(model, layers=[torch.nn.Linear, LinearBase])

    print(f"---------Found {len(layers_to_monitor)} layers to monitor---------")
    for name in list(layers_to_monitor.keys())[:5]:  # Print first 5
        print(f"  {name}")
    if len(layers_to_monitor) > 5:
        print(f"  ... and {len(layers_to_monitor) - 5} more")

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
    except Exception as e:
        print(f"Warning: Could not perform all-reduce: {e}")

    # Convert tensors to Python scalars for easier use
    stats_dict = {}
    for name, stats in model._activation_stats.items():
        stats_dict[name] = {
            "min": stats["min"].item() if isinstance(stats["min"], torch.Tensor) else stats["min"],
            "max": stats["max"].item() if isinstance(stats["max"], torch.Tensor) else stats["max"],
        }
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
