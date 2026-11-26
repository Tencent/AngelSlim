"""KV-cache FP8 scale search (per-tensor and per-head): capture raw K/V in a hook, then
grid-search the FP8 scale multiplier per layer (or head) that minimises fake-quant MSE."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from ....utils import rank_world
from ....utils.utils import find_modules
from .kvcache import (
    compute_perhead_layout,
    get_kv_role,
    infer_num_kv_heads_total,
    num_heads_from_tensor,
)

__all__ = [
    # Per-tensor scale search
    "KVCacheValueHook",
    "KVScaleSearcher",
    "KVMSEProfileCollector",
    "setup_kvcache_value_hooks",
    "get_kv_scale_search_results",
    "get_kv_mse_profile_results",
    "remove_kv_scale_search_hooks",
    # Per-head scale search
    "KVCachePerHeadValueHook",
    "setup_kvcache_perhead_value_hooks",
    "remove_kvcache_perhead_value_hooks",
    "KVScaleSearcherPerHead",
    "KVPerHeadMSEProfileCollector",
    "get_kv_scale_search_results_perhead",
    "get_kv_mse_profile_results_perhead",
]


# =============================================================================
# Per-tensor KV-cache scale search
# =============================================================================


class KVCacheValueHook:
    """Capture raw (BF16) k/v entering vLLM Attention, one list entry per forward pass."""

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
    """Register k/v capture hooks (run via ``llm.apply_model``); returns a status string."""
    try:
        # vLLM >= 0.20 (Tencent custom): Attention moved under model_executor.
        from vllm.model_executor.layers.attention import Attention
    except ImportError:
        # Older vLLM layout.
        from vllm.attention.layer import Attention

    kvcache_layers = find_modules(model, [Attention])

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
    """Return ``{layer_name: {"k": [tensor, ...], "v": [tensor, ...]}}`` from a worker."""
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
    """Fake-quantize to FP8 with vLLM's convention ``scale = abs_max / fp8_max``."""
    orig_dtype = tensor.dtype
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    fp8_min = torch.finfo(torch.float8_e4m3fn).min  # -448.0

    t = tensor.float()
    q = (t / scale).clamp(fp8_min, fp8_max).to(torch.float8_e4m3fn)
    dq = q.to(torch.float32) * scale
    return dq.to(orig_dtype)


def _mse_fp8_kv(kv_tensors: list, scale: float) -> float:
    """Average MSE between the raw and FP8-fake-quantized kv tensors for ``scale``."""
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
    """Grid-search (log-uniform) the multiplier minimising FP8 MSE on a flat fp32 tensor.
    On CUDA, only one ``argmin().item()`` sync is issued instead of one per step."""
    import math

    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    fp8_min = torch.finfo(torch.float8_e4m3fn).min  # -448.0

    log_min = math.log(min_multiplier)
    log_max = math.log(max_multiplier)

    if flat.is_cuda:
        # GPU path: accumulate on device; single sync at the end.
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
        # CPU path (fallback).
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
    """Convenience wrapper - concatenates tensors then delegates to the flat variant."""
    if not kv_tensors:
        return 1.0
    flat = torch.cat([t.reshape(-1).float() for t in kv_tensors])
    return _search_best_multiplier_flat(
        flat, base_scale, min_multiplier, max_multiplier, num_steps
    )


class KVScaleSearcher:
    """``llm.apply_model`` callable running the per-layer KV scale search inside each worker.
    ``activation_stats`` maps ``<attn>.k_cache`` / ``.v_cache`` to ``{"min", "max"}``."""

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

        # Collect raw kv tensors stored by the value hook
        kv_values = _get_kv_search_values(model)
        if not kv_values:
            print(
                "[KVScaleSearcher] WARNING: No kv values collected. "
                "Did you call setup_kvcache_value_hooks before inference?"
            )
            return {}

        # Concatenate on CPU (avoids OOM during cat), then move one flat tensor at a time
        # to GPU for the search.
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            # torch.cuda.current_device() is correct inside apply_model workers.
            search_device = torch.device("cuda", torch.cuda.current_device())
        else:
            search_device = torch.device("cpu")

        # torch.cat runs serially here, NOT in worker threads: concurrent large allocations
        # cause page-fault storms and multi-minute stalls.
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

        # GPU: run sequentially (CUDA already parallelises); CPU: thread pool.
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
    """Return the rank-0 multiplier dict from ``llm.apply_model`` results."""
    if not results_list:
        return {}
    first = results_list[0]
    if first is None:
        return {}
    return first


def _make_log_uniform_multipliers(
    min_multiplier: float,
    max_multiplier: float,
    num_steps: int,
) -> list[float]:
    """Build the log-uniform multiplier grid used by KV scale search."""
    import math

    if num_steps <= 1:
        return [float(min_multiplier)]

    log_min = math.log(min_multiplier)
    log_max = math.log(max_multiplier)
    return [
        math.exp(log_min + (log_max - log_min) * i / (num_steps - 1)) for i in range(num_steps)
    ]


def _compute_kv_mse_profile_flat(
    flat: torch.Tensor,
    base_scale: float,
    min_multiplier: float,
    max_multiplier: float,
    num_steps: int,
) -> dict:
    """Compute an additive SSE/numel profile for DP KV search aggregation."""
    multipliers = _make_log_uniform_multipliers(
        min_multiplier,
        max_multiplier,
        num_steps,
    )
    if flat.numel() == 0:
        return {
            "multipliers": multipliers,
            "sse": [0.0 for _ in multipliers],
            "numel": 0,
            "base_scale": base_scale,
        }

    fp8_min = torch.finfo(torch.float8_e4m3fn).min
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    sse = []

    if flat.is_cuda:
        sse_vals = torch.empty(len(multipliers), dtype=torch.float64, device=flat.device)
        for i, multiplier in enumerate(multipliers):
            scale = base_scale * multiplier
            q_fp8 = (
                (flat / scale).clamp(fp8_min, fp8_max).to(torch.float8_e4m3fn).to(torch.float32)
            )
            diff = flat - q_fp8 * scale
            sse_vals[i] = (diff.double() * diff.double()).sum()
        sse = [float(v) for v in sse_vals.cpu().tolist()]
        del sse_vals
    else:
        for multiplier in multipliers:
            scale = base_scale * multiplier
            q_fp8 = (
                (flat / scale).clamp(fp8_min, fp8_max).to(torch.float8_e4m3fn).to(torch.float32)
            )
            diff = flat - q_fp8 * scale
            sse.append(float((diff.double() * diff.double()).sum().item()))

    return {
        "multipliers": multipliers,
        "sse": sse,
        "numel": int(flat.numel()),
        "base_scale": base_scale,
    }


class KVMSEProfileCollector:
    """Collect local per-tensor KV MSE profiles for DP aggregation."""

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
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        kv_values = _get_kv_search_values(model)
        if not kv_values:
            print(
                "[KVMSEProfileCollector] WARNING: No kv values collected. "
                "Did you call setup_kvcache_value_hooks before inference?"
            )
            return {}

        use_gpu = torch.cuda.is_available()
        search_device = (
            torch.device("cuda", torch.cuda.current_device()) if use_gpu else torch.device("cpu")
        )
        profiles = {}

        for layer_name, tensors_dict in kv_values.items():
            for kv_slot, tensors in tensors_dict.items():
                stats_key = f"{layer_name}.{kv_slot}_cache"
                if stats_key not in self.activation_stats:
                    print(
                        f"[KVMSEProfileCollector] WARNING: {stats_key} not found in "
                        "activation_stats, skipping."
                    )
                    continue
                if not tensors:
                    continue

                stats = self.activation_stats[stats_key]
                abs_max = max(abs(stats["min"]), abs(stats["max"]))
                base_scale = abs_max / fp8_max * 2.0 if abs_max != 0 else 1e-8
                flat_cpu = torch.cat([t.reshape(-1).float() for t in tensors])
                flat = flat_cpu.to(search_device, non_blocking=True) if use_gpu else flat_cpu
                profile = _compute_kv_mse_profile_flat(
                    flat=flat,
                    base_scale=base_scale,
                    min_multiplier=self.min_multiplier,
                    max_multiplier=self.max_multiplier,
                    num_steps=self.num_steps,
                )
                if use_gpu:
                    del flat
                profiles[stats_key] = profile
                print(
                    f"[KVMSEProfileCollector] {stats_key}: "
                    f"numel={profile['numel']} base_scale={base_scale:.6f}"
                )

        return profiles


def get_kv_mse_profile_results(results_list: list) -> dict:
    """Extract the per-tensor MSE profile dict from ``llm.apply_model`` results."""
    if not results_list:
        return {}
    return results_list[0] or {}


def remove_kv_scale_search_hooks(model):
    """Clean up kv-value hooks after search; pass to ``llm.apply_model``."""
    _remove_kvcache_value_hooks(model)
    return "KV-search hooks removed"


# Public alias.
setup_kvcache_value_hooks = _setup_kvcache_value_hooks

# =============================================================================
# Per-head KV-cache value capture + scale search
# =============================================================================


class KVCachePerHeadValueHook:
    """Like ``KVCacheValueHook`` but stores ``(num_heads, seq_len_local, head_dim)`` tensors."""

    def __init__(
        self, layer_name: str, kvcache_values: dict, num_kv_heads_total: int | None = None
    ):
        self.layer_name = layer_name
        self.kvcache_values = kvcache_values
        self._num_heads: int | None = None
        self._num_kv_heads_total: int | None = num_kv_heads_total
        # Which side (K, V or both) this rank captures; resolved lazily on first call.
        self._role: str | None = None

    def __call__(self, module, input, output):
        _, k, v = input[0], input[1], input[2]
        if not isinstance(k, torch.Tensor):
            return

        with torch.no_grad():
            if self._num_heads is None:
                self._num_heads = num_heads_from_tensor(k, module)
            if self._role is None:
                rank, world_size = rank_world()
                if self._num_kv_heads_total is not None and self._num_kv_heads_total > 0:
                    num_kv_heads_total = self._num_kv_heads_total
                else:
                    num_kv_heads_total = self._num_heads * world_size
                self._role = get_kv_role(rank, world_size, num_kv_heads_total)
            H = self._num_heads

            def _to_head_layout(t: torch.Tensor):
                """Return tensor of shape (H, T, head_dim)."""
                last = t.shape[-1]
                if t.ndim >= 3 and t.shape[-2] == H:
                    # (..., H, D) -> (H, T, D)
                    t_h = t.reshape(-1, H, t.shape[-1])  # (T, H, D)
                    return t_h.permute(1, 0, 2).contiguous()  # (H, T, D)
                elif H > 0 and last % H == 0:
                    head_dim = last // H
                    t_h = t.reshape(-1, H, head_dim)  # (T, H, D)
                    return t_h.permute(1, 0, 2).contiguous()  # (H, T, D)
                else:
                    # Fallback: single pseudo-head
                    return t.reshape(1, -1, 1)

            # Only store the side this rank owns (saves ~50% CPU memory on replicated TP).
            if self._role in ("both", "k"):
                k_heads = _to_head_layout(k).detach().cpu()  # (H, T, D)
                self.kvcache_values[self.layer_name]["k"].append(k_heads)
            if self._role in ("both", "v"):
                v_heads = _to_head_layout(v).detach().cpu()  # (H, T, D)
                self.kvcache_values[self.layer_name]["v"].append(v_heads)


def _setup_kvcache_perhead_value_hooks(model):
    """Register per-head value-capture hooks (run via ``llm.apply_model``)."""
    try:
        # vLLM >= 0.20 (Tencent custom): Attention moved under model_executor.
        from vllm.model_executor.layers.attention import Attention
    except ImportError:
        # Older vLLM layout.
        from vllm.attention.layer import Attention

    kvcache_layers = find_modules(model, [Attention])

    # Prefer the head count saved by ``setup_kvcache_perhead_hooks``, else infer it.
    num_kv_heads_total = getattr(model, "_kvcache_num_kv_heads_total", None)
    if num_kv_heads_total is None:
        num_kv_heads_total = infer_num_kv_heads_total(model)
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
    """``llm.apply_model`` callable running the per-head KV scale search inside each worker.
    Returns ``{stats_key: {global_head_idx: multiplier}}`` for the heads this rank holds."""

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

        # Under K/V split (replication >= 2) each rank captures only its role; activation_stats
        # holds the full global per-head vectors, used to locate this rank's head slice.
        rank, world_size = rank_world()

        # Infer num_kv_heads_total from any entry's min/max length.
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
        role, heads_per_rank, global_head_offset, replication = compute_perhead_layout(
            rank, world_size, num_kv_heads_total
        )

        # Work items: (stats_key, global_head_idx, flat_cpu_tensor, base_scale).
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
                    # Expected on the opposite role; skip silently.
                    continue

                stats = self.activation_stats[stats_key]
                min_vals = stats["min"]  # list[float] of length num_kv_heads_total
                max_vals = stats["max"]  # list[float] of length num_kv_heads_total

                # tensors[i] has shape (H_local, T_i, D) - cat along T dimension
                stacked = torch.cat(tensors, dim=1)  # (H_local, total_T, D)
                H_local = stacked.shape[0]

                for local_h in range(H_local):
                    global_h = global_head_offset + local_h
                    if global_h >= num_kv_heads_total:
                        # Safety guard; should not happen.
                        continue
                    abs_max = max(abs(min_vals[global_h]), abs(max_vals[global_h]))
                    if abs_max == 0:
                        base_scale = 1e-8
                    else:
                        base_scale = abs_max / fp8_max * 2.0
                    # Extract head slice: shape (total_T, D) -> flatten -> (N,)
                    flat_cpu = stacked[local_h].reshape(-1).float()
                    work_items.append((stats_key, global_h, flat_cpu, base_scale))

        print(
            f"[KVScaleSearcherPerHead] rank={rank}/{world_size}, role={role}, "
            f"head_offset={global_head_offset}, H_total={num_kv_heads_total}, "
            f"replication={replication}, {len(work_items)} head-level work items "
            f"({'GPU' if use_gpu else 'CPU'}, {self.num_steps} steps each)"
        )

        # Keyed by global head index so per-rank partial results merge correctly.
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


class KVPerHeadMSEProfileCollector:
    """Collect local per-head KV MSE profiles for DP aggregation."""

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
        kv_values = _get_kv_perhead_search_values(model)
        if not kv_values:
            print(
                "[KVPerHeadMSEProfileCollector] WARNING: No per-head kv values collected. "
                "Did you call setup_kvcache_perhead_value_hooks before inference?"
            )
            return {}

        use_gpu = torch.cuda.is_available()
        search_device = (
            torch.device("cuda", torch.cuda.current_device()) if use_gpu else torch.device("cpu")
        )
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        rank, world_size = rank_world()
        num_kv_heads_total = None
        for stats in self.activation_stats.values():
            min_vals = stats.get("min")
            if isinstance(min_vals, list):
                num_kv_heads_total = len(min_vals)
                break
            if isinstance(min_vals, torch.Tensor):
                num_kv_heads_total = min_vals.numel()
                break
        if num_kv_heads_total is None or num_kv_heads_total <= 0:
            num_kv_heads_total = 1

        role, _, global_head_offset, replication = compute_perhead_layout(
            rank,
            world_size,
            num_kv_heads_total,
        )
        profiles: dict[str, dict] = {}

        for layer_name, tensors_dict in kv_values.items():
            for kv_slot, tensors in tensors_dict.items():
                if role != "both" and kv_slot != role:
                    continue

                stats_key = f"{layer_name}.{kv_slot}_cache"
                if stats_key not in self.activation_stats or not tensors:
                    continue

                stats = self.activation_stats[stats_key]
                min_vals = stats["min"]
                max_vals = stats["max"]
                stacked = torch.cat(tensors, dim=1)
                profiles.setdefault(stats_key, {})

                for local_head_idx in range(stacked.shape[0]):
                    global_head_idx = global_head_offset + local_head_idx
                    if global_head_idx >= num_kv_heads_total:
                        continue

                    abs_max = max(
                        abs(min_vals[global_head_idx]),
                        abs(max_vals[global_head_idx]),
                    )
                    base_scale = abs_max / fp8_max * 2.0 if abs_max != 0 else 1e-8
                    flat_cpu = stacked[local_head_idx].reshape(-1).float()
                    flat = flat_cpu.to(search_device, non_blocking=True) if use_gpu else flat_cpu
                    profile = _compute_kv_mse_profile_flat(
                        flat=flat,
                        base_scale=base_scale,
                        min_multiplier=self.min_multiplier,
                        max_multiplier=self.max_multiplier,
                        num_steps=self.num_steps,
                    )
                    if use_gpu:
                        del flat
                    profiles[stats_key][global_head_idx] = profile

        print(
            f"[KVPerHeadMSEProfileCollector] rank={rank}/{world_size}, role={role}, "
            f"head_offset={global_head_offset}, H_total={num_kv_heads_total}, "
            f"replication={replication}"
        )
        return profiles


def get_kv_scale_search_results_perhead(results_list: list) -> dict:
    """Merge ``{stats_key: {global_head_idx: m}}`` from all TP workers into per-head lists."""
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


def get_kv_mse_profile_results_perhead(results_list: list) -> dict:
    """Merge per-head MSE profiles from all TP workers into lists indexed by global head;
    missing heads are ``None`` so indices stay aligned for cross-DP aggregation."""
    if not results_list:
        return {}

    merged: dict = {}
    for worker_result in results_list:
        if not worker_result:
            continue
        for stats_key, head_profiles in worker_result.items():
            merged.setdefault(stats_key, {}).update(head_profiles)

    final: dict = {}
    for stats_key, head_profiles in merged.items():
        if not head_profiles:
            final[stats_key] = []
            continue
        max_head_idx = max(int(i) for i in head_profiles.keys())
        final[stats_key] = [head_profiles.get(i) for i in range(max_head_idx + 1)]
    return final
