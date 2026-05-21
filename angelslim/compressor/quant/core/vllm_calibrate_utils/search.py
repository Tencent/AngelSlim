"""KV-cache FP8 scale search (per-tensor and per-head).

Combines the original ``kv_scale_search`` and ``kv_perhead_search`` modules.
Both pipelines share the same shape:

1. A *value-capture* hook (``KVCacheValueHook`` / ``KVCachePerHeadValueHook``)
   stashes raw BF16 K/V tensors during a calibration forward pass.
2. A *searcher* class (``KVScaleSearcher`` / ``KVScaleSearcherPerHead``)
   walks those tensors and grid-searches the best FP8 multiplier per layer
   (or per head) by minimising fake-quant MSE.

The internal helpers ``_fp8_quantize_dequant`` / ``_search_best_multiplier``
are kept module-private with the underscore prefix.
"""

import torch

from ._common import _compute_perhead_layout, _find_layers, _get_dist_info, _get_kv_role
from .hooks import _get_num_heads_from_tensor, _infer_num_kv_heads_total

__all__ = [
    # Per-tensor scale search
    "KVCacheValueHook",
    "KVScaleSearcher",
    "setup_kvcache_value_hooks",
    "get_kv_scale_search_results",
    "remove_kv_scale_search_hooks",
    # Per-head scale search
    "KVCachePerHeadValueHook",
    "setup_kvcache_perhead_value_hooks",
    "remove_kvcache_perhead_value_hooks",
    "KVScaleSearcherPerHead",
    "get_kv_scale_search_results_perhead",
]


# =============================================================================
# Per-tensor KV-cache scale search
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
# Per-head KV-cache value capture + scale search
# =============================================================================


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
