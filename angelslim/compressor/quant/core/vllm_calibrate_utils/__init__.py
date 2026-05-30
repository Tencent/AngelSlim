"""AngelSlim vLLM calibration utilities.

Originally a single ``vllm_calibrate_utils.py`` file; split into three
focused sub-modules while keeping the public API 100% backward compatible:

* :mod:`._common`  – low-level helpers (``_find_layers``, distributed
  utilities, all-reduce, table printing, per-head role assignment).
* :mod:`.hooks`    – everything that registers a forward hook on the
  model: linear-layer activations, per-tensor / per-head KV-cache
  statistics, KV-only lightweight calibration, FusedMoE expert stats
  (vLLM patch entry point ``collect_fused_moe_internal_stats``) and the
  MTP draft-model variants.
* :mod:`.search`   – KV-cache FP8 scale grid-search (per-tensor and
  per-head) with the value-capture hooks needed by the searchers.

The vLLM ``fused_moe.py`` patch only imports
``collect_fused_moe_internal_stats`` from this package, which is
re-exported via :mod:`.hooks`.
"""

from .hooks import (
    ActivationHook,
    KVCacheHook,
    KVCachePerHeadHook,
    collect_fused_moe_internal_stats,
    get_activation_stats,
    get_kvcache_only_stats,
    get_kvcache_perhead_stats,
    get_moe_stats,
    get_mtp_activation_stats,
    get_mtp_moe_stats,
    print_activation_stats,
    print_kvcache_only_stats,
    print_kvcache_perhead_stats,
    print_moe_stats,
    print_mtp_activation_stats,
    print_mtp_moe_stats,
    remove_kvcache_only_hooks,
    remove_kvcache_perhead_hooks,
    setup_activation_hooks,
    setup_kvcache_only_hooks,
    setup_kvcache_perhead_hooks,
    setup_kvcache_pertensor_hooks,
    setup_mtp_activation_hooks,
)
from .search import (
    KVCachePerHeadValueHook,
    KVCacheValueHook,
    KVScaleSearcher,
    KVScaleSearcherPerHead,
    get_kv_scale_search_results,
    get_kv_scale_search_results_perhead,
    remove_kv_scale_search_hooks,
    remove_kvcache_perhead_value_hooks,
    setup_kvcache_perhead_value_hooks,
    setup_kvcache_value_hooks,
)

__all__ = [
    # Activation / KV / MoE / MTP hooks
    "ActivationHook",
    "KVCacheHook",
    "KVCachePerHeadHook",
    "setup_activation_hooks",
    "get_activation_stats",
    "print_activation_stats",
    "print_moe_stats",
    "get_moe_stats",
    "collect_fused_moe_internal_stats",
    "setup_mtp_activation_hooks",
    "get_mtp_activation_stats",
    "print_mtp_activation_stats",
    "get_mtp_moe_stats",
    "print_mtp_moe_stats",
    # KV-only calibration (no weight / activation hooks)
    "setup_kvcache_only_hooks",
    "get_kvcache_only_stats",
    "print_kvcache_only_stats",
    "remove_kvcache_only_hooks",
    # Per-tensor KV calibration setup
    "setup_kvcache_pertensor_hooks",
    # Per-head KV calibration
    "setup_kvcache_perhead_hooks",
    "get_kvcache_perhead_stats",
    "print_kvcache_perhead_stats",
    "remove_kvcache_perhead_hooks",
    # KV scale search (per-tensor)
    "KVCacheValueHook",
    "setup_kvcache_value_hooks",
    "KVScaleSearcher",
    "get_kv_scale_search_results",
    "remove_kv_scale_search_hooks",
    # KV scale search (per-head)
    "KVCachePerHeadValueHook",
    "setup_kvcache_perhead_value_hooks",
    "remove_kvcache_perhead_value_hooks",
    "KVScaleSearcherPerHead",
    "get_kv_scale_search_results_perhead",
]
