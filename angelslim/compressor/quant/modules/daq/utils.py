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

"""Utility functions for DAQ model quantization pipeline."""

import json
import os
from collections import OrderedDict, defaultdict
from concurrent.futures import Future

import torch
from safetensors.torch import load_file

from angelslim.compressor.quant.core.kernels import weight_dequant
from angelslim.utils import print_info


def get_available_gpus():
    """Get list of available GPU device IDs."""
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def get_system_memory_gb() -> float:
    """Get total system (CPU) physical memory in GB. Returns 0 if cannot be determined."""
    try:
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return mem_bytes / (1024**3)
    except (ValueError, OSError, AttributeError):
        return 0.0


def get_weight_map(model_path: str) -> dict:
    """Load weight map from model.safetensors.index.json."""
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        return {}
    with open(index_file, "r") as f:
        index_data = json.load(f)
    return index_data.get("weight_map", {})


def load_base_shard(base_shard_path: str, device: str):
    """Load a base shard file to CPU memory."""
    return load_file(base_shard_path, device="cpu")


def _resolve_cache_entry(base_file_cache: OrderedDict, shard_file: str):
    """Resolve a cache entry, waiting for Future if it was prefetched."""
    entry = base_file_cache[shard_file]
    if isinstance(entry, Future):
        entry = entry.result()
        base_file_cache[shard_file] = entry
    return entry


def _evict_if_needed(base_file_cache: OrderedDict, max_cache_size: int):
    """Evict oldest cache entries if cache is full."""
    while len(base_file_cache) >= max_cache_size:
        evicted_key, evicted_val = base_file_cache.popitem(last=False)
        if isinstance(evicted_val, Future):
            evicted_val = evicted_val.result()
        del evicted_val


def load_base_weight(
    weight_name: str,
    base_path: str,
    base_weight_map: dict,
    base_file_cache: OrderedDict,
    device: str,
    max_cache_size: int = 3,
    base_is_fp8: bool = False,
):
    """Load a single base weight tensor, with LRU shard caching and FP8 dequant support."""
    if weight_name not in base_weight_map:
        return None

    base_shard_file = base_weight_map[weight_name]

    if base_shard_file in base_file_cache:
        base_file_cache.move_to_end(base_shard_file)
    else:
        base_shard_path = os.path.join(base_path, base_shard_file)
        if not os.path.exists(base_shard_path):
            return None

        _evict_if_needed(base_file_cache, max_cache_size)
        base_file_cache[base_shard_file] = load_file(base_shard_path, device="cpu")

    shard_dict = _resolve_cache_entry(base_file_cache, base_shard_file)
    weight = shard_dict.get(weight_name, None)
    if weight is not None:
        if base_is_fp8 and weight.element_size() == 1:
            scale_inv_name = f"{weight_name}_scale_inv"
            scale_inv = shard_dict.get(scale_inv_name, None)
            if scale_inv is None:
                if scale_inv_name in base_weight_map:
                    scale_shard_file = base_weight_map[scale_inv_name]
                    if scale_shard_file != base_shard_file:
                        if scale_shard_file in base_file_cache:
                            base_file_cache.move_to_end(scale_shard_file)
                        else:
                            scale_shard_path = os.path.join(base_path, scale_shard_file)
                            if os.path.exists(scale_shard_path):
                                _evict_if_needed(base_file_cache, max_cache_size)
                                base_file_cache[scale_shard_file] = load_file(
                                    scale_shard_path, device="cpu"
                                )
                        scale_shard_dict = _resolve_cache_entry(base_file_cache, scale_shard_file)
                        scale_inv = scale_shard_dict.get(scale_inv_name, None)

            if scale_inv is not None:
                weight = weight.to(device, non_blocking=True)
                scale_inv = scale_inv.to(device, non_blocking=True)
                weight = weight_dequant(weight, scale_inv)
            else:
                print_info(
                    f"Warning: scale_inv not found for FP8 base weight {weight_name}, "
                    f"casting FP8->BF16 without dequant"
                )
                weight = weight.to(device, non_blocking=True).to(torch.bfloat16)
        else:
            weight = weight.to(device, non_blocking=True)
    return weight


def prefetch_base_shard(
    base_shard_file: str,
    base_path: str,
    base_file_cache: OrderedDict,
    device: str,
    executor,
    max_cache_size: int = 3,
):
    """Submit a prefetch task to load a base shard file in the background."""
    if base_shard_file in base_file_cache:
        return

    base_shard_path = os.path.join(base_path, base_shard_file)
    if os.path.exists(base_shard_path):
        _evict_if_needed(base_file_cache, max_cache_size)

        future = executor.submit(load_base_shard, base_shard_path, "cpu")
        base_file_cache[base_shard_file] = future


def compute_dynamic_cache_size(
    sft_index_file: str,
    base_weight_map: dict,
    base_path: str = "",
    num_workers_per_gpu: int = 2,
    cpu_memory_budget_gb: float = 0,
) -> int:
    """Compute optimal max_cache_size for CPU-based base shard caching."""
    if not os.path.exists(sft_index_file) or not base_weight_map:
        return 3

    try:
        with open(sft_index_file, "r") as f:
            sft_index = json.load(f)
        sft_weight_map = sft_index.get("weight_map", {})
    except (json.JSONDecodeError, KeyError):
        return 3

    sft_shard_to_weights = defaultdict(list)
    for weight_name, sft_shard in sft_weight_map.items():
        sft_shard_to_weights[sft_shard].append(weight_name)

    max_base_shards_needed = 0
    for _sft_shard, weight_names in sft_shard_to_weights.items():
        base_shards_needed = set()
        for wn in weight_names:
            base_shard = base_weight_map.get(wn)
            if base_shard:
                base_shards_needed.add(base_shard)
        max_base_shards_needed = max(max_base_shards_needed, len(base_shards_needed))

    unique_base_shards = set(base_weight_map.values())
    num_base_shards = len(unique_base_shards)

    if num_base_shards == 0:
        return 3

    avg_shard_size_gb = 5.0  # fallback
    if base_path:
        sample_sizes = []
        for shard_name in list(unique_base_shards)[:10]:
            shard_path = os.path.join(base_path, shard_name)
            if os.path.exists(shard_path):
                sample_sizes.append(os.path.getsize(shard_path))
        if sample_sizes:
            avg_shard_size_gb = (sum(sample_sizes) / len(sample_sizes)) / (1024**3)

    system_memory_gb = get_system_memory_gb()
    if cpu_memory_budget_gb <= 0:
        if system_memory_gb > 0:
            cpu_memory_budget_gb = system_memory_gb * 0.6
        else:
            cpu_memory_budget_gb = 250.0
    else:
        if system_memory_gb > 0:
            safe_limit = system_memory_gb * 0.6
            if cpu_memory_budget_gb > safe_limit:
                print_info(
                    f"Warning: User budget {cpu_memory_budget_gb:.0f}GB exceeds safe limit "
                    f"({safe_limit:.0f}GB = 60% of {system_memory_gb:.0f}GB), capping."
                )
                cpu_memory_budget_gb = safe_limit

    print_info(
        f"System memory: {system_memory_gb:.0f}GB, "
        f"CPU cache budget: {cpu_memory_budget_gb:.0f}GB"
    )

    total_num_workers = num_workers_per_gpu * max(1, torch.cuda.device_count())
    per_worker_overhead_gb = avg_shard_size_gb * 1.5
    total_overhead_gb = per_worker_overhead_gb * total_num_workers
    available_for_cache_gb = max(0, cpu_memory_budget_gb - total_overhead_gb)

    per_worker_cache_budget_gb = available_for_cache_gb / max(1, total_num_workers)

    if avg_shard_size_gb > 0:
        memory_limited_cache = max(3, int(per_worker_cache_budget_gb / avg_shard_size_gb))
    else:
        memory_limited_cache = 3

    memory_limited_cache = min(memory_limited_cache, 50)

    ideal_cache = max_base_shards_needed + 1

    cache_size = max(3, min(ideal_cache, memory_limited_cache))

    print_info(
        f"Cache sizing: max_shards_needed={max_base_shards_needed}, "
        f"avg_shard={avg_shard_size_gb:.2f}GB, "
        f"workers={total_num_workers}, "
        f"worker_overhead={total_overhead_gb:.0f}GB, "
        f"avail_for_cache={available_for_cache_gb:.0f}GB, "
        f"per_worker_cache={per_worker_cache_budget_gb:.1f}GB, "
        f"memory_limit={memory_limited_cache}, ideal={ideal_cache}, "
        f"final={cache_size}"
    )

    if cache_size < ideal_cache:
        print_info(
            f"Warning: Cache size ({cache_size}) < ideal ({ideal_cache}). "
            f"This will cause cache thrashing and slower performance. "
            f"Consider: reducing num_workers, or using a machine with more RAM."
        )

    return cache_size
