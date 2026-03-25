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

"""Delta-Aware Quantization (DAQ) main class."""

import copy
import json
import multiprocessing as mp
import os
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from glob import glob

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from angelslim.compressor.quant.core.kernels import FP8_E4M3_SCHEME, create_quant_kernel
from angelslim.utils import print_info

from .scale_search import scale_search_weight_quant
from .utils import (
    compute_dynamic_cache_size,
    get_available_gpus,
    get_weight_map,
    load_base_weight,
    prefetch_base_shard,
)

__all__ = ["DAQ"]


@dataclass
class ProcessingStats:
    """Statistics collected during DAQ quantization of a single file."""

    total_blocks: int = 0
    blocks_improved: int = 0
    total_channels: int = 0
    channels_improved: int = 0
    preservation_rate_sum: float = 0.0
    baseline_rate_sum: float = 0.0
    weight_count: int = 0
    scale_search_count: int = 0
    standard_quant_count: int = 0

    def merge(self, other: "ProcessingStats"):
        """Accumulate statistics from another ProcessingStats instance."""
        self.total_blocks += other.total_blocks
        self.blocks_improved += other.blocks_improved
        self.total_channels += other.total_channels
        self.channels_improved += other.channels_improved
        self.preservation_rate_sum += other.preservation_rate_sum
        self.baseline_rate_sum += other.baseline_rate_sum
        self.weight_count += other.weight_count
        self.scale_search_count += other.scale_search_count
        self.standard_quant_count += other.standard_quant_count

    @property
    def avg_preservation_rate(self) -> float:
        return self.preservation_rate_sum / self.weight_count if self.weight_count > 0 else 0.0

    @property
    def avg_baseline_rate(self) -> float:
        return self.baseline_rate_sum / self.weight_count if self.weight_count > 0 else 0.0


class DAQ:
    """Delta-Aware Quantization (DAQ) module."""

    # compressed-tensors config template for per_channel quantization
    _PER_CHANNEL_CONFIG_TEMPLATE = {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "memoryless",
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "float",
                },
                "output_activations": None,
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "float",
                },
                "targets": ["Linear"],
            }
        },
        "format": "float-quantized",
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }

    def __init__(self, quant_config, sft_model_path: str):
        self.sft_model_path = sft_model_path
        self.base_model_path = quant_config.base_model_path
        self.base_is_fp8 = quant_config.base_is_fp8
        self.metric = quant_config.metric
        self.quantization_method = quant_config.quantization_method
        self.num_workers = quant_config.num_workers
        self.ignore_layers = getattr(quant_config, "ignore_layers", []) or []
        self.base_model_repo = quant_config.base_model_repo

        gpus_str = quant_config.gpus
        if gpus_str:
            self.gpus = [int(x.strip()) for x in gpus_str.split(",")]
        else:
            self.gpus = None

        scale_search = quant_config.scale_search or {}
        self.scale_search_kwargs = {
            "min_range": scale_search.get("min_range", 0.8),
            "max_range": scale_search.get("max_range", 1.5),
            "coarse_intervals": scale_search.get("coarse_intervals", 5),
            "fine_intervals": scale_search.get("fine_intervals", 10),
            "delta_threshold": scale_search.get("delta_threshold", 1e-5),
        }

        self.quant_scheme = FP8_E4M3_SCHEME

        self._new_weight_map = {}

        self._validate_config()

    def _validate_config(self):
        if not self.base_model_path:
            raise ValueError(
                "DAQ requires 'base_model_path' to be specified. "
                "This should point to the base model directory."
            )
        if not os.path.isdir(self.base_model_path):
            raise FileNotFoundError(
                f"DAQ base_model_path does not exist or is not a directory: "
                f"{self.base_model_path}"
            )

        base_safetensors = glob(os.path.join(self.base_model_path, "*.safetensors"))
        if not base_safetensors:
            raise FileNotFoundError(
                f"No .safetensors files found in base_model_path: {self.base_model_path}"
            )

        available_gpus = get_available_gpus()
        if not available_gpus:
            raise RuntimeError("No CUDA GPUs available for DAQ quantization!")

        if self.gpus is None:
            self.gpus = available_gpus
        else:
            for gpu_id in self.gpus:
                if gpu_id not in available_gpus:
                    raise ValueError(
                        f"GPU {gpu_id} is not available. " f"Available GPUs: {available_gpus}"
                    )

        max_workers_per_gpu = 2
        max_workers = len(self.gpus) * max_workers_per_gpu
        if self.num_workers > max_workers:
            print_info(
                f"Warning: Reducing num_workers from {self.num_workers} to {max_workers} "
                f"(based on {len(self.gpus)} GPUs x {max_workers_per_gpu} workers/GPU)"
            )
            self.num_workers = max_workers

    def run(self, save_path: str):
        """Execute the DAQ quantization pipeline."""
        torch.set_default_dtype(torch.bfloat16)
        os.makedirs(save_path, exist_ok=True)

        if not os.path.isdir(self.sft_model_path):
            raise FileNotFoundError(
                f"DAQ SFT model path does not exist or is not a directory: "
                f"{self.sft_model_path}"
            )
        sft_safetensors = glob(os.path.join(self.sft_model_path, "*.safetensors"))
        if not sft_safetensors:
            raise FileNotFoundError(
                f"No .safetensors files found in SFT model path: {self.sft_model_path}"
            )

        print_info("=" * 60)
        print_info("Delta-Aware Quantization (DAQ)")
        print_info("=" * 60)
        print_info(f"Metric: {self.metric}")
        print_info(f"Quantization method: {self.quantization_method}")
        print_info(f"Scale search kwargs: {self.scale_search_kwargs}")
        print_info(f"SFT model path: {self.sft_model_path}")
        print_info(f"Base model path: {self.base_model_path}")
        print_info(
            f"Base model format: "
            f"{'FP8 (will dequantize on the fly)' if self.base_is_fp8 else 'BF16'}"
        )
        print_info(f"Output path: {save_path}")
        print_info(f"GPUs: {self.gpus}")
        print_info(f"Workers: {self.num_workers}")
        print_info("=" * 60)

        self._prepare_output_dir(save_path)

        model_index_file = os.path.join(save_path, "model.safetensors.index.json")
        with open(model_index_file, "r") as f:
            model_index = json.load(f)
        weight_map = model_index["weight_map"]

        base_weight_map = get_weight_map(self.base_model_path)
        if not base_weight_map:
            print_info(
                "Warning: Base model index not found, "
                "will fall back to standard quantization for all weights"
            )

        safetensor_files = sorted(glob(os.path.join(self.sft_model_path, "*.safetensors")))
        print_info(f"Found {len(safetensor_files)} safetensor files to process")

        sft_index_file = os.path.join(self.sft_model_path, "model.safetensors.index.json")
        dynamic_cache_size = compute_dynamic_cache_size(
            sft_index_file,
            base_weight_map,
            self.base_model_path,
            num_workers_per_gpu=max(1, self.num_workers // len(self.gpus)),
        )

        if self.num_workers > 1:
            results = self._run_multiprocess(
                safetensor_files,
                self.base_model_path,
                save_path,
                weight_map,
                base_weight_map,
                dynamic_cache_size,
            )
        else:
            results = self._run_single_process(
                safetensor_files,
                self.base_model_path,
                save_path,
                weight_map,
                base_weight_map,
                dynamic_cache_size,
            )

        total_stats = ProcessingStats()
        for file_weight_map, file_stats in results:
            self._new_weight_map.update(file_weight_map)
            total_stats.merge(file_stats)

        self._print_statistics(total_stats)

        model_index["weight_map"] = self._new_weight_map
        with open(model_index_file, "w", encoding="utf-8") as f:
            json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)
        print_info(f"Updated model.safetensors.index.json in {save_path}")

        self._update_config_json(save_path)

        print_info("DAQ quantization complete!")

    def _prepare_output_dir(self, save_path: str):
        # TODO: Currently we only support quantizing BF16 DeepSeek V3/R1 models to FP8.
        # To support all model architectures, the logic for determining which weights
        # to quantize should be changed from referencing the target model's
        # model.safetensors.index.json to using regex-based include/exclude lists
        # (e.g. regex patterns for weights to quantize and weights to ignore).
        model_index_file = os.path.join(save_path, "model.safetensors.index.json")
        config_file = os.path.join(save_path, "config.json")

        # Check if files need to be downloaded
        if not os.path.exists(model_index_file) or not os.path.exists(config_file):
            print(f"Model index or config file not found in {save_path}")
            print(f"Downloading config files from HuggingFace: {self.base_model_repo}")
            try:
                snapshot_download(
                    repo_id=self.base_model_repo,
                    ignore_patterns=["*.safetensors"],
                    local_dir=save_path,
                    local_dir_use_symlinks=False,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download config files from HuggingFace repo "
                    f"'{self.base_model_repo}'. Please check your network connection "
                    f"and ensure the repo_id is correct. Original error: {e}"
                ) from e
            print(f"✓ Model index file and config file downloaded to {save_path}")

    def _update_config_json(self, save_path: str):
        config_file = os.path.join(save_path, "config.json")
        if not os.path.exists(config_file):
            return

        with open(config_file, "r") as f:
            config = json.load(f)

        if self.quantization_method == "blockwise":
            quant_config = config.get("quantization_config", {})
            quant_config.pop("fmt", None)
            quant_config["quant_method"] = "fp8"
            quant_config["activation_scheme"] = "dynamic"
            quant_config["weight_block_size"] = [128, 128]
            config["quantization_config"] = quant_config
        elif self.quantization_method == "per_channel":
            per_channel_config = copy.deepcopy(self._PER_CHANNEL_CONFIG_TEMPLATE)
            per_channel_config["ignore"] = self.ignore_layers
            config["quantization_config"] = per_channel_config

        config["quantization_config"]["scale_search"] = {
            "enabled": True,
            "metric": self.metric,
            "kwargs": self.scale_search_kwargs,
        }

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
        print_info(f"Updated config.json in {save_path}")

    def _run_single_process(
        self,
        safetensor_files,
        base_path,
        save_path,
        weight_map,
        base_weight_map,
        dynamic_cache_size,
    ):
        device = f"cuda:{self.gpus[0]}"
        results = []
        for safetensor_file in tqdm(safetensor_files, desc="Worker 0", unit="file"):
            result = _process_single_file(
                safetensor_file,
                base_path,
                save_path,
                weight_map,
                base_weight_map,
                self.scale_search_kwargs,
                True,
                device,
                self.metric,
                self.quantization_method,
                dynamic_cache_size,
                self.base_is_fp8,
                self.ignore_layers,
                self.quant_scheme,
            )
            results.append(result)
        return results

    def _run_multiprocess(
        self,
        safetensor_files,
        base_path,
        save_path,
        weight_map,
        base_weight_map,
        dynamic_cache_size,
    ):
        ctx = mp.get_context("spawn")
        os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_main"

        # Group files by worker_id so each worker gets its own batch
        worker_file_groups = [[] for _ in range(self.num_workers)]
        worker_devices = [None] * self.num_workers
        for i, f in enumerate(safetensor_files):
            wid = i % self.num_workers
            gpu_id = self.gpus[i % len(self.gpus)]
            worker_file_groups[wid].append(f)
            worker_devices[wid] = f"cuda:{gpu_id}"

        worker_args = []
        for wid in range(self.num_workers):
            if not worker_file_groups[wid]:
                continue
            worker_args.append(
                (
                    wid,
                    worker_file_groups[wid],
                    base_path,
                    save_path,
                    weight_map,
                    base_weight_map,
                    self.scale_search_kwargs,
                    worker_devices[wid],
                    self.metric,
                    self.quantization_method,
                    dynamic_cache_size,
                    self.base_is_fp8,
                    self.ignore_layers,
                    self.quant_scheme,
                    self.num_workers,
                )
            )

        with ctx.Pool(processes=self.num_workers) as pool:
            nested_results = pool.map(_worker_process_files, worker_args)

        # Print newlines to avoid overwriting the last tqdm bars
        print("\n" * self.num_workers)

        # Flatten results from all workers
        results = []
        for worker_results in nested_results:
            results.extend(worker_results)

        return results

    def _print_statistics(self, stats: ProcessingStats):
        print_info("")
        print_info("=" * 60)
        print_info("Quantization Statistics:")
        print_info("=" * 60)
        print_info(f"  Scale search quantized: {stats.scale_search_count} weight tensors")
        print_info(f"  Standard quantized:     {stats.standard_quant_count} weight tensors")
        print_info(
            f"  Total quantized:        "
            f"{stats.scale_search_count + stats.standard_quant_count} weight tensors"
        )
        print_info("=" * 60)
        print_info(f"Scale Search Statistics (metric={self.metric}):")

        if self.quantization_method == "per_channel":
            print_info(f"  Total channels:         {stats.total_channels:,}")
            print_info(f"  Channels improved:      {stats.channels_improved:,}")
            if stats.total_channels > 0:
                rate = 100 * stats.channels_improved / stats.total_channels
                print_info(f"  Improvement rate:       {rate:.2f}%")
        else:
            print_info(f"  Total blocks:           {stats.total_blocks:,}")
            print_info(f"  Blocks improved:        {stats.blocks_improved:,}")
            if stats.total_blocks > 0:
                rate = 100 * stats.blocks_improved / stats.total_blocks
                print_info(f"  Improvement rate:       {rate:.2f}%")

        if stats.weight_count > 0:
            avg_rate = stats.avg_preservation_rate
            avg_baseline = stats.avg_baseline_rate
            metric_name = (
                "sign preservation"
                if self.metric == "sign"
                else ("cosine similarity" if self.metric == "cosine" else "negative MSE")
            )
            if self.metric == "mse":
                print_info(f"  Avg {metric_name} (before opt):  {avg_baseline:.6f}")
                print_info(f"  Avg {metric_name} (after opt):   {avg_rate:.6f}")
                print_info(f"  Improvement:            {avg_rate - avg_baseline:+.6f}")
            else:
                print_info(f"  Avg {metric_name} (before opt):  {avg_baseline:.2%}")
                print_info(f"  Avg {metric_name} (after opt):   {avg_rate:.2%}")
                print_info(f"  Improvement:            {avg_rate - avg_baseline:+.2%}")

        print_info("=" * 60)


# Standalone functions for multiprocessing


def _worker_process_files(args):
    """Entry point for each worker process. Processes a batch of files with a tqdm bar."""
    (
        worker_id,
        file_list,
        base_path,
        save_path,
        weight_map,
        base_weight_map,
        scale_search_kwargs,
        device,
        metric,
        quantization_method,
        max_cache_size,
        base_is_fp8,
        ignore_layers,
        quant_scheme,
        num_workers,
    ) = args

    results = []
    for safetensor_file in tqdm(
        file_list,
        desc=f"Worker {worker_id}",
        unit="file",
        position=worker_id,
        leave=True,
    ):
        result = _process_single_file(
            safetensor_file,
            base_path,
            save_path,
            weight_map,
            base_weight_map,
            scale_search_kwargs,
            False,
            device,
            metric,
            quantization_method,
            max_cache_size,
            base_is_fp8,
            ignore_layers,
            quant_scheme,
        )
        results.append(result)
    return results


def _process_single_file(
    safetensor_file,
    base_path,
    fp8_path,
    weight_map,
    base_weight_map,
    scale_search_kwargs,
    verbose,
    device,
    metric,
    quantization_method,
    max_cache_size,
    base_is_fp8,
    ignore_layers,
    quant_scheme=None,
):
    """Process a single safetensor file with DAQ quantization."""
    gpu_id = int(device.split(":")[1])
    torch.cuda.set_device(gpu_id)

    # Set unique Triton cache directory per process
    pid = os.getpid()
    triton_cache_dir = f"/tmp/triton_cache_worker_{pid}"
    os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
    os.makedirs(triton_cache_dir, exist_ok=True)

    quant_kernel = create_quant_kernel(quant_scheme or FP8_E4M3_SCHEME)

    file_name = os.path.basename(safetensor_file)
    scale_search_kwargs = scale_search_kwargs or {}

    sft_state_dict = load_file(safetensor_file, device=device)

    base_file_cache = OrderedDict()

    new_state_dict = {}
    new_weight_map = {}
    stats = ProcessingStats()

    weight_names = list(sft_state_dict.keys())

    shard_to_weights = defaultdict(list)
    no_shard_weights = []
    for wn in weight_names:
        base_shard = base_weight_map.get(wn, None)
        if base_shard is not None:
            shard_to_weights[base_shard].append(wn)
        else:
            no_shard_weights.append(wn)

    sorted_weight_names = []
    sorted_shards_order = []
    for shard_file, weights in shard_to_weights.items():
        sorted_shards_order.append(shard_file)
        sorted_weight_names.extend(weights)
    sorted_weight_names.extend(no_shard_weights)

    executor = ThreadPoolExecutor(max_workers=1)

    if sorted_shards_order:
        prefetch_base_shard(
            sorted_shards_order[0],
            base_path,
            base_file_cache,
            device,
            executor,
            max_cache_size,
        )

    last_shard = None

    for weight_name in sorted_weight_names:
        current_base_shard = base_weight_map.get(weight_name, None)
        if current_base_shard is not None and current_base_shard != last_shard:
            last_shard = current_base_shard
            try:
                idx = sorted_shards_order.index(current_base_shard)
                next_shard_idx = idx + 1
                if next_shard_idx < len(sorted_shards_order):
                    prefetch_base_shard(
                        sorted_shards_order[next_shard_idx],
                        base_path,
                        base_file_cache,
                        device,
                        executor,
                        max_cache_size,
                    )
            except ValueError:
                pass

        weight = sft_state_dict[weight_name]
        scale_inv_name = f"{weight_name}_scale_inv"

        should_ignore = any(ignore_pattern in weight_name for ignore_pattern in ignore_layers)

        if scale_inv_name in weight_map and not should_ignore:
            assert weight.element_size() == 2, f"Expected BF16, got {weight.dtype}"

            base_weight = load_base_weight(
                weight_name,
                base_path,
                base_weight_map,
                base_file_cache,
                device,
                max_cache_size,
                base_is_fp8=base_is_fp8,
            )

            if base_weight is not None and base_weight.shape == weight.shape:
                fp8_weight, scale_out, search_stats = scale_search_weight_quant(
                    weight,
                    base_weight,
                    quantization_method=quantization_method,
                    block_size=128,
                    metric=metric,
                    verbose=verbose,
                    quant_kernel=quant_kernel,
                    **scale_search_kwargs,
                )

                if quantization_method == "per_channel":
                    stats.total_channels += search_stats["total_channels"]
                    stats.channels_improved += search_stats["channels_improved"]
                else:
                    stats.total_blocks += search_stats["total_blocks"]
                    stats.blocks_improved += search_stats["blocks_improved"]

                stats.preservation_rate_sum += search_stats["avg_preservation_rate"]
                stats.baseline_rate_sum += search_stats.get("avg_baseline_rate", 0.0)
                stats.weight_count += 1

                if verbose:
                    print_info(
                        f"  {weight_name}: "
                        f"baseline={search_stats.get('avg_baseline_rate', 0.0):.2%}, "
                        f"optimized={search_stats['avg_preservation_rate']:.2%}"
                    )

                stats.scale_search_count += 1
                del base_weight
            else:
                if verbose:
                    if base_weight is None:
                        print_info(
                            f"  Base weight not found for {weight_name}, "
                            f"using standard quantization"
                        )
                    else:
                        print_info(
                            f"  Shape mismatch for {weight_name}: "
                            f"SFT={weight.shape}, Base={base_weight.shape}"
                        )
                        del base_weight
                elif base_weight is not None:
                    del base_weight

                fp8_weight, scale_out = quant_kernel.quantize_with_scale_compute(
                    weight,
                    quantization_method=quantization_method,
                    block_size=128,
                )
                stats.standard_quant_count += 1

            if quantization_method == "per_channel":
                scale_inv_name = scale_inv_name.replace("_scale_inv", "_scale")

            new_state_dict[weight_name] = fp8_weight.cpu()
            new_state_dict[scale_inv_name] = scale_out.cpu()
            new_weight_map[weight_name] = file_name
            new_weight_map[scale_inv_name] = file_name

            del fp8_weight, scale_out
        else:
            new_state_dict[weight_name] = weight.cpu()
            new_weight_map[weight_name] = file_name

        del weight

    del sft_state_dict
    del base_file_cache
    executor.shutdown(wait=True)
    torch.cuda.empty_cache()

    new_safetensor_file = os.path.join(fp8_path, file_name)
    save_file(new_state_dict, new_safetensor_file)

    return new_weight_map, stats
