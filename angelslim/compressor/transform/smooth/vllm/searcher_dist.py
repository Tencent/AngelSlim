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

"""Distributed (TP-aware) wrapper around the per-layer alpha grid search.

The pure-tensor inner loop lives in ``smooth/core/searcher.py``; this
module adds:

* per-layer module discovery (dense MLP via ``model.get_submodule``,
  MoE expert via ``FusedMoE.w2_weight[expert_id]``);
* reading the TP-local activation absmax shard from
  ``model._smooth_stats``;
* concatenating per-rank ``smooth_weight`` shards and ``alpha`` values
  via ``dist.all_gather``;
* writing the result JSON on rank 0.
"""

import os
import re

import torch

from angelslim.compressor.quant.core.vllm_calibrate_utils._common import _get_dist_info

from ..config import SmoothAlphaSearchConfig
from ..core.stats_search import smooth_alpha_search_layer

__all__ = [
    "SmoothAlphaSearcher",
]


def _get_down_proj_weight(model, layer_key):
    """Resolve the ``down_proj`` weight tensor (TP-local shard) for a
    given stat key.

    Dense MLP key  : ``"model.layers.0.mlp.down_proj"``
                     → ``model.get_submodule(key).weight``  ``[out, in_shard]``

    MoE expert key : ``"model.layers.3.mlp.experts.42.down_proj"``
                     → ``moe = model.get_submodule("model.layers.3.mlp.experts")``
                     → ``moe.w2_weight[42]``  ``[out, in_shard]``
    """
    # Try direct submodule access (dense MLP).
    try:
        mod = model.get_submodule(layer_key)
        if hasattr(mod, "weight"):
            return mod.weight.detach()
    except (AttributeError, ValueError):
        pass

    # MoE pattern: *.experts.<expert_id>.down_proj
    moe_match = re.match(r"^(.+)\.(\d+)\.down_proj$", layer_key)
    if moe_match:
        moe_prefix = moe_match.group(1)
        expert_id = int(moe_match.group(2))
        try:
            moe_module = model.get_submodule(moe_prefix)
        except (AttributeError, ValueError):
            parent = moe_prefix.rsplit(".", 1)[0] if "." in moe_prefix else moe_prefix
            try:
                moe_module = model.get_submodule(parent)
            except (AttributeError, ValueError):
                return None

        # FusedMoE stores w2 (down_proj) as w2_weight: [num_experts, out, in_shard]
        w2 = getattr(moe_module, "w2_weight", None)
        if w2 is not None and expert_id < w2.shape[0]:
            return w2[expert_id].detach()

    return None


class SmoothAlphaSearcher:
    """Per-layer alpha grid search for SmoothQuant ``down_proj``.

    Must be called via ``llm.apply_model()`` **before**
    ``get_smooth_stats()`` so that ``model._smooth_stats`` still contains
    the TP-local shard of absmax.

    Each rank independently searches its own shard, then ``all_gather``
    is used to concatenate ``smooth_weight`` shards into the full
    ``[intermediate_size]`` vector.

    Usage::

        searcher = SmoothAlphaSearcher(config, output_path="path/to/output.json")
        summary_list = llm.apply_model(searcher)
        # Results are saved to output_path by rank 0 inside the worker.
    """

    def __init__(self, config: SmoothAlphaSearchConfig, output_path: str = None):
        self.config = config
        self.output_path = output_path

    def __call__(self, model):
        import torch.distributed as dist

        rank, world_size = _get_dist_info()
        smooth_stats = getattr(model, "_smooth_stats", {})
        alpha_values = getattr(model, "_alpha_search_values", {})

        if not alpha_values:
            print("[SmoothAlphaSearcher] No activation tensors collected, skipping.")
            return {}

        cfg = self.config
        use_gpu = torch.cuda.is_available()
        device = (
            torch.device("cuda", torch.cuda.current_device()) if use_gpu else torch.device("cpu")
        )

        results = {}
        processed = 0
        skipped = 0

        for layer_key, stored in alpha_values.items():
            x_cpu = stored.get("tokens")
            if x_cpu is None or x_cpu.shape[0] == 0:
                skipped += 1
                continue

            print(f"[SmoothAlphaSearcher] Processing layer {layer_key}")
            w_tensor = _get_down_proj_weight(model, layer_key)
            if w_tensor is None:
                skipped += 1
                continue

            # Get local shard of absmax (pre-all_gather).
            stat_key = layer_key
            stat = smooth_stats.get(stat_key)
            if stat is None or stat.get("absmax") is None:
                skipped += 1
                continue
            local_absmax = stat["ema" if cfg.use_ema_for_absmax else "absmax"]
            if local_absmax is None:
                local_absmax = stat["absmax"]
            if local_absmax is None:
                skipped += 1
                continue

            # Move to device
            x = x_cpu.to(device)
            w = w_tensor.float().to(device)

            best_alpha, best_smooth, best_loss = smooth_alpha_search_layer(
                x=x,
                w=w,
                act_absmax=local_absmax,
                config=cfg,
            )

            results[layer_key] = {
                "alpha": best_alpha,
                "smooth_weight_shard": best_smooth.cpu(),
                "loss": best_loss,
            }
            processed += 1

            del x, w, x_cpu

        print(
            f"[SmoothAlphaSearcher] rank={rank}: processed={processed}, "
            f"skipped={skipped}, total_keys={len(alpha_values)}"
        )

        # all_gather: concatenate shards across TP ranks.
        if world_size > 1 and dist.is_initialized():
            keys_list = sorted(results.keys())
            if keys_list:
                # smooth_weight shards: stack into one flat tensor for
                # a single NCCL call.
                sw_list = [results[k]["smooth_weight_shard"] for k in keys_list]
                sw_sizes = [s.numel() for s in sw_list]
                sw_flat = torch.cat([s.reshape(-1) for s in sw_list]).cuda()
                gathered_flat = [torch.zeros_like(sw_flat) for _ in range(world_size)]
                dist.all_gather(gathered_flat, sw_flat)
                offset = 0
                for i, key in enumerate(keys_list):
                    sz = sw_sizes[i]
                    chunks = [g[offset : offset + sz].cpu() for g in gathered_flat]
                    results[key]["smooth_weight"] = torch.cat(chunks, dim=0).tolist()
                    offset += sz

                # alpha values
                alpha_flat = torch.tensor(
                    [results[k]["alpha"] for k in keys_list],
                    dtype=torch.float32,
                ).cuda()
                gathered_alpha = [torch.zeros_like(alpha_flat) for _ in range(world_size)]
                dist.all_gather(gathered_alpha, alpha_flat)
                for i, key in enumerate(keys_list):
                    results[key]["alpha"] = [g[i].item() for g in gathered_alpha]

                for key in keys_list:
                    results[key].pop("smooth_weight_shard", None)

                del sw_flat, gathered_flat, alpha_flat, gathered_alpha
                if use_gpu:
                    torch.cuda.empty_cache()

            dist.barrier()
        else:
            for key in results:
                entry = results[key]
                entry["smooth_weight"] = entry.pop("smooth_weight_shard").tolist()
                entry["alpha"] = [entry["alpha"]]

        # Save results to file on rank 0 (avoids pickling the large
        # dict back through the mp executor).
        num_results = len(results)
        if self.output_path and rank == 0 and results:
            import json as _json

            output_data = {
                "config": {
                    "alpha_min": cfg.alpha_min,
                    "alpha_max": cfg.alpha_max,
                    "alpha_steps": cfg.alpha_steps,
                    "act_quant_method": cfg.act_quant_method,
                    "act_quant_type": cfg.act_quant_type,
                    "weight_quant_method": cfg.weight_quant_method,
                    "weight_quant_type": cfg.weight_quant_type,
                    "weight_quant_bits": cfg.weight_quant_bits,
                },
                "results": results,
            }
            os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
            with open(self.output_path, "w") as f:
                _json.dump(output_data, f, indent=2)
            print(
                f"[SmoothAlphaSearcher] rank=0 saved {num_results} results "
                f"to {self.output_path}"
            )

        return f"processed={num_results}"
