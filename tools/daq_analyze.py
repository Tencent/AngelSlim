#!/usr/bin/env python3
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

"""DAQ Delta Weight Analysis Tool for AngelSLIM."""
import json
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import OrderedDict
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file
from tqdm import tqdm

from angelslim.compressor.quant.core.quant_func import weight_dequant
from angelslim.compressor.quant.modules.daq.utils import compute_dynamic_cache_size


class Dequantizer(ABC):
    """Abstract base class for weight dequantization strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the quantization format."""

    @abstractmethod
    def is_auxiliary_key(self, key: str) -> bool:
        """Return True if `key` is a scale/auxiliary tensor (not a real weight)."""

    @abstractmethod
    def dequantize(self, weight: torch.Tensor, state_dict: dict, weight_name: str) -> torch.Tensor:
        """Dequantize a weight tensor to float."""


class IdentityDequantizer(Dequantizer):
    """No-op dequantizer for models already in BF16/FP32."""

    @property
    def name(self) -> str:
        return "bf16"

    def is_auxiliary_key(self, key: str) -> bool:
        return False

    def dequantize(self, weight: torch.Tensor, state_dict: dict, weight_name: str) -> torch.Tensor:
        return weight.to(torch.float32)


class BlockFP8Dequantizer(Dequantizer):
    """Block-wise FP8 dequantizer."""

    def __init__(self, block_size: int = 128):
        self.block_size = block_size

    @property
    def name(self) -> str:
        return "block_fp8"

    def is_auxiliary_key(self, key: str) -> bool:
        return key.endswith("_scale_inv")

    def dequantize(self, weight: torch.Tensor, state_dict: dict, weight_name: str) -> torch.Tensor:
        scale_name = f"{weight_name}_scale_inv"
        if scale_name not in state_dict:
            return weight.to(torch.float32)
        scale = state_dict[scale_name]
        return weight_dequant(weight, scale, block_size=self.block_size)


class ChannelFP8Dequantizer(Dequantizer):
    """Per-channel FP8 dequantizer."""

    @property
    def name(self) -> str:
        return "channel_fp8"

    def is_auxiliary_key(self, key: str) -> bool:
        return key.endswith("_scale")

    def dequantize(self, weight: torch.Tensor, state_dict: dict, weight_name: str) -> torch.Tensor:
        scale_name = f"{weight_name}_scale"
        if scale_name not in state_dict:
            return weight.to(torch.float32)
        scale = state_dict[scale_name].to(torch.float32)
        return weight.to(torch.float32) * scale


class ChannelINT8Dequantizer(Dequantizer):
    """Per-channel INT8 dequantizer. Supports both _scale and _scale_inv."""

    @property
    def name(self) -> str:
        return "channel_int8"

    def is_auxiliary_key(self, key: str) -> bool:
        return key.endswith("_scale") or key.endswith("_scale_inv")

    def dequantize(self, weight: torch.Tensor, state_dict: dict, weight_name: str) -> torch.Tensor:
        scale_name = f"{weight_name}_scale"
        scale_inv_name = f"{weight_name}_scale_inv"
        wf = weight.to(torch.float32)
        if scale_name in state_dict:
            return wf * state_dict[scale_name].to(torch.float32)
        elif scale_inv_name in state_dict:
            return wf / state_dict[scale_inv_name].to(torch.float32)
        return wf


DEQUANTIZER_REGISTRY: Dict[str, type] = {
    "bf16": IdentityDequantizer,
    "block_fp8": BlockFP8Dequantizer,
    "channel_fp8": ChannelFP8Dequantizer,
    "channel_int8": ChannelINT8Dequantizer,
}


def create_dequantizer(quant_type: str, **kwargs) -> Dequantizer:
    if quant_type not in DEQUANTIZER_REGISTRY:
        supported = ", ".join(DEQUANTIZER_REGISTRY.keys())
        raise ValueError(f"Unknown quant_type '{quant_type}'. Supported: {supported}")
    return DEQUANTIZER_REGISTRY[quant_type](**kwargs)


MAGNITUDE_BINS = {
    "tiny": (0.0, 1e-5),
    "small": (1e-5, 1e-4),
    "medium": (1e-4, 1e-3),
    "large": (1e-3, float("inf")),
}


@dataclass
class SignCounts:
    """Aggregated sign-preservation counters."""

    preserved: int = 0
    flipped: int = 0
    lost: int = 0
    noise: int = 0
    both_zero: int = 0

    @property
    def total(self) -> int:
        return self.preserved + self.flipped + self.lost + self.noise + self.both_zero

    @property
    def non_zero_delta(self) -> int:
        return self.preserved + self.flipped + self.lost

    @property
    def preservation_rate(self) -> float:
        d = self.non_zero_delta
        return (self.preserved / d * 100.0) if d > 0 else 0.0

    @property
    def flip_rate(self) -> float:
        d = self.non_zero_delta
        return (self.flipped / d * 100.0) if d > 0 else 0.0

    @property
    def loss_rate(self) -> float:
        d = self.non_zero_delta
        return (self.lost / d * 100.0) if d > 0 else 0.0

    def accumulate(self, other: "SignCounts") -> None:
        self.preserved += other.preserved
        self.flipped += other.flipped
        self.lost += other.lost
        self.noise += other.noise
        self.both_zero += other.both_zero


@dataclass
class DistanceAccumulator:
    """Accumulates distance metrics across multiple weight tensors."""

    l2_sum: float = 0.0
    weighted_mse_sum: float = 0.0
    weighted_mae_sum: float = 0.0
    total_elements: int = 0
    cosine_similarities: List[float] = field(default_factory=list)

    @property
    def avg_mse(self) -> float:
        return self.weighted_mse_sum / self.total_elements if self.total_elements > 0 else 0.0

    @property
    def avg_mae(self) -> float:
        return self.weighted_mae_sum / self.total_elements if self.total_elements > 0 else 0.0

    @property
    def avg_cosine(self) -> float:
        return (
            sum(self.cosine_similarities) / len(self.cosine_similarities)
            if self.cosine_similarities
            else 0.0
        )

    def add(self, a: torch.Tensor, b: torch.Tensor) -> None:
        metrics = _compute_distance(a, b)
        numel = a.numel()
        self.l2_sum += metrics["l2"]
        self.weighted_mse_sum += metrics["mse"] * numel
        self.weighted_mae_sum += metrics["mae"] * numel
        self.total_elements += numel
        self.cosine_similarities.append(metrics["cosine"])

    def merge(self, other: "DistanceAccumulator") -> None:
        self.l2_sum += other.l2_sum
        self.weighted_mse_sum += other.weighted_mse_sum
        self.weighted_mae_sum += other.weighted_mae_sum
        self.total_elements += other.total_elements
        self.cosine_similarities.extend(other.cosine_similarities)

    def summary_dict(self) -> dict:
        return {
            "total_l2": self.l2_sum,
            "avg_mse": self.avg_mse,
            "avg_mae": self.avg_mae,
            "avg_cosine_similarity": self.avg_cosine,
        }


@dataclass
class FileResult:
    """Analysis result for a single safetensor shard file."""

    file_name: str = ""
    sign_counts: SignCounts = field(default_factory=SignCounts)
    magnitude_counts: Dict[str, SignCounts] = field(
        default_factory=lambda: {k: SignCounts() for k in MAGNITUDE_BINS}
    )
    dist_quant_base: DistanceAccumulator = field(default_factory=DistanceAccumulator)
    dist_quant_sft: DistanceAccumulator = field(default_factory=DistanceAccumulator)
    dist_sft_base: DistanceAccumulator = field(default_factory=DistanceAccumulator)
    delta_cosines: List[float] = field(default_factory=list)
    per_weight: Dict[str, dict] = field(default_factory=dict)
    error: Optional[str] = None


def _compute_distance(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compute L2, MSE, MAE, and cosine similarity between two tensors."""
    af = a.flatten().float()
    bf = b.flatten().float()
    diff = af - bf

    l2 = torch.norm(diff, p=2).item()
    mse = torch.mean(diff**2).item()
    mae = torch.mean(torch.abs(diff)).item()

    norm_a = torch.norm(af, p=2)
    norm_b = torch.norm(bf, p=2)
    cosine = (torch.dot(af, bf) / (norm_a * norm_b)).item() if (norm_a > 0 and norm_b > 0) else 0.0

    return {"l2": l2, "mse": mse, "mae": mae, "cosine": cosine}


def _compute_sign_counts(
    delta_sft: torch.Tensor, delta_quant: torch.Tensor, mask: torch.Tensor = None
) -> SignCounts:
    """Compare signs of two delta tensors element-wise."""
    s_sft = torch.sign(delta_sft)
    s_qnt = torch.sign(delta_quant)

    if mask is not None:
        s_sft = s_sft[mask]
        s_qnt = s_qnt[mask]

    counts = SignCounts()
    counts.preserved = int(((s_sft == s_qnt) & (s_sft != 0)).sum())
    counts.flipped = int(((s_sft != s_qnt) & (s_sft != 0) & (s_qnt != 0)).sum())
    counts.lost = int(((s_sft != 0) & (s_qnt == 0)).sum())
    counts.noise = int(((s_sft == 0) & (s_qnt != 0)).sum())
    counts.both_zero = int(((s_sft == 0) & (s_qnt == 0)).sum())
    return counts


def _load_weight_by_map(
    weight_name: str,
    model_dir: str,
    weight_map: dict,
    file_cache: OrderedDict,
    device: str,
    max_cache_size: int = 5,
) -> Optional[torch.Tensor]:
    """Load a single weight tensor by looking up its shard file via weight_map."""
    if weight_name not in weight_map:
        return None

    shard_file = weight_map[weight_name]

    if shard_file in file_cache:
        file_cache.move_to_end(shard_file)
    else:
        shard_path = os.path.join(model_dir, shard_file)
        if not os.path.exists(shard_path):
            return None
        while len(file_cache) >= max_cache_size:
            _, evicted_sd = file_cache.popitem(last=False)
            del evicted_sd
        file_cache[shard_file] = load_file(shard_path, device=device)

    shard_sd = file_cache[shard_file]
    return shard_sd.get(weight_name, None)


def analyze_shard(
    quantized_file: str,
    sft_dir: str,
    base_dir: str,
    device: str = "cuda:0",
    dequantizer: Dequantizer = None,
    verbose: bool = False,
    sft_weight_map: dict = None,
    base_weight_map: dict = None,
    max_cache_size: int = 5,
) -> FileResult:
    """Analyze sign preservation for all weights in a single safetensor shard."""
    if dequantizer is None:
        dequantizer = IdentityDequantizer()

    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":")[1]))

    fname = os.path.basename(quantized_file)
    result = FileResult(file_name=fname)

    quant_sd = load_file(quantized_file, device=device)

    use_weight_map = sft_weight_map is not None and base_weight_map is not None

    if use_weight_map:
        sft_file_cache: OrderedDict = OrderedDict()
        base_file_cache: OrderedDict = OrderedDict()

        for name, q_weight in quant_sd.items():
            if dequantizer.is_auxiliary_key(name):
                continue

            sft_w = _load_weight_by_map(
                name, sft_dir, sft_weight_map, sft_file_cache, device, max_cache_size
            )
            if sft_w is None:
                continue

            base_w = _load_weight_by_map(
                name, base_dir, base_weight_map, base_file_cache, device, max_cache_size
            )
            if base_w is None:
                continue

            if q_weight.shape != sft_w.shape or sft_w.shape != base_w.shape:
                continue

            q_weight = dequantizer.dequantize(q_weight, quant_sd, name)

            q_f = q_weight.to(torch.float32)
            s_f = sft_w.to(torch.float32)
            b_f = base_w.to(torch.float32)

            result.dist_quant_base.add(q_f, b_f)
            result.dist_quant_sft.add(q_f, s_f)
            result.dist_sft_base.add(s_f, b_f)

            delta_sft = s_f - b_f
            delta_quant = q_f - b_f

            delta_dist = _compute_distance(delta_quant, delta_sft)
            result.delta_cosines.append(delta_dist["cosine"])

            sc = _compute_sign_counts(delta_sft, delta_quant)
            result.sign_counts.accumulate(sc)

            abs_delta = torch.abs(delta_sft)
            for bin_name, (lo, hi) in MAGNITUDE_BINS.items():
                mask = (abs_delta >= lo) & (abs_delta < hi)
                bin_sc = _compute_sign_counts(delta_sft, delta_quant, mask)
                result.magnitude_counts[bin_name].accumulate(bin_sc)

            if verbose:
                result.per_weight[name] = {
                    "shape": list(q_f.shape),
                    "total_elements": q_f.numel(),
                    "sign_preserved": sc.preserved,
                    "sign_flipped": sc.flipped,
                    "sign_lost": sc.lost,
                    "preservation_rate": sc.preservation_rate,
                    "delta_cosine_similarity": delta_dist["cosine"],
                }

        del sft_file_cache, base_file_cache, quant_sd
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        sft_file = os.path.join(sft_dir, fname)
        base_file = os.path.join(base_dir, fname)
        if not os.path.exists(sft_file):
            result.error = f"SFT file not found: {sft_file}"
            return result
        if not os.path.exists(base_file):
            result.error = f"Base file not found: {base_file}"
            return result

        sft_sd = load_file(sft_file, device=device)
        base_sd = load_file(base_file, device=device)

        for name, q_weight in quant_sd.items():
            if dequantizer.is_auxiliary_key(name):
                continue
            if name not in sft_sd or name not in base_sd:
                continue

            sft_w = sft_sd[name]
            base_w = base_sd[name]
            if q_weight.shape != sft_w.shape or sft_w.shape != base_w.shape:
                continue

            q_weight = dequantizer.dequantize(q_weight, quant_sd, name)

            q_f = q_weight.to(torch.float32)
            s_f = sft_w.to(torch.float32)
            b_f = base_w.to(torch.float32)

            result.dist_quant_base.add(q_f, b_f)
            result.dist_quant_sft.add(q_f, s_f)
            result.dist_sft_base.add(s_f, b_f)

            delta_sft = s_f - b_f
            delta_quant = q_f - b_f

            delta_dist = _compute_distance(delta_quant, delta_sft)
            result.delta_cosines.append(delta_dist["cosine"])

            sc = _compute_sign_counts(delta_sft, delta_quant)
            result.sign_counts.accumulate(sc)

            abs_delta = torch.abs(delta_sft)
            for bin_name, (lo, hi) in MAGNITUDE_BINS.items():
                mask = (abs_delta >= lo) & (abs_delta < hi)
                bin_sc = _compute_sign_counts(delta_sft, delta_quant, mask)
                result.magnitude_counts[bin_name].accumulate(bin_sc)

            if verbose:
                result.per_weight[name] = {
                    "shape": list(q_f.shape),
                    "total_elements": q_f.numel(),
                    "sign_preserved": sc.preserved,
                    "sign_flipped": sc.flipped,
                    "sign_lost": sc.lost,
                    "preservation_rate": sc.preservation_rate,
                    "delta_cosine_similarity": delta_dist["cosine"],
                }

        del quant_sd, sft_sd, base_sd
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return result


def _analyze_shard_wrapper(args: tuple) -> FileResult:
    """Pickle-friendly wrapper for multiprocessing."""
    (
        quantized_file,
        sft_dir,
        base_dir,
        device,
        quant_type,
        quant_kwargs,
        verbose,
        sft_weight_map,
        base_weight_map,
        max_cache_size,
    ) = args
    dequantizer = create_dequantizer(quant_type, **quant_kwargs)
    return analyze_shard(
        quantized_file,
        sft_dir,
        base_dir,
        device,
        dequantizer,
        verbose,
        sft_weight_map=sft_weight_map,
        base_weight_map=base_weight_map,
        max_cache_size=max_cache_size,
    )


def _merge_results(results: List[FileResult], verbose: bool):
    """Merge per-shard results into a single summary report dict."""
    total_sign = SignCounts()
    mag_sign = {k: SignCounts() for k in MAGNITUDE_BINS}
    d_qb = DistanceAccumulator()
    d_qs = DistanceAccumulator()
    d_sb = DistanceAccumulator()
    all_delta_cos: List[float] = []
    all_weights: Dict[str, dict] = {}

    for r in results:
        if r.error:
            print(f"  [WARNING] {r.error}")
            continue
        total_sign.accumulate(r.sign_counts)
        for k in MAGNITUDE_BINS:
            mag_sign[k].accumulate(r.magnitude_counts[k])
        d_qb.merge(r.dist_quant_base)
        d_qs.merge(r.dist_quant_sft)
        d_sb.merge(r.dist_sft_base)
        all_delta_cos.extend(r.delta_cosines)
        if verbose:
            all_weights.update(r.per_weight)

    avg_delta_cos = sum(all_delta_cos) / len(all_delta_cos) if all_delta_cos else 0.0

    report = {
        "summary": {
            "total_elements": total_sign.total,
            "non_zero_delta_elements": total_sign.non_zero_delta,
            "sign_preserved": total_sign.preserved,
            "sign_flipped": total_sign.flipped,
            "sign_lost": total_sign.lost,
            "noise_introduced": total_sign.noise,
            "both_zero": total_sign.both_zero,
            "preservation_rate_pct": round(total_sign.preservation_rate, 4),
            "flip_rate_pct": round(total_sign.flip_rate, 4),
            "loss_rate_pct": round(total_sign.loss_rate, 4),
        },
        "distance_metrics": {
            "quant_vs_base": d_qb.summary_dict(),
            "quant_vs_sft": d_qs.summary_dict(),
            "sft_vs_base": d_sb.summary_dict(),
            "delta_cosine": {
                "avg_cosine_similarity": avg_delta_cos,
                "weight_count": len(all_delta_cos),
            },
        },
        "magnitude_breakdown": {},
    }

    for k in MAGNITUDE_BINS:
        sc = mag_sign[k]
        pf = sc.preserved + sc.flipped
        report["magnitude_breakdown"][k] = {
            "total": sc.total,
            "preserved": sc.preserved,
            "flipped": sc.flipped,
            "preservation_rate_pct": round(sc.preserved / max(pf, 1) * 100, 4),
        }

    if verbose:
        report["per_weight"] = all_weights

    return report, total_sign, mag_sign, d_qb, d_qs, d_sb, avg_delta_cos


def _print_report(
    quant_type: str,
    total_sign: SignCounts,
    mag_sign: Dict[str, SignCounts],
    d_qb: DistanceAccumulator,
    d_qs: DistanceAccumulator,
    d_sb: DistanceAccumulator,
    avg_delta_cos: float,
) -> None:
    """Pretty-print the analysis summary to stdout."""
    sep = "=" * 72

    print(f"\n{sep}")
    print(f"  Sign Preservation Analysis — Results  [quant_type={quant_type}]")
    print(sep)

    print(f"\n  Total elements:           {total_sign.total:>15,}")
    print(f"  Non-zero ΔW elements:     {total_sign.non_zero_delta:>15,}")
    print(f"  Both-zero (ΔW=ΔW_q=0):   {total_sign.both_zero:>15,}")

    print(f"\n{sep}")
    print("  Sign Preservation (non-zero ΔW)")
    print(sep)
    print(
        f"  ✓ Preserved:  {total_sign.preserved:>14,}  " f"({total_sign.preservation_rate:>6.2f}%)"
    )
    print(f"  ✗ Flipped:    {total_sign.flipped:>14,}  " f"({total_sign.flip_rate:>6.2f}%)")
    print(f"  ○ Lost (→0):  {total_sign.lost:>14,}  " f"({total_sign.loss_rate:>6.2f}%)")
    print(f"  + Noise (0→):  {total_sign.noise:>13,}")

    print(f"\n{sep}")
    print("  Distance Metrics")
    print(sep)
    header = f"  {'Pair':<22} {'Total L2':>14} {'Avg MSE':>14} " f"{'Avg MAE':>14} {'Avg Cos':>10}"
    print(header)
    print(f"  {'-' * 70}")
    for label, acc in [
        ("Quant ↔ Base", d_qb),
        ("Quant ↔ SFT", d_qs),
        ("SFT ↔ Base (ref)", d_sb),
    ]:
        print(
            f"  {label:<22} {acc.l2_sum:>14.4f} {acc.avg_mse:>14.2e} "
            f"{acc.avg_mae:>14.2e} {acc.avg_cosine:>10.6f}"
        )

    print(f"\n{sep}")
    print("  Delta Cosine Similarity (ΔW_quant vs ΔW_sft)")
    print(sep)
    print(f"  Average: {avg_delta_cos:.6f}   (1.0 = perfect direction preservation)")

    print(f"\n{sep}")
    print("  Breakdown by |ΔW| Magnitude")
    print(sep)
    labels = {
        "tiny": "|ΔW| < 1e-5",
        "small": "1e-5 ~ 1e-4",
        "medium": "1e-4 ~ 1e-3",
        "large": "|ΔW| ≥ 1e-3",
    }
    print(f"  {'Bin':<15} {'Total':>14} {'Preserved':>14} " f"{'Flipped':>14} {'Pres.Rate':>10}")
    print(f"  {'-' * 70}")
    for k, lbl in labels.items():
        sc = mag_sign[k]
        pf = sc.preserved + sc.flipped
        rate = sc.preserved / max(pf, 1) * 100
        print(
            f"  {lbl:<15} {sc.total:>14,} {sc.preserved:>14,} " f"{sc.flipped:>14,} {rate:>9.2f}%"
        )

    print(f"\n{sep}\n")


def main(
    quantized_path: str,
    sft_path: str,
    base_path: str,
    quant_type: str = "block_fp8",
    output_report: Optional[str] = None,
    num_workers: int = 1,
    gpus: Optional[List[int]] = None,
    verbose: bool = False,
    **quant_kwargs,
) -> dict:
    """Run the full sign-preservation analysis across all shard files."""
    torch.set_default_dtype(torch.bfloat16)

    dequantizer = create_dequantizer(quant_type, **quant_kwargs)

    if gpus is None:
        gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]

    print(f"\n{'=' * 72}")
    print("  Sign Preservation Analysis")
    print(f"{'=' * 72}")
    print(f"  Quantized model : {quantized_path}")
    print(f"  Quant type      : {dequantizer.name}")
    print(f"  SFT model       : {sft_path}")
    print(f"  Base model      : {base_path}")
    print(f"  GPUs            : {gpus}")
    print(f"{'=' * 72}\n")

    # Detect weight-map mode
    sft_index_path = os.path.join(sft_path, "model.safetensors.index.json")
    base_index_path = os.path.join(base_path, "model.safetensors.index.json")
    sft_weight_map: Optional[dict] = None
    base_weight_map: Optional[dict] = None
    max_cache_size = 5

    if os.path.exists(sft_index_path) and os.path.exists(base_index_path):
        with open(sft_index_path, "r") as f:
            sft_weight_map = json.load(f).get("weight_map", {})
        with open(base_index_path, "r") as f:
            base_weight_map = json.load(f).get("weight_map", {})

        num_sft_shards = len(set(sft_weight_map.values()))
        num_base_shards = len(set(base_weight_map.values()))

        quant_index_path = os.path.join(quantized_path, "model.safetensors.index.json")
        if os.path.exists(quant_index_path):
            max_cache_size = compute_dynamic_cache_size(
                sft_index_file=quant_index_path,
                base_weight_map=sft_weight_map,
                base_path=sft_path,
            )
            max_cache_size = max(max_cache_size, 5)

        print("  Using weight-map mode")
        print(f"    SFT shards : {num_sft_shards}")
        print(f"    Base shards: {num_base_shards}")
        print(f"    Cache size : {max_cache_size}\n")
    else:
        print("  Using legacy mode (same-filename matching)\n")

    shard_files = sorted(glob(os.path.join(quantized_path, "*.safetensors")))
    print(f"  Found {len(shard_files)} safetensor shard(s) to analyze.\n")

    if num_workers > 1 and torch.cuda.is_available():
        ctx = mp.get_context("spawn")
        args_list = [
            (
                f,
                sft_path,
                base_path,
                f"cuda:{gpus[i % len(gpus)]}",
                quant_type,
                quant_kwargs,
                verbose,
                sft_weight_map,
                base_weight_map,
                max_cache_size,
            )
            for i, f in enumerate(shard_files)
        ]
        with ctx.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_analyze_shard_wrapper, args_list),
                    total=len(shard_files),
                    desc="  Analyzing",
                )
            )
    else:
        device = f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu"
        results = [
            analyze_shard(
                f,
                sft_path,
                base_path,
                device,
                dequantizer,
                verbose,
                sft_weight_map=sft_weight_map,
                base_weight_map=base_weight_map,
                max_cache_size=max_cache_size,
            )
            for f in tqdm(shard_files, desc="  Analyzing")
        ]

    report, total_sign, mag_sign, d_qb, d_qs, d_sb, avg_dc = _merge_results(results, verbose)
    report["paths"] = {
        "quantized": quantized_path,
        "sft": sft_path,
        "base": base_path,
    }
    report["quant_type"] = quant_type

    _print_report(quant_type, total_sign, mag_sign, d_qb, d_qs, d_sb, avg_dc)

    if output_report:
        os.makedirs(os.path.dirname(output_report) or ".", exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  Report saved to: {output_report}\n")

    return report


if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze sign preservation of a quantized model")
    parser.add_argument(
        "--quantized-model",
        type=str,
        required=True,
        help="Directory of the quantized model",
    )
    parser.add_argument(
        "--sft-model",
        type=str,
        required=True,
        help="Directory of the SFT model (BF16)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Directory of the base model (BF16)",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="block_fp8",
        choices=list(DEQUANTIZER_REGISTRY.keys()),
        help="Quantization format of the model (default: block_fp8)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Path to write JSON report (optional)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include per-weight detail in report",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Block size for block-wise FP8 dequantization (default: 128)",
    )

    args = parser.parse_args()

    gpu_list = [int(x) for x in args.gpus.split(",")] if args.gpus else None

    extra_kwargs = {}
    if args.quant_type == "block_fp8":
        extra_kwargs["block_size"] = args.block_size

    main(
        quantized_path=args.quantized_model,
        sft_path=args.sft_model,
        base_path=args.base_model,
        quant_type=args.quant_type,
        output_report=args.output_report,
        num_workers=args.num_workers,
        gpus=gpu_list,
        verbose=args.verbose,
        **extra_kwargs,
    )

    print("  Done!")
