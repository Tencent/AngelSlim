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

"""Scale search optimization for Delta-Aware Quantization (DAQ)."""

from typing import Tuple

import torch

from angelslim.compressor.quant.core.kernels import FP8E4M3Kernel, QuantKernel

from .utils import print_info


class ScaleSearchHelper:
    """Helper class for scale search optimization."""

    def __init__(
        self,
        min_range,
        max_range,
        coarse_intervals,
        fine_intervals,
        delta_threshold,
        metric,
        quantization_method="blockwise",
        verbose=False,
        quant_kernel: QuantKernel = None,
    ):
        self.min_range = min_range
        self.max_range = max_range
        self.coarse_intervals = coarse_intervals
        self.fine_intervals = fine_intervals
        self.delta_threshold = delta_threshold
        self.metric = metric
        self.quantization_method = quantization_method
        self.verbose = verbose
        self.quant_kernel = quant_kernel or FP8E4M3Kernel()

        if metric not in ("sign", "cosine", "mse"):
            raise ValueError(f"Unknown metric: {metric}. Choose from 'sign', 'cosine', or 'mse'")

        if quantization_method not in ("blockwise", "per_channel"):
            raise ValueError(
                f"Unknown quantization method: {quantization_method}. "
                f"Choose from 'blockwise' or 'per_channel'"
            )

    def batch_search_optimal_scales(
        self,
        w_sft: torch.Tensor,
        w_base: torch.Tensor,
        default_scale: torch.Tensor,
        block_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch search for optimal scales across all blocks or channels.

        Returns:
            optimal_scales: The optimized scale tensor.
            best_metrics: The best metric value per block/channel after optimization.
            default_metrics: The baseline metric value per block/channel (before optimization).
        """
        if self.quantization_method == "blockwise":
            return self._batch_search_optimal_scales_blockwise(
                w_sft, w_base, default_scale, block_size
            )
        else:  # per_channel
            return self._batch_search_optimal_scales_per_channel(w_sft, w_base, default_scale)

    def _batch_search_optimal_scales_blockwise(
        self,
        w_sft: torch.Tensor,
        w_base: torch.Tensor,
        default_scale: torch.Tensor,
        block_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M, N = w_sft.shape
        sM, sN = default_scale.shape
        device = w_sft.device

        pad_M = (block_size - M % block_size) % block_size
        pad_N = (block_size - N % block_size) % block_size

        if pad_M > 0 or pad_N > 0:
            w_sft_padded = torch.nn.functional.pad(w_sft, (0, pad_N, 0, pad_M), value=0)
            w_base_padded = torch.nn.functional.pad(w_base, (0, pad_N, 0, pad_M), value=0)
        else:
            w_sft_padded = w_sft
            w_base_padded = w_base

        w_sft_blocks = w_sft_padded.view(sM, block_size, sN, block_size).permute(0, 2, 1, 3)
        w_base_blocks = w_base_padded.view(sM, block_size, sN, block_size).permute(0, 2, 1, 3)

        num_blocks = sM * sN
        w_sft_flat = w_sft_blocks.reshape(num_blocks, -1).float()
        w_base_flat = w_base_blocks.reshape(num_blocks, -1).float()
        default_scale_flat = default_scale.flatten()

        coarse_ratios = torch.linspace(
            self.min_range, self.max_range, self.coarse_intervals, device=device
        )
        coarse_scales = default_scale_flat.unsqueeze(1) * coarse_ratios.unsqueeze(0)
        coarse_metrics = self._batch_compute_metrics(w_sft_flat, w_base_flat, coarse_scales)

        best_coarse_idx = coarse_metrics.argmax(dim=1)
        best_coarse_ratios = coarse_ratios[best_coarse_idx]

        interval_size = (self.max_range - self.min_range) / max(self.coarse_intervals - 1, 1)

        fine_min = torch.clamp(best_coarse_ratios - interval_size / 2, min=self.min_range)
        fine_max = torch.clamp(best_coarse_ratios + interval_size / 2, max=self.max_range)

        fine_t = torch.linspace(0, 1, self.fine_intervals, device=device).unsqueeze(0)
        fine_ratios = fine_min.unsqueeze(1) + fine_t * (fine_max - fine_min).unsqueeze(1)

        fine_scales = default_scale_flat.unsqueeze(1) * fine_ratios
        fine_metrics = self._batch_compute_metrics(w_sft_flat, w_base_flat, fine_scales)

        best_fine_idx = fine_metrics.argmax(dim=1)

        optimal_scales_flat = fine_scales.gather(1, best_fine_idx.unsqueeze(1)).squeeze(1)
        best_metrics_flat = fine_metrics.gather(1, best_fine_idx.unsqueeze(1)).squeeze(1)

        # Protection: if optimized scale is worse than default, keep default
        default_metrics_flat = self._batch_compute_metrics(
            w_sft_flat, w_base_flat, default_scale_flat.unsqueeze(1)
        ).squeeze(1)

        worse_mask = best_metrics_flat < default_metrics_flat
        optimal_scales_flat = torch.where(worse_mask, default_scale_flat, optimal_scales_flat)
        best_metrics_flat = torch.where(worse_mask, default_metrics_flat, best_metrics_flat)

        if worse_mask.any() and self.verbose:
            num_protected = worse_mask.sum().item()
            print_info(
                f"[ScaleSearch] Protected {num_protected}/{num_blocks} blocks "
                f"from degradation (kept default scale)"
            )

        optimal_scales = optimal_scales_flat.view(sM, sN)
        best_metrics = best_metrics_flat.view(sM, sN)
        default_metrics = default_metrics_flat.view(sM, sN)

        return optimal_scales, best_metrics, default_metrics

    def _batch_search_optimal_scales_per_channel(
        self,
        w_sft: torch.Tensor,
        w_base: torch.Tensor,
        default_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M, N = w_sft.shape
        device = w_sft.device

        w_sft_flat = w_sft.float()
        w_base_flat = w_base.float()
        default_scale_flat = default_scale.squeeze(1)

        coarse_ratios = torch.linspace(
            self.min_range, self.max_range, self.coarse_intervals, device=device
        )
        coarse_scales = default_scale_flat.unsqueeze(1) * coarse_ratios.unsqueeze(0)
        coarse_metrics = self._batch_compute_metrics(w_sft_flat, w_base_flat, coarse_scales)

        best_coarse_idx = coarse_metrics.argmax(dim=1)
        best_coarse_ratios = coarse_ratios[best_coarse_idx]

        interval_size = (self.max_range - self.min_range) / max(self.coarse_intervals - 1, 1)

        fine_min = torch.clamp(best_coarse_ratios - interval_size / 2, min=self.min_range)
        fine_max = torch.clamp(best_coarse_ratios + interval_size / 2, max=self.max_range)

        fine_t = torch.linspace(0, 1, self.fine_intervals, device=device).unsqueeze(0)
        fine_ratios = fine_min.unsqueeze(1) + fine_t * (fine_max - fine_min).unsqueeze(1)
        fine_scales = default_scale_flat.unsqueeze(1) * fine_ratios

        fine_metrics = self._batch_compute_metrics(w_sft_flat, w_base_flat, fine_scales)
        best_fine_idx = fine_metrics.argmax(dim=1)

        optimal_scales_flat = fine_scales.gather(1, best_fine_idx.unsqueeze(1)).squeeze(1)
        best_metrics_flat = fine_metrics.gather(1, best_fine_idx.unsqueeze(1)).squeeze(1)

        # Protection: if optimized scale is worse than default, keep default
        default_metrics_flat = self._batch_compute_metrics(
            w_sft_flat, w_base_flat, default_scale_flat.unsqueeze(1)
        ).squeeze(1)

        worse_mask = best_metrics_flat < default_metrics_flat
        optimal_scales_flat = torch.where(worse_mask, default_scale_flat, optimal_scales_flat)
        best_metrics_flat = torch.where(worse_mask, default_metrics_flat, best_metrics_flat)

        if worse_mask.any() and self.verbose:
            num_protected = worse_mask.sum().item()
            print_info(
                f"[ScaleSearch] Protected {num_protected}/{M} channels "
                f"from degradation (kept default scale)"
            )

        optimal_scales = optimal_scales_flat.unsqueeze(1)
        best_metrics = best_metrics_flat.unsqueeze(1)
        default_metrics = default_metrics_flat.unsqueeze(1)

        if self.verbose:
            scale_ratio = optimal_scales / default_scale
            avg_ratio = scale_ratio.mean().item()
            min_ratio = scale_ratio.min().item()
            max_ratio = scale_ratio.max().item()
            var_ratio = scale_ratio.var().item()
            print_info(
                f"[ScaleRatio] Optimal/Default scale ratio - "
                f"Avg: {avg_ratio:.4f}, Min: {min_ratio:.4f}, "
                f"Max: {max_ratio:.4f}, Var: {var_ratio:.4f}"
            )

        return optimal_scales, best_metrics, default_metrics

    def _batch_compute_metrics(
        self,
        w_sft_flat: torch.Tensor,
        w_base_flat: torch.Tensor,
        candidate_scales: torch.Tensor,
    ) -> torch.Tensor:
        """Batch compute metrics for all blocks/channels and candidate scales."""
        if self.metric == "sign":
            return self._batch_compute_sign_preservation(w_sft_flat, w_base_flat, candidate_scales)
        elif self.metric == "cosine":
            return self._batch_compute_cosine_similarity(w_sft_flat, w_base_flat, candidate_scales)
        elif self.metric == "mse":
            return self._batch_compute_negative_mse(w_sft_flat, candidate_scales)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _batch_compute_sign_preservation(
        self,
        w_sft_flat: torch.Tensor,
        w_base_flat: torch.Tensor,
        candidate_scales: torch.Tensor,
    ) -> torch.Tensor:
        """Batch compute sign preservation rate."""
        delta_w = w_sft_flat - w_base_flat

        significant_mask = delta_w.abs() > self.delta_threshold
        total_significant = significant_mask.sum(dim=1, keepdim=True).clamp(min=1)

        expected_sign = torch.sign(delta_w)

        w_sft_expanded = w_sft_flat.unsqueeze(1)
        scales_expanded = candidate_scales.unsqueeze(2)

        w_quantized = self.quant_kernel.simulate_quant_dequant(w_sft_expanded, scales_expanded)

        w_base_expanded = w_base_flat.unsqueeze(1)
        delta_quant = w_quantized - w_base_expanded
        sign_quant = torch.sign(delta_quant)

        expected_sign_expanded = expected_sign.unsqueeze(1)
        significant_mask_expanded = significant_mask.unsqueeze(1)

        correct = (
            (significant_mask_expanded & (sign_quant == expected_sign_expanded)).sum(dim=2).float()
        )

        preservation_rates = correct / total_significant

        return preservation_rates

    def _batch_compute_cosine_similarity(
        self,
        w_sft_flat: torch.Tensor,
        w_base_flat: torch.Tensor,
        candidate_scales: torch.Tensor,
    ) -> torch.Tensor:
        """Batch compute cosine similarity between original and quantized ΔW."""
        delta_w = w_sft_flat - w_base_flat
        delta_w_norm = torch.norm(delta_w, dim=1, keepdim=True).clamp(min=1e-10)

        delta_w_normalized = delta_w / delta_w_norm

        w_sft_expanded = w_sft_flat.unsqueeze(1)
        scales_expanded = candidate_scales.unsqueeze(2)

        w_quantized = self.quant_kernel.simulate_quant_dequant(w_sft_expanded, scales_expanded)

        w_base_expanded = w_base_flat.unsqueeze(1)
        delta_quant = w_quantized - w_base_expanded

        delta_w_normalized_expanded = delta_w_normalized.unsqueeze(1)
        dot_product = (delta_w_normalized_expanded * delta_quant).sum(dim=2)
        delta_quant_norm = torch.norm(delta_quant, dim=2).clamp(min=1e-10)

        cosine_sim = dot_product / delta_quant_norm

        return cosine_sim

    def _batch_compute_negative_mse(
        self,
        w_sft_flat: torch.Tensor,
        candidate_scales: torch.Tensor,
    ) -> torch.Tensor:
        """Batch compute negative MSE (higher is better)."""
        w_sft_expanded = w_sft_flat.unsqueeze(1)
        scales_expanded = candidate_scales.unsqueeze(2)

        w_quantized = self.quant_kernel.simulate_quant_dequant(w_sft_expanded, scales_expanded)

        mse = ((w_sft_expanded - w_quantized) ** 2).mean(dim=2)

        return -mse


def scale_search_weight_quant(
    w_sft: torch.Tensor,
    w_base: torch.Tensor,
    quantization_method: str = "blockwise",
    block_size: int = 128,
    min_range: float = 0.8,
    max_range: float = 1.5,
    coarse_intervals: int = 5,
    fine_intervals: int = 10,
    delta_threshold: float = 1e-5,
    metric: str = "sign",
    verbose: bool = False,
    quant_kernel: QuantKernel = None,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """FP8 quantization with scale search optimization."""
    assert w_sft.is_contiguous() and w_base.is_contiguous()
    assert w_sft.shape == w_base.shape
    assert w_sft.dim() == 2
    assert quantization_method in ("blockwise", "per_channel")

    _kernel = quant_kernel or FP8E4M3Kernel()

    if quantization_method == "blockwise":
        M, N = w_sft.shape
        sM = int(torch.tensor(1.0 * M / block_size).ceil().item())
        sN = int(torch.tensor(1.0 * N / block_size).ceil().item())

        default_scale = _kernel.compute_scale(w_sft, block_size)

        helper = ScaleSearchHelper(
            min_range=min_range,
            max_range=max_range,
            coarse_intervals=coarse_intervals,
            fine_intervals=fine_intervals,
            delta_threshold=delta_threshold,
            metric=metric,
            quantization_method=quantization_method,
            verbose=verbose,
            quant_kernel=_kernel,
        )

        optimal_scales, best_metrics, default_metrics = helper.batch_search_optimal_scales(
            w_sft, w_base, default_scale, block_size
        )

        w_quant = _kernel.quantize(w_sft, optimal_scales, block_size)

        total_blocks = sM * sN
        avg_preservation_rate = best_metrics.mean().item()
        avg_baseline_rate = default_metrics.mean().item()
        blocks_improved = (best_metrics > default_metrics).sum().item()

        search_stats = {
            "total_blocks": total_blocks,
            "blocks_improved": blocks_improved,
            "avg_preservation_rate": avg_preservation_rate,
            "avg_baseline_rate": avg_baseline_rate,
        }
    else:
        M, N = w_sft.size()

        qmax = _kernel.scheme.max_val
        abs_max = torch.abs(w_sft).max(dim=1, keepdim=True)[0].clamp(min=1e-12)
        default_scale = abs_max / qmax

        helper = ScaleSearchHelper(
            min_range=min_range,
            max_range=max_range,
            coarse_intervals=coarse_intervals,
            fine_intervals=fine_intervals,
            delta_threshold=delta_threshold,
            metric=metric,
            quantization_method=quantization_method,
            verbose=verbose,
            quant_kernel=_kernel,
        )

        optimal_scales, best_metrics, default_metrics = helper.batch_search_optimal_scales(
            w_sft, w_base, default_scale
        )

        w_scaled = w_sft / optimal_scales
        w_quantized = torch.clamp(w_scaled, -qmax, qmax)
        w_quant = w_quantized.to(_kernel.scheme.dtype)

        total_channels = M
        avg_preservation_rate = best_metrics.squeeze(1).mean().item()
        avg_baseline_rate = default_metrics.squeeze(1).mean().item()
        channels_improved = (best_metrics > default_metrics).sum().item()

        search_stats = {
            "total_channels": total_channels,
            "channels_improved": channels_improved,
            "avg_preservation_rate": avg_preservation_rate,
            "avg_baseline_rate": avg_baseline_rate,
        }

    return w_quant, optimal_scales, search_stats
