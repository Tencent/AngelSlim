import math
from typing import Any, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .cache_helper import CacheHelper


class TaylorCacheHelper(CacheHelper):
    """
    TaylorCache helper class that extends CacheHelper with Taylor expansion-based.

    This class implements a caching strategy using Taylor series expansion to predict
    future outputs based on derivatives computed from previous steps. It decomposes
    tensors into low and high frequency components for more accurate approximation.
    """

    def __init__(
        self,
        double_blocks: Optional[List] = None,
        single_blocks: Optional[List] = None,
        no_cache_steps: Optional[Set[int]] = None,
        max_order: int = 2,
        low_freqs_order: int = 2,
        high_freqs_order: int = 2,
    ):
        """
        Initialize TaylorCacheHelper.

        Args:
            double_blocks: List of double block modules, can be None
            single_blocks: List of single block modules, can be None
            no_cache_steps: Set of steps where caching should not be used
            max_order: Maximum order of Taylor expansion
            low_freqs_order: Order for computing low frequency derivatives
            high_freqs_order: Order for computing high frequency derivatives
        """
        super().__init__(
            double_blocks=double_blocks,
            single_blocks=single_blocks,
            no_cache_steps=no_cache_steps,
        )
        self.max_order = max_order
        self.low_freqs_order = low_freqs_order
        self.high_freqs_order = high_freqs_order
        self.counter = 0

        self.taylor_cache = CacheWithFreqsContainer(self.max_order)

    def is_skip(self) -> bool:
        # For some pipelines, the first timestep may not be 0
        if self.start_timestep is None:
            self.start_timestep = self.cur_timestep
            self.last_full_computation_step = self.start_timestep

        # If current step is in no_cache_steps, do not use cache
        if self.cur_timestep - self.start_timestep in self.no_cache_steps:
            return False

        return True

    def wrap_block_forward(self, block: Any, block_id: int, blocktype: str) -> None:
        # Save the original forward method
        self.function_dict[(blocktype, block_id)] = block.forward

        def wrapped_forward(*args, **kwargs):
            """Wrapped forward method with caching logic."""
            skip = self.is_skip()

            if skip:
                # Use cached output
                if blocktype == "double_blocks":
                    is_last_double_block = block_id == len(self.double_blocks) - 1
                    if not self.single_blocks and is_last_double_block:
                        self.counter += 1
                        output = self.taylor_cache.taylor_formula(distance=self.counter)
                        result = [output, self.cached_output[(blocktype, block_id)][1]]
                    else:
                        result = self.cached_output[(blocktype, block_id)]
                if blocktype == "single_blocks":
                    is_last_single_block = block_id == len(self.single_blocks) - 1
                    if is_last_single_block:
                        self.counter += 1
                        result = self.taylor_cache.taylor_formula(distance=self.counter)
                    else:
                        result = self.cached_output[(blocktype, block_id)]
            else:
                # Recompute and cache the result
                self.counter = 0
                result = self.function_dict[(blocktype, block_id)](*args, **kwargs)
                self.cached_output[(blocktype, block_id)] = result
                if not self.single_blocks:
                    is_last_double_block = block_id == len(self.double_blocks) - 1
                    if blocktype == "double_blocks" and is_last_double_block:
                        cached_output = result[0]
                        distance = self.cur_timestep - self.last_full_computation_step
                        if self.cur_timestep != self.start_timestep:
                            self.taylor_cache.derivatives_computation(
                                cached_output,
                                distance=distance,
                                low_freqs_order=self.low_freqs_order,
                                high_freqs_order=self.high_freqs_order,
                            )
                        self.last_full_computation_step = self.cur_timestep
                else:
                    is_last_single_block = block_id == len(self.single_blocks) - 1
                    if blocktype == "single_blocks" and is_last_single_block:
                        cached_output = result
                        distance = self.cur_timestep - self.last_full_computation_step
                        if self.cur_timestep != self.start_timestep:
                            self.taylor_cache.derivatives_computation(
                                cached_output,
                                distance=distance,
                                low_freqs_order=self.low_freqs_order,
                                high_freqs_order=self.high_freqs_order,
                            )
                        self.last_full_computation_step = self.cur_timestep

            return result

        block.forward = wrapped_forward

    def reset_states(self) -> None:
        """Reset all internal states."""
        self.start_timestep = None
        self.cur_timestep = 0
        self.function_dict = {}
        self.cached_output = {}
        self.taylor_cache.clear_derivatives()

    def clear_states(self) -> None:
        """Clear cache states but preserve function_dict."""
        self.cur_timestep = 0
        self.start_timestep = None
        self.cached_output = {}
        self.taylor_cache.clear_derivatives()


@torch.compile
def decomposition_FFT(
    x: torch.Tensor, cutoff_ratio: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose tensor into low and high frequency components
    using Fast Fourier Transform.

    Args:
        x: Input tensor of shape [B, H*W, D]
        cutoff_ratio: Cutoff frequency ratio for separating
        low/high frequencies (0~0.5, default: 0.1)

    Returns:
        Tuple of (low_freq, high_freq) tensors with same shape and dtype as input
    """
    orig_dtype = x.dtype
    device = x.device

    x_fp32 = x.to(torch.float32)

    B, HW, D = x_fp32.shape

    # FFT on spatial dimension
    freq = torch.fft.fft(x_fp32, dim=1)

    freqs = torch.fft.fftfreq(HW, d=1.0, device=device)
    cutoff = cutoff_ratio * freqs.abs().max()

    # Create frequency masks
    low_mask = freqs.abs() <= cutoff
    high_mask = ~low_mask

    # Broadcast masks to match tensor shape (B, HW, D)
    low_mask = low_mask[None, :, None]
    high_mask = high_mask[None, :, None]

    low_freq_complex = freq * low_mask
    high_freq_complex = freq * high_mask

    # IFFT and take real part
    low_fp32 = torch.fft.ifft(low_freq_complex, dim=1).real
    high_fp32 = torch.fft.ifft(high_freq_complex, dim=1).real

    # Convert back to original dtype
    low = low_fp32.to(device=device, dtype=orig_dtype)
    high = high_fp32.to(device=device, dtype=orig_dtype)

    return low, high


@torch.compile
def reconstruction(low_freq: torch.Tensor, high_freq: torch.Tensor) -> torch.Tensor:
    return low_freq + high_freq


class CacheWithFreqsContainer(nn.Module):
    def __init__(self, max_order: int):
        super().__init__()
        self.max_order = max_order

        # Register buffers for derivatives and temporary derivatives
        for i in range(max_order + 1):
            self.register_buffer(f"derivative_{i}_low_freqs", None, persistent=False)
            self.register_buffer(f"derivative_{i}_high_freqs", None, persistent=False)
            self.register_buffer(
                f"temp_derivative_{i}_low_freqs", None, persistent=False
            )
            self.register_buffer(
                f"temp_derivative_{i}_high_freqs", None, persistent=False
            )

    def get_derivative(self, order: int, freqs: str) -> Optional[torch.Tensor]:
        return getattr(self, f"derivative_{order}_{freqs}")

    def set_derivative(self, order: int, freqs: str, tensor: torch.Tensor) -> None:
        setattr(self, f"derivative_{order}_{freqs}", tensor)

    def set_temp_derivative(self, order: int, freqs: str, tensor: torch.Tensor) -> None:
        setattr(self, f"temp_derivative_{order}_{freqs}", tensor)

    def get_temp_derivative(self, order: int, freqs: str) -> Optional[torch.Tensor]:
        return getattr(self, f"temp_derivative_{order}_{freqs}")

    def move_temp_to_derivative(self) -> None:
        for i in range(self.max_order + 1):
            if self.get_temp_derivative(i, "low_freqs") is not None:
                setattr(
                    self,
                    f"derivative_{i}_low_freqs",
                    self.get_temp_derivative(i, "low_freqs"),
                )
            if self.get_temp_derivative(i, "high_freqs") is not None:
                setattr(
                    self,
                    f"derivative_{i}_high_freqs",
                    self.get_temp_derivative(i, "high_freqs"),
                )
        self.clear_temp_derivative()

    def get_all_filled_derivatives(self, freqs: str) -> List[torch.Tensor]:
        return [
            self.get_derivative(i, freqs)
            for i in range(self.max_order + 1)
            if self.get_derivative(i, freqs) is not None
        ]

    def taylor_formula(self, distance: int) -> torch.Tensor:
        low_freqs_output = 0
        high_freqs_output = 0
        for i in range(len(self.get_all_filled_derivatives("low_freqs"))):
            coefficient = 1 / math.factorial(i)
            low_freqs_output += (
                coefficient * self.get_derivative(i, "low_freqs") * (distance**i)
            )
        for i in range(len(self.get_all_filled_derivatives("high_freqs"))):
            coefficient = 1 / math.factorial(i)
            high_freqs_output += (
                coefficient * self.get_derivative(i, "high_freqs") * (distance**i)
            )

        return reconstruction(low_freqs_output, high_freqs_output)

    def derivatives_computation(
        self,
        x: torch.Tensor,
        distance: int,
        low_freqs_order: int,
        high_freqs_order: int,
    ) -> None:
        x_low, x_high = decomposition_FFT(x, cutoff_ratio=0.1)
        self.set_temp_derivative(0, "low_freqs", x_low)
        self.set_temp_derivative(0, "high_freqs", x_high)
        for i in range(low_freqs_order):
            if self.get_derivative(i, "low_freqs") is not None:
                derivative_diff = self.get_temp_derivative(
                    i, "low_freqs"
                ) - self.get_derivative(i, "low_freqs")
                self.set_temp_derivative(i + 1, "low_freqs", derivative_diff / distance)
        for i in range(high_freqs_order):
            if self.get_derivative(i, "high_freqs") is not None:
                derivative_diff = self.get_temp_derivative(
                    i, "high_freqs"
                ) - self.get_derivative(i, "high_freqs")
                self.set_temp_derivative(
                    i + 1, "high_freqs", derivative_diff / distance
                )
        self.move_temp_to_derivative()

    def clear_temp_derivative(self) -> None:
        for i in range(self.max_order + 1):
            setattr(self, f"temp_derivative_{i}_low_freqs", None)
            setattr(self, f"temp_derivative_{i}_high_freqs", None)

    def clear_derivatives(self) -> None:
        for i in range(self.max_order + 1):
            setattr(self, f"derivative_{i}_low_freqs", None)
            setattr(self, f"derivative_{i}_high_freqs", None)
            setattr(self, f"temp_derivative_{i}_low_freqs", None)
            setattr(self, f"temp_derivative_{i}_high_freqs", None)
