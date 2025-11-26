from typing import Any, List, Optional, Set

import torch

from .cache_helper import CacheHelper


class TeaCacheHelper(CacheHelper):
    """
    TeaCache helper class that extends CacheHelper with residual-based caching.

    This class implements a caching strategy that stores the residual (difference)
    between the input and output of the last block, allowing for efficient cache
    reuse by adding the residual to the cached input.
    """

    def __init__(
        self,
        double_blocks: Optional[List] = None,
        single_blocks: Optional[List] = None,
        no_cache_steps: Optional[Set[int]] = None,
        cache_name: str = "img",
    ):
        """
        Initialize TeaCacheHelper.

        Args:
            double_blocks: List of double block modules, can be None
            single_blocks: List of single block modules, can be None
            no_cache_steps: Set of steps where caching should not be used
            cache_name: Name of the input field in kwargs to cache (default: "img")
        """
        super().__init__(
            double_blocks=double_blocks,
            single_blocks=single_blocks,
            no_cache_steps=no_cache_steps,
        )
        self.cache_name = cache_name
        self.cached_input: Optional[torch.Tensor] = None
        self.previous_residual: Optional[torch.Tensor] = None

    def wrap_block_forward(self, block: Any, block_id: int, blocktype: str) -> None:
        # Save the original forward method
        self.function_dict[(blocktype, block_id)] = block.forward

        def wrapped_forward(*args, **kwargs):
            """Wrapped forward method with residual-based caching logic."""
            skip = self.is_skip()

            if skip:
                # Use cached output
                if blocktype == "double_blocks":
                    is_last_double_block = block_id == len(self.double_blocks) - 1
                    if not self.single_blocks and is_last_double_block:
                        result = [
                            self.cached_input + self.previous_residual,
                            self.cached_output[(blocktype, block_id)][1],
                        ]
                    else:
                        result = self.cached_output[(blocktype, block_id)]

                elif blocktype == "single_blocks":
                    is_last_single_block = block_id == len(self.single_blocks) - 1
                    if is_last_single_block:
                        result = self.cached_input + self.previous_residual
                    else:
                        result = self.cached_output[(blocktype, block_id)]
            else:
                # Recompute and cache the result
                result = self.function_dict[(blocktype, block_id)](*args, **kwargs)
                self.cached_output[(blocktype, block_id)] = result

                if blocktype == "double_blocks" and block_id == 0:
                    self.cached_input = kwargs[self.cache_name]

                if not self.single_blocks:
                    is_last_double_block = block_id == len(self.double_blocks) - 1
                    if blocktype == "double_blocks" and is_last_double_block:
                        cached_output = result[0]
                        self.previous_residual = cached_output - self.cached_input
                else:
                    is_last_single_block = block_id == len(self.single_blocks) - 1
                    if blocktype == "single_blocks" and is_last_single_block:
                        img_seq_len = self.cached_output[("double_blocks", 0)][0].shape[
                            1
                        ]
                        cached_output = result[:, :img_seq_len, ...]
                        self.previous_residual = cached_output - self.cached_input

            return result

        block.forward = wrapped_forward
