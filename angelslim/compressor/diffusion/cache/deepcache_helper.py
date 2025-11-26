from typing import Any, Dict, List, Optional, Set

from .cache_helper import CacheHelper


class DeepCacheHelper(CacheHelper):
    """
    DeepCache helper class that extends CacheHelper with block-level caching control.

    This class inherits from CacheHelper and adds the ability to skip caching for
    specific blocks based on their IDs and types, providing fine-grained control
    over the caching behavior.
    """

    def __init__(
        self,
        double_blocks: Optional[List] = None,
        single_blocks: Optional[List] = None,
        no_cache_steps: Optional[Set[int]] = None,
        no_cache_block_id: Optional[Dict[str, Set[int]]] = None,
    ):
        """
        Initialize DeepCacheHelper.

        Args:
            double_blocks: List of double block modules, can be None
            single_blocks: List of single block modules, can be None
            no_cache_steps: Set of steps where caching should not be used
            no_cache_block_id: Dictionary mapping block types("double_blocks",
            "single_blocks") to sets of block IDs that should not be cached
        """
        super().__init__(
            double_blocks=double_blocks,
            single_blocks=single_blocks,
            no_cache_steps=no_cache_steps,
        )
        self.no_cache_block_id = (
            no_cache_block_id if no_cache_block_id is not None else {}
        )

    def is_skip(self, block_id: int, blocktype: str) -> bool:
        # For some pipelines, the first timestep may not be 0
        if self.start_timestep is None:
            self.start_timestep = self.cur_timestep

        # If current step is in no_cache_steps, do not use cache
        if self.cur_timestep - self.start_timestep in self.no_cache_steps:
            return False

        # If current block is in no_cache_block_id, do not use cache
        if self.no_cache_block_id and blocktype in self.no_cache_block_id:
            if block_id in self.no_cache_block_id[blocktype]:
                return False

        return True

    def wrap_block_forward(self, block: Any, block_id: int, blocktype: str) -> None:
        # Save the original forward method
        self.function_dict[(blocktype, block_id)] = block.forward

        def wrapped_forward(*args, **kwargs):
            """Wrapped forward method with caching logic."""
            skip = self.is_skip(block_id, blocktype)

            if skip:
                # Use cached output
                result = self.cached_output[(blocktype, block_id)]
            else:
                # Recompute and cache the result
                result = self.function_dict[(blocktype, block_id)](*args, **kwargs)
                self.cached_output[(blocktype, block_id)] = result

            return result

        block.forward = wrapped_forward
