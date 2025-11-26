from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class CacheHelper:
    """
    Cache helper class for managing caching of Diffusion models.

    This class wraps the forward methods of modules to cache output
    results at specified steps, enabling cache reuse in subsequent steps
    to improve inference efficiency.
    """

    def __init__(
        self,
        double_blocks: Optional[List] = None,
        single_blocks: Optional[List] = None,
        no_cache_steps: Optional[Set[int]] = None,
    ):
        """
        Initialize CacheHelper.

        Args:
            double_blocks: List of double block modules
            single_blocks: List of single block modules
            no_cache_steps: Set of steps where caching should not be used
        """
        self.double_blocks = double_blocks if double_blocks is not None else []
        self.single_blocks = single_blocks if single_blocks is not None else []
        self.no_cache_steps = no_cache_steps if no_cache_steps is not None else set()

        self.start_timestep: Optional[int] = None
        self.cur_timestep: int = 0
        self.function_dict: Dict[Tuple[str, int], Callable] = {}
        self.cached_output: Dict[Tuple[str, int], List] = {}

    def enable(self) -> None:
        """
        Enable caching functionality.

        Raises:
            ValueError: Raised when both double_blocks and single_blocks are empty
        """
        if not self.double_blocks and not self.single_blocks:
            raise ValueError(
                "At least one of double_blocks or single_blocks must be provided"
            )

        self.reset_states()
        self.wrap_modules()

    def disable(self) -> None:
        """Disable caching functionality and restore original forward methods."""
        self.unwrap_modules()
        self.reset_states()

    def is_skip(self) -> bool:
        """
        Determine whether the current step should skip.

        Returns:
            bool: True means use cache, False means recompute
        """
        # For some pipelines, the first timestep may not be 0
        if self.start_timestep is None:
            self.start_timestep = self.cur_timestep

        # If current step is in no_cache_steps, do not use cache
        if self.cur_timestep - self.start_timestep in self.no_cache_steps:
            return False

        return True

    def wrap_block_forward(self, block: Any, block_id: int, blocktype: str) -> None:
        """
        Wrap a single block's forward method with caching logic.

        Args:
            block: The block module to wrap
            block_id: The index ID of the block
            blocktype: The type of block ("double_blocks" or "single_blocks")
        """
        # Save the original forward method
        self.function_dict[(blocktype, block_id)] = block.forward

        def wrapped_forward(*args, **kwargs):
            """Wrapped forward method with caching logic."""
            skip = self.is_skip()

            if skip:
                # Use cached output
                result = self.cached_output[(blocktype, block_id)]
            else:
                # Recompute and cache the result
                result = self.function_dict[(blocktype, block_id)](*args, **kwargs)
                self.cached_output[(blocktype, block_id)] = result

            return result

        block.forward = wrapped_forward

    def wrap_modules(self) -> None:
        """Wrap forward methods of all blocks."""
        # Wrap double blocks
        if self.double_blocks:
            for block_id, block in enumerate(self.double_blocks):
                self.wrap_block_forward(block, block_id, blocktype="double_blocks")

        # Wrap single blocks
        if self.single_blocks:
            for block_id, block in enumerate(self.single_blocks):
                self.wrap_block_forward(block, block_id, blocktype="single_blocks")

    def unwrap_modules(self) -> None:
        """Restore original forward methods of all blocks."""
        # Restore double blocks
        if self.double_blocks:
            for block_id, block in enumerate(self.double_blocks):
                key = ("double_blocks", block_id)
                if key in self.function_dict:
                    block.forward = self.function_dict[key]

        # Restore single blocks
        if self.single_blocks:
            for block_id, block in enumerate(self.single_blocks):
                key = ("single_blocks", block_id)
                if key in self.function_dict:
                    block.forward = self.function_dict[key]

    def reset_states(self) -> None:
        """Reset all internal states."""
        self.start_timestep = None
        self.cur_timestep = 0
        self.function_dict = {}
        self.cached_output = {}

    def clear_states(self) -> None:
        """Clear cache states but preserve function_dict."""
        self.cur_timestep = 0
        self.start_timestep = None
        self.cached_output = {}
