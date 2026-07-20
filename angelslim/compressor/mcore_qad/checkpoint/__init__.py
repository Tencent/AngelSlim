"""Checkpointing: reshardable FP weights (dist) + trainable quantizer scales."""

from angelslim.compressor.mcore_qad.checkpoint.scales import (
    load_initial_scales,
    save_scales,
)

__all__ = [
    "save_dist_checkpoint",
    "load_dist_checkpoint",
    "save_scales",
    "load_initial_scales",
]


def __getattr__(name):
    """Keep Megatron-Core optional until distributed checkpoint I/O is used."""
    if name in ("save_dist_checkpoint", "load_dist_checkpoint"):
        from angelslim.compressor.mcore_qad.checkpoint.dist import (
            load_dist_checkpoint,
            save_dist_checkpoint,
        )

        return {
            "save_dist_checkpoint": save_dist_checkpoint,
            "load_dist_checkpoint": load_dist_checkpoint,
        }[name]
    raise AttributeError(name)
