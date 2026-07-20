from angelslim.compressor.mcore_qad.train.config import TrainConfig

__all__ = ["TrainConfig", "Trainer"]


def __getattr__(name):
    """Keep Megatron-Core optional until the trainer is actually requested."""
    if name == "Trainer":
        from angelslim.compressor.mcore_qad.train.trainer import Trainer

        return Trainer
    raise AttributeError(name)
