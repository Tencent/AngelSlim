"""AngelSlim adapter for the isolated Megatron-Core QAT/QAD backend."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ..compressor_factory import CompressorFactory
from .train.config import OptimConfig, ParallelConfig, TrainConfig


def validate_parallel_world_size(world_size: int, parallel: ParallelConfig) -> None:
    """Validate Megatron 0.18 dense and expert parallel-folding topologies."""
    dense_fold_size = (
        parallel.tensor_parallel * parallel.pipeline_parallel * parallel.context_parallel
    )
    if world_size % dense_fold_size != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must be divisible by TP*PP*CP={dense_fold_size}."
        )

    # The grouped-expert path fixes expert tensor parallelism (ETP) to one.
    expert_fold_size = parallel.expert_parallel * parallel.pipeline_parallel
    if world_size % expert_fold_size != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must be divisible by ETP*EP*PP="
            f"{expert_fold_size} (ETP=1)."
        )


def _teardown_distributed() -> None:
    if not dist.is_initialized():
        return
    try:
        from .mcore.dist import teardown
    except ImportError:
        dist.destroy_process_group()
    else:
        teardown()


@CompressorFactory.register("MCoreQAD")
class MCoreQAD:
    """Run the mcore scale-only training lifecycle from an AngelSlim config."""

    _SUPPORTED_MODEL_TYPES = ("qwen3_moe", "hy_v3")

    def __init__(self, model: Any, slim_config: Any) -> None:
        if model is not None:
            raise ValueError(
                "MCoreQAD builds its own Megatron-Core model and does not accept "
                "an AngelSlim/Hugging Face model instance."
            )
        self.full_config = slim_config
        self.train_config = self._build_train_config(slim_config)
        self.trainer = None

    @staticmethod
    def _build_train_config(config: Any) -> TrainConfig:
        mcore = config.compression_config.MCoreQAD
        dataset = config.dataset_config
        parallel = ParallelConfig(**asdict(mcore.parallel))
        optim = OptimConfig(**asdict(mcore.optim))
        return TrainConfig(
            ckpt_path=mcore.checkpoint_path,
            hf_path=config.model_config.model_path,
            fmt=mcore.format,
            data_path=dataset.data_path if dataset is not None else None,
            init_scales_path=mcore.init_scales_path,
            lm_weight=mcore.lm_weight,
            distill_weight=mcore.distill_weight,
            distill_type=mcore.distill_type,
            distill_temperature=mcore.distill_temperature,
            distill_topk=mcore.distill_topk,
            experts_only=mcore.experts_only,
            parallel=parallel,
            optim=optim,
            seq_len=mcore.seq_len,
            micro_batch_size=mcore.micro_batch_size,
            train_iters=mcore.train_iters,
            recompute=mcore.recompute,
            save_path=config.global_config.save_path,
            save_every=mcore.save_every,
        )

    def _validate_paths(self) -> None:
        config_path = Path(self.train_config.hf_path) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"MCoreQAD requires a Hugging Face config at {config_path}.")
        with config_path.open() as config_file:
            model_type = json.load(config_file).get("model_type")
        if model_type not in self._SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported MCoreQAD model_type {model_type!r}. "
                f"Supported: {list(self._SUPPORTED_MODEL_TYPES)}"
            )

        checkpoint = Path(self.train_config.ckpt_path)
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"MCoreQAD distributed checkpoint not found: {checkpoint}.")
        if self.train_config.data_path and not Path(self.train_config.data_path).is_file():
            raise FileNotFoundError(f"MCoreQAD dataset not found: {self.train_config.data_path}.")
        if (
            self.train_config.init_scales_path
            and not Path(self.train_config.init_scales_path).is_file()
        ):
            raise FileNotFoundError(
                f"MCoreQAD initial scales not found: {self.train_config.init_scales_path}."
            )

    def _validate_runtime(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("MCoreQAD requires CUDA.")
        try:
            import megatron.core  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "MCoreQAD optional dependencies are missing. "
                "Install AngelSlim with the 'mcore-qad' extra."
            ) from error
        if self.train_config.parallel.context_parallel > 1:
            try:
                import transformer_engine  # noqa: F401
            except ImportError as error:
                raise ImportError(
                    "MCoreQAD context_parallel>1 requires Transformer Engine. "
                    "Install AngelSlim with the 'mcore-qad' extra."
                ) from error

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        validate_parallel_world_size(world_size, self.train_config.parallel)

    def run(self, dataloader: Any = None) -> None:
        """Build the mcore model and run scale-only QAT/QAD."""
        try:
            if dataloader is not None:
                raise ValueError(
                    "MCoreQAD owns its distributed data iterator; configure "
                    "dataset.data_path instead of passing an AngelSlim dataloader."
                )
            self._validate_paths()
            self._validate_runtime()

            from .train.trainer import Trainer

            self.trainer = Trainer(self.train_config).setup()
            self.trainer.train()
        finally:
            _teardown_distributed()

    def convert(self) -> None:
        """MCoreQAD is fake-quant training; deployment conversion is out of scope."""

    def save(self, save_path: str) -> None:
        """Scales are persisted by the trainer, including periodic snapshots."""
        if save_path != self.train_config.save_path:
            raise ValueError(
                "MCoreQAD save_path must match global.save_path because scales are "
                "written during training."
            )
