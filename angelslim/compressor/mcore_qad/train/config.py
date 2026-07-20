"""Top-level training configuration for the mcore QAT/QAD path.

Built from CLI args in train.py (all knobs live in the launch bash); no YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParallelConfig:
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    context_parallel: int = 1
    expert_parallel: int = 1
    sequence_parallel: bool = True


@dataclass
class OptimConfig:
    lr: float = 2e-4  # scale-only -> usually higher LR than weight training
    weight_decay: float = 0.0  # generally 0 for scales
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0


@dataclass
class TrainConfig:
    ckpt_path: str  # reshardable mcore dist-checkpoint (FP weights)
    hf_path: str  # HF dir (config + tokenizer)
    fmt: str = "nvfp4"  # nvfp4 | nvfp4a16 | w4a16 | w8a8 | fp8 | w4afp8
    data_path: Optional[str] = None  # ShareGPT jsonl (None -> random tokens)
    init_scales_path: Optional[str] = None  # external/PTQ initial scales
    # loss: total = lm_weight * lm + distill_weight * distill. distill off by default.
    lm_weight: float = 1.0
    distill_weight: float = 0.0
    distill_type: str = "kl"  # kl | rkl | cakld
    distill_temperature: float = 1.0
    distill_topk: int = 0  # >0: KL over teacher top-k only (large-vocab memory)
    experts_only: bool = False  # quantize ONLY routed MoE experts; everything else stays BF16
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    seq_len: int = 256
    micro_batch_size: int = 1
    train_iters: int = 100
    recompute: bool = True  # full activation recompute (see mcore.patches); needed
    # to fit long seq / big models, amortized by a large mbs.
    save_path: Optional[str] = None
    save_every: int = 0  # >0: also snapshot scales every N steps (tiny; scales only)
