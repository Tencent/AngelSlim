"""Per-model adapters. Importing registers each model_type into the adapter registry."""

from angelslim.compressor.mcore_qad.models import (  # noqa: F401  (registration side-effects)
    hy_v3,
    qwen3,
)
from angelslim.compressor.mcore_qad.models.base import (
    ModelMeta,
    auto_config,
    get_adapter,
    load_hf_config,
    load_hf_into_mcore,
    register,
)

__all__ = [
    "ModelMeta",
    "load_hf_config",
    "auto_config",
    "load_hf_into_mcore",
    "get_adapter",
    "register",
]
