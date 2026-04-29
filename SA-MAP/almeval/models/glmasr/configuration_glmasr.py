from typing import Any, Dict, List, Optional

from transformers import LlamaConfig, PretrainedConfig, WhisperConfig


class GlmasrConfig(PretrainedConfig):
    model_type = "Glmasr"
    is_composition = True

    def __init__(
        self,
        lm_config: Optional[Dict[str, Any] | LlamaConfig] = None,
        whisper_config: Optional[Dict[str, Any] | WhisperConfig] = None,
        adapter_type: str = "mlp",
        merge_factor: int = 2,
        spec_aug: bool = False,
        use_rope: bool = False,
        max_whisper_length: int = 1500,
        max_length: int = 1024,
        mlp_adapter_act: str = "gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(lm_config, LlamaConfig):
            self.lm_config = lm_config
        else:
            self.lm_config = LlamaConfig.from_dict(lm_config or {})
        if isinstance(whisper_config, WhisperConfig):
            self.whisper_config = whisper_config
        else:
            self.whisper_config = WhisperConfig.from_dict(whisper_config or {})

        self.adapter_type = adapter_type
        self.merge_factor = merge_factor
        self.spec_aug = spec_aug
        self.use_rope = use_rope
        self.max_whisper_length = max_whisper_length
        self.max_length = max_length
        self.mlp_adapter_act = mlp_adapter_act


__all__ = ["GlmasrConfig"]
