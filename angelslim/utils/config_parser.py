# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from .utils import get_hf_config


@dataclass
class GlobalConfig:
    """
    Global configuration for LLM compression.
    Attributes:
        save_path: Directory to save compressed models
        max_seq_length: Maximum sequence length for calibration data
        hidden_size: Hidden size of the model
        model_arch_type: Model architecture type
        deploy_backend: Backend for deployment (e.g., "vllm", "huggingface")
    """

    save_path: str = field(default="./output")
    # Shared max_seq_length configuration
    max_seq_length: int = field(default=2048)
    hidden_size: int = field(default=2048)
    model_arch_type: str = field(default=None)
    deploy_backend: str = field(default="vllm")

    def update(self, model_path: str = None, max_seq_length: int = None):
        """
        Update global configuration with model and dataset properties.

        Args:
            model_path: Path to the model for extracting hidden size and architecture
            max_seq_length: Maximum sequence length for the model

        Returns:
            Updated GlobalConfig object
        """
        if model_path:
            self.set_model_hidden_size(model_path)
            self.set_model_arch_type(model_path)
        if max_seq_length:
            self.set_max_seq_length(max_seq_length)

    def set_max_seq_length(self, value: int):
        self.max_seq_length = value

    def get_max_seq_length(self) -> int:
        return self.max_seq_length

    def set_model_hidden_size(self, model_path) -> int:
        json_data = get_hf_config(model_path)
        self.hidden_size = json_data["hidden_size"]

    def set_model_arch_type(self, model_path) -> str:
        json_data = get_hf_config(model_path)
        self.model_arch_type = json_data["model_type"]


@dataclass
class ModelConfig:
    """
    Configuration for the LLM model to be compressed.

    Attributes:
        name: Model name (e.g., "Qwen3-8B")
        model_path: Path to model weights/directory
        trust_remote_code: Trust remote code for HuggingFace
        torch_dtype: PyTorch dtype for loading (e.g., "bfloat16")
        device_map: Strategy for device placement (e.g., "auto", "cpu", "cuda")
        low_cpu_mem_usage: Use low memory loading for large models
        use_cache: Whether to use cache during model loading
        cache_dir: Directory for caching model files
    """

    name: str
    model_path: str
    trust_remote_code: bool = field(default=True)
    torch_dtype: str = field(default="auto")
    device_map: str = field(default="auto")
    low_cpu_mem_usage: bool = field(default=True)
    use_cache: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)


@dataclass
class DatasetConfig:
    """
    Configuration for LLM dataset used in compression.

    Attributes:
        name: Dataset identifier (e.g., "wikitext")
        data_path: Directory path to dataset files
        max_length: Context length for processing
        max_samples: Maximum samples for calibration
        batch_size: Batch size for processing
        shuffle: whether to shuffle dataset
    """

    name: str
    data_path: str
    max_seq_length: int = field(default=2048)
    num_samples: int = field(default=256)
    batch_size: int = field(default=1)
    shuffle: bool = field(default=False)


@dataclass
class QuantizationConfig:
    """
    Quantization-specific configurations for LLMs.

    Attributes:
        name: Quantization method (e.g., "awq", "gptq")
        bits: Quantization bit-width (4/8)
        group_size: Group size for grouped quantization
        quant_method: Algorithm used for quantization
        modules_to_quantize: List of module types to quantize
        ignore_layers: List of layer names to skip
    """

    name: str = field(default="fp8_dynamic")
    bits: int = field(default=8)
    quant_method: Dict[str, Any] = field(
        default_factory=lambda: {
            "weight": "per-tensor",
            "activation": "per-tensor",
            "group_size": -1,
        }
    )
    quant_helpers: List[str] = field(default_factory=list)
    smooth_alpha: float = field(default=0.5)
    low_memory: bool = field(default=False)
    modules_to_quantize: List[str] = field(default_factory=list)
    zero_point: bool = field(default=True)
    mse_range: bool = field(default=False)
    ignore_layers: List[str] = field(default_factory=list)
    quant_analyse: bool = field(default=False)
    quant_vit: bool = field(default=False)


@dataclass
class CacheConfig:
    """
    Configuration for caching in LLM compression.

    Attributes:
        name: Cache method (e.g., "DeepCache")
        no_cache_steps: List of steps where caching is disabled
    """

    name: str = field(default="DeepCache")
    use_cache_helper: bool = field(default=False)
    no_cache_steps: List[int] = field(default_factory=list)
    no_cache_block_id: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "single": [],
            "multi": [],
        }
    )
    cnt: int = field(default=0)
    num_steps: int = field(default=50)
    rel_l1_thresh: float = field(default=0.6)  # Threshold for relative L1 distance
    # Accumulated distance for caching decisions
    accumulated_rel_l1_distance: float = field(default=0.0)


@dataclass
class CompressionConfig:
    """
    Compression configurations container for LLM.

    Attributes:
        method: Selected compression method
        quantization: Quantization configurations
        speculative_decoding: Training settings for quantization-aware compression
    """

    name: str
    quantization: Optional[QuantizationConfig] = None
    cache: Optional[CacheConfig] = None
    # speculative_decoding: Optional[SpeculativeDecodingConfig] = None

    @property
    def need_dataset(self) -> bool:
        return (
            "dynamic" not in self.quantization.name
            or "smooth" in self.quantization.quant_helpers
        )


@dataclass
class InferenceConfig:
    """Configuration for inference parameters.
    Attributes:
        height: Height of the generated image
        width: Width of the generated image
        guidance_scale: Guidance scale for inference
        num_inference_steps: Number of inference steps
        max_sequence_length: Maximum sequence length for the model
        seed: Random seed for reproducibility
    """

    height: Optional[int]
    width: Optional[int]
    guidance_scale: Optional[float]
    num_inference_steps: Optional[int]
    max_sequence_length: Optional[float]
    seed: Optional[int]


@dataclass
class FullConfig:
    """
    Top-level configuration container for LLM compression.

    Attributes:
        model_config: Model configuration parameters
        compression_config: Compression configuration parameters
        dataset_config: Dataset configuration parameters
    """

    model_config: ModelConfig
    compression_config: CompressionConfig
    dataset_config: DatasetConfig
    global_config: GlobalConfig
    infer_config: InferenceConfig


class SlimConfigParser:
    """
    Parser for LLM compression YAML configurations.

    Methods:
        parse: Load and validate configuration from YAML
    """

    def __init__(self):
        # Supported compression methods
        self.supported_methods = ["PTQ", "QAT", "Cache", "speculative_decoding"]
        # Supported quantization methods
        self.supported_quant_methods = [
            "fp8_static",
            "fp8_dynamic",
            "int4_awq",
            "int4_gptq",
            "int8_dynamic",
            "w4a8_fp8",
        ]
        # Supported speculative decoding methods
        self.supported_speculative_decoding_methods = ["EAGLE", "EAGLE2", "EAGLE3"]

    def parse(self, yaml_path: str) -> FullConfig:
        """
        Load and parse YAML configuration file for LLM compression.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Fully populated FullConfig object

        Raises:
            ValueError: On invalid configuration or unsupported methods
        """
        try:
            with open(yaml_path, "r") as f:
                config_dict = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file '{yaml_path}' not found. Using defaults.")
            return self.get_default_config()

        return self._get_configs(config_dict)

    def _get_configs(self, config_dict: dict) -> FullConfig:
        # Parse base configurations
        model_dict = config_dict.get("model", {})
        if not model_dict:
            raise ValueError("Missing 'model' section in configuration")
        # Initialize model config
        model_conf = ModelConfig(**model_dict)

        dataset_conf = None
        if "dataset" in config_dict:
            dataset_dict = config_dict["dataset"]
            dataset_conf = DatasetConfig(**dataset_dict)

        # Get compression section
        compression_dict = config_dict.get("compression", {})
        if not compression_dict:
            raise ValueError("Missing 'compression' section in configuration")

        # Validate compression method
        compress_name = compression_dict.get("name")
        if compress_name not in self.supported_methods:
            raise ValueError(
                f"Unsupported compression method: {compress_name}. "
                f"Supported methods: {self.supported_methods}"
            )

        # Initialize compression config
        compression_conf = CompressionConfig(name=compress_name)

        # Parse method-specific configurations
        if compress_name in ["PTQ", "QAT"]:
            # Validate quantization type
            quant_dict = compression_dict.get("quantization", {})
            quant_method = quant_dict.get("name")
            if quant_method not in self.supported_quant_methods:
                raise ValueError(
                    f"Unsupported quantization method: {quant_method}. "
                    f"Supported: {self.supported_quant_methods}"
                )

            # Parse quantization config
            compression_conf.quantization = QuantizationConfig(**quant_dict)
        elif compress_name == "Cache":
            # Parse cache configuration
            cache_dict = compression_dict.get("cache", {})
            compression_conf.cache = CacheConfig(**cache_dict)
        else:
            raise ValueError(
                f"Unsupported compression method: {compress_name}. "
                f"Supported methods: {self.supported_methods}"
            )

        # Global properties
        global_config = self._get_global_config(config_dict, model_conf, dataset_conf)

        # Inference configuration
        inference_conf = None
        if "inference" in config_dict:
            inference_dict = config_dict["inference"]
            inference_conf = InferenceConfig(**inference_dict)

        return FullConfig(
            model_config=model_conf,
            compression_config=compression_conf,
            dataset_config=dataset_conf,
            global_config=global_config,
            infer_config=inference_conf,
        )

    def _get_global_config(
        self, config_dict, model_conf, dataset_conf=None
    ) -> GlobalConfig:
        """
        Extract global configuration settings from the provided dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            GlobalConfig object with populated fields
        """
        global_dict = config_dict.get("global", {})
        global_config = GlobalConfig(**global_dict)
        return global_config

    @staticmethod
    def get_default_config() -> FullConfig:
        """Return a default configuration for Qwen model"""
        model_config = ModelConfig(
            name="Qwen",
            model_path="Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True,
        )
        # Global properties
        global_config = GlobalConfig()
        global_config.set_model_hidden_size(model_config.model_path)
        global_config.set_model_arch_type(model_config.model_path)
        return FullConfig(
            model_config=model_config,
            compression_config=CompressionConfig(
                name="PTQ",
                quantization=QuantizationConfig(
                    name="fp8_dynamic",
                    bits=8,
                    ignore_layers=["lm_head", "model.embed_tokens"],
                ),
            ),
            dataset_config=None,
            global_config=global_config,
        )


def print_config(config, indent=0):
    """
    Print the configuration in a structured YAML-like format

    Args:
        config: Configuration object to print
        indent: Current indentation level
    """
    prefix = " " * indent
    next_indent = indent + 2

    # Special handling for FullLLMConfig
    if (
        hasattr(config, "model_config")
        and hasattr(config, "compression_config")
        and hasattr(config, "dataset_config")
        and hasattr(config, "global_config")
        and hasattr(config, "infer_config")
    ):
        print(f"{prefix}model:")
        print_config(config.model_config, next_indent)

        print(f"{prefix}compression:")
        print_config(config.compression_config, next_indent)

        print(f"{prefix}dataset:")
        if config.dataset_config:
            print_config(config.dataset_config, next_indent)
        else:
            print(f"{prefix}None")

        print(f"{prefix}Global:")
        if config.global_config:
            print_config(config.global_config, next_indent)
        else:
            print(f"{prefix}None")

        print(f"{prefix}Inference:")
        if config.infer_config:
            print_config(config.infer_config, next_indent)
        else:
            print(f"{prefix}None")
        return

    # Handle dataclass instances
    if hasattr(config, "__dataclass_fields__"):
        for _field in config.__dataclass_fields__:
            value = getattr(config, _field)
            # Skip uninteresting default values
            if value is None or (isinstance(value, list) and len(value) == 0):
                continue

            # Special case for models with path in name
            if _field == "name" and hasattr(config, "path") and config.path != "":
                value = f"{value} @ {config.path}"

            # Print field with appropriate formatting
            if hasattr(value, "__dataclass_fields__"):
                print(f"{prefix}{_field}:")
                print_config(value, next_indent)
            elif isinstance(value, list):
                print(f"{prefix}{_field}:")
                for item in value:
                    print(f"{prefix}- {item}")
            elif isinstance(value, bool):
                print(f"{prefix}{_field}: {'true' if value else 'false'}")
            else:
                print(f"{prefix}{_field}: {value}")
        return

    # Fallback for other types
    print(f"{prefix}{str(config)}")


# Example usage
if __name__ == "__main__":
    parser = SlimConfigParser()
    config = parser.parse("llm_compression_config.yaml")

    # Access configuration values
    print(f"Compressing model: {config.model_config.name}")
    print(f"Device map: {config.model_config.device_map}")

    comp_conf = config.compression_config
    print(f"\nCompression Method: {comp_conf.name}")

    if comp_conf.name == "quantization":
        quant_conf = comp_conf.quantization
        print(f"Quantization Type: {quant_conf.name}")
        print(f"Bit Width: {quant_conf.bits}-bit")

    _dataset_conf = config.dataset_config
    if _dataset_conf:
        print(f"\nDataset: {_dataset_conf.name}")
        print(f"Max Context Length: {_dataset_conf.max_seq_length}")
