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

import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch

from .compressor import CompressorFactory
from .compressor.speculative.benchmark import pytorch as pytorch_benchmark
from .compressor.speculative.benchmark import vllm as vllm_benchmark
from .data.dataloader import DataLoaderFactory
from .models import SlimModelFactory
from .utils import (
    default_compress_config,
    get_package_info,
    parse_json_full_config,
    print_info,
)

DEFAULT_COMPRESSION_CONFIG = {
    "fp8_static": default_compress_config.default_fp8_static_config(),
    "fp8_dynamic": default_compress_config.default_fp8_dynamic_config(),
    "int8_dynamic": default_compress_config.default_int8_dynamic_config(),
    "int4_awq": default_compress_config.default_int4_awq_config(),
    "int4_gptq": default_compress_config.default_int4_gptq_config(),
    "w4a8_fp8": default_compress_config.default_w4a8_fp8_static_config(),
}


def get_supported_compress_method():
    return DEFAULT_COMPRESSION_CONFIG.keys()


class Engine:
    def __init__(self):
        """
        Initialize engine configuration
        """
        self.slim_model = None
        self.tokenizer = None
        self.dataloader = None
        self.compressor = None
        self.compress_type = None
        self.only_inference = False
        self.model_path = None
        self.max_seq_length = None

    def prepare_model(
        self,
        model_name="Qwen",
        model=None,
        tokenizer=None,
        model_path=None,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
        cache_dir=None,
        deploy_backend="vllm",
        using_multi_nodes=False,
    ) -> Any:
        """Load pretrained model and tokenizer
        Args:
            model_name (str): Name of the model to load.
            model (Any, optional): Preloaded model instance.
                If provided, `model_path` is ignored.
            tokenizer (Any, optional): Preloaded tokenizer instance.
                If model is set, tokenizer must be also set in LLM and VLM.
            model_path (str, optional): Path to the pretrained model.
            torch_dtype (str): Data type for the model weights.
            device_map (str): Device map for the model.
            trust_remote_code (bool): Whether to trust remote code.
            low_cpu_mem_usage (bool): Whether to use low CPU memory usage mode.
            use_cache (bool): Whether to use cache during loading.
            cache_dir (str, optional): Directory to cache the model.
            deploy_backend (str): Backend for deployment, e.g., "torch", "vllm".
            using_multi_nodes (bool): Whether to use multi-nodes for calibration.
        """
        assert model_name, "model_name must be specified."
        assert model_path, "model_path must be specified."

        # Initialize slim model by ModelFactory
        self.slim_model = SlimModelFactory.create(
            model_name, model=model, deploy_backend=deploy_backend
        )

        self.series = SlimModelFactory.get_series_by_models(model_name)

        if self.series in ["LLM", "VLM"]:
            if model:
                assert tokenizer, " If model is set, tokenizer must be also set."
                self.slim_model.tokenizer = tokenizer
            else:
                self.slim_model.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    use_cache=use_cache,
                    using_multi_nodes=using_multi_nodes,
                )
                self.model_path = model_path
        else:
            raise ValueError(f"Unsupported series: {self.series}")

        return self.slim_model

    def prepare_data(
        self,
        data_path=None,
        data_type="TextDataset",
        custom_dataloader=None,
        max_length=2048,
        batch_size=1,
        num_samples=128,
        shuffle=True,
        inference_settings=None,
        model_name=None,
    ) -> Optional[Any]:
        """Prepare compression dataset"""
        if custom_dataloader is not None:
            print_info("Using custom provided dataloader...")
            self.dataloader = custom_dataloader
            return self.dataloader

        assert data_path, "data_path must be specified."
        # Dynamically create dataloader by DataLoaderFactory
        self.dataloader = DataLoaderFactory.create_data_loader(
            data_type=data_type,
            processor=(
                self.slim_model.processor
                if self.series == "VLM"
                else self.slim_model.tokenizer
            ),
            device=self.slim_model.model.device,
            max_length=max_length,
            batch_size=batch_size,
            shuffle=shuffle,
            num_samples=num_samples,
            data_source=data_path,
            inference_settings=inference_settings,
            model_name=model_name,
        )
        self.max_seq_length = max_length

        return self.dataloader

    def prepare_compressor(
        self,
        compress_name="PTQ",
        global_config=None,
        compress_config=None,
        default_method=None,
    ) -> Any:
        """
        Initialize compression components.
        Args:
            compress_name (str): Name of the compression method to use.
            global_config (dict, optional): Global configuration for the model.
            compress_config (dict, optional): Configuration for the compression method.
            default_method (str, optional): Default compression method if not specified.
               If set default_method, compress_config and global_config will be ignored.
        """
        if isinstance(compress_name, str):
            compress_names = [compress_name]
        elif isinstance(compress_name, list):
            compress_names = compress_name
        for method_name in compress_names:
            if method_name not in CompressorFactory.get_available_compressor():
                raise ValueError(
                    f"Compression method '{method_name}' not registered. "
                    f"Available methods: {CompressorFactory.get_available_compressor()}"
                )
        if self.series in ["LLM", "VLM"]:
            global_config.update(self.model_path, self.max_seq_length)

        if default_method:
            assert (
                default_method in DEFAULT_COMPRESSION_CONFIG
            ), f"`default_method` not found in : {DEFAULT_COMPRESSION_CONFIG.keys()}."
            slim_config = DEFAULT_COMPRESSION_CONFIG[default_method]
        else:
            slim_config = {
                "global_config": global_config,
                "compress_config": compress_config,
            }
        self.compress_type = compress_names
        self.only_inference = (
            compress_config.only_inference if compress_config else False
        )
        # Create compressor by CompressorFactory
        self.compressor = CompressorFactory.create(
            compress_names, self.slim_model, slim_config=slim_config
        )
        return self.compressor

    def run(self) -> Any:
        """Execute compression pipeline"""
        if not self.compressor:
            raise RuntimeError(
                "Compressor not initialized. Call prepare_compressor() first"
            )
        if isinstance(self.compressor, str):
            compressors = [self.compressor]
        elif isinstance(self.compressor, list):
            compressors = self.compressor
        for idx, compress_type in enumerate(self.compress_type):
            if self.only_inference[idx]:
                continue
            if compress_type == "PTQ":
                compressors[idx].calibrate(self.dataloader)
            else:
                raise NotImplementedError(
                    f"Compression type {self.compress_type} is not implemented"
                )

    def save(
        self, save_path: Optional[str] = None, config: Optional[dataclass] = None
    ) -> None:
        """Save compressed model and tokenizer
        Args:
            save_path (str, optional): Path to save the compressed model and tokenizer.
        """
        assert save_path, "Save path must be provided in model_config or as an argument"
        if isinstance(self.compressor, str):
            compressors = [self.compressor]
        elif isinstance(self.compressor, list):
            compressors = self.compressor
        for idx, compress_type in enumerate(self.compress_type):
            if self.only_inference[idx]:
                continue
            if compress_type == "PTQ":
                # Execute model conversion
                compressors[idx].convert()

            # Save quantized model
            compressors[idx].save(save_path)

        # Save all config
        if config is not None:
            config_dict = asdict(config)
            config_dict["debug_info"] = {
                "python": sys.version,
                "angelslim": get_package_info("angelslim"),
                "torch": get_package_info("torch"),
                "transformers": get_package_info("transformers"),
                "torch_cuda_version": (
                    torch.version.cuda if torch.cuda.is_available() else None
                ),
            }
            config_dict["model_config"]["model_path"] = "Base Model Path"
            config_dict["global_config"]["save_path"] = "Save Model Path"
            if "dataset_config" in config_dict and isinstance(
                config_dict["dataset_config"], dict
            ):
                config_dict["dataset_config"]["data_path"] = "Data Path"
            with open(os.path.join(save_path, "angelslim_config.json"), "w") as f:
                json.dump(config_dict, f, indent=4)

        print_info(f"Compressed model saved to {save_path}")


class InferEngine(Engine):
    def __init__(self):
        """
        Initialize engine configuration
        """
        super().__init__()
        self.slim_model = None
        self.tokenizer = None
        self.dataloader = None
        self.compressor = None
        self.compress_type = None
        self.model_path = None
        self.max_seq_length = None

    def from_pretrained(
        self,
        model_path,
        torch_dtype=None,
        device_map=None,
        trust_remote_code=None,
        low_cpu_mem_usage=None,
        use_cache=None,
    ) -> Any:
        """Load pretrained model and tokenizer
        Args:
            model_path (str): Path to the pretrained model.
            torch_dtype (str): Data type for the model weights.
            device_map (str): Device map for the model.
            trust_remote_code (bool): Whether to trust remote code.
            low_cpu_mem_usage (bool): Whether to use low CPU memory usage mode.
            use_cache (bool): Whether to use cache during loading.
            cache_dir (str, optional): Directory to cache the model.
        """
        assert model_path, "model_path must be specified."
        # load slim config
        slim_config_path = os.path.join(model_path, "angelslim_config.json")
        if not os.path.exists(slim_config_path):
            raise FileNotFoundError(
                f"angelslim_config.json not found in {model_path}. "
                "Please ensure the model is compressed with Angelslim."
            )
        slim_config = parse_json_full_config(slim_config_path)
        if torch_dtype:
            slim_config.model_config.torch_dtype = torch_dtype
        if device_map:
            slim_config.model_config.device_map = device_map
        if trust_remote_code is not None:
            slim_config.model_config.trust_remote_code = trust_remote_code
        if low_cpu_mem_usage is not None:
            slim_config.model_config.low_cpu_mem_usage = low_cpu_mem_usage
        if use_cache is not None:
            slim_config.model_config.use_cache = use_cache

        self.slim_model = SlimModelFactory.create(
            slim_config.model_config.name, deploy_backend="huggingface"
        )

        self.slim_model.from_pretrained(
            model_path=model_path,
            torch_dtype=slim_config.model_config.torch_dtype,
            device_map=slim_config.model_config.device_map,
            trust_remote_code=slim_config.model_config.trust_remote_code,
            low_cpu_mem_usage=slim_config.model_config.low_cpu_mem_usage,
            use_cache=slim_config.model_config.use_cache,
            compress_config=slim_config.compression_config,
        )

        self.series = SlimModelFactory.get_series_by_models(
            slim_config.model_config.name
        )

    def generate(self, input_prompt: str, **kwargs) -> Any:
        """Run inference with the compressed model
        Args:
            input_prompt (str): Input prompt for the model.
        """
        if not self.slim_model or not self.slim_model.model:
            raise RuntimeError("Model not initialized. Call from_pretrained() first")

        if self.series in ["LLM", "VLM"]:
            return self.slim_model.generate(
                input_ids=self.slim_model.tokenizer(
                    input_prompt, return_tensors="pt"
                ).input_ids,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                f"Series {self.series} is not implemented for inference"
            )


class SpecEngine:
    """
    High-level interface for speculative decoding benchmarks
    Integrates BenchmarkEngine with additional workflow management
    """

    def __init__(self, config=None, deploy_backend: str = "pytorch"):
        """
        Initialize SpecEngine

        Args:
            config: BenchmarkConfig instance (optional)
            deploy_backend: Backend to use ('pytorch' or 'vllm')
        """
        self.config = config
        self.benchmark_engine = None
        self.results = {}
        self.deploy_backend = deploy_backend.lower()

        if self.deploy_backend == "pytorch":
            self.BenchmarkConfig = pytorch_benchmark.BenchmarkConfig
            self.BenchmarkEngine = pytorch_benchmark.BenchmarkEngine
            self.BenchmarkMode = pytorch_benchmark.BenchmarkMode
        elif self.deploy_backend == "vllm":
            self.BenchmarkConfig = vllm_benchmark.BenchmarkConfig
            self.BenchmarkEngine = vllm_benchmark.BenchmarkEngine
            self.BenchmarkMode = vllm_benchmark.BenchmarkMode
        else:
            raise ValueError(f"Unsupported deploy_backend: {deploy_backend}")

    def setup_benchmark(
        self,
        base_model_path: str,
        eagle_model_path: str,
        model_id: str,
        bench_name: str = "mt_bench",
        output_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Setup benchmark configuration

        Args:
            base_model_path: Path to base model
            eagle_model_path: Path to Eagle model
            model_id: Model identifier
            bench_name: Benchmark dataset name
            output_dir: Output directory for results
            **kwargs: Additional configuration parameters

        Returns:
            BenchmarkConfig instance
        """
        config_dict = {
            "base_model_path": base_model_path,
            "eagle_model_path": eagle_model_path,
            "model_id": model_id,
            "bench_name": bench_name,
            "output_dir": output_dir,
        }
        config_dict.update(kwargs)

        self.config = self.BenchmarkConfig(**config_dict)
        self.benchmark_engine = self.BenchmarkEngine(self.config)

        return self.config

    def run_eagle_benchmark(self) -> Dict[str, Any]:
        """Run Eagle speculative decoding benchmark only"""
        if not self.benchmark_engine:
            raise RuntimeError(
                "Benchmark not configured. Call setup_benchmark() first."
            )

        self.results = self.benchmark_engine.run_benchmark(self.BenchmarkMode.EAGLE)
        return self.results

    def run_baseline_benchmark(self) -> Dict[str, Any]:
        """Run baseline benchmark only"""
        if not self.benchmark_engine:
            raise RuntimeError(
                "Benchmark not configured. Call setup_benchmark() first."
            )

        self.results = self.benchmark_engine.run_benchmark(self.BenchmarkMode.BASELINE)
        return self.results

    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run complete benchmark (both Eagle and baseline) with automatic analysis

        Returns:
            Dictionary containing all results and metrics
        """
        if not self.benchmark_engine:
            raise RuntimeError(
                "Benchmark not configured. Call setup_benchmark() first."
            )

        self.results = self.benchmark_engine.run_benchmark(self.BenchmarkMode.BOTH)
        return self.results

    def calculate_acceptance_length(self, eagle_file: Optional[str] = None) -> float:
        """
        Calculate acceptance length from Eagle benchmark results

        Args:
            eagle_file: Path to Eagle results file
                (optional, uses default if not provided)

        Returns:
            Average acceptance length
        """
        if not self.benchmark_engine:
            raise RuntimeError(
                "Benchmark not configured. Call setup_benchmark() first."
            )

        if eagle_file is None:
            eagle_file = self.benchmark_engine.eagle_file

        return self.benchmark_engine._calculate_acceptance_length(eagle_file)

    def calculate_speedup_ratio(
        self,
        baseline_file: Optional[str] = None,
        eagle_file: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> float:
        """
        Calculate speedup ratio between baseline and Eagle

        Args:
            baseline_file: Path to baseline results file
            eagle_file: Path to Eagle results file
            model_path: Path to model for tokenization

        Returns:
            Speedup ratio
        """
        if not self.benchmark_engine:
            raise RuntimeError(
                "Benchmark not configured. Call setup_benchmark() first."
            )

        if baseline_file is None:
            baseline_file = self.benchmark_engine.baseline_file
        if eagle_file is None:
            eagle_file = self.benchmark_engine.eagle_file
        if model_path is None:
            model_path = self.config.base_model_path

        return self.benchmark_engine._calculate_speedup_ratio(
            model_path, baseline_file, eagle_file
        )

    def get_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        if not self.benchmark_engine:
            return "Benchmark not configured."

        return self.benchmark_engine.get_performance_summary()

    def cleanup_results(self):
        """Clean up temporary result files"""
        if self.benchmark_engine:
            for file_path in [
                self.benchmark_engine.eagle_file,
                self.benchmark_engine.baseline_file,
                self.benchmark_engine.analysis_file,
            ]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
