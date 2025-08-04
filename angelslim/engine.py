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
from typing import Any, Optional

import torch

from .compressor import CompressorFactory
from .data.dataloader import DataLoaderFactory
from .models import SlimModelFactory
from .utils import default_compress_config, get_package_info, print_info

DEFAULT_COMPRESSION_CONFIG = {
    "fp8_static": default_compress_config.default_fp8_static_config(),
    "fp8_dynamic": default_compress_config.default_fp8_dynamic_config(),
    "int8_dynamic": default_compress_config.default_int8_dynamic_config(),
    "int4_awq": default_compress_config.default_int4_awq_config(),
    "int4_gptq": default_compress_config.default_int4_gptq_config(),
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
                )
                self.model_path = model_path
        elif self.series == "Diffusion":
            if not model:
                self.slim_model.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    cache_dir=cache_dir,
                )
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
        if compress_name not in CompressorFactory.get_available_compressor():
            raise ValueError(
                f"Compression method '{compress_name}' not registered. "
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
        self.compress_type = compress_name
        # Create compressor by CompressorFactory
        self.compressor = CompressorFactory.create(
            compress_name, self.slim_model, slim_config=slim_config
        )
        return self.compressor

    def run(self) -> Any:
        """Execute compression pipeline"""
        if not self.compressor:
            raise RuntimeError(
                "Compressor not initialized. Call prepare_compressor() first"
            )

        if self.compress_type == "PTQ":
            self.compressor.calibrate(self.dataloader)
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
        if self.compress_type == "PTQ":
            # Execute model conversion
            self.compressor.convert()

        # Save quantized model
        self.compressor.save(save_path)

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
            with open(os.path.join(save_path, "angelslim_config.json"), "w") as f:
                json.dump(config_dict, f, indent=4)

        print_info(f"Compressed model saved to {save_path}")

    def infer(self, input_prompt: str, **kwargs) -> Any:
        """Run inference with the compressed model
        Args:
            input_prompt (str): Input prompt for the model.
        """
        if not self.slim_model or not self.slim_model.model:
            raise RuntimeError("Model not initialized. Call prepare_model() first")

        if self.series in ["LLM", "VLM"]:
            return self.slim_model.generate(
                input_ids=self.slim_model.tokenizer(
                    input_prompt, return_tensors="pt"
                ).input_ids,
                **kwargs,
            )
        elif self.series == "Diffusion":
            return self.slim_model.generate(input_prompt, **kwargs)
        else:
            raise NotImplementedError(
                f"Series {self.series} is not implemented for inference"
            )
