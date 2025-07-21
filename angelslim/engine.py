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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        self.model = None
        self.tokenizer = None
        self.dataloader = None
        self.compressor = None
        self.slim_model = None
        self.compress_type = None

    def prepare_model(
        self,
        model_name="Qwen",
        model_path=None,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
        deploy_backend="vllm",
    ) -> AutoModelForCausalLM:
        """Load pretrained model and tokenizer"""
        assert model_name, "model_name must be specified."
        assert model_path, "model_path must be specified."
        assert deploy_backend in [
            "vllm",
            "huggingface",
        ], "deploy_backend must be vllm or huggingface"

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_cache=use_cache,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

        # Initialize slim model by ModelFactory
        self.slim_model = SlimModelFactory.create(
            model_name, model=self.model, deploy_backend=deploy_backend
        )
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
            processor=self.tokenizer,
            device=self.model.device,
            max_length=max_length,
            batch_size=batch_size,
            shuffle=shuffle,
            num_samples=num_samples,
            data_source=data_path,
        )

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

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)

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
            with open(os.path.join(save_path, "angelslim_config.json"), "w") as f:
                json.dump(config_dict, f, indent=4)

        print_info(f"Compressed model saved to {save_path}")
