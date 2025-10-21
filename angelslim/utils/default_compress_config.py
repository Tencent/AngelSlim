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

from .config_parser import CompressionConfig, GlobalConfig, QuantizationConfig

__all__ = [
    "default_fp8_dynamic_config",
    "default_fp8_static_config",
    "default_int8_dynamic_config",
    "default_int4_gptq_config",
    "default_int4_awq_config",
    "default_int4_gptaq_config",
]


def default_fp8_dynamic_config() -> dict:
    """
    Returns a default configuration dictionary for model compression.

    This configuration includes global settings and specific compression parameters.
    """
    return {
        "global_config": GlobalConfig(),
        "compress_config": CompressionConfig(
            name="PTQ",
            quantization=QuantizationConfig(
                name="fp8_dynamic",
                bits=8,
                quant_method={"weight": "per-tensor", "activation": "per-tensor"},
                ignore_layers=["lm_head"],
            ),
        ),
    }


def default_fp8_static_config() -> dict:
    """
    Returns a default configuration dictionary for model compression.

    This configuration includes global settings and specific compression parameters.
    """
    return {
        "global_config": GlobalConfig(),
        "compress_config": CompressionConfig(
            name="PTQ",
            quantization=QuantizationConfig(
                name="fp8_static",
                bits=8,
                quant_method={"weight": "per-tensor", "activation": "per-tensor"},
                ignore_layers=["lm_head"],
            ),
        ),
    }


def default_int8_dynamic_config() -> dict:
    """
    Returns a default configuration dictionary for model compression.

    This configuration includes global settings and specific compression parameters.
    """
    return {
        "global_config": GlobalConfig(),
        "compress_config": CompressionConfig(
            name="PTQ",
            quantization=QuantizationConfig(
                name="int8_dynamic",
                bits=8,
                quant_method={"weight": "per-channel", "activation": "per-token"},
                ignore_layers=["lm_head"],
            ),
        ),
    }


def default_int4_gptq_config() -> dict:
    """
    Returns a default configuration dictionary for model compression.

    This configuration includes global settings and specific compression parameters.
    """
    return {
        "global_config": GlobalConfig(),
        "compress_config": CompressionConfig(
            name="PTQ",
            quantization=QuantizationConfig(
                name="int4_gptq",
                bits=4,
                quant_method={"weight": "per-group", "group_size": 128},
                ignore_layers=["lm_head"],
            ),
        ),
    }


def default_int4_awq_config() -> dict:
    """
    Returns a default configuration dictionary for model compression.

    This configuration includes global settings and specific compression parameters.
    """
    return {
        "global_config": GlobalConfig(),
        "compress_config": CompressionConfig(
            name="PTQ",
            quantization=QuantizationConfig(
                name="int4_awq",
                bits=4,
                quant_method={
                    "weight": "per-group",
                    "group_size": 128,
                    "zero_point": True,
                    "mse_range": False,
                },
                ignore_layers=["lm_head"],
            ),
        ),
    }


def default_w4a8_fp8_static_config() -> dict:
    """
    Returns a default configuration dictionary for model compression.

    This configuration includes global settings and specific compression parameters.
    """
    return {
        "global_config": GlobalConfig(),
        "compress_config": CompressionConfig(
            name="PTQ",
            quantization=QuantizationConfig(
                name="w4a8_fp8",
                bits=4,
                quant_method={
                    "weight": "per-group",
                    "group_size": 128,
                    "activation": "per-tensor",
                },
                ignore_layers=["lm_head"],
            ),
        ),
    }


def default_int4_gptaq_config() -> dict:
    """
    Returns a default configuration dictionary for model compression.

    This configuration includes global settings and specific compression parameters.
    """
    return {
        "global_config": GlobalConfig(),
        "compress_config": CompressionConfig(
            name="PTQ",
            quantization=QuantizationConfig(
                name="int4_gptaq",
                bits=4,
                quant_method={"weight": "per-group", "group_size": 128},
                ignore_layers=["lm_head"],
            ),
        ),
    }
