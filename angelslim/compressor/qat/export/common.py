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

"""Shared helpers for FOCUS FP4 checkpoint export."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch


def load_focus_checkpoint(
    checkpoint_path: str | os.PathLike,
) -> dict[str, torch.Tensor]:
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        state_dict = torch.load(checkpoint_path, mmap=True, **kwargs)
    except TypeError:
        state_dict = torch.load(checkpoint_path, **kwargs)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict mapping, got {type(state_dict).__name__}")
    return state_dict


def weight_key_from_scale(scale_key: str) -> str:
    suffix = ".weight_quantizer.max_scale"
    if not scale_key.endswith(suffix):
        raise ValueError(f"Not a FOCUS weight scale key: {scale_key}")
    return scale_key[: -len(suffix)] + ".weight"


def copy_model_metadata(
    model_path: Path,
    output_path: Path,
    quantization_config: dict,
) -> None:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing base model config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["quantization_config"] = quantization_config
    config["torch_dtype"] = config.get("torch_dtype", "bfloat16")
    with (output_path / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    metadata_names = {
        "added_tokens.json",
        "chat_template.json",
        "chat_template.jinja",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
    for name in metadata_names:
        source = model_path / name
        if source.is_file():
            shutil.copy2(source, output_path / name)
