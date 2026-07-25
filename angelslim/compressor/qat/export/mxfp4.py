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

"""Export FOCUS MXFP4 checkpoints to vLLM compressed-tensors format."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from huggingface_hub import save_torch_state_dict
from safetensors import safe_open

from ....utils import print_info
from ....utils.zero3_io import iter_safetensors_shards, resolve_safetensors_model_path
from ...quant.modules.mxfp4 import MXFP4_GROUP_SIZE, mxfp4_quantize_pack
from .common import copy_model_metadata, load_focus_checkpoint, weight_key_from_scale


def build_mxfp4_quantization_config(ignored_layers: list[str] | None = None) -> dict:
    """Build the vLLM compressed-tensors W4A4 MXFP4 checkpoint contract."""
    ignored_layers = list(ignored_layers or ["lm_head", "embed_tokens"])
    scheme = {
        "num_bits": 4,
        "group_size": MXFP4_GROUP_SIZE,
        "strategy": "group",
        "symmetric": True,
        "type": "float",
    }
    return {
        "quant_method": "compressed-tensors",
        "format": "mxfp4-pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {**scheme, "dynamic": False},
                "input_activations": {**scheme, "dynamic": True},
                "output_activations": None,
            }
        },
        "ignore": ignored_layers,
        "kv_cache_scheme": None,
    }


class FocusMXFP4SaveVllmHF:
    """Model save adapter used by ``save_via_model_save_func``."""

    def __init__(self, quant_model):
        self.quant_model = quant_model

    def save(self, save_path):
        ignored_layers = self.quant_model.skip_layer_names()
        quantization_config = build_mxfp4_quantization_config(ignored_layers)
        self.quant_model.get_model().config.update(
            {
                "quantization_config": quantization_config,
                "use_cache": True,
            }
        )
        print_info(
            "Save FOCUS MXFP4 compressed-tensors quantization_config: " f"{quantization_config}"
        )
        os.makedirs(save_path, exist_ok=True)
        self.quant_model.get_model().save_pretrained(save_path, max_shard_size="5GB")
        self.quant_model.tokenizer.save_pretrained(save_path)


def export_focus_mxfp4_checkpoint(
    checkpoint_path: str | os.PathLike,
    model_path: str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    ignored_layers: list[str] | None = None,
    max_shard_size: str = "5GB",
) -> dict:
    """Convert a baked FOCUS fake checkpoint using base weights + learned scales.

    Fake checkpoints contain baked fake-quantized weights. Real packing must
    instead use the frozen base weights together with the learned ``max_scale``
    tensors, otherwise applying ``max_scale`` a second time changes the trained
    quantization result.
    """
    resolved_model_path = Path(resolve_safetensors_model_path(model_path))
    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    focus_state = load_focus_checkpoint(checkpoint_path)
    scale_by_weight = {
        weight_key_from_scale(key): value
        for key, value in focus_state.items()
        if key.endswith(".weight_quantizer.max_scale")
    }
    subgroup_suffix = ".weight_quantizer.quant_max_scale"
    subgroup_by_weight = {
        key[: -len(subgroup_suffix)] + ".weight": value
        for key, value in focus_state.items()
        if key.endswith(subgroup_suffix)
    }
    if not scale_by_weight:
        raise RuntimeError("No FOCUS MXFP4 weight max_scale tensors found")

    exported_state: dict[str, torch.Tensor] = {}
    exported_layers: list[str] = []
    found_weight_keys: set[str] = set()

    for shard_path, keys in iter_safetensors_shards(str(resolved_model_path)):
        with safe_open(shard_path, framework="pt", device="cpu") as reader:
            for key in keys:
                tensor = reader.get_tensor(key)
                max_scale = scale_by_weight.get(key)
                if max_scale is None:
                    exported_state[key] = tensor
                    continue

                found_weight_keys.add(key)
                if tensor.ndim != 2:
                    raise ValueError(f"FOCUS MXFP4 export expects 2D Linear weight {key}")
                packed, encoded_scale = mxfp4_quantize_pack(
                    tensor,
                    max_scale,
                    quant_max_scale=subgroup_by_weight.get(key),
                )
                prefix = key[: -len(".weight")]
                exported_state[f"{prefix}.weight_packed"] = packed.cpu()
                exported_state[f"{prefix}.weight_scale"] = encoded_scale.cpu()
                exported_layers.append(prefix)

    missing_weights = sorted(set(scale_by_weight) - found_weight_keys)
    if missing_weights:
        raise KeyError(
            f"{len(missing_weights)} FOCUS scales have no matching base weight: "
            f"{missing_weights[:5]}"
        )

    quantization_config = build_mxfp4_quantization_config(ignored_layers)
    copy_model_metadata(resolved_model_path, output_path, quantization_config)
    save_torch_state_dict(
        state_dict=exported_state,
        save_directory=output_path,
        max_shard_size=max_shard_size,
        safe_serialization=True,
        force_contiguous=True,
    )

    summary = {
        "format": "mxfp4-pack-quantized",
        "source_checkpoint": str(Path(checkpoint_path).resolve()),
        "base_model": str(resolved_model_path.resolve()),
        "output_path": str(output_path.resolve()),
        "exported_layer_count": len(exported_layers),
        "subgroup_layer_count": len(subgroup_by_weight),
        "exported_layers": sorted(exported_layers),
    }
    with (output_path / "focus_mxfp4_export.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary
