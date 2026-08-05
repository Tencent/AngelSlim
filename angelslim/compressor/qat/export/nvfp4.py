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

"""Export FOCUS NVFP4 checkpoints to vLLM compressed-tensors format."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from huggingface_hub import save_torch_state_dict
from safetensors import safe_open

from ....utils import print_info
from ....utils.zero3_io import iter_safetensors_shards, resolve_safetensors_model_path
from ...quant.modules.nvfp4 import (
    NVFP4_BLOCK_SIZE,
    harmonize_nvfp4_fused_scales,
    nvfp4_quantize_pack,
)
from .common import copy_model_metadata, load_focus_checkpoint, weight_key_from_scale


def build_nvfp4_quantization_config(ignored_layers: list[str] | None = None) -> dict:
    """Build the vLLM compressed-tensors W4A4 NVFP4 checkpoint contract."""
    ignored_layers = list(ignored_layers or ["lm_head", "embed_tokens"])
    scheme = {
        "num_bits": 4,
        "group_size": NVFP4_BLOCK_SIZE,
        "strategy": "tensor_group",
        "symmetric": True,
        "type": "float",
    }
    return {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {**scheme, "dynamic": False},
                # The calibrated global scale is static; per-token block scales
                # are generated dynamically by the NVFP4 inference kernel.
                "input_activations": {**scheme, "dynamic": "local"},
                "output_activations": None,
            }
        },
        "ignore": ignored_layers,
        "kv_cache_scheme": None,
    }


class FocusNVFP4SaveVllmHF:
    """Model save adapter used by ``save_via_model_save_func``."""

    def __init__(self, quant_model):
        self.quant_model = quant_model

    def save(self, save_path):
        ignored_layers = self.quant_model.skip_layer_names()
        quantization_config = build_nvfp4_quantization_config(ignored_layers)
        self.quant_model.get_model().config.update(
            {
                "quantization_config": quantization_config,
                "use_cache": True,
            }
        )
        print_info(
            "Save FOCUS NVFP4 compressed-tensors quantization_config: " f"{quantization_config}"
        )
        os.makedirs(save_path, exist_ok=True)
        self.quant_model.get_model().save_pretrained(save_path, max_shard_size="5GB")
        self.quant_model.tokenizer.save_pretrained(save_path)


def _scale2_key_from_weight(weight_key: str, quantizer_name: str) -> str:
    return weight_key[: -len(".weight")] + f".{quantizer_name}.scale_2"


def export_focus_nvfp4_checkpoint(
    checkpoint_path: str | os.PathLike,
    model_path: str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    ignored_layers: list[str] | None = None,
    max_shard_size: str = "5GB",
) -> dict:
    """Convert a baked FOCUS fake checkpoint to packed NVFP4."""
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
        raise RuntimeError("No FOCUS NVFP4 weight max_scale tensors found")

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
                    raise ValueError(f"FOCUS NVFP4 export expects 2D Linear weight {key}")
                weight_scale_2_key = _scale2_key_from_weight(key, "weight_quantizer")
                input_scale_2_key = _scale2_key_from_weight(key, "act_quantizer")
                if weight_scale_2_key not in focus_state:
                    raise KeyError(f"Missing NVFP4 weight scale_2: {weight_scale_2_key}")
                if input_scale_2_key not in focus_state:
                    raise KeyError(f"Missing NVFP4 activation scale_2: {input_scale_2_key}")

                packed, local_scale, weight_global_scale = nvfp4_quantize_pack(
                    tensor,
                    max_scale,
                    focus_state[weight_scale_2_key],
                    quant_max_scale=subgroup_by_weight.get(key),
                )
                input_scale_2 = focus_state[input_scale_2_key].float().reshape(-1)
                if (
                    input_scale_2.numel() != 1
                    or not torch.isfinite(input_scale_2).all()
                    or not (input_scale_2 > 0).all()
                ):
                    raise ValueError(
                        f"Invalid NVFP4 activation scale_2 for {key}: {input_scale_2}"
                    )

                prefix = key[: -len(".weight")]
                exported_state[f"{prefix}.weight_packed"] = packed.cpu()
                exported_state[f"{prefix}.weight_scale"] = local_scale.cpu()
                exported_state[f"{prefix}.weight_global_scale"] = (
                    weight_global_scale.float().reshape(1)
                )
                exported_state[f"{prefix}.input_global_scale"] = (1.0 / input_scale_2).reshape(1)
                exported_layers.append(prefix)

    missing_weights = sorted(set(scale_by_weight) - found_weight_keys)
    if missing_weights:
        raise KeyError(
            f"{len(missing_weights)} FOCUS scales have no matching base weight: "
            f"{missing_weights[:5]}"
        )

    fusion_summary = harmonize_nvfp4_fused_scales(exported_state)
    if fusion_summary["fused_group_count"]:
        print_info(f"Harmonized FOCUS NVFP4 fused scales: {fusion_summary}")

    quantization_config = build_nvfp4_quantization_config(ignored_layers)
    copy_model_metadata(resolved_model_path, output_path, quantization_config)
    save_torch_state_dict(
        state_dict=exported_state,
        save_directory=output_path,
        max_shard_size=max_shard_size,
        safe_serialization=True,
        force_contiguous=True,
    )

    summary = {
        "format": "nvfp4-pack-quantized",
        "source_checkpoint": str(Path(checkpoint_path).resolve()),
        "base_model": str(resolved_model_path.resolve()),
        "output_path": str(output_path.resolve()),
        "exported_layer_count": len(exported_layers),
        "subgroup_layer_count": len(subgroup_by_weight),
        "fusion_scale_harmonization": fusion_summary,
        "exported_layers": sorted(exported_layers),
    }
    with (output_path / "focus_nvfp4_export.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary
