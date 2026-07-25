#!/usr/bin/env python3
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

"""Validate a vLLM compressed-tensors FOCUS NVFP4 export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from angelslim.compressor.qat.export.common import load_focus_checkpoint  # noqa: E402
from angelslim.compressor.quant.modules.nvfp4 import (  # noqa: E402
    NVFP4_BLOCK_SIZE,
    nvfp4_quantize_pack,
)
from angelslim.utils.zero3_io import resolve_safetensors_model_path  # noqa: E402


def _weight_map(model_path: Path) -> dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        return json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    result = {}
    for shard_path in sorted(model_path.glob("*.safetensors")):
        with safe_open(shard_path, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                result[key] = shard_path.name
    if not result:
        raise FileNotFoundError(f"No safetensors under {model_path}")
    return result


def _load_tensor(model_path: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    shard_name = weight_map.get(key)
    if shard_name is None:
        raise KeyError(key)
    with safe_open(model_path / shard_name, framework="pt", device="cpu") as reader:
        return reader.get_tensor(key)


def _tensor_shape(model_path: Path, weight_map: dict[str, str], key: str) -> tuple[int, ...]:
    shard_name = weight_map.get(key)
    if shard_name is None:
        raise KeyError(key)
    with safe_open(model_path / shard_name, framework="pt", device="cpu") as reader:
        return tuple(reader.get_slice(key).get_shape())


def _sample_evenly(values: list[str], limit: int) -> list[str]:
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    return [values[round(index * (len(values) - 1) / (limit - 1))] for index in range(limit)]


def validate_export(
    export_path: str,
    checkpoint_path: str,
    model_path: str,
    max_weights: int = 3,
) -> dict:
    export_path = Path(export_path)
    base_path = Path(resolve_safetensors_model_path(model_path))
    export_map = _weight_map(export_path)
    base_map = _weight_map(base_path)
    focus_state = load_focus_checkpoint(checkpoint_path)

    config = json.loads((export_path / "config.json").read_text(encoding="utf-8"))
    quant_config = config.get("quantization_config", {})
    if quant_config.get("quant_method") != "compressed-tensors":
        raise ValueError("Export config is not compressed-tensors")
    if quant_config.get("format") != "nvfp4-pack-quantized":
        raise ValueError("Export config is not nvfp4-pack-quantized")
    group = quant_config.get("config_groups", {}).get("group_0", {})
    if group.get("weights", {}).get("strategy") != "tensor_group":
        raise ValueError("NVFP4 weights must use tensor_group strategy")
    if group.get("input_activations", {}).get("dynamic") != "local":
        raise ValueError("NVFP4 activation local scales must be dynamic")

    scale_suffix = ".weight_quantizer.max_scale"
    scale_keys = sorted(key for key in focus_state if key.endswith(scale_suffix))
    if not scale_keys:
        raise RuntimeError("No FOCUS max_scale tensors found")

    layer_results = []
    weight_keys = []
    for scale_key in scale_keys:
        prefix = scale_key[: -len(scale_suffix)]
        base_weight_key = f"{prefix}.weight"
        required = {
            "packed": f"{prefix}.weight_packed",
            "scale": f"{prefix}.weight_scale",
            "weight_global": f"{prefix}.weight_global_scale",
            "input_global": f"{prefix}.input_global_scale",
        }
        if base_weight_key not in base_map:
            raise KeyError(f"Missing base weight: {base_weight_key}")
        if base_weight_key in export_map:
            raise ValueError(f"Export retained unpacked quantized weight: {base_weight_key}")
        missing = [key for key in required.values() if key not in export_map]
        if missing:
            raise KeyError(f"Missing packed tensors for {prefix}: {missing}")

        base_shape = _tensor_shape(base_path, base_map, base_weight_key)
        packed = _load_tensor(export_path, export_map, required["packed"])
        local_scale = _load_tensor(export_path, export_map, required["scale"])
        weight_global = _load_tensor(export_path, export_map, required["weight_global"])
        input_global = _load_tensor(export_path, export_map, required["input_global"])
        expected_scale_shape = (
            *base_shape[:-1],
            base_shape[-1] // NVFP4_BLOCK_SIZE,
        )
        if packed.dtype != torch.uint8 or packed.shape[-1] * 2 != base_shape[-1]:
            raise ValueError(f"Invalid packed weight layout for {prefix}")
        if (
            local_scale.dtype != torch.float8_e4m3fn
            or tuple(local_scale.shape) != expected_scale_shape
        ):
            raise ValueError(f"Invalid FP8 local scale layout for {prefix}")
        for name, value in (
            ("weight_global_scale", weight_global),
            ("input_global_scale", input_global),
        ):
            if value.dtype != torch.float32 or value.numel() != 1:
                raise ValueError(f"Invalid {name} for {prefix}")

        weight_keys.append(base_weight_key)
        layer_results.append(
            {
                "layer": prefix,
                "weight_shape": list(base_shape),
                "packed_shape": list(packed.shape),
                "scale_shape": list(local_scale.shape),
            }
        )

    samples = []
    for weight_key in _sample_evenly(weight_keys, max_weights):
        prefix = weight_key[: -len(".weight")]
        expected = nvfp4_quantize_pack(
            _load_tensor(base_path, base_map, weight_key),
            focus_state[f"{prefix}{scale_suffix}"],
            focus_state[f"{prefix}.weight_quantizer.scale_2"],
            quant_max_scale=focus_state.get(f"{prefix}.weight_quantizer.quant_max_scale"),
        )
        actual = (
            _load_tensor(export_path, export_map, f"{prefix}.weight_packed"),
            _load_tensor(export_path, export_map, f"{prefix}.weight_scale"),
            _load_tensor(export_path, export_map, f"{prefix}.weight_global_scale"),
        )
        if any(not torch.equal(left, right) for left, right in zip(actual, expected)):
            raise ValueError(f"NVFP4 packed tensors mismatch for {prefix}")
        expected_input_global = (
            1.0 / focus_state[f"{prefix}.act_quantizer.scale_2"].float()
        ).reshape(1)
        actual_input_global = _load_tensor(export_path, export_map, f"{prefix}.input_global_scale")
        if not torch.equal(actual_input_global, expected_input_global):
            raise ValueError(f"NVFP4 input global scale mismatch for {prefix}")
        samples.append({"layer": prefix, "bit_exact_match": True})

    return {
        "export_path": str(export_path.resolve()),
        "format": "nvfp4-pack-quantized",
        "validated_layer_count": len(layer_results),
        "samples": samples,
        "status": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-weights", type=int, default=3)
    args = parser.parse_args()
    result = validate_export(
        args.export_path,
        args.checkpoint,
        args.model_path,
        max_weights=args.max_weights,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
