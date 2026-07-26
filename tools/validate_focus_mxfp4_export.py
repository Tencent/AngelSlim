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

"""Validate a vLLM compressed-tensors FOCUS MXFP4 export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from angelslim.compressor.qat.export.common import load_focus_checkpoint  # noqa: E402
from angelslim.compressor.quant.modules.mxfp4 import (  # noqa: E402
    MXFP4_GROUP_SIZE,
    mxfp4_quantize_pack,
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
    checkpoint_path: str | None = None,
    model_path: str | None = None,
    max_weights: int = 3,
) -> dict:
    if model_path is None:
        raise ValueError("model_path is required")
    export_path = Path(export_path)
    base_path = Path(resolve_safetensors_model_path(model_path))
    export_map = _weight_map(export_path)
    base_map = _weight_map(base_path)
    focus_state = load_focus_checkpoint(checkpoint_path) if checkpoint_path else None

    config = json.loads((export_path / "config.json").read_text(encoding="utf-8"))
    quant_config = config.get("quantization_config", {})
    if quant_config.get("quant_method") != "compressed-tensors":
        raise ValueError("Export config is not compressed-tensors")
    if quant_config.get("format") != "mxfp4-pack-quantized":
        raise ValueError("Export config is not mxfp4-pack-quantized")

    scale_suffix = ".weight_quantizer.max_scale"
    if focus_state is not None:
        scale_keys = sorted(key for key in focus_state if key.endswith(scale_suffix))
        if not scale_keys:
            raise RuntimeError("No FOCUS max_scale tensors found")
        prefixes = [key[: -len(scale_suffix)] for key in scale_keys]
        validation_mode = "bit_exact"
    else:
        packed_suffix = ".weight_packed"
        packed_keys = sorted(key for key in export_map if key.endswith(packed_suffix))
        if not packed_keys:
            raise RuntimeError("No MXFP4 packed weight tensors found")
        prefixes = [key[: -len(packed_suffix)] for key in packed_keys]
        validation_mode = "schema_only"

    layer_results = []
    weight_keys = []
    for prefix in prefixes:
        base_weight_key = f"{prefix}.weight"
        packed_key = f"{prefix}.weight_packed"
        export_scale_key = f"{prefix}.weight_scale"
        if base_weight_key not in base_map:
            raise KeyError(f"Missing base weight: {base_weight_key}")
        if base_weight_key in export_map:
            raise ValueError(f"Export retained unpacked quantized weight: {base_weight_key}")
        if packed_key not in export_map or export_scale_key not in export_map:
            raise KeyError(f"Missing packed tensors for {prefix}")

        base_shape = _tensor_shape(base_path, base_map, base_weight_key)
        if base_shape[-1] % MXFP4_GROUP_SIZE:
            raise ValueError(
                f"Base weight width must be divisible by {MXFP4_GROUP_SIZE}: " f"{base_weight_key}"
            )
        packed = _load_tensor(export_path, export_map, packed_key)
        encoded_scale = _load_tensor(export_path, export_map, export_scale_key)
        expected_packed_shape = (*base_shape[:-1], base_shape[-1] // 2)
        expected_scale_shape = (
            *base_shape[:-1],
            base_shape[-1] // MXFP4_GROUP_SIZE,
        )
        if packed.dtype != torch.uint8 or tuple(packed.shape) != expected_packed_shape:
            raise ValueError(f"Invalid packed weight layout for {prefix}")
        if (
            encoded_scale.dtype != torch.uint8
            or tuple(encoded_scale.shape) != expected_scale_shape
        ):
            raise ValueError(f"Invalid E8M0 scale layout for {prefix}")

        weight_keys.append(base_weight_key)
        layer_results.append(
            {
                "layer": prefix,
                "weight_shape": list(base_shape),
                "packed_shape": list(packed.shape),
                "scale_shape": list(encoded_scale.shape),
            }
        )

    samples = []
    if focus_state is not None:
        for weight_key in _sample_evenly(weight_keys, max_weights):
            prefix = weight_key[: -len(".weight")]
            base_weight = _load_tensor(base_path, base_map, weight_key)
            max_scale = focus_state[f"{prefix}{scale_suffix}"]
            quant_max_scale = focus_state.get(f"{prefix}.weight_quantizer.quant_max_scale")
            expected_packed, expected_scale = mxfp4_quantize_pack(
                base_weight,
                max_scale,
                quant_max_scale=quant_max_scale,
            )
            actual_packed = _load_tensor(export_path, export_map, f"{prefix}.weight_packed")
            actual_scale = _load_tensor(export_path, export_map, f"{prefix}.weight_scale")
            if not torch.equal(actual_packed, expected_packed):
                raise ValueError(f"Packed weight mismatch for {prefix}")
            if not torch.equal(actual_scale, expected_scale):
                raise ValueError(f"E8M0 scale mismatch for {prefix}")
            samples.append(
                {
                    "layer": prefix,
                    "packed_exact_match": True,
                    "scale_exact_match": True,
                }
            )

    return {
        "export_path": str(export_path.resolve()),
        "format": "mxfp4-pack-quantized",
        "validation_mode": validation_mode,
        "validated_layer_count": len(layer_results),
        "samples": samples,
        "status": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-path", required=True)
    parser.add_argument(
        "--checkpoint",
        help="FOCUS fake checkpoint; omit for schema-only direct-real validation",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-weights", type=int, default=3)
    args = parser.parse_args()
    result = validate_export(
        export_path=args.export_path,
        checkpoint_path=args.checkpoint,
        model_path=args.model_path,
        max_weights=args.max_weights,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
