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

"""Validate a FOCUS fake-quant checkpoint against its frozen base model."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open

_QWEN_DENSE_WEIGHT_SUFFIXES = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
)


def _load_checkpoint(checkpoint_path: str):
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        state_dict = torch.load(checkpoint_path, mmap=True, **kwargs)
    except TypeError:
        state_dict = torch.load(checkpoint_path, **kwargs)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict mapping, got {type(state_dict).__name__}")
    return state_dict


def _resolve_model_path(model_path: str) -> Path:
    local_path = Path(os.path.expanduser(model_path))
    if local_path.is_dir():
        return local_path
    if local_path.exists() or local_path.is_absolute() or model_path.startswith("."):
        raise FileNotFoundError(f"Base model path is not a directory: {local_path}")

    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=model_path,
        allow_patterns=["*.safetensors", "*.safetensors.index.json"],
    )
    return Path(snapshot_path)


def _build_weight_map(model_path: Path):
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)["weight_map"]

    weight_map = {}
    for shard_path in sorted(model_path.glob("*.safetensors")):
        with safe_open(shard_path, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                weight_map[key] = shard_path.name
    if not weight_map:
        raise FileNotFoundError(f"No safetensors found under {model_path}")
    return weight_map


def _load_base_tensor(model_path: Path, weight_map, key: str):
    shard_name = weight_map.get(key)
    if shard_name is None:
        return None
    shard_path = model_path / shard_name
    with safe_open(shard_path, framework="pt", device="cpu") as reader:
        return reader.get_tensor(key)


def _load_base_tensor_shape(model_path: Path, weight_map, key: str):
    shard_name = weight_map.get(key)
    if shard_name is None:
        return None
    shard_path = model_path / shard_name
    with safe_open(shard_path, framework="pt", device="cpu") as reader:
        return tuple(reader.get_slice(key).get_shape())


def _sample_evenly(values, limit: int):
    if limit >= len(values):
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    indices = sorted({round(i * (len(values) - 1) / (limit - 1)) for i in range(limit)})
    return [values[index] for index in indices]


def validate_checkpoint(
    checkpoint_path: str,
    model_path: str,
    qtype: str,
    group_size: int,
    max_weights: int = 3,
    num_sub: int | None = None,
):
    if qtype not in ("mxfp4", "nvfp4"):
        raise ValueError(f"Unsupported qtype: {qtype}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if max_weights <= 0:
        raise ValueError(f"max_weights must be positive, got {max_weights}")
    if num_sub is not None:
        if num_sub <= 0:
            raise ValueError(f"num_sub must be positive, got {num_sub}")
        if group_size % num_sub:
            raise ValueError(f"group_size ({group_size}) must be divisible by num_sub ({num_sub})")

    state_dict = _load_checkpoint(checkpoint_path)
    resolved_model_path = _resolve_model_path(model_path)
    weight_map = _build_weight_map(resolved_model_path)

    expected_weight_keys = sorted(
        key
        for key in weight_map
        if any(key.endswith(suffix) for suffix in _QWEN_DENSE_WEIGHT_SUFFIXES)
    )
    if not expected_weight_keys:
        raise RuntimeError("No Qwen dense projection weights found in the base model")

    expected_scale_keys = []
    quant_max_scale_keys = []
    inferred_num_subs = set()
    for weight_key in expected_weight_keys:
        fake_weight = state_dict.get(weight_key)
        if not isinstance(fake_weight, torch.Tensor):
            raise RuntimeError(f"Missing fake weight tensor: {weight_key}")
        if fake_weight.ndim != 2 or not fake_weight.is_floating_point():
            raise RuntimeError(f"Expected a 2D dense weight: {weight_key}")
        base_shape = _load_base_tensor_shape(resolved_model_path, weight_map, weight_key)
        if tuple(fake_weight.shape) != base_shape:
            raise RuntimeError(
                f"Shape mismatch for {weight_key}: fake={tuple(fake_weight.shape)} "
                f"base={base_shape}"
            )
        if not torch.isfinite(fake_weight).all():
            raise RuntimeError(f"Non-finite fake weight tensor: {weight_key}")

        prefix = weight_key[: -len(".weight")]
        scale_key = f"{prefix}.weight_quantizer.max_scale"
        expected_scale_keys.append(scale_key)
        max_scale = state_dict.get(scale_key)
        if not isinstance(max_scale, torch.Tensor) or not max_scale.is_floating_point():
            raise RuntimeError(f"Missing weight scale tensor: {scale_key}")
        if not torch.isfinite(max_scale).all():
            raise RuntimeError(f"Non-finite weight scale tensor: {scale_key}")

        expected_scale_shape = (
            fake_weight.shape[0],
            math.ceil(fake_weight.shape[1] / group_size),
        )
        if tuple(max_scale.shape) != expected_scale_shape:
            raise RuntimeError(
                f"Scale shape mismatch for {scale_key}: actual={tuple(max_scale.shape)} "
                f"expected={expected_scale_shape}"
            )

        quant_scale_key = f"{prefix}.weight_quantizer.quant_max_scale"
        quant_max_scale = state_dict.get(quant_scale_key)
        if num_sub is not None and not isinstance(quant_max_scale, torch.Tensor):
            raise RuntimeError(f"Missing subgroup scale tensor: {quant_scale_key}")
        if isinstance(quant_max_scale, torch.Tensor):
            if not quant_max_scale.is_floating_point():
                raise RuntimeError(f"Invalid subgroup scale tensor: {quant_scale_key}")
            if not torch.isfinite(quant_max_scale).all():
                raise RuntimeError(f"Non-finite subgroup scale tensor: {quant_scale_key}")

            layer_num_sub = num_sub
            if layer_num_sub is None:
                parent_group_count = max_scale.shape[1]
                if (
                    quant_max_scale.ndim != 2
                    or quant_max_scale.shape[0] != max_scale.shape[0]
                    or quant_max_scale.shape[1] % parent_group_count
                ):
                    raise RuntimeError(
                        f"Cannot infer num_sub from {quant_scale_key}: "
                        f"actual={tuple(quant_max_scale.shape)} "
                        f"parent_scale={tuple(max_scale.shape)}"
                    )
                layer_num_sub = quant_max_scale.shape[1] // parent_group_count
                if layer_num_sub <= 0 or group_size % layer_num_sub:
                    raise RuntimeError(
                        f"Invalid inferred num_sub ({layer_num_sub}) for {quant_scale_key}"
                    )

            expected_quant_scale_shape = (
                max_scale.shape[0],
                max_scale.shape[1] * layer_num_sub,
            )
            if tuple(quant_max_scale.shape) != expected_quant_scale_shape:
                raise RuntimeError(
                    f"Subgroup scale shape mismatch for {quant_scale_key}: "
                    f"actual={tuple(quant_max_scale.shape)} "
                    f"expected={expected_quant_scale_shape}"
                )
            quant_max_scale_keys.append(quant_scale_key)
            inferred_num_subs.add(layer_num_sub)

        if qtype == "mxfp4":
            act_scale_key = f"{prefix}.act_quantizer.max_scale"
        else:
            scale_2_key = f"{prefix}.weight_quantizer.scale_2"
            scale_2 = state_dict.get(scale_2_key)
            if not isinstance(scale_2, torch.Tensor) or not scale_2.is_floating_point():
                raise RuntimeError(f"Missing NVFP4 tensor scale: {scale_2_key}")
            if tuple(scale_2.shape) != (1,) or not torch.isfinite(scale_2).all():
                raise RuntimeError(f"Invalid NVFP4 tensor scale: {scale_2_key}")
            act_scale_key = f"{prefix}.act_quantizer.scale_2"

        act_scale = state_dict.get(act_scale_key)
        if not isinstance(act_scale, torch.Tensor) or not act_scale.is_floating_point():
            raise RuntimeError(f"Missing activation scale tensor: {act_scale_key}")
        if tuple(act_scale.shape) != (1,) or not torch.isfinite(act_scale).all():
            raise RuntimeError(f"Invalid activation scale tensor: {act_scale_key}")

    samples = []
    for weight_key in _sample_evenly(expected_weight_keys, max_weights):
        fake_weight = state_dict[weight_key].detach().cpu()
        base_weight = _load_base_tensor(resolved_model_path, weight_map, weight_key)
        if base_weight is None:
            raise RuntimeError(f"Base weight disappeared from weight map: {weight_key}")
        if tuple(fake_weight.shape) != tuple(base_weight.shape):
            raise RuntimeError(
                f"Shape mismatch for {weight_key}: fake={tuple(fake_weight.shape)} "
                f"base={tuple(base_weight.shape)}"
            )
        if not torch.isfinite(base_weight).all():
            raise RuntimeError(f"Non-finite base weight tensor: {weight_key}")

        delta = (fake_weight.float() - base_weight.float()).abs()
        if not torch.isfinite(delta).all():
            raise RuntimeError(f"Non-finite weight delta: {weight_key}")
        changed_fraction = float((delta != 0).float().mean())
        if changed_fraction == 0:
            raise RuntimeError(
                f"Fake weight is identical to the frozen base model: {weight_key}; "
                "quant_inplace may not have persisted"
            )
        samples.append(
            {
                "weight": weight_key,
                "shape": list(fake_weight.shape),
                "changed_fraction": changed_fraction,
                "mean_abs_delta": float(delta.mean()),
                "max_abs_delta": float(delta.max()),
                "fake_abs_mean": float(fake_weight.float().abs().mean()),
            }
        )

    if not samples:
        raise RuntimeError("No comparable dense weights were loaded")
    if len(inferred_num_subs) > 1:
        raise RuntimeError(
            f"Inconsistent num_sub values across checkpoint: {sorted(inferred_num_subs)}"
        )

    validated_num_sub = num_sub
    if validated_num_sub is None and inferred_num_subs:
        validated_num_sub = next(iter(inferred_num_subs))

    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "base_model": str(resolved_model_path.resolve()),
        "qtype": qtype,
        "group_size": group_size,
        "num_sub": validated_num_sub,
        "tensor_count": len(state_dict),
        "validated_dense_layer_count": len(expected_weight_keys),
        "max_scale_count": len(expected_scale_keys),
        "quant_max_scale_count": len(quant_max_scale_keys),
        "samples": samples,
        "status": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate FOCUS scale tensors and prove that saved fake weights "
            "differ from the frozen base model."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="FOCUS *_fake_quant_model.pt")
    parser.add_argument("--model-path", required=True, help="Local base model path or Hub ID")
    parser.add_argument("--qtype", required=True, choices=("mxfp4", "nvfp4"))
    parser.add_argument("--group-size", required=True, type=int)
    parser.add_argument(
        "--num-sub",
        type=int,
        help="Require and validate subgroup quant_max_scale tensors with this num_sub",
    )
    parser.add_argument("--max-weights", type=int, default=3)
    args = parser.parse_args()

    summary = validate_checkpoint(
        checkpoint_path=args.checkpoint,
        model_path=args.model_path,
        qtype=args.qtype,
        group_size=args.group_size,
        max_weights=args.max_weights,
        num_sub=args.num_sub,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
