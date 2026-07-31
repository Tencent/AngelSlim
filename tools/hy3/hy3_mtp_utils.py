#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.

"""Utilities shared by the HY3 calibration and NVFP4 merge pipelines.

HY3 checkpoints store MTP blocks as transformer layers appended after the
``num_hidden_layers`` main-model layers.  The helpers in this file deliberately
discover those layers from the checkpoint index instead of relying on a fixed
layer number such as 61 or 80.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable

import torch
from safetensors import safe_open

FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_LAYER_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.")
_MTP_BLOCK_RE = re.compile(r"\.mtp_block\.")

# Keep this aligned with tools/fp8_quant_with_vllm_activation.py.  These are
# GEMM weights for which vLLM's ModelOpt FP8 loader has a matching quantized
# Linear/FusedMoE implementation.
MTP_FP8_WEIGHT_SUFFIXES = (
    ".gate_and_up_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
    ".q_a_proj.weight",
    ".q_b_proj.weight",
    ".kv_a_proj_with_mqa.weight",
    ".kv_b_proj.weight",
    ".qkv_proj.weight",
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".eh_proj.weight",
    ".experts.gate_up_proj",
    ".experts.down_proj",
)


@dataclass(frozen=True)
class MtpLayout:
    """MTP layout discovered from an HF checkpoint."""

    base_num_hidden_layers: int
    layer_ids: tuple[int, ...]
    explicit_prefixes: tuple[str, ...]
    weight_keys: tuple[str, ...]

    @property
    def has_mtp(self) -> bool:
        return bool(self.layer_ids or self.explicit_prefixes)


def read_weight_map(model_path: str) -> Dict[str, str]:
    """Return the checkpoint weight map for indexed or single-file models."""

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid weight_map in {index_path}")
        return weight_map

    single_path = os.path.join(model_path, "model.safetensors")
    if os.path.isfile(single_path):
        with safe_open(single_path, framework="pt", device="cpu") as f:
            return {key: "model.safetensors" for key in f.keys()}

    raise FileNotFoundError(
        f"No model.safetensors.index.json or model.safetensors found under {model_path}"
    )


def detect_hy3_mtp(model_path: str) -> MtpLayout:
    """Detect appended HY3 MTP layers from config.json and checkpoint keys."""

    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "num_hidden_layers" not in config:
        raise ValueError(f"Missing num_hidden_layers in {config_path}")
    base_num_hidden_layers = int(config["num_hidden_layers"])

    weight_map = read_weight_map(model_path)
    mtp_layer_ids = set()
    explicit_prefixes = set()
    mtp_keys = []

    for key in weight_map:
        match = _LAYER_KEY_RE.match(key)
        if match and int(match.group(1)) >= base_num_hidden_layers:
            mtp_layer_ids.add(int(match.group(1)))
            mtp_keys.append(key)
        elif key.startswith("mtp."):
            explicit_prefixes.add("mtp")
            mtp_keys.append(key)

    return MtpLayout(
        base_num_hidden_layers=base_num_hidden_layers,
        layer_ids=tuple(sorted(mtp_layer_ids)),
        explicit_prefixes=tuple(sorted(explicit_prefixes)),
        weight_keys=tuple(sorted(mtp_keys)),
    )


def is_mtp_key(key: str, layout: MtpLayout) -> bool:
    """Return whether a checkpoint key belongs to a detected MTP block."""

    if key.startswith("mtp.") and layout.explicit_prefixes:
        return True
    match = _LAYER_KEY_RE.match(key)
    return bool(match and int(match.group(1)) in layout.layer_ids)


def normalize_mtp_stats(stats: Dict[str, dict]) -> Dict[str, dict]:
    """Map vLLM draft-module names to HF checkpoint names."""

    return {_MTP_BLOCK_RE.sub(".", key): value for key, value in stats.items()}


def load_mtp_stats(statistics_path: str) -> Dict[str, dict]:
    """Load and merge MTP activation and per-expert calibration statistics."""

    merged: Dict[str, dict] = {}
    for filename in ("mtp_activation_stats.json", "mtp_moe_expert_stats.json"):
        path = os.path.join(statistics_path, filename)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Top level of {path} must be a mapping")
        merged.update(normalize_mtp_stats(data))
    return merged


def is_mtp_fp8_weight(weight_name: str) -> bool:
    """Return whether a weight uses one of the supported FP8 GEMM layouts."""

    return any(weight_name.endswith(suffix) for suffix in MTP_FP8_WEIGHT_SUFFIXES)


def module_name_from_weight(weight_name: str) -> str:
    """Convert an HF weight/packed-weight key to its module prefix."""

    if weight_name.endswith(".weight"):
        return weight_name[: -len(".weight")]
    if weight_name.endswith(".experts.gate_up_proj"):
        return weight_name[: -len(".gate_up_proj")]
    if weight_name.endswith(".experts.down_proj"):
        return weight_name[: -len(".down_proj")]
    raise ValueError(f"Unsupported quantized weight name: {weight_name}")


def resolve_mtp_activation_key(weight_name: str) -> str:
    """Resolve the calibration-stat key used by one MTP weight."""

    module_name = module_name_from_weight(weight_name)

    if module_name.endswith((".q_proj", ".k_proj", ".v_proj")):
        return module_name.rsplit(".", 1)[0] + ".qkv_proj"

    if module_name.endswith((".gate_proj", ".up_proj")) and ".mlp." in module_name:
        return module_name.rsplit(".", 1)[0] + ".gate_up_proj"

    if module_name.endswith(".experts"):
        if weight_name.endswith(".experts.gate_up_proj"):
            return module_name + ".gate_up_proj"
        return module_name + ".down_proj"

    return module_name


def fp8_scale_from_stats(stats: dict) -> torch.Tensor:
    """Create a static per-tensor FP8 input scale from min/max statistics."""

    min_value = stats["min"]
    max_value = stats["max"]
    if isinstance(min_value, list) or isinstance(max_value, list):
        raise ValueError("Linear input statistics must be scalar, not per-head lists")
    absmax = max(abs(float(min_value)), abs(float(max_value)))
    return torch.tensor([max(absmax / FP8_MAX, 1e-12)], dtype=torch.float32)


def quantize_weight_per_tensor_fp8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one BF16/FP16 weight tensor to serialized ModelOpt FP8."""

    absmax = float(weight.float().abs().max().item())
    scale_value = max(absmax / FP8_MAX, 1e-12)
    scale = torch.tensor([scale_value], dtype=torch.float32)
    quantized = (weight.float() / scale_value).clamp(-FP8_MAX, FP8_MAX)
    return quantized.to(torch.float8_e4m3fn).contiguous(), scale


def layer_prefixes(layer_ids: Iterable[int]) -> tuple[str, ...]:
    """Return canonical ``model.layers.N`` prefixes."""

    return tuple(f"model.layers.{layer_id}" for layer_id in sorted(set(layer_ids)))
