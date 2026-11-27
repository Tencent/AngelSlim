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

"""Smooth statistics I/O and per-layer alpha grid search.

Combines:

* The smooth-stat data structure & JSON I/O — the single bridge between
  Phase 1 (vLLM online stat collection) and Phase 2 (offline weight
  conversion).  The JSON file written by ``get_smooth_stats`` is consumed
  by ``apply_qk_smooth`` / ``apply_vo_smooth`` / ``apply_down_proj_smooth``.
* The *pure-tensor* core of :class:`SmoothAlphaSearcher`: it takes
  already-collected ``x``, ``w``, ``act_absmax`` tensors (typically on a
  single TP shard) and returns the best alpha + smooth_weight by
  minimising fake-quant MSE.  No distributed primitives, no module
  discovery, no FusedMoE awareness — those concerns live in
  ``smooth/vllm/searcher_dist.py``.
"""

import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

from .tensor_math import (
    bf16_equivalence_error,
    consumer_error,
    norm_scale,
    power_of_two_smooth_scale,
    producer_equivalence_error,
    smooth_default,
    smooth_per_tensor_act_first,
    smooth_qdq_act,
    smooth_qdq_weight,
)

__all__ = [
    "TensorReader",
    "offline_group_specs",
    "build_plan",
    "load_smooth_plan",
    "search_offline_scales",
    # smooth_stats
    "SmoothStats",
    "load_smooth_stats",
    "save_smooth_stats",
    "load_alpha_search_results",
    "save_alpha_search_results",
    # searcher
    "smooth_alpha_search_layer",
]


# ===========================================================================
# Smooth-stat data structure & JSON I/O
# ===========================================================================


@dataclass
class SmoothStats:
    """A single layer's smooth statistics."""

    absmax: Optional[torch.Tensor]  # [C] per-channel absmax
    ema: Optional[torch.Tensor]  # [C] per-channel EMA
    call_count: int = 0


def load_smooth_stats(json_path: str, use_ema: bool = False) -> dict:
    """Load smooth stats from JSON file and convert lists back to tensors.

    The original ``smooth_stats.json`` schema is::

        {
          "<layer_key>": {
            "absmax": [float, ...],
            "ema":    [float, ...],
            "call_count": int
          },
          ...
        }

    Args:
        json_path: path to smooth_stats.json.
        use_ema: if True, use the EMA field as the per-channel scale;
            otherwise use absmax.

    Returns:
        ``dict[layer_key, {"scale": Tensor[C], "call_count": int}]``.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    field = "ema" if use_ema else "absmax"
    out = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        vals = entry.get(field)
        if vals is None:
            continue
        out[key] = {
            "scale": torch.tensor(vals, dtype=torch.float32),
            "call_count": int(entry.get("call_count", 0)),
        }
    return out


def save_smooth_stats(stats: dict, json_path: str) -> None:
    """Serialise ``model._smooth_stats``-shaped data to JSON.

    Args:
        stats: the dict returned by ``get_smooth_stats(model)`` — already
            JSON-serialisable (lists, ints).
        json_path: target file path.
    """
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)


def load_alpha_search_results(json_path: str) -> dict:
    """Load smooth_alpha_search.json.

    Returns the inner ``"results"`` dict if present, otherwise the whole
    payload (for forwards compat with older / flatter formats)::

        {
          "<layer_key>": {
            "alpha": [float, ...],     # per-rank list under TP
            "smooth_weight": [float],  # full [intermediate_size]
            "loss": float
          },
          ...
        }
    """
    with open(json_path, "r") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "results" in raw:
        return raw["results"]
    return raw


def save_alpha_search_results(payload: dict, json_path: str) -> None:
    """Save alpha search payload (config + results) to JSON."""
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)


# ===========================================================================
# Per-layer alpha grid search
# ===========================================================================


def smooth_alpha_search_layer(
    x: torch.Tensor,
    w: torch.Tensor,
    act_absmax: torch.Tensor,
    config,
) -> Tuple[float, torch.Tensor, float]:
    """Run the per-layer grid search.

    Args:
        x: ``[N, in_features]`` raw activation (float, on the search device).
        w: ``[out_features, in_features]`` weight tensor (float, on device).
        act_absmax: ``[in_features]`` per-channel activation absmax/EMA.
        config: ``SmoothAlphaSearchConfig`` instance.

    Returns:
        Tuple of ``(best_alpha_or_mul, best_smooth_weight, best_loss)``.
        For ``smooth_search_mode="per-tensor-act-first"`` the first
        element is the optimal multiplier; for ``"default"`` it is the
        optimal ``alpha``.
    """
    cfg = config
    device = x.device
    act_absmax = act_absmax.float().to(device).clamp(min=1e-8)
    weight_absmax = w.abs().max(dim=0).values.clamp(min=1e-8)

    # Reference output (computed once)
    y_ref = x @ w.T  # [N, out_feat]

    if cfg.smooth_search_mode == "default":
        alphas = torch.linspace(cfg.alpha_min, cfg.alpha_max, cfg.alpha_steps).tolist()
        losses = torch.empty(len(alphas), device=device)
        for i, alpha in enumerate(alphas):
            smooth = smooth_default(
                act_absmax,
                weight_absmax,
                alpha=alpha,
                smooth_min=cfg.smooth_min,
                smooth_max=cfg.smooth_max,
            )
            x_smoothed = x / smooth.unsqueeze(0)
            w_smoothed = w * smooth.unsqueeze(0)
            x_q = smooth_qdq_act(
                x_smoothed,
                method=cfg.act_quant_method,
                quant_type=cfg.act_quant_type,
                bits=cfg.weight_quant_bits,
            )
            w_q = smooth_qdq_weight(
                w_smoothed,
                method=cfg.weight_quant_method,
                quant_type=cfg.weight_quant_type,
                bits=cfg.weight_quant_bits,
                group_size=cfg.weight_group_size,
                block_size=cfg.block_size,
            )
            y_q = x_q @ w_q.T
            losses[i] = ((y_ref - y_q) ** 2).mean()

        best_idx = int(losses.argmin().item())
        best_alpha = alphas[best_idx]
        best_smooth = smooth_default(
            act_absmax,
            weight_absmax,
            alpha=best_alpha,
            smooth_min=cfg.smooth_min,
            smooth_max=cfg.smooth_max,
        )
        return best_alpha, best_smooth, losses[best_idx].item()

    elif cfg.smooth_search_mode == "per-tensor-act-first":
        muls = torch.linspace(cfg.act_mul_min, cfg.act_mul_max, cfg.alpha_steps).tolist()
        losses = torch.empty(len(muls), device=device)
        for i, mul in enumerate(muls):
            smooth = smooth_per_tensor_act_first(
                act_absmax,
                mul=mul,
                smooth_min=cfg.smooth_min,
                smooth_max=cfg.smooth_max,
            )
            x_smoothed = x / smooth.unsqueeze(0)
            w_smoothed = w * smooth.unsqueeze(0)
            x_q = smooth_qdq_act(
                x_smoothed,
                method=cfg.act_quant_method,
                quant_type=cfg.act_quant_type,
                bits=cfg.weight_quant_bits,
            )
            w_q = smooth_qdq_weight(
                w_smoothed,
                method=cfg.weight_quant_method,
                quant_type=cfg.weight_quant_type,
                bits=cfg.weight_quant_bits,
                group_size=cfg.weight_group_size,
                block_size=cfg.block_size,
            )
            y_q = x_q @ w_q.T
            losses[i] = ((y_ref - y_q) ** 2).mean()

        best_idx = int(losses.argmin().item())
        best_mul = muls[best_idx]
        best_smooth = smooth_per_tensor_act_first(
            act_absmax,
            mul=best_mul,
            smooth_min=cfg.smooth_min,
            smooth_max=cfg.smooth_max,
        )
        return best_mul, best_smooth, losses[best_idx].item()

    else:
        raise ValueError(f"Unknown smooth_search_mode: {cfg.smooth_search_mode}")


# Offline Smooth.
class TensorReader:
    """Read BF16 checkpoint tensors on demand."""

    def __init__(self, root, model_name=""):
        from safetensors import safe_open

        from ..struct import LAYER_PREFIXES, resolve_model_name

        self.root = Path(root).resolve()
        config = json.loads((self.root / "config.json").read_text())
        self.config = config.get("text_config") or config
        self.model_name = resolve_model_name(model_name)
        if any(c.get("quantization_config") is not None for c in (config, self.config)):
            raise ValueError("Offline Smooth requires original BF16 input")
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(self.root / "model.safetensors", framework="pt") as handle:
                self.weight_map = {key: "model.safetensors" for key in handle.keys()}
        # Exclude ViT/DFly norms; accept the VLM language_model prefix.
        prefixes = [
            base
            for base in LAYER_PREFIXES
            if f"{base}.0.input_layernorm.weight" in self.weight_map
        ]
        if len(prefixes) != 1:
            raise ValueError("expected one main input_layernorm")
        self.layer_prefix = prefixes[0]

    def has(self, key):
        return key in self.weight_map

    def shape(self, key):
        from safetensors import safe_open

        with safe_open(self.root / self.weight_map[key], framework="pt") as handle:
            return tuple(handle.get_slice(key).get_shape())

    def get(self, key):
        from safetensors import safe_open

        with safe_open(self.root / self.weight_map[key], framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"Offline Smooth requires original BF16 tensor: {key}")
        return tensor


def offline_group_specs(
    reader, modules=None, config_groups=None, algorithm="rtn", gptq=None, ignore=()
):
    """Select supported targets; all consumers of their input must be compensated."""
    from ....quant.ckpt_convert.convert import Router, build_groups, compile_pattern
    from ..struct import DEFAULT_MODULES, get_smooth_pairs

    if modules is None and config_groups is None:
        modules = DEFAULT_MODULES
    patterns = [modules] if isinstance(modules, str) else modules
    if patterns is not None and (
        not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns)
    ):
        raise ValueError("smooth.modules must be a regex or a list of regexes")
    matchers = [re.compile(p.removeprefix("re:")) for p in patterns or []]
    router = Router(build_groups(config_groups, algorithm, gptq)) if config_groups else None
    ignored = [compile_pattern(p) for p in ignore]
    requested = set()
    for key in reader.weight_map:
        name = input_name(key)
        if any(m(name) for m in ignored):
            continue
        group = router.route(name) if router else None
        if router and (group is None or not group.smooth):
            continue
        if patterns is not None and not any(m.search(name) for m in matchers):
            continue
        requested.add(key)
    supported, specs = set(), []
    for spec in get_smooth_pairs(reader.model_name, reader):
        targets = [key for key in spec["consumers"] if key in requested]
        if targets:
            spec["targets"] = targets
            supported.update(targets)
            specs.append(spec)
    unsupported = requested - supported
    if unsupported:
        warnings.warn(
            f"Offline Smooth skips {len(unsupported)} unsupported targets: "
            f"{sorted(unsupported)[:5]}",
            stacklevel=2,
        )
    return specs


def smooth_selection(options):
    keys = ("modules", "config_groups", "algorithm", "gptq", "ignore")
    return {key: options[key] for key in keys if key in options}


def input_name(weight_name):
    return weight_name.removesuffix(".weight").removeprefix("language_model.")


def target_absmax(spec, stats, device="cpu"):
    return torch.stack(
        [
            torch.as_tensor(stats[input_name(key)]["absmax"], dtype=torch.float32, device=device)
            for key in spec["targets"]
        ]
    ).amax(0)


def _merge_operations(target, operations):
    for key, ops in operations.items():
        target.setdefault(key, []).extend(ops)


def weight_input_absmax(reader, keys):
    maximum = None
    for key in keys:
        weight = reader.get(key)
        if weight.ndim != 2 or not weight.numel():
            raise ValueError(f"expected nonempty matrix: {key}")
        current = weight.float().abs().amax(0)
        if maximum is not None and current.shape != maximum.shape:
            raise ValueError(f"consumer input dimension mismatch: {key}")
        maximum = current if maximum is None else torch.maximum(maximum, current)
    if maximum is None:
        raise ValueError("no Smooth target weights")
    return maximum


def _offline_operations(spec):
    """Map a Smooth group to per-tensor compensation operations."""
    name = spec["name"]
    scale = name + (".actual" if "norm" in spec else ".scale")
    operations = {key: [dict(op="mul", axis=1, scale=scale)] for key in spec["consumers"]}
    if "norm" in spec:
        operations[spec["norm"]] = [dict(op="replace", axis=0, scale=name + ".norm_replacement")]
    else:
        operations[spec["producer"]] = [
            dict(op="div", axis=0, scale=scale, selector=spec["selector"])
        ]
    return operations


def _validate_offline_plan(reader, operations, scales):
    """Check scales and compensation shapes against shard headers."""
    for key, ops in operations.items():
        shape = reader.shape(key)
        for op in ops:
            value = scales[op["scale"]]
            if value.ndim != 1 or not value.numel() or not torch.isfinite(value).all():
                raise ValueError(f"invalid Smooth scale: {op['scale']}")
            if op["op"] == "replace":
                valid = shape == tuple(value.shape)
            else:
                if (value <= 0).any() or not torch.equal(value.log2(), value.log2().round()):
                    raise ValueError(
                        f"Smooth scale must be a positive power of two: {op['scale']}"
                    )
                selector = op.get("selector")
                if selector:
                    valid = (
                        len(shape) == 2
                        and shape[0] == selector["heads"] * selector["head_width"]
                        and value.numel() == selector["heads"] * selector["length"]
                        and 0 <= selector["offset"]
                        and selector["offset"] + selector["length"] <= selector["head_width"]
                    )
                else:
                    valid = len(shape) == 2 and shape[op["axis"]] == value.numel()
            if not valid:
                raise ValueError(f"Smooth compensation shape mismatch: {key}")


def load_smooth_plan(path, selection=None, model_name=""):
    """Load and validate (operations, scales, manifest)."""
    from safetensors import safe_open

    from ..struct import resolve_model_name

    path = Path(path)
    manifest = json.loads(path.read_text())
    plan_model = resolve_model_name(
        manifest.get("model_name") or model_name, profile=manifest.get("profile")
    )
    if model_name and resolve_model_name(model_name) != plan_model:
        raise ValueError("Smooth plan belongs to a different model.name")
    reader = TensorReader(manifest["model_path"], plan_model)
    expected = {}
    specs = offline_group_specs(
        reader, **(selection if selection is not None else manifest.get("selection", {}))
    )
    for spec in specs:
        _merge_operations(expected, _offline_operations(spec))
    if (
        manifest.get("scale_policy") != "power_of_two"
        or manifest.get("fp8_scale_dtype") != "float32"
        or manifest["operations"] != expected
        or {g["name"]: g["targets"] for g in manifest["groups"]}
        != {s["name"]: s["targets"] for s in specs}
    ):
        raise ValueError("plan differs from the required FP32-FP8 attention compensation graph")
    with safe_open(path.parent / manifest["scale_file"], framework="pt", device="cpu") as handle:
        scales = {key: handle.get_tensor(key).float() for key in handle.keys()}
    _validate_offline_plan(reader, expected, scales)
    return manifest["operations"], scales, manifest


def build_plan(
    model_path, stats, selected_alpha, lower=1 / 12, upper=2, selection=None, model_name=""
):
    """Build (manifest, scales) from selected Smooth alphas."""
    reader = TensorReader(model_path, model_name)
    stats = stats.get("stats", stats)
    operations, scales, groups = {}, {}, []
    for spec in offline_group_specs(reader, **(selection or {})):
        name, alpha = spec["name"], float(selected_alpha[spec["name"]])
        act = target_absmax(spec, stats)
        requested = power_of_two_smooth_scale(
            act, weight_input_absmax(reader, spec["targets"]), alpha, lower, upper
        )
        scale_key = name + (".actual" if "norm" in spec else ".scale")
        if "norm" in spec:
            replacement, actual = norm_scale(reader.get(spec["norm"]), requested)
            scales[name + ".norm_replacement"] = replacement
            scales[scale_key] = actual
        else:
            scales[scale_key] = requested
        _merge_operations(operations, _offline_operations(spec))
        groups.append(dict(name=name, alpha=alpha, scale=scale_key, targets=spec["targets"]))
    _validate_offline_plan(reader, operations, scales)
    return (
        dict(
            model_path=str(reader.root),
            model_name=reader.model_name,
            scale_policy="power_of_two",
            fp8_scale_dtype="float32",
            objective="local_fp8_rtn_qdq_without_gptq_hessian",
            scale_file="smooth_scales.safetensors",
            groups=groups,
            operations=operations,
            selection=selection or {},
        ),
        scales,
    )


def search_offline_scales(args):
    """Search Smooth alpha by local FP8 RTN error, without Hessian weighting."""
    from safetensors import safe_open

    alphas = sorted(set(float(x) for x in args.alphas))
    if not alphas or alphas[0] != 0 or any(not 0 <= x <= 1 for x in alphas):
        raise ValueError("alpha grid must include identity 0 and stay in [0,1]")
    if not 0 <= args.minimum_gain < 1:
        raise ValueError("minimum_gain must be in [0,1)")
    reader = TensorReader(args.model_path, getattr(args, "model_name", ""))
    raw = json.loads(Path(args.stats).read_text())
    stats = raw.get("stats", raw)
    errors, selected = {}, {}
    with safe_open(args.samples, framework="pt", device="cpu") as sample_file:
        for spec in offline_group_specs(reader, **smooth_selection(vars(args))):
            name = spec["name"]
            inputs = [
                sample_file.get_tensor(input_name(k)).to(args.device).float()
                for k in spec["targets"]
            ]
            act = target_absmax(spec, stats, args.device)
            if any(x.ndim != 2 or not x.numel() or x.shape[1:] != act.shape for x in inputs):
                raise ValueError(f"{name}: activation sample/statistic shape mismatch")
            weights = {k: reader.get(k).to(args.device) for k in spec["consumers"]}
            targets = [weights[k] for k in spec["targets"]]
            maximum = torch.stack([w.float().abs().amax(0) for w in targets]).amax(0)
            norm = reader.get(spec["norm"]).to(args.device) if "norm" in spec else None
            baseline = consumer_error(inputs, targets, torch.ones_like(act), args.device)
            if not math.isfinite(baseline):
                raise ValueError(f"{name}: nonfinite baseline error")
            best_error, best_alpha, best_scale = baseline, 0.0, torch.ones_like(act)
            cache, candidates = {}, {}
            for alpha in alphas:
                scale = power_of_two_smooth_scale(
                    act, maximum, alpha, args.smooth_min, args.smooth_max
                )
                actual = norm_scale(norm, scale)[1] if norm is not None else scale
                key = actual.cpu().numpy().tobytes()
                if key not in cache:
                    cache[key] = consumer_error(inputs, targets, actual, args.device)
                error = cache[key]
                if not math.isfinite(error):
                    raise ValueError(f"{name}: nonfinite candidate error")
                candidates[str(alpha)] = error
                if error < best_error and error < baseline * (1 - args.minimum_gain):
                    best_error, best_alpha, best_scale = error, alpha, scale
            eq = max(
                bf16_equivalence_error(x, list(weights.values()), best_scale, norm) for x in inputs
            )
            producer_error = 0.0
            if "producer" in spec:
                selector = spec["selector"]
                producer_error = producer_equivalence_error(
                    sample_file.get_tensor(input_name(spec["producer"])).to(args.device),
                    reader.get(spec["producer"]),
                    best_scale,
                    tuple(selector[k] for k in ("heads", "head_width", "offset", "length")),
                )
            if (
                not math.isfinite(eq)
                or not math.isfinite(producer_error)
                or max(eq, producer_error) > 1e-6
            ):
                raise RuntimeError(f"{name}: non-exact BF16 compensation {eq}, {producer_error}")
            selected[name] = best_alpha
            errors[name] = dict(
                baseline=baseline,
                smooth=best_error,
                candidates=candidates,
                compensation=max(eq, producer_error),
                targets=spec["targets"],
            )
            print(f"[smooth] {name} alpha={best_alpha} error={best_error:.6g}", flush=True)
            del weights, targets, maximum, inputs
    report = dict(
        selected_alpha=selected,
        errors=errors,
        objective="local_fp8_rtn_qdq_without_gptq_hessian",
        fp8_scale_dtype="float32",
        activation_quantization="dynamic_per_token_group_128_fp8_e4m3",
    )
    save_alpha_search_results(report, args.output)
    return report
