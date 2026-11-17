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


"""Format routing, checkpoint layout and the per-shard quantizer."""

import json
import math
import os
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..modules.gptq.hessian_store import HessianStore
from .formats import FORMATS

__all__ = [
    "ActStats",
    "Converter",
    "Layout",
    "MTP_PREFIX",
    "MTP_SHARD",
    "QuantGroup",
    "RECORDS_DIR",
    "Router",
    "Unit",
    "build_groups",
    "ckpt_keys",
    "compile_pattern",
    "dequant_fp8_block",
    "module_name",
    "template",
]

_GROUP_KEYS = {"format", "targets", "algorithm", "alpha_list", "gptq", "block_size",
               "group_size", "scale_fmt", "act"}  # fmt: skip
_GPTQ_KEYS = {"damp", "batch_rows", "mode"}
# mode: per_column (strict GPTQ) or static_groups (one cast per group, between-group updates)
GPTQ_DEFAULTS = {"damp": 0.05, "batch_rows": 131072, "mode": "per_column"}


def compile_pattern(pattern):
    """``re:<regex>`` (``re.match``, as compressed-tensors does), else a substring match."""
    if pattern.startswith("re:"):
        regex = re.compile(pattern[3:])
        return lambda name: regex.match(name) is not None
    return lambda name: pattern in name


@dataclass
class QuantGroup:
    name: str
    fmt: object
    targets: List[str]
    algorithm: str = "rtn"
    alphas: Optional[List[float]] = None
    gptq: dict = field(default_factory=lambda: dict(GPTQ_DEFAULTS))

    def __post_init__(self):
        self._match = [compile_pattern(t) for t in self.targets]

    def matches(self, name):
        return any(m(name) for m in self._match)


class Router:
    """Ignore first, then the first group whose target matches; ``None`` means passthrough."""

    def __init__(self, groups, ignore=()):
        self.groups = groups
        self._ignore = [compile_pattern(p) for p in ignore]

    def route(self, name):
        if any(m(name) for m in self._ignore):
            return None
        return next((g for g in self.groups if g.matches(name)), None)


def build_groups(config_groups, algorithm="rtn", gptq=None, legacy=False):
    """``quantization.config_groups`` -> [QuantGroup]; group keys override the defaults."""
    if not config_groups:
        raise ValueError("quantization.config_groups is empty")
    defaults = {**GPTQ_DEFAULTS, **(gptq or {})}
    groups = []
    for name, spec in config_groups.items():
        spec = dict(spec)
        unknown = set(spec) - _GROUP_KEYS
        if unknown:
            raise ValueError(f"config_groups.{name}: unknown keys {sorted(unknown)}")
        fmt_name = spec.get("format")
        if fmt_name not in FORMATS:
            raise ValueError(f"config_groups.{name}: format must be one of {sorted(FORMATS)}")
        targets = spec.get("targets") or []
        if not targets or not all(isinstance(t, str) for t in targets):
            raise ValueError(f"config_groups.{name}: targets must be a non-empty list of strings")
        algo = spec.get("algorithm", algorithm)
        if algo not in ("rtn", "gptq"):
            raise ValueError(f"config_groups.{name}: algorithm must be rtn or gptq, got {algo}")
        group_gptq = spec.get("gptq") or {}
        for src in (defaults, group_gptq):
            if set(src) - _GPTQ_KEYS:
                raise ValueError(f"gptq options are {sorted(_GPTQ_KEYS)}, got {sorted(src)}")
        if {**defaults, **group_gptq}["mode"] not in ("per_column", "static_groups"):
            raise ValueError(
                f"config_groups.{name}: gptq.mode must be per_column or static_groups"
            )
        if fmt_name == "fp8_block":
            fmt = FORMATS[fmt_name](
                tuple(spec.get("block_size", (128, 128))), spec.get("scale_fmt", "float"), legacy
            )
        else:
            kwargs = {k: spec[k] for k in ("group_size", "act") if k in spec}
            fmt = FORMATS[fmt_name](**kwargs)
        groups.append(
            QuantGroup(
                name=name,
                fmt=fmt,
                targets=list(targets),
                algorithm=algo,
                alphas=[float(a) for a in spec.get("alpha_list") or []] or None,
                gptq={**defaults, **group_gptq},
            )
        )
    return groups


_MTP_RE = re.compile(r"^model\.mtp_layers\.(\d+)\.")
_EXPERT_STAGES = ("gate_up_proj", "down_proj")
_PAIRED = {"gate_proj": "up_proj", "up_proj": "gate_proj"}


def dequant_fp8_block(weight, scale_inv, block_size):
    """FP8 ``weight`` with a per-block ``weight_scale_inv`` -> BF16 (``fp8 * scale``)."""
    rows, cols = weight.shape
    bm, bn = block_size
    s = scale_inv.float().reshape(-1, (cols + bn - 1) // bn) if scale_inv.dim() < 2 else scale_inv
    s = s.float().repeat_interleave(bm, 0)[:rows].repeat_interleave(bn, 1)[:, :cols]
    return (weight.float() * s.to(weight.device)).to(torch.bfloat16)


def module_name(key):
    """Checkpoint key -> module name (drop a trailing ``.weight``)."""
    return key[: -len(".weight")] if key.endswith(".weight") else key


@dataclass
class Unit:
    """One matrix to quantize: output names, data, and where its H / stats live."""

    base: str  # output module name
    wkey: str  # output weight key (legacy fp8 names scales after it)
    weight: Any
    hkey: Optional[str]  # vLLM module name of its Hessian
    share: Any  # units with the same key share one global scale
    stats_key: str  # input module name for the activation stats lookup
    siblings: List[str] = field(default_factory=list)  # keys sharing the global scale


class Layout:
    """Checkpoint naming: batched experts are split per expert, and ``model.mtp_layers.<i>``
    is ``model.layers.<num_hidden_layers + i>`` inside vLLM."""

    def __init__(self, model_path, split_experts=True):
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        text = cfg.get("text_config") or {}
        self.model_type = cfg.get("model_type", "")
        self.depth = int(cfg.get("num_hidden_layers") or text.get("num_hidden_layers") or 0)
        self.split_experts = split_experts
        qc = cfg.get("quantization_config") or {}
        # FP8 block checkpoints (e.g. GLM-5) are dequantized to BF16 on read.
        self.fp8_block = None
        if qc.get("quant_method") == "fp8" and qc.get("weight_block_size"):
            self.fp8_block = tuple(qc["weight_block_size"])

    def runtime_name(self, name):
        m = _MTP_RE.match(name)
        if m is None:
            return name
        return f"model.layers.{self.depth + int(m.group(1))}.{name[m.end():]}"

    def units(self, module, key, tensor):
        stage = module.rpartition(".experts.")[2]
        if tensor.dim() == 3 and self.split_experts and stage in _EXPERT_STAGES:
            prefix = module[: -len(stage)]  # "<...>.experts."
            if stage == "gate_up_proj":
                gate, up = tensor.chunk(2, dim=1)  # [E, 2I, H]
                parts = (("gate_proj", gate), ("up_proj", up))
            else:
                parts = (("down_proj", tensor),)
            units = []
            for e in range(tensor.shape[0]):
                hkey = self.runtime_name(f"{prefix}{e}.{stage}")
                for leaf, w in parts:
                    base = f"{prefix}{e}.{leaf}"
                    units.append(Unit(base, base + ".weight", w[e], hkey, (module, e), module))
            return units
        hkey = self.runtime_name(module) if tensor.dim() == 2 else None
        parent, _, leaf = module.rpartition(".")
        siblings = [f"{parent}.{_PAIRED[leaf]}.weight"] if leaf in _PAIRED else []
        return [Unit(module, key, tensor, hkey, (module, None), module, siblings)]


FP8_MAX = 448.0

# FP8 checkpoints are calibrated scale-only: ``<stage>.scale_amax`` holds max(scale) = amax / 448.
_TRUNK_RE = re.compile(
    r"^(?:language_model\.)?model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?:(?P<expert>\d+)\.)?(?P<stage>gate_up_proj|down_proj)(?P<fp8>\.scale_amax)?$"
)
# MTP stats come keyed four ways; ``.mtp_block`` hangs off a trunk layer.
_MTP_STATS_RE = re.compile(
    r"^(?:language_model\.)?model\.(?:"
    r"mtp_layers\.(?P<mtp>\d+)(?:\.mtp_block)?"
    r"|layers\.(?P<flat>\d+)(?P<block>\.mtp_block)?"
    r")\.mlp\.experts\.(?:(?P<expert>\d+)\.)?(?P<stage>gate_up_proj|down_proj)"
    r"(?P<fp8>\.scale_amax)?$"
)
_MODULE_RE = re.compile(
    r"^model\.(?P<kind>layers|mtp_layers)\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?:\d+\.)?(?P<leaf>gate_up_proj|gate_proj|up_proj|down_proj)$"
)


def _amax(v, m):
    if not isinstance(v, dict) or "max" not in v:
        return None
    amax = abs(float(v["max"]))
    if isinstance(v.get("min"), (int, float)):
        amax = max(amax, abs(float(v["min"])))
    if m.group("fp8"):
        amax *= FP8_MAX
    return amax if math.isfinite(amax) and amax > 0 else None


def _merge(layer_agg, per_expert):
    """The layer aggregate wins; else the max over that layer's per-expert entries."""
    return {k: layer_agg.get(k, per_expert.get(k)) for k in set(layer_agg) | set(per_expert)}


def _load_trunk(path):
    with open(path, encoding="utf-8") as f:
        stats = json.load(f)
    agg, per_expert = {}, {}
    for k, v in stats.items():
        m = _TRUNK_RE.match(k)
        amax = _amax(v, m) if m else None
        if amax is None:
            continue
        key = (int(m.group("layer")), m.group("stage"))
        if m.group("expert") is None:
            agg[key] = amax
        else:
            per_expert[key] = max(amax, per_expert.get(key, 0.0))
    out = _merge(agg, per_expert)
    if not out:
        raise ValueError(f"{path}: no MoE expert stats matched, sample keys {list(stats)[:3]}")
    return out


def _load_mtp(path, depth):
    with open(path, encoding="utf-8") as f:
        stats = json.load(f)
    agg, per_expert = {}, {}
    for k, v in stats.items():
        m = _MTP_STATS_RE.match(k)
        amax = _amax(v, m) if m else None
        if amax is None:
            continue
        if m.group("mtp") is not None:
            key = (int(m.group("mtp")), m.group("stage"))
        elif m.group("block") is not None:
            key = ("mtp_block", int(m.group("flat")), m.group("stage"))
        elif int(m.group("flat")) >= depth:
            key = (int(m.group("flat")) - depth, m.group("stage"))
        else:
            continue  # a trunk layer
        if m.group("expert") is None:
            agg[key] = amax
        else:
            per_expert[key] = max(amax, per_expert.get(key, 0.0))
    raw = _merge(agg, per_expert)
    out = {k: v for k, v in raw.items() if k[0] != "mtp_block"}
    # ``layers.<N>.mtp_block`` -> MTP layer index, in trunk-layer order.
    for idx, trunk in enumerate(sorted({k[1] for k in raw if k[0] == "mtp_block"})):
        for stage in ("gate_up_proj", "down_proj"):
            if ("mtp_block", trunk, stage) in raw:
                out.setdefault((idx, stage), raw[("mtp_block", trunk, stage)])
    if not out:
        raise ValueError(f"{path}: no MTP expert stats matched, sample keys {list(stats)[:3]}")
    return out


class ActStats:
    """MoE input absmax from ``moe_expert_stats.json`` / ``mtp_moe_expert_stats.json``."""

    def __init__(self, stats_dir, depth):
        self.trunk = _load_trunk(os.path.join(stats_dir, "moe_expert_stats.json"))
        mtp_path = os.path.join(stats_dir, "mtp_moe_expert_stats.json")
        self.mtp = _load_mtp(mtp_path, depth) if os.path.isfile(mtp_path) else {}

    def amax(self, module):
        """Input absmax of a checkpoint expert module (batched or per-expert)."""
        m = _MODULE_RE.match(module)
        if m is None:
            raise KeyError(f"{module}: activation stats exist only for MoE experts")
        stage = "down_proj" if m.group("leaf") == "down_proj" else "gate_up_proj"
        table = self.mtp if m.group("kind") == "mtp_layers" else self.trunk
        value = table.get((int(m.group("layer")), stage))
        if value is None:
            raise KeyError(f"{module}: no activation stats for this {m.group('kind')} layer")
        return value


MTP_PREFIX = "model.mtp_layers."
# Output shard for --mtp_file keys the checkpoint does not have.
MTP_SHARD = "model-mtp-00000-of-00001.safetensors"
RECORDS_DIR = "_convert_records"


def template(name):
    return re.sub(r"(?<=\.)\d+(?=\.|$)", "N", name)


def ckpt_keys(model_path):
    """{key: shard} of the input checkpoint."""
    index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as f:
            return json.load(f)["weight_map"]
    with safe_open(os.path.join(model_path, "model.safetensors"), framework="pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


class Converter:
    """Quantizes one checkpoint shard at a time on ``device``."""

    def __init__(self, spec, device):
        self.spec = spec
        self.device = device
        self.groups = build_groups(spec.config_groups, spec.algorithm, spec.gptq, spec.legacy)
        self.router = Router(self.groups, spec.ignore)
        self.layout = Layout(spec.model_path, spec.split_experts)
        need_h = any(g.algorithm == "gptq" for g in self.groups)
        self.hstore = HessianStore(spec.hessian_dir) if need_h and spec.hessian_dir else None
        need_stats = any(g.fmt.needs_act_stats for g in self.groups)
        self.stats = ActStats(spec.stats_dir, self.layout.depth) if need_stats else None
        self.index = ckpt_keys(spec.model_path)
        self._files = {}
        self.mtp_keys = set()
        if spec.mtp_file:
            self.mtp_keys = set(self._open(spec.mtp_file).keys())

    def _open(self, path):
        if path not in self._files:
            self._files[path] = safe_open(path, framework="pt")
        return self._files[path]

    def _raw(self, key):
        if key in self.mtp_keys and key.startswith(MTP_PREFIX):
            return self._open(self.spec.mtp_file).get_tensor(key)
        return self._open(os.path.join(self.spec.model_path, self.index[key])).get_tensor(key)

    def _has(self, key):
        return key in self.index or key in self.mtp_keys

    def _is_fp8_scale(self, key):
        """A ``weight_scale_inv`` consumed by dequantizing its FP8 weight on read."""
        return (
            self.layout.fp8_block is not None
            and key.endswith(".weight_scale_inv")
            and (self._has(key[: -len("_scale_inv")]))
        )

    def _source_fp8(self, key):
        return self.layout.fp8_block is None or self._has(key + "_scale_inv")

    def read(self, key):
        """Tensor of ``key`` (MTP keys from ``mtp_file``), FP8 block weights as BF16."""
        tensor = self._raw(key)
        scale_key = key + "_scale_inv"
        if tensor.dtype == torch.float8_e4m3fn and self.layout.fp8_block and self._has(scale_key):
            tensor = dequant_fp8_block(tensor, self._raw(scale_key), self.layout.fp8_block)
        return tensor

    def convert_shard(self, file_name):
        """Writes ``<save_path>/<file_name>`` and returns its record."""
        record = {"index": {}, "quantized": {}, "plain": [], "fallback": Counter()}
        out = OrderedDict()
        if file_name == MTP_SHARD:
            keys = sorted(self.mtp_keys - set(self.index))
        else:
            keys = [k for k, f in self.index.items() if f == file_name]
        for key in keys:
            if self._is_fp8_scale(key):
                continue
            tensor = self.read(key)
            module = module_name(key)
            group = self.router.route(module) if tensor.dim() >= 2 else None
            if group and self.spec.skip_unquantized_source and not self._source_fp8(key):
                group = None
            if group is None:
                out[key] = tensor
                if tensor.dim() == 2 and key.endswith(".weight"):
                    record["plain"].append(module)
                continue
            units = self.layout.units(module, key, tensor)
            out.update(self._quantize(group, units, record))
            del tensor, units
        dst = os.path.join(self.spec.save_path, file_name)
        tmp = f"{dst}.{os.getpid()}.tmp"
        save_file(out, tmp)
        os.replace(tmp, dst)
        record["index"] = {k: file_name for k in out}
        record["fallback"] = [[*k, v] for k, v in record["fallback"].items()]
        return record

    def _hinv(self, group, unit, record):
        """Unit's GPTQ factor and Hessian entry, or (None, None) after counting the fallback."""
        reason = "no_hessian"
        if self.hstore is not None and unit.hkey and unit.weight.dim() == 2:
            entry = self.hstore.entry(unit.hkey)
            if entry is not None:
                hinv = self.hstore.get_hinv(unit.hkey, self.device, group.gptq["damp"])
                k = unit.weight.shape[-1]
                if hinv is None:
                    reason = "not_pd"
                elif hinv.shape[-1] != k or k % group.fmt.group:
                    reason = "shape"
                else:
                    return hinv, entry
        record["fallback"][(group.name, template(unit.base), reason)] += 1
        return None, None

    def _quantize(self, group, units, record):
        fmt, dev = group.fmt, self.device
        shares = OrderedDict()
        for u in units:
            shares.setdefault(u.share, []).append(u.weight)
        # Rounding of amax / 448 / 6 differs between CPU and GPU; this split (fused gate/up
        # on CPU, single matrices on the device) reproduces the gptq_alg1 tool bit for bit.
        for u in units:  # e.g. per-expert gate/up stored as separate keys
            if fmt.has_global_scale and u.siblings and len(shares[u.share]) == 1:
                shares[u.share] += [self.read(k) for k in u.siblings if self._has(k)]
        gscale = {
            s: fmt.global_scale(ws, "cpu" if len(ws) > 1 else dev) for s, ws in shares.items()
        }
        act = self.stats.amax(units[0].stats_key) if fmt.needs_act_stats else None
        results = {}
        sg = group.gptq.get("mode") == "static_groups"

        def flush(batch, shared):
            """One GPTQ sweep over a batch: rows stacked under a shared H, else one H each."""
            ws = [u.weight.to(dev) for _, u, _ in batch]
            gs = [gscale[u.share] for _, u, _ in batch]
            if len(batch) == 1:
                parts = [fmt.gptq(ws[0], batch[0][2], gs[0], group.alphas, sg)]
            elif shared:
                g = None
                if gs[0] is not None:
                    g = torch.cat([g.reshape(1).expand(w.shape[0]) for g, w in zip(gs, ws)])
                q, s = fmt.gptq(torch.cat(ws), batch[0][2], g, group.alphas, sg)
                bounds = torch.tensor([0] + [w.shape[0] for w in ws]).cumsum(0).tolist()
                parts = [fmt.row_slice(q, s, r0, r1) for r0, r1 in zip(bounds, bounds[1:])]
            else:
                g = None if gs[0] is None else torch.stack([g.reshape(()) for g in gs])
                hinv = torch.stack([h for _, _, h in batch])
                q, s = fmt.gptq(torch.stack(ws), hinv, g, group.alphas, sg)
                parts = list(zip(q, s))
            for (i, _, _), (q, s) in zip(batch, parts):
                results[i] = (q.cpu(), s.cpu())

        if group.algorithm == "gptq":
            max_rows = int(group.gptq["batch_rows"])
            batch, rows, bkey = [], 0, None
            for i, u in enumerate(units):
                hinv, entry = self._hinv(group, u, record)
                if hinv is None:
                    continue
                n = u.weight.shape[0]
                shared = entry.get("expert") is None and fmt.can_stack_rows(n)
                key = (shared, entry["group"] if shared else None, tuple(u.weight.shape))
                if batch and (key != bkey or rows + n > max_rows):
                    flush(batch, bkey[0])
                    batch, rows = [], 0
                batch.append((i, u, hinv))
                bkey, rows = key, rows + n
            if batch:
                flush(batch, bkey[0])

        out = OrderedDict()
        for i, u in enumerate(units):
            q, s = results.pop(i, None) or fmt.rtn(u.weight.to(dev), gscale[u.share], group.alphas)
            out.update(fmt.tensors(u.base, u.wkey, q, s, gscale[u.share], act))
            record["quantized"][u.base] = group.name
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        return out
