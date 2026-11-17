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


"""Shard scheduling across nodes and the final index / quantization_config."""

import heapq
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import struct
import time
from collections import Counter, defaultdict

import torch
from safetensors import safe_open

from .convert import (
    MTP_PREFIX,
    MTP_SHARD,
    RECORDS_DIR,
    Converter,
    Layout,
    Router,
    build_groups,
    ckpt_keys,
    module_name,
)

__all__ = [
    "build_cs_config",
    "cs_ignore",
    "finalize",
    "fp8_blockwise_targets",
    "list_shards",
    "record_path",
    "run_node",
]

_SUFFIXES = [
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
    ".experts.gate_up_proj",
    ".experts.down_proj",
]
# Linears a model has on top of _SUFFIXES, keyed by the YAML ``model.name`` (case-insensitive).
# Anything not listed stays BF16; use a mixed config's config_groups for finer control.
_MODEL_SUFFIXES = {
    # Qwen3.5 linear attention; in_proj_a/b, conv1d, mtp.fc and the routers stay BF16.
    "qwen3_5": [
        ".linear_attn.in_proj_qkv.weight",
        ".linear_attn.in_proj_z.weight",
        ".linear_attn.out_proj.weight",
    ],
    # Hy4 gated MLA and DSA indexer; indexer.wk / weights_proj and MTP eh_proj stay BF16.
    "hy4": [
        ".self_attn.linear_gate.weight",
        ".self_attn.indexer.wq_b.weight",
    ],
}
_BATCHED_EXPERTS = (".experts.gate_up_proj", ".experts.down_proj")


def _model_key(model_name):
    return (model_name or "").lower()


def _suffixes(model_name):
    return _SUFFIXES + _MODEL_SUFFIXES.get(_model_key(model_name), [])


def fp8_blockwise_targets(model_name):
    """The suffix table as module-name targets (a key minus its ``.weight``)."""
    out = []
    for s in _suffixes(model_name):
        s = s[: -len(".weight")] if s.endswith(".weight") else s
        out.append("re:.*" + re.escape(s) + "$")
    return out


def _meta_model(input_path):
    import accelerate
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(input_path)
    # Instantiate the class the checkpoint names: for multimodal ones
    # AutoModelForCausalLM picks the text-only variant or refuses outright, and the module
    # tree is what the BF16 module list below is built from.
    arch = (getattr(config, "architectures", None) or [None])[0]
    model_class = getattr(transformers, arch) if arch and hasattr(transformers, arch) else None
    with accelerate.init_empty_weights():
        if model_class is not None:
            return config, model_class._from_config(config)
        return config, AutoModelForCausalLM.from_config(config)


def fp8_blockwise_quant_config(input_path, block_size, model_name=""):
    """``quant_method: fp8`` config with the BF16 modules of an empty-weight model."""
    from torch import nn

    from ....utils import find_layers

    config, model = _meta_model(input_path)
    suffixes = _suffixes(model_name)
    if _model_key(model_name) == "qwen3_5":
        # Every 2-D weight module counts, matching the official Qwen3.5 FP8 checkpoints.
        ignored = []
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if weight is None or weight.ndim < 2:
                continue
            wname = f"{name}.weight"
            if not any(wname.endswith(s) for s in suffixes + list(_BATCHED_EXPERTS)):
                ignored.append(name)
        for i in range(getattr(config.text_config, "mtp_num_hidden_layers", 0)):
            p = f"mtp.layers.{i}.mlp"
            ignored += ["mtp.fc", f"{p}.gate", f"{p}.shared_expert_gate"]
        quant_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_per_tensor": False,
            "act_per_tensor": False,
            "modules_to_not_convert": ignored,
        }
    else:
        ignored = [
            name
            for name in find_layers(model, [nn.Linear])
            if not name.endswith("mlp.experts")
            and not any(f"{name}.weight".endswith(s) for s in suffixes)
        ]
        quant_config = {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "modules_to_not_convert": ignored,
        }
    if -1 not in tuple(block_size):
        quant_config["weight_block_size"] = list(block_size)
    return quant_config


# Seconds per byte of I/O and per weight element quantized; only used to balance workers.
_COST_IO = 2.0e-9
_COST = {"rtn": 2.0e-10, "gptq": 5.0e-9}


def record_path(spec, file_name):
    return os.path.join(spec.save_path, RECORDS_DIR, file_name + ".json")


def list_shards(spec):
    """Input shards, plus ``MTP_SHARD`` when ``mtp_file`` brings keys the checkpoint lacks."""
    keys = ckpt_keys(spec.model_path)
    shards = sorted(set(keys.values()))
    if spec.mtp_file:
        with safe_open(spec.mtp_file, framework="pt") as f:
            mtp_keys = set(f.keys())
        short = sorted(k for k in keys if k.startswith(MTP_PREFIX) and k not in mtp_keys)
        if short:
            raise ValueError(f"mtp_file lacks {len(short)} checkpoint MTP keys, e.g. {short[:3]}")
        if mtp_keys - set(keys):
            shards.append(MTP_SHARD)
    return shards


def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def _shard_cost(spec, router, file_name):
    if file_name == MTP_SHARD:
        header = _header(spec.mtp_file)
    else:
        header = _header(os.path.join(spec.model_path, file_name))
    cost = 0.0
    for key, meta in header.items():
        numel = math.prod(meta.get("shape") or [1])
        cost += _COST_IO * 2 * numel
        group = router.route(module_name(key)) if len(meta.get("shape") or []) >= 2 else None
        if group is not None:
            cost += _COST[group.algorithm] * numel
    return cost


def _balance(shards, costs, num_workers):
    """Longest-processing-time-first assignment; each list runs most expensive first."""
    heap = [(0.0, i) for i in range(num_workers)]
    lists = [[] for _ in range(num_workers)]
    for name in sorted(shards, key=lambda s: -costs[s]):
        load, i = heapq.heappop(heap)
        lists[i].append(name)
        heapq.heappush(heap, (load + costs[name], i))
    return lists


def _worker(i, spec, file_names):
    n_gpu = torch.cuda.device_count()
    device = f"cuda:{i % n_gpu}" if n_gpu else "cpu"
    if n_gpu:
        torch.cuda.set_device(i % n_gpu)
    converter = Converter(spec, device)
    for k, file_name in enumerate(file_names, 1):
        t0 = time.time()
        record = converter.convert_shard(file_name)
        # The record marks the shard done: it is what --skip-existing and finalize read.
        path = record_path(spec, file_name)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(record, f)
        os.replace(path + ".tmp", path)
        print(f"[w{i} {device}] {k}/{len(file_names)} {file_name} {time.time() - t0:.1f}s",
              flush=True)  # fmt: skip


def run_node(spec, node_rank=0, num_nodes=1, skip_existing=False):
    """Convert ``shards[node_rank::num_nodes]`` with ``spec.num_workers`` processes."""
    shards = list_shards(spec)[node_rank::num_nodes]
    os.makedirs(os.path.join(spec.save_path, RECORDS_DIR), exist_ok=True)
    if skip_existing:
        done = [s for s in shards if os.path.isfile(os.path.join(spec.save_path, s))
                and os.path.isfile(record_path(spec, s))]  # fmt: skip
        shards = [s for s in shards if s not in done]
        print(f"node {node_rank}: skipping {len(done)} converted shards", flush=True)
    if not shards:
        return
    router = Router(build_groups(spec.config_groups, spec.algorithm, spec.gptq, spec.legacy),
                    spec.ignore)  # fmt: skip
    costs = {s: _shard_cost(spec, router, s) for s in shards}
    num_workers = max(1, min(spec.num_workers, len(shards)))
    lists = _balance(shards, costs, num_workers)
    print(f"node {node_rank}/{num_nodes}: {len(shards)} shards over {num_workers} workers",
          flush=True)  # fmt: skip

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(i, spec, lst)) for i, lst in enumerate(lists)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    bad = [f"w{i} exit={p.exitcode}" for i, p in enumerate(procs) if p.exitcode != 0]
    if bad:
        raise RuntimeError(f"node {node_rank}: worker(s) failed ({', '.join(bad)}), see log above")


# compressed-tensors matches with re.match on the runtime prefix, which carries however many
# wrapper modules the architecture adds (Hy4-VL mounts the trunk at
# language_model.model.model.layers.N). Keep the model. anchor so visual.layers.N never matches.
_VLM_PREFIX = r".*"


def _cs_match(name, entries):
    """compressed-tensors semantics: ``re:`` entries use ``re.match``, others equality."""
    for e in entries:
        if e.startswith("re:"):
            if re.match(e[3:], name):
                return True
        elif e == name:
            return True
    return False


def cs_ignore(patterns):
    """YAML ignore_layers -> compressed-tensors ``ignore``: substrings become regexes and
    ``model.``-anchored regexes tolerate the wrapper prefixes the runtime adds."""
    out = []
    for p in patterns:
        if not p.startswith("re:"):
            p = "re:.*" + re.escape(p) + ".*"
        elif p.startswith("re:model\\."):
            p = "re:" + _VLM_PREFIX + p[3:]
        out.append(p)
    if not any(re.search("lm_head", p) for p in out):
        out.append(r"re:.*lm_head$")
    return out


def _first_number(name):
    return next((int(s) for s in name.split(".") if s.isdigit()), None)


def _regex(name, first=None):
    """Regex of ``name`` with every number as ``\\d+``; the first number becomes an
    alternation of ``first`` when given."""
    parts, seen = [], False
    for seg in name.split("."):
        if seg.isdigit():
            if first is not None and not seen:
                alt = sorted(first)
                parts.append(str(alt[0]) if len(alt) == 1 else "(?:%s)" % "|".join(map(str, alt)))
            else:
                parts.append(r"\d+")
            seen = True
        else:
            parts.append(re.escape(seg))
    return parts


def _split(parts):
    return parts[:-1], parts[-1]


def _join(entries):
    """Merge ``[(prefix_parts, leaf)]`` sharing a prefix into ``prefix.(?:a|b)$`` regexes."""
    by_prefix = defaultdict(set)
    for prefix, leaf in entries:
        by_prefix[r"\.".join(prefix)].add(leaf)
    out = []
    for prefix, leaves in sorted(by_prefix.items()):
        leaves = sorted(leaves)
        leaf = leaves[0] if len(leaves) == 1 else "(?:%s)" % "|".join(leaves)
        vlm = _VLM_PREFIX if prefix.startswith("model") else ""
        out.append(f"re:{vlm}{prefix}\\.{leaf}$")
    return out


def build_cs_config(quantized, plain, ignore, schemes, runtime_name=lambda n: n):
    """compressed-tensors config from the written modules ({name: group}) and BF16 ``plain``
    ones, in vLLM names. Equal schemes merge; a ``\\d+`` target's BF16 instances go to
    ``ignore`` unless they outnumber the quantized ones, then layer numbers are listed."""
    ignore = cs_ignore(ignore)
    merged, owner = {}, {}
    for g, scheme in schemes.items():
        key = json.dumps(scheme, sort_keys=True)
        owner[g] = merged.setdefault(key, g)
    q_rt = {runtime_name(m): owner[g] for m, g in quantized.items()}
    by_tpl = defaultdict(lambda: {"plain": set(), "groups": defaultdict(set)})
    for name, g in q_rt.items():
        by_tpl[re.sub(r"(?<=\.)\d+(?=\.|$)", "N", name)]["groups"][g].add(name)
    for m in plain:
        name = runtime_name(m)
        tpl = re.sub(r"(?<=\.)\d+(?=\.|$)", "N", name)
        if name not in q_rt and tpl in by_tpl and not _cs_match(name, ignore):
            by_tpl[tpl]["plain"].add(name)

    targets, extra_ignore = defaultdict(list), []
    for _tpl, info in sorted(by_tpl.items()):
        for g, members in sorted(info["groups"].items()):
            others = info["plain"].union(*(ns for g2, ns in info["groups"].items() if g2 != g))
            rep = min(members)
            mine_first = {_first_number(n) for n in members}
            other_first = {_first_number(n) for n in others}
            if not others:
                targets[g].append(_split(_regex(rep)))
            elif mine_first & other_first:  # same layer split across groups: list modules
                targets[g] += [_split([re.escape(s) for s in n.split(".")]) for n in members]
            elif others <= info["plain"] and len(others) <= len(members):
                targets[g].append(_split(_regex(rep)))
                extra_ignore.append(_split(_regex(rep, other_first)))
            else:
                targets[g].append(_split(_regex(rep, mine_first)))

    groups = {}
    for g in sorted(targets, key=lambda g: list(schemes).index(g)):
        scheme = dict(schemes[g])
        fmt = scheme.pop("format")
        groups[g] = {**scheme, "targets": _join(targets[g])}
        if len(targets) > 1:
            groups[g]["format"] = fmt
    formats = {schemes[g]["format"] for g in groups}
    return {
        "quant_method": "compressed-tensors",
        "format": formats.pop() if len(formats) == 1 else "mixed-precision",
        "quantization_status": "compressed",
        "global_compression_ratio": None,
        "kv_cache_scheme": None,
        "config_groups": groups,
        "ignore": ignore + _join(extra_ignore),
    }


def _read_records(spec):
    shards = list_shards(spec)
    missing = [s for s in shards if not os.path.isfile(record_path(spec, s))]
    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(shards)} shards not converted, e.g. {missing[:3]}; "
            "rerun the quant stage (--skip-existing resumes)"
        )
    merged = {"index": {}, "quantized": {}, "plain": set(), "fallback": Counter()}
    for s in shards:
        with open(record_path(spec, s), encoding="utf-8") as f:
            rec = json.load(f)
        merged["index"].update(rec["index"])
        merged["quantized"].update(rec["quantized"])
        merged["plain"].update(rec["plain"])
        for g, tpl, reason, n in rec["fallback"]:
            merged["fallback"][(g, tpl, reason)] += n
    return merged


def _copy_aux(src, dst):
    skip = {"config.json", "model.safetensors.index.json", "__pycache__"}
    for name in sorted(os.listdir(src)):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        hidden_or_weights = name.startswith(".") or name.endswith(".safetensors")
        if name in skip or hidden_or_weights or os.path.exists(d):
            continue
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)


def finalize(spec):
    """Merge the per-shard records into the index and write ``config.json``."""
    rec = _read_records(spec)
    for (g, tpl, reason), n in sorted(rec["fallback"].items()):
        print(f"[gptq->rtn] {g}: {tpl} x{n} ({reason})", flush=True)
    with open(os.path.join(spec.save_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": dict(sorted(rec["index"].items()))}, f, indent=2)
    _copy_aux(spec.model_path, spec.save_path)

    with open(os.path.join(spec.model_path, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    groups = build_groups(spec.config_groups, spec.algorithm, spec.gptq, spec.legacy)
    if spec.legacy:
        qc = fp8_blockwise_quant_config(spec.model_path, groups[0].fmt.block_size, spec.model_name)
        skipped = sorted(m for m in rec["plain"] if groups[0].matches(m))  # YAML ignore_layers
        known = set(qc["modules_to_not_convert"])
        qc["modules_to_not_convert"] += [m for m in skipped if m not in known]
        config["quantization_config"] = qc
        indent = 4
    else:
        schemes = {g.name: {**g.fmt.cs_scheme(), "format": g.fmt.cs_format} for g in groups}
        layout = Layout(spec.model_path, spec.split_experts)
        config["quantization_config"] = build_cs_config(
            rec["quantized"], rec["plain"], spec.ignore, schemes, layout.runtime_name
        )
        if spec.mtp_file:
            config.setdefault("angelspec_mtp", {})["mtp_source_path"] = spec.mtp_file
        indent = 2
    with open(os.path.join(spec.save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=indent)
    shutil.rmtree(os.path.join(spec.save_path, RECORDS_DIR), ignore_errors=True)
    print(f"quantization_config written to {spec.save_path}/config.json", flush=True)
