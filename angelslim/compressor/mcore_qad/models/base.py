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

"""Per-model adapter machinery: registry + shared HF->mcore conversion helpers.

A model adapter is `(config_fn, convert_fn)` registered under its HF ``model_type``:
  * config_fn(hf_dict, **parallel/dtype) -> (TransformerConfig, ModelMeta)
  * convert_fn(hf_state_dict, cfg, meta)  -> mcore state dict

Add a new model = a directory ``models/<name>/`` with config.py + convert.py that
calls `register(...)`. Everything (training, tools) dispatches by model_type via
`auto_config` / `load_hf_into_mcore`, so no core code changes per model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict

import torch
from torch import Tensor


@dataclass
class ModelMeta:
    vocab_size: int
    max_sequence_length: int
    rotary_base: float
    tie_embeddings: bool


def load_hf_config(path: str) -> Dict[str, Any]:
    with open(f"{path}/config.json") as f:
        return json.load(f)


# --- shared HF->mcore tensor converters (reused across model adapters) ---------


def interleave_qkv(q: Tensor, k: Tensor, v: Tensor, n_heads: int, n_kv: int, hd: int) -> Tensor:
    """[q|k|v] (HF separate) -> mcore fused GQA-interleaved linear_qkv weight."""
    h = q.shape[-1]
    q, k, v = q.view(n_heads, hd, h), k.view(n_kv, hd, h), v.view(n_kv, hd, h)
    hpg = n_heads // n_kv
    rows = []
    for g in range(n_kv):
        rows += [q[g * hpg : (g + 1) * hpg], k[g : g + 1], v[g : g + 1]]
    return torch.cat(rows, dim=0).reshape(-1, h)


def gated_fc1(hf, gate_key, up_key) -> Tensor:
    return torch.cat([hf[gate_key], hf[up_key]], dim=0).contiguous()  # mcore fc1 = [gate; up]


def attn_to_mcore(hf, p, m, cfg, out) -> None:
    out[f"{m}.input_layernorm.weight"] = hf[f"{p}.input_layernorm.weight"]
    out[f"{m}.pre_mlp_layernorm.weight"] = hf[f"{p}.post_attention_layernorm.weight"]
    out[f"{m}.self_attention.linear_qkv.weight"] = interleave_qkv(
        hf[f"{p}.self_attn.q_proj.weight"],
        hf[f"{p}.self_attn.k_proj.weight"],
        hf[f"{p}.self_attn.v_proj.weight"],
        cfg.num_attention_heads,
        cfg.num_query_groups,
        cfg.kv_channels,
    )
    if cfg.qk_layernorm:
        out[f"{m}.self_attention.q_layernorm.weight"] = hf[f"{p}.self_attn.q_norm.weight"]
        out[f"{m}.self_attention.k_layernorm.weight"] = hf[f"{p}.self_attn.k_norm.weight"]
    out[f"{m}.self_attention.linear_proj.weight"] = hf[f"{p}.self_attn.o_proj.weight"]


def experts_to_mcore(hf, p, m, num_experts, out) -> None:
    fused = f"{p}.mlp.experts.gate_up_proj"
    for e in range(num_experts):
        fc1 = f"{m}.mlp.experts.local_experts.{e}.linear_fc1.weight"
        fc2 = f"{m}.mlp.experts.local_experts.{e}.linear_fc2.weight"
        if fused in hf:  # fused 3D layout
            out[fc1] = hf[fused][e].contiguous()
            out[fc2] = hf[f"{p}.mlp.experts.down_proj"][e].contiguous()
        else:  # per-expert layout
            out[fc1] = gated_fc1(
                hf, f"{p}.mlp.experts.{e}.gate_proj.weight", f"{p}.mlp.experts.{e}.up_proj.weight"
            )
            out[fc2] = hf[f"{p}.mlp.experts.{e}.down_proj.weight"].contiguous()


def globals_to_mcore(hf, meta, out) -> None:
    out["embedding.word_embeddings.weight"] = hf["model.embed_tokens.weight"]
    out["decoder.final_layernorm.weight"] = hf["model.norm.weight"]
    if not meta.tie_embeddings:
        out["output_layer.weight"] = hf["lm_head.weight"]


# --- adapter registry ---------------------------------------------------------


@dataclass
class ModelAdapter:
    config_fn: Callable
    convert_fn: Callable


REGISTRY: Dict[str, ModelAdapter] = {}


def register(model_type: str, config_fn: Callable, convert_fn: Callable) -> None:
    REGISTRY[model_type] = ModelAdapter(config_fn, convert_fn)


def get_adapter(model_type: str) -> ModelAdapter:
    if model_type not in REGISTRY:
        raise KeyError(f"unsupported model_type {model_type!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[model_type]


def auto_config(hf: Dict[str, Any], **kw):
    return get_adapter(hf["model_type"]).config_fn(hf, **kw)


def load_hf_into_mcore(model, hf: Dict[str, Tensor], cfg, meta, model_type: str):
    """Convert (by model_type) + load into the mcore model. Returns (missing, unexpected)."""
    sd = get_adapter(model_type).convert_fn(hf, cfg, meta)
    target = model.state_dict()
    sd = {k: v.to(target[k].dtype) for k, v in sd.items() if k in target}
    return model.load_state_dict(sd, strict=False)
