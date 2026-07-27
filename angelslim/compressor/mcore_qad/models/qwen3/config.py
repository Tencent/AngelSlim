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

"""Qwen3-MoE: HF config -> mcore TransformerConfig (GQA + qk-norm + softmax-router MoE)."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from megatron.core.transformer.transformer_config import TransformerConfig

from angelslim.compressor.mcore_qad.models.base import ModelMeta


def qwen3_moe_config(
    hf: Dict[str, Any],
    *,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    cp: int = 1,
    sequence_parallel: bool = False,
    params_dtype: torch.dtype = torch.bfloat16,
):
    head_dim = hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"])
    cfg = TransformerConfig(
        num_layers=hf["num_hidden_layers"],
        hidden_size=hf["hidden_size"],
        num_attention_heads=hf["num_attention_heads"],
        num_query_groups=hf["num_key_value_heads"],
        kv_channels=head_dim,
        ffn_hidden_size=hf["intermediate_size"],
        normalization="RMSNorm",
        layernorm_epsilon=hf.get("rms_norm_eps", 1e-6),
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        add_qkv_bias=False,
        qk_layernorm=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_moe_experts=hf["num_experts"],
        moe_router_topk=hf["num_experts_per_tok"],
        moe_ffn_hidden_size=hf["moe_intermediate_size"],
        moe_grouped_gemm=True,
        # Qwen3 norm_topk_prob == softmax over selected top-k == mcore pre_softmax=False.
        moe_router_pre_softmax=(not hf.get("norm_topk_prob", True)),
        moe_router_load_balancing_type="none",
        moe_router_dtype="fp32",  # fp32 routing: stable softmax over many experts
        moe_router_score_function="softmax",
        moe_token_dispatcher_type="alltoall",
        moe_router_topk_scaling_factor=1.0,
        moe_router_enable_expert_bias=False,
        gradient_accumulation_fusion=False,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
        context_parallel_size=cp,
        sequence_parallel=sequence_parallel,
        params_dtype=params_dtype,
        bf16=(params_dtype == torch.bfloat16),
        pipeline_dtype=params_dtype,
    )
    meta = ModelMeta(
        vocab_size=hf["vocab_size"],
        max_sequence_length=hf.get("max_position_embeddings", 4096),
        rotary_base=hf.get("rope_theta", 10000.0),
        tie_embeddings=hf.get("tie_word_embeddings", False),
    )
    return cfg, meta
