"""Analytic FLOPs estimate for throughput (TFLOPS) reporting.

Returns the forward FLOPs to process ONE token through the whole model (all ranks
together), GQA- and MoE-aware (only the top-k activated experts count). A training
step costs `factor * fwd_per_token * tokens`, with factor = 3 (fwd+bwd) for pure LM,
or 4 for QAD (+1 for the quant-off teacher forward). Per-GPU TFLOPS then divides the
global step FLOPs by world_size and the step time. This is the standard ~6ND estimate
plus the attention seq-length term; it is an approximation (ignores norms, router,
biases, activations).
"""

from __future__ import annotations


def _is_moe_layer(cfg, i: int) -> bool:
    ne = cfg.num_moe_experts
    if not ne:
        return False
    f = cfg.moe_layer_freq
    if isinstance(f, (list, tuple)):
        return bool(f[i])
    if isinstance(f, int) and f > 0:
        return (i % f) == 0
    return True


def fwd_flops_per_token(cfg, vocab_size: int, seq_len: int) -> float:
    h, hd = cfg.hidden_size, cfg.kv_channels
    nh, ng = cfg.num_attention_heads, (cfg.num_query_groups or cfg.num_attention_heads)
    attn_params = 2 * h * hd * (nh + ng)  # q,k,v,o projections (GQA-aware)
    n_active = 0
    for i in range(cfg.num_layers):
        n_active += attn_params
        if _is_moe_layer(cfg, i):  # SwiGLU: gate+up+down = 3*h*ffn
            n_active += cfg.moe_router_topk * 3 * h * cfg.moe_ffn_hidden_size
            if cfg.moe_shared_expert_intermediate_size:
                n_active += 3 * h * cfg.moe_shared_expert_intermediate_size
        else:
            n_active += 3 * h * cfg.ffn_hidden_size
    gemm = 2 * (n_active + h * vocab_size)  # 2 FLOPs / MAC (incl. output proj)
    attn_seq = 4 * cfg.num_layers * nh * hd * seq_len  # QK^T + A·V, scales with seq
    return float(gemm + attn_seq)
