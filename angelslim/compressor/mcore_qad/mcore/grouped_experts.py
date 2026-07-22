"""Grouped (stacked) MoE experts with one-shot weight fake-quant + grouped GEMM.

Replaces mcore's per-expert SequentialMLP (which launches ~6000 tiny quant+GEMM ops per
forward -- the measured bottleneck). All local experts live in stacked 3D weights; the
whole stack is fake-quantized in ONE vectorized call, then a single grouped GEMM
(`torch._grouped_mm`, native ATen/cutlass) computes all experts. Expert tensor-parallel is
assumed 1 (experts are sharded by EP only), so there is no in-expert TP/SP reduce to replicate.

Quantization is format-driven (the same QuantSpec pair used for the dense linears):
  * weight: a pluggable grouped weight quantizer (NVFP4 / INT4-group), see grouped_quant.py.
  * activation: an optional per-token quantizer applied to BOTH grouped-GEMM inputs (the
    A in W4A8). Block activation schemes (e.g. NVFP4 two-level) are not wired on the expert
    input yet, so those formats keep weight-only experts (unchanged prior behavior).

Interface matches mcore experts: forward(permuted_hidden, tokens_per_expert, permuted_probs)
-> (output, bias). Tokens arrive already permuted/grouped by expert from the dispatcher;
`tokens_per_expert` are the grouped-GEMM group sizes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from angelslim.compressor.mcore_qad.quant.formats import FORMAT_REGISTRY
from angelslim.compressor.mcore_qad.quant.functional import grad_scale, lsq_grad_scale
from angelslim.compressor.mcore_qad.quant.grouped_quant import (
    build_grouped_weight_quant,
)
from angelslim.compressor.mcore_qad.quant.quantizer import Quantizer
from angelslim.compressor.mcore_qad.quant.schemes import SCHEME_REGISTRY
from angelslim.compressor.mcore_qad.quant.spec import QuantSpec

#: activation schemes the grouped-expert input supports (token-wise; no block kernel yet).
GROUPED_ACT_SCHEMES = ("per_token", "per_tensor")


def build_grouped_act_quant(act_spec: QuantSpec):
    """Per-token (or per-tensor) activation quantizer for the grouped-GEMM input, or None.

    Dynamic -> no trainable params; returns a Quantizer so the QAD teacher's quant_disabled
    toggles it too. Block activation schemes (NVFP4) stay None (weight-only experts) for now.
    """
    if act_spec.is_identity() or act_spec.scheme not in GROUPED_ACT_SCHEMES:
        return None
    fmt = FORMAT_REGISTRY.create(act_spec.fmt)
    scheme = SCHEME_REGISTRY.create(act_spec.scheme, source=act_spec.source)
    return Quantizer(fmt, scheme)


class GroupedNVFP4Activation(nn.Module):
    """Per-expert static global scale + dynamic per-token block-16 NVFP4 QDQ."""

    def __init__(self, num_local_experts: int, group_size: int = 16) -> None:
        super().__init__()
        self.num_local_experts = num_local_experts
        self.g = group_size
        self.alpha = nn.Parameter(torch.ones(num_local_experts))
        self.register_buffer("ref", torch.ones(num_local_experts))
        self.register_buffer("_initialized", torch.zeros((), dtype=torch.bool))
        self.e2m1 = FORMAT_REGISTRY.create("e2m1")
        self.e4m3 = FORMAT_REGISTRY.create("e4m3")
        self.lsq = lsq_grad_scale(group_size, self.e2m1.max_repr)
        self.enabled = True

    def global_scale(self) -> Tensor:
        return self.ref * self.alpha.clamp(0.25, 4.0)

    @torch.no_grad()
    def _initialize(self, x: Tensor, tokens_per_expert: Tensor) -> None:
        start = 0
        fallback = (
            x.float().abs().amax() / (self.e2m1.max_repr * self.e4m3.max_repr)
            if x.numel()
            else x.new_tensor(1.0, dtype=torch.float32)
        ).clamp_min(1e-10)
        for expert, count in enumerate(tokens_per_expert.tolist()):
            if count:
                value = x[start : start + count].float().abs().amax()
                value = value / (self.e2m1.max_repr * self.e4m3.max_repr)
                self.ref[expert].copy_(value.clamp_min(1e-10))
            else:
                self.ref[expert].copy_(fallback)
            start += count
        self.alpha.fill_(1.0)
        self._initialized.fill_(True)

    def forward(self, x: Tensor, tokens_per_expert: Tensor) -> Tensor:
        if not self.enabled or x.numel() == 0:
            return x
        if x.shape[-1] % self.g:
            raise ValueError(f"activation K={x.shape[-1]} is not divisible by {self.g}")
        if not bool(self._initialized):
            self._initialize(x, tokens_per_expert)
        counts = tokens_per_expert.to(device=x.device, dtype=torch.long)
        total_tokens = int(counts.sum())
        if total_tokens != x.shape[0]:
            raise ValueError(
                f"tokens_per_expert sums to {total_tokens}, expected {x.shape[0]}"
            )
        token_global = torch.repeat_interleave(self.global_scale(), counts).view(-1, 1)
        xb = x.float().reshape(x.shape[0], x.shape[-1] // self.g, self.g)
        block_amax = xb.detach().abs().amax(dim=-1)
        block_raw = (block_amax / self.e2m1.max_repr) / token_global
        block_raw = torch.where(block_amax > 0, block_raw, torch.ones_like(block_raw))
        block_scale = self.e4m3.quantize_scale(block_raw)
        eff = (token_global * block_scale).unsqueeze(-1).clamp_min(1e-10)
        eff = grad_scale(eff, self.lsq)
        qdq = self.e2m1.to_grid(xb / eff) * eff
        return qdq.reshape_as(x).to(x.dtype)


class QuantGroupedExperts(nn.Module):
    def __init__(
        self,
        num_local_experts: int,
        hidden_size: int,
        moe_ffn: int,
        weight_spec: QuantSpec,
        act_spec: QuantSpec,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        E, H, F_ = num_local_experts, hidden_size, moe_ffn
        # mcore linear convention: weight [out, in]. fc1 out=2F (gate|up), fc2 out=H.
        self.weight1 = nn.Parameter(
            torch.empty(E, 2 * F_, H, dtype=params_dtype), requires_grad=False
        )
        self.weight2 = nn.Parameter(torch.empty(E, H, F_, dtype=params_dtype), requires_grad=False)
        self.weight_q1 = build_grouped_weight_quant(weight_spec, (E, 2 * F_, H))
        self.weight_q2 = build_grouped_weight_quant(weight_spec, (E, H, F_))
        if act_spec.fmt == "e2m1" and act_spec.scheme == "two_level_block":
            self.act_q1 = GroupedNVFP4Activation(E, act_spec.group_size or 16)
            self.act_q2 = GroupedNVFP4Activation(E, act_spec.group_size or 16)
            self.act_q = None
        else:
            self.act_q1 = self.act_q2 = None
            self.act_q = build_grouped_act_quant(act_spec)  # None -> weight-only experts

    def _q_act(
        self, x: Tensor, tokens_per_expert: Tensor | None = None, *, second: bool = False
    ) -> Tensor:
        nvfp4_q = self.act_q2 if second else self.act_q1
        if nvfp4_q is not None:
            if tokens_per_expert is None:
                raise ValueError("NVFP4 grouped activation requires tokens_per_expert")
            return nvfp4_q(x, tokens_per_expert)
        return self.act_q(x.float()).to(x.dtype) if self.act_q is not None else x

    def _q_w(self, q, W: Tensor) -> Tensor:
        return q(W) if q is not None else W

    def forward(
        self,
        permuted_hidden: Tensor,
        tokens_per_expert: Tensor,
        permuted_probs: Tensor | None = None,
    ):
        # ONE-SHOT fake-quant of the whole expert stack (the measured bottleneck), then a
        # single grouped GEMM via torch._grouped_mm -- a NATIVE ATen op (cutlass), so it
        # composes with mcore's activation checkpoint (no cross-iter leak) and is fast, unlike
        # the 3rd-party grouped_gemm package whose custom autograd Function leaked under
        # checkpointing. This is the same grouped-GEMM strategy mcore/TE use for MoE.
        Wq1 = self._q_w(self.weight_q1, self.weight1)  # [E,2F,H] (out,in), quantized once
        Wq2 = self._q_w(self.weight_q2, self.weight2)  # [E,H,F]
        x = self._q_act(permuted_hidden, tokens_per_expert)  # [M, H]
        offs = torch.cumsum(tokens_per_expert.to(x.device), 0).to(torch.int32)  # group boundaries
        # _grouped_mm(a[M,K], b[E,K,N], offs) = per-group a@b -> [M,N]; b = W^T = [E,in,out]
        h = torch._grouped_mm(x, Wq1.transpose(-1, -2), offs=offs)  # [M, 2F]
        gate, up = torch.chunk(h, 2, dim=-1)
        a = F.silu(gate) * up  # SwiGLU -> [M, F]
        a = self._q_act(a, tokens_per_expert, second=True)  # quantize fc2 input too
        y = torch._grouped_mm(a, Wq2.transpose(-1, -2), offs=offs)  # [M, H]
        # Apply routing after the down projection in FP32. Applying it before the
        # fc2 input quantizer changes the activation distribution and diverges from
        # the HF/deployment semantics.
        if permuted_probs is not None:
            y = (y.float() * permuted_probs.unsqueeze(-1).float()).to(y.dtype)
        return y, None


@torch.no_grad()
def replace_moe_experts_with_grouped(model, weight_spec: QuantSpec, act_spec: QuantSpec) -> int:
    """Swap each MoELayer's per-expert SequentialMLP for a stacked QuantGroupedExperts,
    copying the (etp=1, full) per-expert weights into the 3D stack. Returns #layers swapped.

    Run AFTER load_dist_checkpoint (so weights are loaded) and AFTER quantize_mcore_model
    (which must skip routed experts); the grouped module carries its own fake-quant.
    """
    from megatron.core.transformer.moe.moe_layer import MoELayer

    n = 0
    for m in model.modules():
        if not isinstance(m, MoELayer):
            continue
        seq = m.experts  # SequentialMLP
        E = m.num_local_experts
        cfg = m.config
        dev = seq.local_experts[0].linear_fc1.weight.device
        dt = seq.local_experts[0].linear_fc1.weight.dtype
        qge = QuantGroupedExperts(
            E, cfg.hidden_size, cfg.moe_ffn_hidden_size, weight_spec, act_spec, params_dtype=dt
        ).to(dev)
        for e in range(E):
            qge.weight1[e].copy_(seq.local_experts[e].linear_fc1.weight)  # [2F,H]
            qge.weight2[e].copy_(seq.local_experts[e].linear_fc2.weight)  # [H,F]
        m.experts = qge
        del seq
        n += 1
    import gc

    gc.collect()
    torch.cuda.empty_cache()  # release the old SequentialMLP weights
    return n
