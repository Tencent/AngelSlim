"""QuantGroupedExperts (grouped GEMM) must equal a per-expert reference using the SAME
fake-quantized weights AND activations -- validates grouped-GEMM + SwiGLU + routing-probs
+ (for W4A8) the per-token activation quant on both GEMM inputs."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("megatron.core")

from angelslim.compressor.mcore_qad.mcore.grouped_experts import (  # noqa: E402
    QuantGroupedExperts,
)
from angelslim.compressor.mcore_qad.quant.presets import get_format  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped experts run on GPU")
@pytest.mark.parametrize("fmt", ["nvfp4", "w4afp8"])
def test_grouped_experts_match_per_expert_reference(fmt):
    dev = "cuda"
    torch.manual_seed(0)
    E, H, Fd = 6, 256, 128  # H,Fd divisible by 16 and 128 (int4 group)
    dt = torch.bfloat16
    wspec, aspec = get_format(fmt)
    m = QuantGroupedExperts(E, H, Fd, wspec, aspec, params_dtype=dt).to(dev)
    with torch.no_grad():
        m.weight1.copy_(torch.randn(E, 2 * Fd, H, device=dev, dtype=dt) * 0.1)
        m.weight2.copy_(torch.randn(E, H, Fd, device=dev, dtype=dt) * 0.1)

    tokens_per_expert = torch.tensor([5, 0, 11, 7, 3, 9], device=dev)
    tot = int(tokens_per_expert.sum())
    x = torch.randn(tot, H, device=dev, dtype=dt) * 0.2
    probs = torch.rand(tot, device=dev, dtype=dt)

    y, _ = m(x, tokens_per_expert, probs)  # grouped path (inits quant on 1st call)

    # reference: same quantized weights/activations, per-expert torch matmul
    Wq1, Wq2 = m._q_w(m.weight_q1, m.weight1), m._q_w(m.weight_q2, m.weight2)
    xq = m._q_act(x)  # per-token act quant == split-then-quant
    xs = torch.split(xq, tokens_per_expert.tolist())
    ps = torch.split(probs, tokens_per_expert.tolist())
    outs = []
    for e in range(E):
        if xs[e].shape[0] == 0:
            continue
        gate, up = torch.chunk(xs[e] @ Wq1[e].t(), 2, dim=-1)
        a = (F.silu(gate) * up) * ps[e].unsqueeze(-1).to(dt)
        outs.append(m._q_act(a) @ Wq2[e].t())
    y_ref = torch.cat(outs, 0)

    rel = (y.float() - y_ref.float()).abs().max() / y_ref.float().abs().max()
    assert rel < 1e-2, (fmt, rel)

    # gradient sanity: the expert weight scales receive a finite, non-zero gradient
    y.float().pow(2).mean().backward()
    grad = m.weight_q1.alpha.grad
    assert grad is not None and torch.isfinite(grad).all() and float(grad.abs().sum()) > 0
