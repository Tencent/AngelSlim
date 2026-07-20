"""GroupedInt4GroupWeight (stacked, one call) must match the per-expert per_group scheme
exactly, both in forward value and in gradient w.r.t. the learnable per-group alpha."""

import pytest
import torch

from angelslim.compressor.mcore_qad.quant.formats import FORMAT_REGISTRY
from angelslim.compressor.mcore_qad.quant.grouped_quant import GroupedInt4GroupWeight
from angelslim.compressor.mcore_qad.quant.schemes.per_group import PerGroupScheme


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped int4 uses a triton kernel (GPU)"
)
def test_grouped_int4_matches_per_expert_forward_and_grad():
    dev = "cuda"
    torch.manual_seed(0)
    E, OUT, IN, g = 5, 32, 256, 128
    W = torch.randn(E, OUT, IN, device=dev) * 0.1
    fmt = FORMAT_REGISTRY.create("int4")

    gq = GroupedInt4GroupWeight((E, OUT, IN), group_size=g).to(dev)
    gq(W)  # 1st call inits ref/alpha=1
    with torch.no_grad():  # perturb alpha to exercise gradients
        gq.alpha.copy_(1.0 + 0.3 * torch.randn_like(gq.alpha))

    # reference: per-expert per_group scheme with the SAME alpha slice
    per = []
    for e in range(E):
        sch = PerGroupScheme(source="learnable", group_size=g, block_shape=(OUT, IN // g)).to(dev)
        sch.quantize(W[e], fmt)  # init store (ref, alpha=1)
        with torch.no_grad():
            sch.store.alpha.copy_(gq.alpha[e])
        per.append(sch)

    Wq_g = gq(W)
    ref_out = torch.stack([per[e].quantize(W[e], fmt) for e in range(E)], 0)
    fwd_err = (Wq_g - ref_out).abs().max().item()
    assert fwd_err < 1e-4, fwd_err

    go = torch.randn_like(W)
    gq.alpha.grad = None
    (Wq_g * go).sum().backward()
    grad_grouped = gq.alpha.grad.clone()

    grad_ref = torch.empty_like(grad_grouped)
    for e in range(E):
        per[e].store.alpha.grad = None
        (per[e].quantize(W[e], fmt) * go[e]).sum().backward()
        grad_ref[e] = per[e].store.alpha.grad
    grad_err = (grad_grouped - grad_ref).abs().max().item()
    grad_scale = grad_ref.abs().max().item()
    assert grad_err < max(1e-5, 1e-3 * grad_scale), (grad_err, grad_scale)
