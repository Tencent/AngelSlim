"""End-to-end NVFP4 fake-quant via the composed Quantizer (format x scheme x source).

Validates:
  * shape preservation + block-size constraint,
  * the two-level (block-16 + global) scheme beats single-level per-tensor on
    outlier-heavy tensors (the whole reason NVFP4 exists),
  * STE passes gradient to the input,
  * the learnable (weight-like) path: only scale params train, and optimizing the
    scales reduces reconstruction error of the frozen tensor.
"""

import pytest
import torch

from angelslim.compressor.mcore_qad.quant.formats import FORMAT_REGISTRY
from angelslim.compressor.mcore_qad.quant.quantizer import Quantizer
from angelslim.compressor.mcore_qad.quant.schemes import SCHEME_REGISTRY


def _nvfp4(source="dynamic", block_shape=None):
    fmt = FORMAT_REGISTRY.create("e2m1")
    scheme = SCHEME_REGISTRY.create(
        "two_level_block", group_size=16, source=source, block_shape=block_shape
    )
    return Quantizer(fmt, scheme)


def test_nvfp4_preserves_shape_and_requires_block_multiple():
    q = _nvfp4()
    x = torch.randn(8, 256)
    assert q(x).shape == x.shape
    with pytest.raises(AssertionError):
        q(torch.randn(8, 30))  # 30 not divisible by 16


def test_nvfp4_reconstruction_is_reasonable():
    q = _nvfp4()
    x = torch.randn(16, 512)
    rel = (q(x) - x).pow(2).mean() / x.var()
    assert rel < 0.1, rel  # 4-bit block fake-quant: small relative error


def test_two_level_beats_per_tensor_on_outliers():
    # tensor with a heavy-outlier column: per-tensor scale is wrecked, block isn't.
    x = torch.randn(32, 256)
    x[:, 0] *= 100.0
    nvfp4 = _nvfp4()
    per_tensor = Quantizer(
        FORMAT_REGISTRY.create("e2m1"), SCHEME_REGISTRY.create("per_tensor", source="dynamic")
    )
    mse_block = (nvfp4(x) - x).pow(2).mean().item()
    mse_pt = (per_tensor(x) - x).pow(2).mean().item()
    assert mse_block < mse_pt, (mse_block, mse_pt)


def test_nvfp4_ste_passes_input_gradient():
    q = _nvfp4()
    x = torch.randn(4, 64, requires_grad=True)
    q(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_dynamic_nvfp4_has_no_trainable_params():
    q = _nvfp4()
    q(torch.randn(4, 64))  # trigger any lazy init
    assert sum(p.numel() for p in q.parameters() if p.requires_grad) == 0


def test_learnable_per_block_scale_reduces_downstream_loss():
    """The real NVFP4 lever: train the per-block E4M3 scale (frozen weight) to match
    a full-precision teacher's linear output. This is the QAD signal in miniature.
    """
    torch.manual_seed(0)
    out, K, N = 64, 256, 128
    W = torch.randn(out, K)  # frozen "weight"
    X = torch.randn(N, K)
    Yt = X @ W.t()  # full-precision teacher output
    nb = K // 16

    q = _nvfp4(source="learnable", block_shape=(out, nb))
    q(W)  # lazy-init per-block scale at min-max
    params = [p for p in q.parameters() if p.requires_grad]
    assert len(params) == 1 and tuple(params[0].shape) == (out, nb)

    def loss():
        return (X @ q(W).t() - Yt).pow(2).mean()

    # perturb the per-block scale off its data init, then verify training recovers.
    with torch.no_grad():
        q.scheme.block_store.alpha *= 1.6
    bad = loss().item()
    opt = torch.optim.Adam(q.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        current_loss = loss()
        current_loss.backward()
        assert params[0].grad is not None and torch.isfinite(params[0].grad).all()
        opt.step()
    recovered = loss().item()
    assert recovered < 0.95 * bad, (bad, recovered)  # per-block scale is a real lever
