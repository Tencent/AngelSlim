"""Single-rank numerical correctness of each Format's grid + STE gradient.

These pin the fake-quant *fidelity* (esp. NVFP4: E2M1 grid + E4M3 block-scale
round-trip) and the straight-through gradient behaviour the LSQ scale training
relies on.
"""

import torch

from angelslim.compressor.mcore_qad.quant.formats import FORMAT_REGISTRY
from angelslim.compressor.mcore_qad.quant.formats.fp8 import E4M3Format


def _fmt(key):
    return FORMAT_REGISTRY.create(key)


# ---------------------------------------------------------------- E2M1 (NVFP4)


def test_e2m1_snaps_to_grid_levels():
    fmt = _fmt("e2m1")
    x = torch.tensor([0.0, 0.2, 0.4, 0.6, 1.1, 1.4, 2.6, 3.4, 5.5, -0.6, -3.4])
    q = fmt.to_grid(x)
    expected = torch.tensor([0.0, 0.0, 0.5, 0.5, 1.0, 1.5, 3.0, 3.0, 6.0, -0.5, -3.0])
    assert torch.allclose(q, expected), q


def test_e2m1_clamps_at_max():
    fmt = _fmt("e2m1")
    x = torch.tensor([6.0, 7.5, 100.0, -50.0])
    q = fmt.to_grid(x)
    assert torch.allclose(q, torch.tensor([6.0, 6.0, 6.0, -6.0]))


def test_e2m1_only_representable_values():
    fmt = _fmt("e2m1")
    x = torch.randn(2000) * 3
    q = fmt.to_grid(x)
    allowed = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    allowed = torch.cat([-allowed, allowed]).unique()
    assert torch.isin(q, allowed).all()


def test_e2m1_ste_gradient_inside_is_identity_outside_is_zero():
    fmt = _fmt("e2m1")
    x = torch.tensor([0.4, 1.4, 7.0, -8.0], requires_grad=True)  # last two saturate
    fmt.to_grid(x).sum().backward()
    assert torch.allclose(x.grad, torch.tensor([1.0, 1.0, 0.0, 0.0]))


# ---------------------------------------------------------------- E4M3


def test_e4m3_representable_values_roundtrip_exactly():
    fmt = _fmt("e4m3")
    # values exactly on the E4M3 grid must be preserved.
    x = torch.tensor([0.0, 1.0, 2.0, 0.5, 8.0, 16.0, -4.0])
    assert torch.allclose(fmt.to_grid(x), x)


def test_e4m3_is_lossy_offgrid_and_clamps():
    fmt = _fmt("e4m3")
    assert not torch.allclose(fmt.to_grid(torch.tensor([1.234567])), torch.tensor([1.234567]))
    assert fmt.to_grid(torch.tensor([1e6])).item() == 448.0


def test_e4m3_block_scale_quantization_is_faithful_not_fp32():
    """The NVFP4 block scale MUST pass through E4M3 (not stay FP32)."""
    fmt = E4M3Format()
    raw = torch.tensor([0.123456, 7.77, 33.3])
    q = fmt.quantize_scale(raw)
    assert not torch.allclose(q, raw)  # genuinely quantized
    # idempotent: quantizing an already-E4M3 value is a no-op.
    assert torch.allclose(fmt.quantize_scale(q), q)


def test_e4m3_quantize_scale_passes_gradient():
    fmt = E4M3Format()
    s = torch.tensor([2.0, 10.0], requires_grad=True)
    fmt.quantize_scale(s).sum().backward()
    assert torch.allclose(s.grad, torch.ones_like(s))  # STE identity in-range


# ---------------------------------------------------------------- INT


def test_int8_clamp_round_and_ste():
    fmt = _fmt("int8")
    x = torch.tensor([0.4, 1.6, 200.0, -200.0], requires_grad=True)
    q = fmt.to_grid(x)
    assert torch.allclose(q.detach(), torch.tensor([0.0, 2.0, 127.0, -127.0]))
    q.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_int4_range():
    fmt = _fmt("int4")
    assert fmt.qmax() == 7.0 and fmt.qmin() == -7.0
