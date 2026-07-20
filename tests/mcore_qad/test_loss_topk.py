"""Top-k distillation: with k == vocab it must reproduce the full-vocab KL exactly, and
with k < vocab it must stay finite and pass gradients (the large-vocab memory path)."""

import torch

from angelslim.compressor.mcore_qad.train.loss import build_loss


def _logits(B=2, S=8, V=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(B, S, V, generator=g, requires_grad=True)
    t = torch.randn(B, S, V, generator=g)  # teacher (no grad)
    targets = torch.randint(0, V, (B, S), generator=g)
    return s, t, targets


def test_topk_equals_full_when_k_is_vocab():
    V = 512
    s, t, tgt = _logits(V=V)
    for dt in ("kl", "rkl", "cakld"):
        full = build_loss(lm_weight=1.0, distill_weight=1.0, distill_type=dt, temperature=1.0)
        topk = build_loss(
            lm_weight=1.0, distill_weight=1.0, distill_type=dt, temperature=1.0, distill_topk=V
        )  # k == vocab -> identical support
        lf, _, mf = full(s, t, tgt)
        lk, _, mk = topk(s, t, tgt)
        assert abs(lf.item() - lk.item()) < 1e-4, (dt, lf.item(), lk.item())
        for key in ("fkl", "rkl", "cakld"):
            assert abs(mf[key].item() - mk[key].item()) < 1e-4, (dt, key)


def test_topk_subset_is_finite_and_differentiable():
    V = 512
    s, t, tgt = _logits(V=V)
    loss_fn = build_loss(lm_weight=0.0, distill_weight=1.0, distill_type="cakld", distill_topk=16)
    total, n, m = loss_fn(s, t, tgt)
    assert torch.isfinite(total) and n == 2 * 8
    total.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    # grad is sparse: only the teacher's top-16 columns per token receive signal
    nz_cols = (s.grad.abs().sum(dim=(0, 1)) > 0).sum().item()
    assert 16 <= nz_cols <= 16 * 2 * 8, nz_cols  # <= union of per-token top-k


def test_topk_close_to_full_for_peaked_teacher():
    """A peaked teacher (the QAD regime) -> top-k captures ~all the forward-KL mass.

    forward KL(teacher||student) weights each term by the TEACHER prob, which for a peaked
    teacher concentrates on its top tokens -> top-k recovers most of the value. (We use a
    student that is a mild perturbation of the teacher, as in QAD, so the dropped-tail terms
    are small.)
    """
    V = 1024
    s, t, tgt = _logits(V=V, seed=3)
    t = t * 6.0  # sharpen the teacher distribution
    s = (t.detach() + 0.5 * (s.detach() - t.detach())).requires_grad_(True)  # student ~ teacher
    full = build_loss(lm_weight=0.0, distill_weight=1.0, distill_type="kl")
    topk = build_loss(lm_weight=0.0, distill_weight=1.0, distill_type="kl", distill_topk=128)
    lf, _, _ = full(s, t, tgt)
    lk, _, _ = topk(s, t, tgt)
    assert abs(lf.item() - lk.item()) / (abs(lf.item()) + 1e-6) < 0.15, (lf.item(), lk.item())
