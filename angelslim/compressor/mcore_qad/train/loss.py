"""Training loss: LM (default) + optional distillation, each with its own weight.

    total = lm_weight * lm_loss + distill_weight * distill_loss

* lm_loss     : next-token cross-entropy (always available; default weight 1.0).
* distill_loss: teacher/student logit match (default weight 0.0 -> off). Types:
    - kl    : forward KL(teacher || student)
    - rkl   : reverse KL(student || teacher)
    - cakld : Contextual Asymmetric KL -- per-token mix of fwd/rev KL weighted by the
              teacher's confidence on the gold label (conf*rkl + (1-conf)*fkl).
  `temperature` scales the softmaxes (KD-style, *T^2).

`distill_topk` (0 = full vocab) restricts the distill KL to the teacher's top-k tokens,
renormalized over that support. This is what makes large-vocab distillation fit: the
KL then never materializes/retains a [tokens, vocab] tensor (only [tokens, k]), which the
full path must keep for backward. For QAD (student ~= teacher) the top-k carries ~all the
mass, so it tracks the full KL closely. NOTE: top-k only shrinks the DISTILL term; an
exact LM cross-entropy still needs the full vocab (so under lm_weight>0 the LM term keeps
the [tokens, vocab] cost -- use TP/vocab-parallel for that, not top-k).

The distill term needs a teacher forward; the trainer runs a quant-OFF pass to get it
only when `needs_teacher` is true. Everything is computed in fp32 on causal-shifted,
non-padding tokens (bf16 full-vocab softmax can underflow and make KL grads Inf/NaN).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

IGNORE_INDEX = -100


def _kl_from_logps(log_p_src: Tensor, log_p_tgt: Tensor) -> Tensor:
    """Per-token KL(tgt || src) from log-probs (gradient-safe)."""
    return (log_p_tgt.exp() * (log_p_tgt - log_p_src)).sum(dim=-1)


class TrainLoss:
    def __init__(
        self,
        *,
        lm_weight: float = 1.0,
        distill_weight: float = 0.0,
        distill_type: str = "kl",
        temperature: float = 1.0,
        distill_topk: int = 0,
    ) -> None:
        self.lm_weight = float(lm_weight)
        self.distill_weight = float(distill_weight)
        self.distill_type = distill_type.lower()
        self.temperature = float(temperature)
        self.distill_topk = int(distill_topk)

    @property
    def needs_teacher(self) -> bool:
        return self.distill_weight > 0.0

    _SEL = {"kl": "fkl", "rkl": "rkl", "cakld": "cakld"}

    def _distill_all(self, s: Tensor, t: Tensor, valid: Tensor) -> dict:
        """All distill variants: fkl=KL(teacher||student), rkl=KL(student||teacher),
        cakld=conf-mixed. With distill_topk>0 the KL is over the teacher's top-k tokens
        (renormalized on that support) so no [tokens, vocab] tensor is built/retained;
        conf is then the teacher mass on the gold token within that top-k (else 0)."""
        T = max(self.temperature, 1e-6)
        scale = T * T
        k = self.distill_topk
        if k and k < t.shape[-1]:
            tv, ti = t.topk(k, dim=-1)  # teacher top-k [N,k] (+ indices)
            sv = s.gather(-1, ti)  # student at same indices [N,k]
            s_lp = F.log_softmax(sv.float() / T, dim=-1)
            t_lp = F.log_softmax(tv.float() / T, dim=-1)
            conf = (t_lp.exp() * (ti == valid.unsqueeze(-1))).sum(
                dim=-1
            )  # teacher P(gold) in top-k
        else:
            conf = F.softmax(t.float(), dim=-1).gather(-1, valid.unsqueeze(-1)).squeeze(-1)
            s_lp, t_lp = F.log_softmax(s.float() / T, dim=-1), F.log_softmax(t.float() / T, dim=-1)
        fkl_pt, rkl_pt = _kl_from_logps(s_lp, t_lp), _kl_from_logps(t_lp, s_lp)
        cakld = (conf * rkl_pt + (1.0 - conf) * fkl_pt).mean() * scale
        return {"fkl": fkl_pt.mean() * scale, "rkl": rkl_pt.mean() * scale, "cakld": cakld}

    def __call__(self, student_logits: Tensor, teacher_logits, targets: Tensor):
        """Return (loss, n_tokens, metrics). `loss` is the per-token-mean training objective
        (lm_weight*lm + distill_weight*<distill_type>); `metrics` always reports lm and (if a
        teacher is given) fkl/rkl/cakld for monitoring -- all detached. Targets are pre-shifted.
        """
        mask = (targets != IGNORE_INDEX).reshape(-1)
        n_tokens = int(mask.sum())
        if n_tokens == 0:  # keep grad-connected (CP shard may be empty)
            return student_logits.sum() * 0.0, 0, {}
        s = student_logits.flatten(0, -2)[mask]  # [N,V] (kept in model dtype; float per-use)
        valid = targets.reshape(-1)[mask]

        # LM cross-entropy needs the full vocab; when it is not in the objective (weight 0) we
        # still log it but under no_grad so it neither joins the graph nor retains [N,V] state.
        if self.lm_weight != 0.0:
            lm = F.cross_entropy(s.float(), valid)
            total = self.lm_weight * lm
        else:
            with torch.no_grad():
                lm = F.cross_entropy(s.float(), valid)
            total = student_logits.new_zeros(())
        metrics = {"lm": lm.detach()}
        if self.needs_teacher:
            t = teacher_logits.flatten(0, -2)[mask]  # [N,V] teacher (detached, no grad)
            d = self._distill_all(s, t, valid)
            metrics.update({k: v.detach() for k, v in d.items()})
            total = total + self.distill_weight * d[self._SEL[self.distill_type]]
        return total, n_tokens, metrics


def build_loss(
    *,
    lm_weight: float = 1.0,
    distill_weight: float = 0.0,
    distill_type: str = "kl",
    temperature: float = 1.0,
    distill_topk: int = 0,
) -> TrainLoss:
    return TrainLoss(
        lm_weight=lm_weight,
        distill_weight=distill_weight,
        distill_type=distill_type,
        temperature=temperature,
        distill_topk=distill_topk,
    )
