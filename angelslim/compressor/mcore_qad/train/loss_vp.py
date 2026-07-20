"""Vocab-parallel loss: LM + distillation on TP-SHARDED logits (no full-vocab gather).

For long context (256k) the gathered full-vocab logits ([seq, vocab]) dominate memory
(tens of GB). Here logits stay sharded across tensor-parallel ranks ([.., vocab/tp]) and
the loss is computed with TP all-reduces:
  * LM    : mcore vocab_parallel_cross_entropy (online softmax across TP).
  * distill: vocab-parallel log-softmax (all-reduce max + sum-exp) and KL (all-reduce the
             per-token vocab sum). cakld's teacher P(gold) is gathered across TP.

Same interface/semantics as train.loss.TrainLoss: __call__ -> (loss, n_tokens, metrics),
targets are pre-shifted next-token ids (-100 = ignore).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel import vocab_parallel_cross_entropy
from torch import Tensor

IGNORE_INDEX = -100


def _tp():
    return ps.get_tensor_model_parallel_group()


def _vp_log_softmax(x: Tensor) -> Tensor:
    """log-softmax over the vocab dim that is sharded across TP. x: [N, V/tp] fp32."""
    g = _tp()
    m = x.max(dim=-1, keepdim=True).values
    dist.all_reduce(m, op=dist.ReduceOp.MAX, group=g)
    x = x - m
    Z = x.exp().sum(dim=-1, keepdim=True)
    dist.all_reduce(Z, op=dist.ReduceOp.SUM, group=g)
    return x - Z.log()


def _vp_kl(log_p_src: Tensor, log_p_tgt: Tensor) -> Tensor:
    """Per-token KL(tgt || src) with the vocab sum reduced across TP. -> [N]."""
    s = (log_p_tgt.exp() * (log_p_tgt - log_p_src)).sum(dim=-1)
    dist.all_reduce(s, op=dist.ReduceOp.SUM, group=_tp())
    return s


def _vp_gold_prob(log_p_tgt: Tensor, gold: Tensor) -> Tensor:
    """Teacher P(gold) when the vocab is TP-sharded: pick the logprob from whichever rank
    owns the gold id (others contribute -inf), then exp. -> [N]."""
    g = _tp()
    vtp = log_p_tgt.shape[-1]
    off = ps.get_tensor_model_parallel_rank() * vtp
    local = gold - off
    inrange = (local >= 0) & (local < vtp)
    lp = log_p_tgt.gather(-1, local.clamp(0, vtp - 1).unsqueeze(-1)).squeeze(-1)
    lp = torch.where(inrange, lp, torch.full_like(lp, float("-inf")))
    dist.all_reduce(lp, op=dist.ReduceOp.MAX, group=g)  # gold lives on exactly one rank
    return lp.exp()


class VocabParallelLoss:
    def __init__(
        self,
        *,
        lm_weight: float = 1.0,
        distill_weight: float = 0.0,
        distill_type: str = "kl",
        temperature: float = 1.0,
        distill_topk: int = 0,
    ) -> None:
        # Under TP>1 the vocab is already sharded (V/tp) for BOTH lm and distill, so memory
        # is bounded without top-k; a correct top-k here would need a cross-rank global top-k
        # merge. Not worth it -- pick one: top-k (TP=1) OR vocab-parallel (TP>1).
        if distill_topk:
            raise NotImplementedError(
                "distill_topk is TP=1 only; under tensor parallelism the vocab-parallel loss "
                "already bounds memory to V/tp -- run with --distill-topk 0."
            )
        self.lm_weight = float(lm_weight)
        self.distill_weight = float(distill_weight)
        self.distill_type = distill_type.lower()
        self.temperature = float(temperature)

    @property
    def needs_teacher(self) -> bool:
        return self.distill_weight > 0.0

    _SEL = {"kl": "fkl", "rkl": "rkl", "cakld": "cakld"}

    def _distill_all(self, s: Tensor, t: Tensor, valid: Tensor) -> dict:
        T = max(self.temperature, 1e-6)
        s_lp, t_lp = _vp_log_softmax(s / T), _vp_log_softmax(t / T)
        scale = T * T
        fkl_pt, rkl_pt = _vp_kl(s_lp, t_lp), _vp_kl(t_lp, s_lp)
        conf = _vp_gold_prob(t_lp, valid)
        cakld = (conf * rkl_pt + (1.0 - conf) * fkl_pt).mean() * scale
        return {"fkl": fkl_pt.mean() * scale, "rkl": rkl_pt.mean() * scale, "cakld": cakld}

    def __call__(self, student_logits: Tensor, teacher_logits, targets: Tensor):
        """student_logits/teacher_logits: [b, s, V/tp] (TP-sharded). targets: [b, s].
        Returns (loss, n_tokens, metrics) like the dense loss."""
        vtp = student_logits.shape[-1]
        mask = (targets != IGNORE_INDEX).reshape(-1)
        n_tokens = int(mask.sum())
        if n_tokens == 0:
            return student_logits.sum() * 0.0, 0, {}

        sl = student_logits.transpose(0, 1).contiguous().float()  # [s, b, V/tp]
        tg = targets.transpose(0, 1).contiguous().clamp_min(0)  # [s, b] (ignore -> 0)
        lm = vocab_parallel_cross_entropy(sl, tg).transpose(0, 1).reshape(-1)[mask].mean()
        metrics = {"lm": lm.detach()}
        total = self.lm_weight * lm if self.lm_weight != 0.0 else student_logits.new_zeros(())
        if self.needs_teacher:
            sV = student_logits.reshape(-1, vtp)[mask].float()
            tV = teacher_logits.reshape(-1, vtp)[mask].float()
            valid = targets.reshape(-1)[mask]
            d = self._distill_all(sV, tV, valid)
            metrics.update({k: v.detach() for k, v in d.items()})
            total = total + self.distill_weight * d[self._SEL[self.distill_type]]
        return total, n_tokens, metrics
