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

import torch
import torch.nn.functional as F


class DistillLoss:
    def __init__(self, loss_type="kl", loss_topk=None, kd_temperature=1.0):
        self.loss_type = str(loss_type).lower()
        self.loss_topk = loss_topk
        self.kd_temperature = float(kd_temperature)

    @staticmethod
    def _kl_from_logps(log_p_src: torch.Tensor, log_p_tgt: torch.Tensor) -> torch.Tensor:
        """Per-token KL(tgt || src) computed from log-probs in a gradient-safe way."""
        p_tgt = log_p_tgt.exp()
        return (p_tgt * (log_p_tgt - log_p_src)).sum(dim=-1)

    def compute(self, student_logits, teacher_logits, labels):
        """Return per-token KD losses computed only on valid labels.

        The returned dict always contains ``loss``, ``forward_kl`` and
        ``backward_kl`` so callers can log diagnostics for every KD variant.
        """
        # CausalLM loss shifts labels internally: logits at position t predict
        # labels at position t + 1. Match that alignment for KD as well.
        student_logits = student_logits[..., :-1, :].contiguous()
        teacher_logits = teacher_logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        flat_mask = (labels != -100).reshape(-1)
        if flat_mask.sum() == 0:
            zero = student_logits.new_zeros(())
            return {"loss": zero, "forward_kl": zero, "backward_kl": zero}

        # Flatten to [N, V] and keep only valid tokens. Compute KD in fp32:
        # bf16/fp16 full-vocab probabilities can underflow to exact zeros, and
        # KL gradients through prob targets can then produce Inf/NaN gradients.
        s_flat = student_logits.flatten(0, -2)[flat_mask].float()
        t_flat = teacher_logits.flatten(0, -2)[flat_mask].float()
        valid_labels = labels.reshape(-1)[flat_mask]

        # Diagnostic KL (always computed at T=1 on valid tokens).
        s_logp = F.log_softmax(s_flat, dim=-1)
        t_logp = F.log_softmax(t_flat, dim=-1)
        t_p = t_logp.exp()
        forward_kl = self._kl_from_logps(s_logp, t_logp).mean()
        backward_kl = self._kl_from_logps(t_logp, s_logp).mean()

        # Main KD loss according to loss_type.
        if self.loss_type == "kl":
            kd = forward_kl
        elif self.loss_type == "rkl":
            kd = backward_kl
        elif self.loss_type == "mse":
            kd = F.mse_loss(s_flat, t_flat)
        elif self.loss_type == "kd":
            # Legacy "kd": temperature-scaled forward KL. The outer trainer
            # combines this value with the LM loss by configured weights.
            temperature = max(self.kd_temperature, 1e-6)
            s_temp_logp = F.log_softmax(s_flat / temperature, dim=-1)
            t_temp_logp = F.log_softmax(t_flat / temperature, dim=-1)
            kd = self._kl_from_logps(s_temp_logp, t_temp_logp).mean() * (temperature * temperature)
        elif self.loss_type == "cakld":
            # Contextual Asymmetric KL-divergence: per-token mixing of
            # forward / reverse KL by teacher's confidence on the label.
            per_tok_fkl = self._kl_from_logps(s_logp, t_logp)
            per_tok_bkl = self._kl_from_logps(t_logp, s_logp)
            conf = torch.gather(t_p, dim=-1, index=valid_labels.unsqueeze(-1)).squeeze(-1)
            kd = (conf * per_tok_bkl + (1.0 - conf) * per_tok_fkl).mean()
        elif self._is_reverse_topk_loss(self.loss_type):
            topk = self._resolve_topk(self.loss_type, s_flat.size(-1))
            top_s, idx = s_flat.topk(topk, dim=-1, sorted=False)
            top_t = t_flat.gather(-1, idx)
            top_s_logp = F.log_softmax(top_s, dim=-1)
            top_t_logp = F.log_softmax(top_t, dim=-1)
            kd = self._kl_from_logps(top_t_logp, top_s_logp).mean()
        elif self._is_forward_topk_loss(self.loss_type):
            topk = self._resolve_topk(self.loss_type, t_flat.size(-1))
            top_t, idx = t_flat.topk(topk, dim=-1, sorted=False)
            top_s = s_flat.gather(-1, idx)
            top_s_logp = F.log_softmax(top_s, dim=-1)
            top_t_logp = F.log_softmax(top_t, dim=-1)
            kd = self._kl_from_logps(top_s_logp, top_t_logp).mean()
        else:
            raise ValueError(
                f"Unsupported QAT kd loss_type: {self.loss_type}. "
                "Valid: kl, rkl, mse, kd, cakld, kl_top[_K], r_kl_top[_K]."
            )

        return {"loss": kd, "forward_kl": forward_kl, "backward_kl": backward_kl}

    @staticmethod
    def _is_forward_topk_loss(loss_type):
        return loss_type.startswith("kl_top")

    @staticmethod
    def _is_reverse_topk_loss(loss_type):
        return loss_type.startswith("r_kl_top") or loss_type.startswith("rkl_top")

    def _resolve_topk(self, loss_type, vocab_size):
        topk = self.loss_topk
        if topk is None and "_top_" in loss_type:
            topk = int(loss_type.rsplit("_", 1)[-1])
        if topk is None:
            topk = 1000
        topk = int(topk)
        if topk <= 0:
            raise ValueError(f"loss_topk must be positive, got: {topk}")
        return min(topk, vocab_size)
