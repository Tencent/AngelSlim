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

from collections import defaultdict

import torch
from transformers import Seq2SeqTrainer

from ..qat.plugins.distill_loss import DistillLoss


class DistillSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        teacher_model=None,
        loss_config=None,
        place_teacher_on_device=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if teacher_model is None:
            raise ValueError("DistillSeq2SeqTrainer requires a teacher_model.")

        loss_config = loss_config or {}
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        if place_teacher_on_device:
            self.teacher_model.to(self.args.device)

        self.loss_type = str(loss_config.get("loss_type", "kd")).lower()
        self.loss_topk = loss_config.get("loss_topk")
        self.kd_temperature = float(loss_config.get("kd_temperature", 1.0))
        self.lm_loss_weight = float(loss_config.get("lm_loss_weight", 1.0))
        self.kd_loss_weight = float(loss_config.get("kd_loss_weight", 1.0))
        self.distill_loss = DistillLoss(
            loss_type=self.loss_type,
            loss_topk=self.loss_topk,
            kd_temperature=self.kd_temperature,
        )
        self._distill_metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def _record(self, name, value):
        if value is None:
            return
        mode = "train" if self.model.training else "eval"
        if isinstance(value, torch.Tensor):
            value = value.detach().float()
            if value.dim() == 0:
                value = value[None]
            try:
                value = self.accelerator.gather_for_metrics(value)
            except Exception:
                pass
            value = value.mean().item()
        else:
            value = float(value)
        self._distill_metrics[mode][name].append(value)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels", None)
        lm_on = self.lm_loss_weight > 0.0
        kd_on = self.kd_loss_weight > 0.0 and self.loss_type != "origin"

        if not lm_on and not kd_on:
            raise ValueError("Both lm_loss_weight and kd_loss_weight are 0; nothing to optimize.")

        student_inputs = dict(inputs)
        if not lm_on:
            student_inputs.pop("labels", None)
        student_outputs = model(**student_inputs)

        lm_loss = (
            student_outputs.loss
            if lm_on and getattr(student_outputs, "loss", None) is not None
            else None
        )
        if lm_on and lm_loss is None:
            raise ValueError(
                "lm_loss_weight > 0 but model did not return a loss. "
                "Check that labels are set in the batch."
            )

        kd_info = None
        if kd_on:
            if labels is None:
                raise ValueError("kd_loss_weight > 0 requires labels in the batch.")
            teacher_inputs = dict(inputs)
            teacher_inputs.pop("labels", None)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**teacher_inputs)
            kd_info = self.distill_loss.compute(
                student_outputs.logits,
                teacher_outputs.logits,
                labels,
            )

        total = student_outputs.logits.new_zeros(())
        if lm_loss is not None:
            total = total + self.lm_loss_weight * lm_loss
        if kd_info is not None:
            total = total + self.kd_loss_weight * kd_info["loss"]

        if lm_on and lm_loss is not None:
            self._record("lm_loss", lm_loss)
        if kd_on and kd_info is not None:
            self._record(f"kd/{self.loss_type}", kd_info["loss"])
            self._record("kd/forward_kl", kd_info["forward_kl"])
            self._record("kd/backward_kl", kd_info["backward_kl"])
        self._record("total_loss", total)

        return (total, student_outputs) if return_outputs else total

    def log(self, logs, start_time=None, *args, **kwargs):
        mode = "train" if self.model.training else "eval"
        bucket = self._distill_metrics.get(mode, {})
        if bucket:
            for key, values in bucket.items():
                if not values:
                    continue
                out_key = key if mode == "train" else f"eval_{key}"
                logs[out_key] = float(sum(values) / len(values))
            bucket.clear()
        if mode == "train" and "total_loss" in logs:
            logs["loss"] = logs["total_loss"]
        elif mode != "train" and "eval_total_loss" in logs:
            logs["eval_loss"] = logs["eval_total_loss"]

        try:
            return super().log(logs, start_time, *args, **kwargs)
        except TypeError:
            return super().log(logs)
