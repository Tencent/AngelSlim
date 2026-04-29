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
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from ....data.qat_dataset import QATDataset
from ....utils import patch_deepspeed_duplicate_check, print_info
from ..plugins.distill_loss import DistillLoss
from ..plugins.learnable_scale import set_quant_state
from .trainer_factory import TrainerFactory


def _unique_named_params(model, predicate):
    """Collect parameters matching ``predicate`` with id-based de-duplication.

    Some QAT setups share a single scale Parameter across multiple
    QuantLinear views (e.g. MoE experts built from a shared tensor). HF /
    DeepSpeed optimizer init rejects duplicates, so we de-dup by ``id``.
    """
    seen = set()
    result = []
    for name, param in model.named_parameters():
        if id(param) in seen or not predicate(name, param):
            continue
        seen.add(id(param))
        result.append(param)
    return result


class QATSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, loss_config=None, quant_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        loss_config = loss_config or {}
        quant_config = quant_config or {}
        self.loss_type = str(loss_config.get("loss_type", "origin")).lower()
        self.loss_topk = loss_config.get("loss_topk")
        self.kd_temperature = float(loss_config.get("kd_temperature", 1.0))
        # ``kd_alpha`` kept for backward compat but IGNORED when
        # ``lm_loss_weight`` / ``kd_loss_weight`` are the (new) source of
        # truth.
        self.kd_alpha = float(loss_config.get("kd_alpha", 0.5))
        self.lm_loss_weight = float(loss_config.get("lm_loss_weight", 1.0))
        self.kd_loss_weight = float(loss_config.get("kd_loss_weight", 0.0))
        self.distill_loss = DistillLoss(
            loss_type=self.loss_type,
            loss_topk=self.loss_topk,
            kd_temperature=self.kd_temperature,
        )
        self.use_weight_quant = quant_config.get("use_weight_quant", False)
        self.use_activation_quant = quant_config.get("use_activation_quant", False)
        self.use_qkv_quant = quant_config.get("use_qkv_quant", False)

        # Running metric aggregator keyed by logger mode.
        from collections import defaultdict

        self._qat_metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def _record(self, name, value):
        if value is None:
            return
        mode = "train" if self.model.training else "eval"
        v = value.detach().float() if isinstance(value, torch.Tensor) else float(value)
        if isinstance(v, torch.Tensor):
            self._qat_metrics[mode][name].append(v.item())
        else:
            self._qat_metrics[mode][name].append(v)

    # ------------------------------------------------------------------
    # compute_loss
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels", None)
        lm_on = self.lm_loss_weight > 0.0
        kd_on = self.kd_loss_weight > 0.0

        # Back-compat: ``loss_type="origin"`` means "pure HF CE loss" → the
        # classic SFT path with no distillation. Honour it even when the
        # user forgot to set kd_loss_weight=0.
        if self.loss_type == "origin":
            kd_on = False

        if not lm_on and not kd_on:
            raise ValueError("Both lm_loss_weight and kd_loss_weight are 0 — nothing to optimise.")

        # Student forward — always needed.
        # HF CausalLM loss is computed when ``labels`` is present in inputs.
        student_inputs = dict(inputs)
        if not lm_on:
            # Still need labels for flat_mask; pop from student kwargs to
            # skip HF's internal CE and save some compute.
            student_inputs.pop("labels", None)
        outputs = model(**student_inputs)

        lm_loss = outputs.loss if lm_on and getattr(outputs, "loss", None) is not None else None
        if lm_on and lm_loss is None:
            raise ValueError(
                "lm_loss_weight > 0 but model did not return a loss — "
                "check that ``labels`` is set in the batch."
            )

        kd_info = None
        if kd_on:
            if labels is None:
                raise ValueError("kd_loss_weight > 0 requires ``labels`` in the batch.")
            teacher_logits = self.get_ori_outputs(model, inputs).logits
            kd_info = self.distill_loss.compute(
                outputs.logits,
                teacher_logits,
                labels,
            )

        # Combine.
        total = outputs.logits.new_zeros(())
        if lm_loss is not None:
            total = total + self.lm_loss_weight * lm_loss
        if kd_info is not None:
            total = total + self.kd_loss_weight * kd_info["loss"]

        # Logging: record every component whose weight is > 0, plus the
        # always-informative forward/backward KL diagnostics when kd is on.
        if lm_on and lm_loss is not None:
            self._record("lm_loss", lm_loss)
        if kd_on and kd_info is not None:
            self._record(f"kd/{self.loss_type}", kd_info["loss"])
            # Diagnostic KL(L/R) for any kd variant. Useful to monitor
            # teacher-student disagreement independent of the combined
            # objective.
            self._record("kd/forward_kl", kd_info["forward_kl"])
            self._record("kd/backward_kl", kd_info["backward_kl"])
        self._record("total_loss", total)

        return (total, outputs) if return_outputs else total

    @torch.no_grad()
    def get_ori_outputs(self, model, inputs):
        teacher_inputs = dict(inputs)
        teacher_inputs.pop("labels", None)
        raw_model = self.accelerator.unwrap_model(model)
        set_quant_state(raw_model, weight_quant=False, act_quant=False, qkv_quant=False)
        try:
            outputs = model(**teacher_inputs)
        finally:
            set_quant_state(
                raw_model,
                weight_quant=self.use_weight_quant,
                act_quant=self.use_activation_quant,
                qkv_quant=self.use_qkv_quant,
            )
        return outputs

    def log(self, logs, start_time=None, *args, **kwargs):
        """Inject running QAT loss components (lm_loss / kd/... / kd/forward_kl
        / kd/backward_kl / total_loss) into HuggingFace Trainer's log dict.

        Each value is averaged across the steps accumulated since the
        previous ``log`` call, then the accumulator is cleared — matching
        HF Trainer's behaviour for its built-in ``loss`` key.
        """
        mode = "train" if self.model.training else "eval"
        bucket = self._qat_metrics.get(mode, {})
        if bucket:
            for key, vals in bucket.items():
                if not vals:
                    continue
                avg = sum(vals) / len(vals)
                out_key = key if mode == "train" else f"eval_{key}"
                logs[out_key] = float(avg)
            bucket.clear()

        # Forward to HF Trainer's log. Signature differs across versions.
        try:
            return super().log(logs, start_time, *args, **kwargs)
        except TypeError:
            return super().log(logs)


@TrainerFactory.register("end2end")
class End2EndTrainer:

    def __init__(self, quant_model, config, plugin_manager):
        self.quant_model = quant_model
        self.config = config
        self.plugin_manager = plugin_manager
        self.training_mode = config["compress_config"].QAT.training_mode
        self.dist_mode = config["compress_config"].QAT.dist_mode
        self.hf_dataset = config["compress_config"].QAT.hf_dataset
        self.hf_cache_dir = config["compress_config"].QAT.hf_cache_dir
        self.resume_ckpt_dir = config["compress_config"].QAT.resume_ckpt_dir
        self.do_train = config["compress_config"].QAT.do_train
        self.external_trainer = None

        self.loss_config = {
            "loss_type": config["compress_config"].QAT.loss_type,
            "loss_topk": config["compress_config"].QAT.loss_topk,
            "kd_temperature": config["compress_config"].QAT.kd_temperature,
            "kd_alpha": config["compress_config"].QAT.kd_alpha,
            "lm_loss_weight": config["compress_config"].QAT.lm_loss_weight,
            "kd_loss_weight": config["compress_config"].QAT.kd_loss_weight,
        }
        self.quant_config = {
            "use_weight_quant": config["compress_config"]
            .QAT.plugin_config.get("quant_config", {})
            .get("use_weight_quant", False),
            "use_activation_quant": config["compress_config"]
            .QAT.plugin_config.get("quant_config", {})
            .get("use_activation_quant", False),
            "use_qkv_quant": config["compress_config"]
            .QAT.plugin_config.get("quant_config", {})
            .get("use_qkv_quant", False),
        }

    def _init_optimizer(self):
        lr = float(self.config["compress_config"].QAT.hf_args.get("learning_rate", 1e-5))
        wd = float(self.config["compress_config"].QAT.hf_args.get("weight_decay", 0))
        scale_params = _unique_named_params(
            self.quant_model.model,
            lambda n, p: p.requires_grad and ("scale" in n or "zero_point" in n),
        )
        params = [
            {
                "params": scale_params,
                "weight_decay": wd,
                "lr": lr,
            }
        ]

        enable_lwc = (
            self.config["compress_config"]
            .QAT.plugin_config.get("quant_config", {})
            .get("lwc", {})
            .get("enable_lwc", False)
        )
        lwc_param_count = 0
        if enable_lwc:
            lwc_lr = float(
                self.config["compress_config"]
                .QAT.plugin_config.get("quant_config", {})
                .get("lwc", {})
                .get("lwc_lr", 1e-1)
            )
            lwc_params = _unique_named_params(
                self.quant_model.model,
                lambda n, p: p.requires_grad
                and ("clip_factor_w_max" in n or "clip_factor_w_min" in n),
            )
            lwc_param_count = len(lwc_params)
            params.append({"params": lwc_params, "weight_decay": wd, "lr": lwc_lr})

        if not any(group["params"] for group in params):
            raise ValueError("QAT optimizer has no trainable parameters.")

        self.optimizer = torch.optim.AdamW(params)
        if enable_lwc:
            print_info(
                f"Init optimizer with {len(scale_params)} scale params, "
                f"{lwc_param_count} lwc params, lr={lr} lwc_lr={lwc_lr} weight_decay={wd}"
            )
        else:
            print_info(
                f"Init optimizer with {len(scale_params)} scale params, "
                f"lr={lr} weight_decay={wd}"
            )

    def prepare_trainer(self):
        if self.training_mode == "blockwise":
            return
        if self.training_mode == "end2end" and self.dist_mode == "hf":
            self._init_optimizer()
            # When DeepSpeed is used, neutralize its duplicate-parameter
            # check: it rejects param-groups that share tensors, which our
            # scale/zero_point setup can legally have (shared tensors
            # across views). Idempotent and a no-op if deepspeed is absent.
            if self.config["compress_config"].QAT.hf_args.get("deepspeed") is not None:
                patch_deepspeed_duplicate_check()
            self.external_trainer = QATSeq2SeqTrainer(
                model=self.quant_model.model,
                processing_class=self.quant_model.tokenizer,
                args=Seq2SeqTrainingArguments(
                    output_dir=self.config["global_config"].save_path,
                    **self.config["compress_config"].QAT.hf_args,
                ),
                train_dataset=self.train_dataset,
                eval_dataset=None,
                optimizers=(self.optimizer, None),
                loss_config=self.loss_config,
                quant_config=self.quant_config,
            )
        else:
            raise NotImplementedError(f"Unsupported distribution mode: {self.dist_mode}")

    def prepare_dataset(self, dataloader):
        if self.hf_dataset is not None:
            parts = self.hf_dataset.split(",")
            dataset = load_dataset(*parts, cache_dir=self.hf_cache_dir)
            self.train_dataset = QATDataset(
                dataset["train"],
                self.quant_model.tokenizer,
                block_size=dataloader.dataset.max_length,
                is_opensource=True,
            )
        else:
            self.train_dataset = QATDataset(dataloader.dataset, self.quant_model.tokenizer)

    def run(self, dataloader):
        self.prepare_dataset(dataloader)
        self.plugin_manager.call_before_train(train_dataset=self.train_dataset)
        self.prepare_trainer()

        if self.resume_ckpt_dir is not None:
            print_info(f"Loading from resume {self.resume_ckpt_dir}")
            save_dict = torch.load(self.resume_ckpt_dir, map_location="cpu")
            self.quant_model.model.load_state_dict(save_dict)

        if self.do_train:
            if self.external_trainer is not None:
                self.external_trainer.train()
            else:
                self.train()

        self.plugin_manager.call_after_train()
