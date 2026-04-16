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
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from ....data.qat_dataset import QATDataset
from ....utils import print_info
from ..plugins.learnable_scale import set_quant_state
from .trainer_factory import TrainerFactory


class QATSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, loss_config=None, quant_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        loss_config = loss_config or {}
        quant_config = quant_config or {}
        self.loss_type = str(loss_config.get("loss_type", "origin")).lower()
        self.loss_topk = loss_config.get("loss_topk")
        self.kd_temperature = float(loss_config.get("kd_temperature", 1.0))
        self.kd_alpha = float(loss_config.get("kd_alpha", 0.5))
        self.use_weight_quant = quant_config.get("use_weight_quant", False)
        self.use_activation_quant = quant_config.get("use_activation_quant", False)
        self.use_qkv_quant = quant_config.get("use_qkv_quant", False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.loss_type == "origin":
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        teacher_logits = self.get_ori_outputs(model, inputs).logits
        student_inputs = dict(inputs)
        if self.loss_type != "kd":
            student_inputs.pop("labels", None)
        outputs = model(**student_inputs)
        student_logits = outputs.logits

        if self.loss_type == "kl":
            loss = F.kl_div(
                F.log_softmax(student_logits.flatten(0, -2), dim=-1),
                F.softmax(teacher_logits.flatten(0, -2), dim=-1),
                reduction="batchmean",
            )
        elif self.loss_type == "rkl":
            loss = F.kl_div(
                F.log_softmax(teacher_logits.flatten(0, -2), dim=-1),
                F.softmax(student_logits.flatten(0, -2), dim=-1),
                reduction="batchmean",
            )
        elif self.loss_type == "mse":
            loss = F.mse_loss(student_logits, teacher_logits)
        elif self.loss_type == "kd":
            if getattr(outputs, "loss", None) is None:
                raise ValueError("loss_type='kd' requires labels to compute CE loss.")
            temperature = max(self.kd_temperature, 1e-6)
            alpha = self.kd_alpha
            distill_loss = F.kl_div(
                F.log_softmax(student_logits.flatten(0, -2) / temperature, dim=-1),
                F.softmax(teacher_logits.flatten(0, -2) / temperature, dim=-1),
                reduction="batchmean",
            )
            loss = outputs.loss * (1 - alpha) + distill_loss * (alpha * temperature * temperature)
        elif self._is_reverse_topk_loss(self.loss_type):
            topk = self._resolve_topk(self.loss_type, student_logits.size(-1))
            top_student_logits, indices = student_logits.topk(topk, dim=-1, sorted=False)
            top_teacher_logits = teacher_logits.gather(-1, indices)
            loss = F.kl_div(
                F.log_softmax(top_teacher_logits.flatten(0, -2), dim=-1),
                F.softmax(top_student_logits.flatten(0, -2), dim=-1),
                reduction="batchmean",
            )
        elif self._is_forward_topk_loss(self.loss_type):
            topk = self._resolve_topk(self.loss_type, teacher_logits.size(-1))
            top_teacher_logits, indices = teacher_logits.topk(topk, dim=-1, sorted=False)
            top_student_logits = student_logits.gather(-1, indices)
            loss = F.kl_div(
                F.log_softmax(top_student_logits.flatten(0, -2), dim=-1),
                F.softmax(top_teacher_logits.flatten(0, -2), dim=-1),
                reduction="batchmean",
            )
        else:
            raise ValueError(f"Unsupported QAT loss_type: {self.loss_type}")

        return (loss, outputs) if return_outputs else loss

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
        params = [
            {
                "params": [
                    p
                    for n, p in self.quant_model.model.named_parameters()
                    if "scale" in n or "zero_point" in n
                ],
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
        if enable_lwc:
            lwc_lr = float(
                self.config["compress_config"]
                .QAT.plugin_config.get("quant_config", {})
                .get("lwc", {})
                .get("lwc_lr", 1e-1)
            )
            lwc_params = [
                {
                    "params": [
                        p
                        for n, p in self.quant_model.model.named_parameters()
                        if "clip_factor_w_max" in n or "clip_factor_w_min" in n
                    ],
                    "weight_decay": wd,
                    "lr": lwc_lr,
                }
            ]
            params.extend(lwc_params)

        self.optimizer = torch.optim.AdamW(params)
        if enable_lwc:
            print_info(f"Init optimizer with learnable lr={lr} lwc_lr={lwc_lr} weight_decay={wd}")
        else:
            print_info(f"Init optimizer with learnable lr={lr} weight_decay={wd}")

    def prepare_trainer(self):
        if self.training_mode == "blockwise":
            return
        if self.training_mode == "end2end" and self.dist_mode == "hf":
            self._init_optimizer()
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
