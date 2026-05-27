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

import os
from types import SimpleNamespace

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Seq2SeqTrainingArguments

from ...data.qat_dataset import QATDataset
from ...utils import patch_deepspeed_duplicate_check, print_info
from ..compressor_factory import CompressorFactory
from .trainer import DistillSeq2SeqTrainer


def _normalize_device_map(device_map):
    if isinstance(device_map, str) and device_map.lower() in ("none", "distributed"):
        return None
    return device_map


@CompressorFactory.register
class Distill:
    """Full-precision knowledge distillation.

    Quantized-student distillation lives in ``angelslim.compressor.qad``.
    Keeping this path fp-only prevents it from inheriting QAT state or save
    semantics by accident.
    """

    def __init__(self, model, slim_config=None):
        self.quant_model = model
        self.config = slim_config
        self.distill_config = slim_config["compress_config"].Distill
        self.student_type = getattr(self.distill_config, "student_type", "fp").lower()
        self.trainable_parameters = self.distill_config.trainable_parameters.lower()
        self.save_fmt = self.distill_config.save_format
        self.trainer = SimpleNamespace(external_trainer=None)
        self.teacher_model = None
        self.train_dataset = None

        self._validate_config()

    def _validate_config(self):
        if not self.distill_config.teacher_model_path:
            raise ValueError("Distill requires compression.Distill.teacher_model_path.")
        if self.student_type != "fp":
            raise ValueError("Distill only supports fp students. Use QAD for quantized students.")
        if self.trainable_parameters != "all":
            raise ValueError(
                "Distill trainable_parameters must be 'all'. Use QAD for quant params."
            )

    def _prepare_dataset(self, dataloader):
        if self.distill_config.hf_dataset is not None:
            parts = self.distill_config.hf_dataset.split(",")
            dataset = load_dataset(*parts, cache_dir=self.distill_config.hf_cache_dir)
            self.train_dataset = QATDataset(
                dataset["train"],
                self.quant_model.tokenizer,
                block_size=dataloader.dataset.max_length,
                is_opensource=True,
            )
        else:
            self.train_dataset = QATDataset(dataloader.dataset, self.quant_model.tokenizer)

    def _load_teacher_model(self):
        teacher_device_map = _normalize_device_map(self.distill_config.teacher_device_map)
        kwargs = {
            "torch_dtype": self.distill_config.teacher_torch_dtype,
            "trust_remote_code": self.distill_config.teacher_trust_remote_code,
            "low_cpu_mem_usage": self.distill_config.teacher_low_cpu_mem_usage,
            "use_cache": self.distill_config.teacher_use_cache,
        }
        if teacher_device_map is not None:
            kwargs["device_map"] = teacher_device_map
        if self.distill_config.teacher_cache_dir is not None:
            kwargs["cache_dir"] = self.distill_config.teacher_cache_dir

        print_info(f"Loading teacher model from {self.distill_config.teacher_model_path}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            self.distill_config.teacher_model_path,
            **kwargs,
        )
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        return teacher_model, teacher_device_map is None

    def _apply_trainable_parameters(self):
        model = self.quant_model.model
        for param in model.parameters():
            param.requires_grad = True

    def _init_optimizer(self):
        return None

    def _prepare_trainer(self, place_teacher_on_device):
        optimizer = self._init_optimizer()
        if self.distill_config.hf_args.get("deepspeed") is not None:
            patch_deepspeed_duplicate_check()

        loss_config = {
            "loss_type": self.distill_config.loss_type,
            "loss_topk": self.distill_config.loss_topk,
            "kd_temperature": self.distill_config.kd_temperature,
            "lm_loss_weight": self.distill_config.lm_loss_weight,
            "kd_loss_weight": self.distill_config.kd_loss_weight,
        }
        trainer_kwargs = {}
        if optimizer is not None:
            trainer_kwargs["optimizers"] = (optimizer, None)

        self.trainer.external_trainer = DistillSeq2SeqTrainer(
            model=self.quant_model.model,
            teacher_model=self.teacher_model,
            processing_class=self.quant_model.tokenizer,
            args=Seq2SeqTrainingArguments(
                output_dir=self.config["global_config"].save_path,
                **self.distill_config.hf_args,
            ),
            train_dataset=self.train_dataset,
            eval_dataset=None,
            loss_config=loss_config,
            place_teacher_on_device=place_teacher_on_device,
            **trainer_kwargs,
        )

    def _load_resume_checkpoint(self):
        if self.distill_config.resume_ckpt_dir is None:
            return
        print_info(f"Loading distill resume checkpoint from {self.distill_config.resume_ckpt_dir}")
        save_dict = torch.load(self.distill_config.resume_ckpt_dir, map_location="cpu")
        self.quant_model.model.load_state_dict(save_dict)

    def run(self, dataloader):
        self._prepare_dataset(dataloader)
        self._apply_trainable_parameters()
        self._load_resume_checkpoint()
        self.teacher_model, place_teacher_on_device = self._load_teacher_model()
        self._prepare_trainer(place_teacher_on_device)

        if self.distill_config.do_train:
            self.trainer.external_trainer.train()

    def convert(self):
        return None

    def save(self, save_path: str):
        if self.save_fmt not in ("hf", "real", "full"):
            print_info("Save format not specified, skip save.")
            return None

        output_dir = os.path.join(save_path, "final_checkpoint")
        if self.trainer.external_trainer is not None:
            self.trainer.external_trainer.save_model(output_dir)
        else:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                os.makedirs(output_dir, exist_ok=True)
                self.quant_model.get_model().save_pretrained(output_dir, max_shard_size="5GB")
                self.quant_model.tokenizer.save_pretrained(output_dir)
        return None
