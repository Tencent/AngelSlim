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

from types import SimpleNamespace

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Seq2SeqTrainingArguments

from ...data.qat_dataset import QATDataset
from ...utils import patch_deepspeed_duplicate_check, print_info
from ..compressor_factory import CompressorFactory
from ..distill.trainer import DistillSeq2SeqTrainer
from ..qat.plugins import PluginManager
from ..qat.qat import QAT


def _unique_named_params(model, predicate):
    seen = set()
    result = []
    for name, param in model.named_parameters():
        if id(param) in seen or not predicate(name, param):
            continue
        seen.add(id(param))
        result.append(param)
    return result


def _normalize_device_map(device_map):
    if isinstance(device_map, str) and device_map.lower() in ("none", "distributed"):
        return None
    return device_map


@CompressorFactory.register
class QAD(QAT):
    """Quantization-aware distillation.

    QAD keeps the quantized-student path separate from fp-only distillation
    while still reusing QAT conversion, plugins, and save logic.
    """

    def __init__(self, model, slim_config=None):
        self.quant_model = model
        self.config = slim_config
        self.qad_config = slim_config["compress_config"].QAD
        self.student_type = getattr(self.qad_config, "student_type", "quantized").lower()
        self.trainable_parameters = self.qad_config.trainable_parameters.lower()
        self.save_fmt = self.qad_config.save_format
        self.plugin_config = self.qad_config.plugin_config
        self.plugin_manager = PluginManager()
        self.trainer = SimpleNamespace(external_trainer=None)
        self._rank0_state_dict = None
        self.teacher_model = None
        self.train_dataset = None

        self._validate_config()
        self.quant_model.init_ptq(slim_config)
        self.quant_info = self.quant_model.quant_config
        self._init_plugins()

    def _validate_config(self):
        if not self.qad_config.teacher_model_path:
            raise ValueError("QAD requires compression.QAD.teacher_model_path.")
        if self.student_type != "quantized":
            raise ValueError("QAD only supports a quantized student. Use Distill for fp students.")
        if self.trainable_parameters not in ("all", "quant"):
            raise ValueError("QAD trainable_parameters must be 'all' or 'quant'.")

    def _init_plugins(self):
        if self.plugin_config.get("enable_scale", False):
            self.plugin_manager.register_plugin(
                "learnable_scale",
                quant_info=self.quant_info,
                ignore_layers=self.config["compress_config"].quantization.ignore_layers,
                resume_ckpt_dir=self.qad_config.resume_ckpt_dir,
                from_ptq_ckpt_dir=self.qad_config.from_ptq_ckpt,
                config=self.plugin_config.get("quant_config", {}),
                quant_model=self.quant_model,
            )

    def _prepare_dataset(self, dataloader):
        if self.qad_config.hf_dataset is not None:
            parts = self.qad_config.hf_dataset.split(",")
            dataset = load_dataset(*parts, cache_dir=self.qad_config.hf_cache_dir)
            self.train_dataset = QATDataset(
                dataset["train"],
                self.quant_model.tokenizer,
                block_size=dataloader.dataset.max_length,
                is_opensource=True,
            )
        else:
            self.train_dataset = QATDataset(dataloader.dataset, self.quant_model.tokenizer)

    def _load_teacher_model(self):
        teacher_device_map = _normalize_device_map(self.qad_config.teacher_device_map)
        kwargs = {
            "torch_dtype": self.qad_config.teacher_torch_dtype,
            "trust_remote_code": self.qad_config.teacher_trust_remote_code,
            "low_cpu_mem_usage": self.qad_config.teacher_low_cpu_mem_usage,
            "use_cache": self.qad_config.teacher_use_cache,
        }
        if teacher_device_map is not None:
            kwargs["device_map"] = teacher_device_map
        if self.qad_config.teacher_cache_dir is not None:
            kwargs["cache_dir"] = self.qad_config.teacher_cache_dir

        print_info(f"Loading QAD teacher model from {self.qad_config.teacher_model_path}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            self.qad_config.teacher_model_path,
            **kwargs,
        )
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        return teacher_model, teacher_device_map is None

    def _apply_trainable_parameters(self):
        model = self.quant_model.model
        if self.trainable_parameters == "all":
            for param in model.parameters():
                param.requires_grad = True
            return

        if not any(param.requires_grad for param in model.parameters()):
            raise ValueError("QAD quant optimizer has no trainable parameters.")

    def _init_optimizer(self):
        if self.trainable_parameters == "all":
            return None

        lr = float(self.qad_config.hf_args.get("learning_rate", 1e-5))
        wd = float(self.qad_config.hf_args.get("weight_decay", 0))
        lwc_names = ("clip_factor_w_max", "clip_factor_w_min")
        base_params = _unique_named_params(
            self.quant_model.model,
            lambda n, p: p.requires_grad and not any(key in n for key in lwc_names),
        )
        params = [{"params": base_params, "weight_decay": wd, "lr": lr}]

        lwc_params = _unique_named_params(
            self.quant_model.model,
            lambda n, p: p.requires_grad and any(key in n for key in lwc_names),
        )
        if lwc_params:
            lwc_lr = float(
                self.plugin_config.get("quant_config", {}).get("lwc", {}).get("lwc_lr", lr)
            )
            params.append({"params": lwc_params, "weight_decay": wd, "lr": lwc_lr})
            print_info(
                f"Init QAD optimizer with {len(base_params)} params, "
                f"{len(lwc_params)} lwc params, lr={lr}, lwc_lr={lwc_lr}, weight_decay={wd}"
            )
        else:
            print_info(
                f"Init QAD optimizer with {len(base_params)} params, "
                f"lr={lr}, weight_decay={wd}"
            )

        if not any(group["params"] for group in params):
            raise ValueError("QAD optimizer has no trainable parameters.")
        return torch.optim.AdamW(params)

    def _prepare_trainer(self, place_teacher_on_device):
        optimizer = self._init_optimizer()
        if self.qad_config.hf_args.get("deepspeed") is not None:
            patch_deepspeed_duplicate_check()

        loss_config = {
            "loss_type": self.qad_config.loss_type,
            "loss_topk": self.qad_config.loss_topk,
            "kd_temperature": self.qad_config.kd_temperature,
            "lm_loss_weight": self.qad_config.lm_loss_weight,
            "kd_loss_weight": self.qad_config.kd_loss_weight,
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
                **self.qad_config.hf_args,
            ),
            train_dataset=self.train_dataset,
            eval_dataset=None,
            loss_config=loss_config,
            place_teacher_on_device=place_teacher_on_device,
            **trainer_kwargs,
        )

    def _load_resume_checkpoint(self):
        if self.qad_config.resume_ckpt_dir is None:
            return
        print_info(f"Loading QAD resume checkpoint from {self.qad_config.resume_ckpt_dir}")
        save_dict = torch.load(self.qad_config.resume_ckpt_dir, map_location="cpu")
        self.quant_model.model.load_state_dict(save_dict)

    def run(self, dataloader):
        self._prepare_dataset(dataloader)
        self.plugin_manager.call_before_train(train_dataset=self.train_dataset)
        self._apply_trainable_parameters()
        self._load_resume_checkpoint()
        self.teacher_model, place_teacher_on_device = self._load_teacher_model()
        self._prepare_trainer(place_teacher_on_device)

        if self.qad_config.do_train:
            self.trainer.external_trainer.train()

        self.plugin_manager.call_after_train()

    def convert(self):
        super().convert()

    def save(self, save_path: str):
        return super().save(save_path)
