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

import torch
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from ....utils import print_info
from ..modules.dataset import QATDataset
from .trainer_factory import TrainerFactory


@TrainerFactory.register("end2end")
class End2EndTrainer:

    def __init__(self, quant_model, config, plugin_manager):
        self.quant_model = quant_model
        self.config = config
        self.plugin_manager = plugin_manager
        self.tc = self.config["training_config"]

    def _get_hf_arg(self, key, default=None):
        hf_args = getattr(self.config.get("training_config", None), "hf_args", None)
        if isinstance(hf_args, dict) and key in hf_args:
            return hf_args[key]
        return default

    def _init_optimizer(self):
        lr = float(self._get_hf_arg("learning_rate", 1e-5))
        wd = float(self._get_hf_arg("weight_decay", 0.0))
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
        self.optimizer = torch.optim.AdamW(params)
        print_info(f"Init optimizer with lr={lr} weight_decay={wd}")

    def prepare_trainer(self):
        if self.tc.training_mode != "end2end":
            self.external_trainer = None
            return
        if self.tc.dist_mode == "hf":
            self._init_optimizer()
            self.external_trainer = Seq2SeqTrainer(
                model=self.quant_model.model,
                tokenizer=self.quant_model.tokenizer,
                args=Seq2SeqTrainingArguments(**self.tc.hf_args),
                train_dataset=self.train_dataset,
                eval_dataset=None,
                optimizers=(self.optimizer, None),
            )
        else:
            raise NotImplementedError(f"Unsupported distribution mode: {self.tc.dist_mode}")

    def prepare_dataset(self, dataloader):
        if self.tc.hf_dataset:
            parts = self.tc.hf_dataset.split(",")
            dataset = load_dataset(*parts, cache_dir=self.tc.cache_dir)
            self.train_dataset = QATDataset(
                dataset["train"],
                self.quant_model.tokenizer,
                block_size=min(self.tc.max_length, 2048),
                is_opensource=True,
            )
        else:
            self.train_dataset = QATDataset(dataloader.dataset, self.quant_model.tokenizer)

    def run(self, dataloader):
        self.prepare_dataset(dataloader)
        self.prepare_trainer()
        self.plugin_manager.call_before_train(train_dataset=self.train_dataset)

        resume_path = self.tc.resume_ckpt_dir
        if os.path.isfile(resume_path):
            print_info(f"Loading from resume {resume_path}")
            save_dict = torch.load(resume_path, map_location="cpu")
            self.quant_model.model.load_state_dict(save_dict)

        if self.tc.do_train:
            if self.external_trainer is not None:
                self.external_trainer.train()
            else:
                self.train()
            self.plugin_manager.call_after_train()
