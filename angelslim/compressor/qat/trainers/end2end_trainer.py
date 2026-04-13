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
from ....utils import print_info
from .trainer_factory import TrainerFactory


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

    def prepare_trainer(self):
        if self.training_mode == "blockwise":
            return
        if self.training_mode == "end2end" and self.dist_mode == "hf":
            self.external_trainer = Seq2SeqTrainer(
                model=self.quant_model.model,
                processing_class=self.quant_model.tokenizer,
                args=Seq2SeqTrainingArguments(
                    output_dir=self.config["global_config"].save_path,
                    **self.config["compress_config"].QAT.hf_args,
                ),
                train_dataset=self.train_dataset,
                eval_dataset=None,
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
        self.prepare_trainer()
        self.plugin_manager.call_before_train(train_dataset=self.train_dataset)

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
