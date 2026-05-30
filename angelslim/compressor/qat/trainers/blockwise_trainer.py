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

import gc
import math
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from ....data.qat_dataset import BlockTrainDataset
from ....utils import print_info
from ..modules.scaler import NativeScalerWithGradNormCount
from ..plugins.learnable_scale import (
    quant_inplace,
    quant_parameters,
    set_quant_parameters,
    set_quant_state,
    set_weight_parameters,
    trainable_parameters,
    weight_parameters,
)
from .end2end_trainer import End2EndTrainer
from .trainer_factory import TrainerFactory

LOSS_AVG_WINDOW = 64


class _Catcher(nn.Module):
    def __init__(self, module, dataset=None):
        super().__init__()
        self.module = module
        self.dataset = dataset
        self.index = 0
        self.layer_kwargs = {}
        # for qwen3
        if hasattr(module, "attention_type"):
            self.attention_type = module.attention_type

    def forward(self, inp, **kwargs):
        if self.dataset is not None:
            self.dataset.update_data(self.index, inp.squeeze(0).to("cpu"))
        self.index += 1
        self.layer_kwargs.update(kwargs)
        raise ValueError


@TrainerFactory.register("blockwise")
class BlockwiseTrainer(End2EndTrainer):
    def __init__(self, quant_model, config, plugin_manager):
        super().__init__(quant_model, config, plugin_manager)
        self.use_act_quant = (
            self.config["compress_config"]
            .QAT.plugin_config.get("quant_config", {})
            .get("use_act_quant", False)
        )
        bc = self.config["compress_config"].QAT.block_wise_config
        self.args = type(
            "Args",
            (),
            {
                "epochs": bc.get("epochs", 20),
                "batch_size": bc.get("batch_size", 1),
                "train_size": bc.get("train_size", 128),
                "val_size": bc.get("val_size", 64),
                "training_seqlen": bc.get("training_seqlen", 2048),
                "quant_lr": bc.get("quant_lr", 1e-4),
                "weight_lr": bc.get("weight_lr", 1e-3),
                "min_lr_factor": bc.get("min_lr_factor", 20),
                "weight_decay": bc.get("wd", 0.0),
            },
        )()

    @torch.no_grad()
    def _update_dataset(self, layer, dataset, dev, **kwargs):
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(dataset):
                inps = inps.to(dev)
                if len(inps.shape) == 2:
                    inps = inps.unsqueeze(0)
                new_data = layer(inps, **kwargs).to("cpu")
                dataset.update_data(index, new_data)

    def _move_embeddings_to(self, dev):
        self.quant_model.model.model.embed_tokens = self.quant_model.model.model.embed_tokens.to(
            dev
        )
        self.quant_model.model.model.norm = self.quant_model.model.model.norm.to(dev)
        if hasattr(self.quant_model.model.model, "rotary_emb"):
            self.quant_model.model.model.rotary_emb = self.quant_model.model.model.rotary_emb.to(
                dev
            )

    def _capture_first_layer_inputs(self, layers, dev):
        self._move_embeddings_to(dev)
        layers[0] = layers[0].to(dev)
        layers[0] = _Catcher(layers[0], dataset=self.fp_train_inps)

        iters = len(self.fp_train_inps)
        with torch.no_grad():
            for i in range(iters):
                data = []
                for j in range(i * self.args.batch_size, (i + 1) * self.args.batch_size):
                    sample = self.train_dataset[j]
                    data.append(torch.tensor(sample["input_ids"]).unsqueeze(0).to(dev))
                data = torch.cat(data, dim=0)

                try:
                    self.quant_model.model(data)
                except ValueError:
                    pass

        layer_kwargs = layers[0].layer_kwargs
        for k, v in layer_kwargs.items():
            # position embeddings
            if isinstance(v, tuple):
                layer_kwargs[k] = tuple(
                    (item.to(dev) if isinstance(item, (torch.Tensor, nn.Module)) else item)
                    for item in v
                )
        layers[0] = layers[0].module
        layers[0] = layers[0].cpu()
        self._move_embeddings_to("cpu")
        torch.cuda.empty_cache()
        return layer_kwargs

    def _make_scheduler(self, lr, total_iters):
        dummy_opt = torch.optim.AdamW([torch.tensor(0)], lr=lr)
        return CosineAnnealingLR(
            dummy_opt, T_max=total_iters, eta_min=lr / self.args.min_lr_factor
        )

    def _train_single_block(self, qlayer, block_index, dev, loss_func, **layer_kwargs):
        with torch.no_grad():
            qlayer.float()

        total_iters = self.args.epochs * self.args.train_size / self.args.batch_size
        param_groups = []
        schedulers = {}

        if float(self.args.quant_lr) > 0:
            set_quant_parameters(qlayer, True)
            idx = len(param_groups)
            param_groups.append(
                {"params": quant_parameters(qlayer), "lr": float(self.args.quant_lr)}
            )
            schedulers[idx] = self._make_scheduler(float(self.args.quant_lr), total_iters)
        else:
            set_quant_parameters(qlayer, False)

        if float(self.args.weight_lr) > 0:
            set_weight_parameters(qlayer, True)
            idx = len(param_groups)
            param_groups.append(
                {"params": weight_parameters(qlayer), "lr": float(self.args.weight_lr)}
            )
            schedulers[idx] = self._make_scheduler(float(self.args.weight_lr), total_iters)
        else:
            set_weight_parameters(qlayer, False)

        assert param_groups
        optimizer = torch.optim.AdamW(param_groups, weight_decay=float(self.args.weight_decay))
        loss_scaler = NativeScalerWithGradNormCount()

        for epoch in range(self.args.epochs):
            loss_list, norm_list = [], []
            start_time = time.time()

            for quant_inps, fp_inps in zip(self.quant_train_inps, self.fp_train_inps):
                with torch.amp.autocast("cuda"):
                    quant_out = qlayer(quant_inps.to(dev), **layer_kwargs)[0]
                    loss = loss_func(fp_inps.to(dev), quant_out)

                if not math.isfinite(loss.item()):
                    print_info(f"Loss is NAN at block {block_index} epoch {epoch}")
                    raise RuntimeError(f"Non-finite loss: {loss.item()}")

                loss_list.append(loss.detach().cpu())
                optimizer.zero_grad()
                norm = loss_scaler(loss, optimizer, parameters=trainable_parameters(qlayer)).cpu()
                norm_list.append(norm.data)

                for idx, sched in schedulers.items():
                    sched.step()
                    optimizer.param_groups[idx]["lr"] = sched.get_lr()[0]

            n = min(len(loss_list), LOSS_AVG_WINDOW)
            loss_mean = torch.stack(loss_list)[-(n - 1) :].mean()
            norm_mean = torch.stack(norm_list).mean()
            quant_sched = schedulers.get(0)
            current_lr = quant_sched.get_lr()[0] if quant_sched else 0.0
            print_info(
                f"blocks {block_index} epoch {epoch} "
                f"recon_loss:{loss_mean} quant_lr:{current_lr} "
                f"norm:{norm_mean:.8f} "
                f"max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024 ** 2} "  # noqa: E501
                f"time {time.time() - start_time}"
            )

        optimizer.zero_grad()
        del optimizer

    def train(self):
        set_quant_state(self.quant_model.model, weight_quant=False, act_quant=False)
        set_quant_parameters(self.quant_model.model, requires_grad=False)
        set_weight_parameters(self.quant_model.model, requires_grad=False)
        ds_kwargs = dict(
            size=min(self.args.train_size, len(self.train_dataset)),
            seqlen=self.args.training_seqlen,
            hidden_size=self.quant_model.model.config.hidden_size,
            batch_size=self.args.batch_size,
            dtype=torch.float16,
        )
        self.fp_train_inps = BlockTrainDataset(**ds_kwargs)
        self.quant_train_inps = BlockTrainDataset(**ds_kwargs)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_cache = self.quant_model.model.config.use_cache
        self.quant_model.model.config.use_cache = False
        layers = self.quant_model.model.model.layers

        layer_kwargs = self._capture_first_layer_inputs(layers, dev)

        for index, data in enumerate(self.fp_train_inps):
            self.quant_train_inps.update_data(index, data)

        loss_func = torch.nn.MSELoss()
        for block_index in range(len(layers)):
            print_info(f"=== Start quantize blocks {block_index} ===")
            qlayer = layers[block_index].to(dev)

            if self.args.epochs > 0:
                self._update_dataset(qlayer, self.fp_train_inps, dev, **layer_kwargs)

            set_quant_state(qlayer, weight_quant=True, act_quant=self.use_act_quant)

            if self.args.epochs > 0:
                self._train_single_block(qlayer, block_index, dev, loss_func, **layer_kwargs)

            qlayer.half()
            quant_inplace(qlayer)
            set_quant_state(qlayer, weight_quant=False, act_quant=self.use_act_quant)

            if self.args.epochs > 0:
                self._update_dataset(qlayer, self.quant_train_inps, dev, **layer_kwargs)

            layers[block_index] = qlayer.to("cpu")
            del qlayer
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        gc.collect()
        self.quant_model.model.config.use_cache = use_cache
