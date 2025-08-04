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

import threadpoolctl as tctl
import torch
import torch.nn as nn
from huggingface_hub import save_torch_state_dict
from tqdm import tqdm

from .....utils import print_info
from ...modules.catcher import Catcher
from ...modules.helper_layer import GPTQQuantLinear
from .gptq_module import GPTQModule

__all__ = ["GPTQ"]


class GPTQ:
    def __init__(
        self, model, seq_length=2048, hidden_size=2560, sym=True, actorder=True
    ):
        super(GPTQ, self).__init__()
        self.model = model
        self.modal_type = self.model.modal_type
        if self.modal_type == "VLM":
            self.layers = self.model.model.model.language_model.layers
        else:
            self.layers = self.model.model.model.layers
        self.layers_block_name = self.model.block_name
        self.quant_bits = self.model.quant_config.quant_bit
        self.group_size = self.model.quant_config.quant_algo_info["group_size"]
        self.ignore_layers = self.model.quant_config.quant_algo_info["ignore_layers"]
        self.percdamp = 0.01
        self.sym = sym
        self.actorder = actorder
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.dtype = next(iter(self.layers.parameters())).dtype
        self.quantizers = {}
        self.gptq = {}

    @torch.no_grad()
    def run(self, dataloader):
        for model_module in self.layers:
            model_module.eval()

        layers = self.layers
        dev = "cuda:0"

        print_info("dev = :{}".format(dev))

        nsamples = len(dataloader)
        inps = torch.zeros(
            (nsamples, self.seq_length, self.hidden_size), device=dev, dtype=self.dtype
        )
        cache = {"i": 0}

        pre_transformer_modules_dict = self.model.get_pre_transformer_modules()
        for _, module in pre_transformer_modules_dict.items():
            module.to(dev)
        layers[0] = layers[0].to(dev)
        layers[0] = Catcher(layers[0], inps, cache)
        # get modle input in dataloader
        self.model.model_forward(dataloader)
        layer_kwargs = layers[0].layer_kwargs

        print_info("cache['i']:{}".format(cache["i"]))

        layers[0] = layers[0].module
        for _, module in pre_transformer_modules_dict.items():
            module.cpu()
        layers[0].cpu()
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        # begin the gptq process
        print_info("Ready.")

        layers = layers.cpu()

        for i in range(len(layers)):
            layer = layers[i].to(inps.device)
            subset = self._find_layers(layer)
            print_info("subset:{}".format(subset))
            self.gptq = {}
            print_info("GPTQMoe start layer {}".format(i))
            for name in subset:
                if name in self.ignore_layers:
                    continue
                self.gptq[name] = GPTQModule(subset[name], quant_bits=self.quant_bits)

            def add_batch(layer_name):
                def tmp(_, inp, out):
                    self.gptq[layer_name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in self.gptq:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            # being hook
            for j in range(nsamples):
                with torch.no_grad():
                    outs[j, :, :] = layer(
                        hidden_states=inps[j, :, :].unsqueeze(0), **layer_kwargs
                    )[0].squeeze(1)

            print_info("HOOK Step{}".format(j))
            for h in handles:
                h.remove()

            for name in subset:
                if name in self.ignore_layers:
                    continue
                print_info("Quant {} ...".format(name))
                scale, zero, g_idx = self.gptq[name].fasterquant(
                    percdamp=self.percdamp,
                    group_size=self.group_size,
                    actorder=self.actorder,
                    sym=self.sym,
                )
                self.quantizers[f"{self.layers_block_name}.{i}.{name}"] = (
                    scale,
                    zero,
                    g_idx,
                )
                self.gptq[name].free()

            for j in range(nsamples):
                with torch.no_grad():
                    outs[j, :, :] = layer(
                        hidden_states=inps[j, :, :].unsqueeze(0), **layer_kwargs
                    )[0].squeeze(1)

            for name in self.gptq:
                del self.gptq[name].layer

            layers[i] = layer.cpu()
            del layer
            # del gptq
            torch.cuda.empty_cache()
            inps, outs = outs, inps
            print_info("GPTQ end layer {}\n".format(i))

        # inps = inps.cpu()
        # outs = outs.cpu()
        del inps, outs
        torch.cuda.empty_cache()
        print_info("GPTQ done.")

    def _make_quant(
        self,
        module,
        names,
        bits,
        group_size,
    ):
        if isinstance(module, GPTQQuantLinear):
            return

        for name, submodule in module.named_modules():
            if name in names:
                ori_layer_device = next(submodule.parameters()).device

                if isinstance(submodule, nn.Linear):
                    in_features = submodule.in_features
                    out_features = submodule.out_features
                bias = submodule.bias is not None
                new_layer = GPTQQuantLinear(
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    bias,
                    weight_dtype=submodule.weight.dtype,
                )
                new_layer.device = ori_layer_device
                self._recurse_setattr(module, name, new_layer.to(ori_layer_device))

    def _pack_model(
        self, model, quantizers, bits, group_size, force_layer_back_to_cpu: bool = False
    ):
        if force_layer_back_to_cpu:
            model.cpu()

        print_info("Packing model...")
        layers = self._find_layers(model)
        layers = {n: layers[n] for n in quantizers}

        self._make_quant(model, quantizers, bits, group_size)

        qlayers = self._find_layers(model, [GPTQQuantLinear])

        with tctl.threadpool_limits(limits=1):
            pbar = tqdm(qlayers.keys(), leave=True)
            for name in pbar:
                pbar.set_description(f"Packing {name}...", refresh=True)

                scale, zero, g_idx = quantizers[name]
                # so far can only pack layer on CPU
                layer_device = qlayers[name].device
                qlayers[name].cpu()
                layers[name], scale, zero, g_idx = (
                    layers[name].cpu(),
                    scale.cpu(),
                    zero.cpu(),
                    g_idx.cpu(),
                )
                qlayers[name].pack(layers[name], scale, zero, g_idx)
                qlayers[name].to(layer_device)
        print_info("Model packed.")

    def _convert_llm(self):
        self._pack_model(
            model=self.model.model,
            quantizers=self.quantizers,
            bits=self.quant_bits,
            group_size=self.group_size,
            force_layer_back_to_cpu=True,
        )

    def convert(self):
        """
        Saves scales and inserts QDQ modules.
        """
        print_info("Start convert model...")
        if self.modal_type in ["LLM", "VLM"]:
            self._convert_llm()
        elif self.modal_type == "AIGC":
            pass
        else:
            print_info("current {} modal type not support".format(self.modal_type))
            raise NotImplementedError
        print_info("convert model done.")

    def save(self, save_dir: str, shard_size="5GB", safetensors=True):
        """save quantized model and configs to local disk"""
        os.makedirs(save_dir, exist_ok=True)

        self.model.model.cpu()

        # Save model
        class EmptyModule(nn.Module):
            def __init__(self):
                super(EmptyModule, self).__init__()

            def forward(self, x):
                return x

        # Save model and config files with empty state dict
        self.model.model.config.quantization_config = {
            "bits": self.quant_bits,
            "checkpoint_format": "gptq",
            "desc_act": True,
            "group_size": self.group_size,
            "quant_method": "gptq",
            "static_groups": True,
            "sym": True,
            "true_sequential": True,
        }
        self.model.model.config.save_pretrained(
            save_dir, state_dict=EmptyModule().state_dict()
        )

        # Remove empty state dict
        default_paths = [
            f"{save_dir}/model.safetensors",
            f"{save_dir}/pytorch_model.bin",
        ]
        for path in default_paths:
            if os.path.exists(path):
                os.remove(path)

        save_torch_state_dict(
            state_dict=self.model.model.state_dict(),
            save_directory=save_dir,
            max_shard_size=shard_size,
            safe_serialization=safetensors,
            force_contiguous=True,
            shared_tensors_to_discard=self.model.model._tied_weights_keys,
        )
        # self.model.model.config.torch_dtype = "float16"
        self.model.model.config.to_json_file(os.path.join(save_dir, "config.json"))

        # save processor and tokenizer
        if self.modal_type == "VLM" and self.model.processor is not None:
            self.model.processor.save_pretrained(save_dir)
        if self.modal_type in ["LLM", "VLM"]:
            self.model.tokenizer.save_pretrained(save_dir)

    def _recurse_setattr(self, module, name, value):
        """A function to recursively set attributes to a module."""
        if "." not in name:
            setattr(module, name, value)
        else:
            name, rest = name.split(".", 1)
            self._recurse_setattr(getattr(module, name), rest, value)

    def _find_layers(self, module, layers=None, name=""):
        if not layers:
            layers = [torch.nn.Linear]
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(
                self._find_layers(
                    child,
                    layers=layers,
                    name=name + "." + name1 if name != "" else name1,
                )
            )
        return res
