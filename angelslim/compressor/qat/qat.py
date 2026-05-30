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
from safetensors.torch import save_file

from ...utils import (
    gathered_param_if_zero3,
    model_has_zero3_params,
    print_info,
    save_via_model_save_func,
    set_op_by_name,
)
from ..compressor_factory import CompressorFactory
from ..quant.modules.helper_layer import QDQModule
from .modules.quantizer import QuantLinear
from .plugins.plugin_manager import PluginManager
from .trainers.trainer_factory import TrainerFactory

__all__ = ["QAT"]


@CompressorFactory.register
class QAT:
    def __init__(self, model, slim_config=None):
        self.quant_model = model
        self.config = slim_config
        self.training_mode = slim_config["compress_config"].QAT.training_mode.lower()
        self.save_fmt = slim_config["compress_config"].QAT.save_format
        self.plugin_config = slim_config["compress_config"].QAT.plugin_config
        self.quant_model.init_ptq(slim_config)
        self.quant_info = self.quant_model.quant_config
        self.plugin_manager = PluginManager()
        # When set, ``save`` will use this rank-0-only state_dict instead of
        # walking the model again. Populated by ``convert`` under ZeRO-3 to
        # avoid keeping a full CPU copy of every layer's QDQModule on every
        # rank.
        self._rank0_state_dict = None
        self._init_plugins()
        self._init_trainer()

    def _init_plugins(self):
        # Register learnable rotation plugin
        if self.plugin_config.get("enable_rotation", False):
            self.plugin_manager.register_plugin(
                "learnable_rotation",
                config=self.plugin_config.get("rotation_config", {}),
                quant_model=self.quant_model,
            )

        # Register learnable scale plugin
        if self.plugin_config.get("enable_scale", False):
            self.plugin_manager.register_plugin(
                "learnable_scale",
                quant_info=self.quant_info,
                ignore_layers=self.config["compress_config"].quantization.ignore_layers,
                resume_ckpt_dir=self.config["compress_config"].QAT.resume_ckpt_dir,
                from_ptq_ckpt_dir=self.config["compress_config"].QAT.from_ptq_ckpt,
                config=self.plugin_config.get("quant_config", {}),
                quant_model=self.quant_model,
            )

    def _init_trainer(self):
        self.trainer = TrainerFactory.create(
            training_mode=self.training_mode,
            quant_model=self.quant_model,
            config=self.config,
            plugin_manager=self.plugin_manager,
        )

    def run(self, dataloader):
        self.trainer.run(dataloader)

    @staticmethod
    def _gather_clone(tensor):
        """Detach + CPU-clone a tensor, gathering if it is a ZeRO-3 shard.

        WARNING: every rank gets a full CPU copy. Only safe for SMALL tensors
        (e.g. scale Parameters). For large weights under ZeRO-3 use
        ``_rank0_gather_clone`` instead.
        """
        if tensor is None:
            return None
        with gathered_param_if_zero3(tensor):
            return tensor.detach().cpu().clone()

    @staticmethod
    def _sym_gather_clone(tensor):
        """Symmetric gather-and-clone: every rank gets a full CPU copy.

        Collective timing is symmetric across ranks (minimising NCCL
        stalls). Caller is responsible for dropping the clone on rank>0
        immediately to avoid keeping 'world_size' copies alive.
        """
        if tensor is None:
            return None
        with gathered_param_if_zero3(tensor):
            return tensor.detach().cpu().clone()

    def convert(self):
        if self.save_fmt not in ("real", "real_and_kvcache"):
            return

        zero3 = model_has_zero3_params(self.quant_model.model)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        quant_algo = self.quant_info.quant_algo

        if not zero3:
            # ----- single-GPU / non-ZeRO-3 path: original behaviour -----
            print_info("Start QAT convert: replacing QuantLinear with QDQModule...")
            for name, module in [
                (n, m)
                for n, m in self.quant_model.model.named_modules()
                if isinstance(m, QuantLinear)
            ]:
                weight_scale = (
                    module.weight_quantizer.scale.data.clone()
                    if hasattr(module, "weight_quantizer")
                    else None
                )
                input_scale = None
                if module.use_act_quant and hasattr(module, "act_quantizer"):
                    aq = module.act_quantizer
                    if getattr(aq, "scale", None) is not None:
                        input_scale = aq.scale.data.clone()
                qdq_module = QDQModule(
                    quant_algo=quant_algo,
                    weight=module.weight,
                    weight_scale=weight_scale,
                    bias=module.bias,
                    group_size=(
                        module.weight_quantizer.group_size
                        if hasattr(module, "weight_quantizer")
                        and hasattr(module.weight_quantizer, "group_size")
                        else 128
                    ),
                    input_scale=input_scale,
                )
                set_op_by_name(self.quant_model.model, name, qdq_module)
            return

        # ----- ZeRO-3 path: every rank gathers + clones per layer (fast,
        # NCCL-symmetric), but only rank0 keeps the data by feeding it into
        # ``_rank0_state_dict``. rank>0 drops the clone immediately so peak
        # CPU remains bounded by ~one layer's worth of tensors per rank.
        # Model structure is NOT modified — we stream straight into the
        # state_dict and let ``save_via_model_save_func`` patch
        # ``state_dict()`` for the underlying save_func.
        print_info(
            f"[rank{rank}] Start QAT convert (ZeRO-3 mode: stream rank0 "
            "state_dict, keep model structure intact)..."
        )
        self._rank0_state_dict = {} if rank == 0 else {}

        quant_linear_modules = [
            (n, m) for n, m in self.quant_model.model.named_modules() if isinstance(m, QuantLinear)
        ]
        consumed_prefixes = set()

        for name, module in quant_linear_modules:

            # Symmetric gather: all ranks clone (memcpy, fast) so NCCL
            # timing stays tight. This is ~world_size× transient CPU RAM
            # for JUST this one layer; we free right after the rank0
            # branch completes.
            weight = self._sym_gather_clone(module.weight)
            bias = self._sym_gather_clone(getattr(module, "bias", None))
            weight_scale = None
            if hasattr(module, "weight_quantizer"):
                weight_scale = self._sym_gather_clone(module.weight_quantizer.scale)
            input_scale = None
            if module.use_act_quant and hasattr(module, "act_quantizer"):
                aq = module.act_quantizer
                if getattr(aq, "scale", None) is not None:
                    input_scale = self._sym_gather_clone(aq.scale)

            consumed_prefixes.add(name)

            if rank != 0:
                # Drop the clone immediately; next iteration will overwrite
                # these locals anyway but be explicit for clarity.
                del weight, bias, weight_scale, input_scale
                continue

            # rank0 only: run the fp8/int quantize path via a throwaway
            # QDQModule, then move its params into the consolidated dict
            # and discard the module.
            qdq_module = QDQModule(
                quant_algo=quant_algo,
                weight=weight,
                weight_scale=weight_scale,
                bias=bias,
                group_size=(
                    module.weight_quantizer.group_size
                    if hasattr(module, "weight_quantizer")
                    and hasattr(module.weight_quantizer, "group_size")
                    else 128
                ),
                input_scale=input_scale,
            )
            for sub_name, p in qdq_module.named_parameters(recurse=False):
                self._rank0_state_dict[f"{name}.{sub_name}"] = p.detach().cpu()
            for sub_name, b in qdq_module.named_buffers(recurse=False):
                if sub_name in qdq_module._non_persistent_buffers_set:
                    continue
                self._rank0_state_dict[f"{name}.{sub_name}"] = b.detach().cpu()
            del qdq_module, weight, bias, weight_scale, input_scale

        # Second pass: params/buffers that are NOT inside a QuantLinear
        # (embeddings, lm_head, layernorms, MoE router gate, ...). The
        # collective order MUST be identical across ranks, so this loop
        # runs on every rank; only rank0 keeps the data.
        for pname, param in self.quant_model.model.named_parameters():
            if any(pname.startswith(p + ".") for p in consumed_prefixes):
                continue
            with gathered_param_if_zero3(param):
                if rank == 0:
                    self._rank0_state_dict[pname] = param.detach().cpu().clone()

        if rank == 0:
            for module_name, mod in self.quant_model.model.named_modules():
                for buf_name, buf in mod.named_buffers(recurse=False):
                    if buf is None or buf_name in mod._non_persistent_buffers_set:
                        continue
                    full_key = f"{module_name}.{buf_name}" if module_name else buf_name
                    if full_key in self._rank0_state_dict:
                        continue
                    self._rank0_state_dict[full_key] = buf.detach().cpu().clone()
            print_info(
                f"[zero3] convert done: rank0 state_dict has "
                f"{len(self._rank0_state_dict)} tensors."
            )

    def _save_kv_cache_scales(self, save_path: str):
        """Extract and save KV cache scales to a safetensors file."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        kv_scales = {}
        for name, module in self.quant_model.model.named_modules():
            if not isinstance(module, QuantLinear):
                continue
            if not module.use_qkv_quant or not hasattr(module, "qkv_quantizer"):
                continue
            # Map k_proj / v_proj to k_cache / v_cache
            if name.endswith(".k_proj"):
                cache_name = name.replace(".k_proj", ".k_cache")
            elif name.endswith(".v_proj"):
                cache_name = name.replace(".v_proj", ".v_cache")
            else:
                continue
            scale_key = f"{cache_name}.scale"
            scale_tensor = self._gather_clone(module.qkv_quantizer.scale)
            if scale_tensor is not None and rank == 0:
                kv_scales[scale_key] = scale_tensor.float()

        if rank != 0:
            return
        os.makedirs(save_path, exist_ok=True)
        out_file = os.path.join(save_path, "kv_cache_scales.safetensors")
        save_file(kv_scales, out_file)
        print_info(f"Saved {len(kv_scales)} KV cache scales to: {out_file}")

    def save(self, save_path: str):
        # "fake": save fake-quant state_dict (NOTE: only supports non-distributed / single-GPU)
        if self.save_fmt == "fake":
            parts = save_path.rsplit("/")
            save_path = os.path.join("/".join(parts[:-1]), parts[-1] + "_fake_quant_model.pt")
            print_info(f"Start save QAT fake ckpt to: {save_path}")

            cpu_state = self.trainer.external_trainer.model.state_dict()
            torch.save(cpu_state, save_path)

        # "real": save real-quant model via model-specific save function
        elif self.save_fmt == "real":
            save_func = self.quant_model.get_save_func()(self.quant_model)
            save_via_model_save_func(
                self.quant_model,
                save_func,
                os.path.join(save_path, "final_quant_checkpoint"),
                prebuilt_state_dict=self._rank0_state_dict,
            )

        # "save_kvcache_only": only export KV cache scales (kv_cache_scales.safetensors)
        elif self.save_fmt == "save_kvcache_only":
            self._save_kv_cache_scales(os.path.join(save_path, "final_quant_checkpoint"))

        # "real_and_kvcache": save real-quant model AND KV cache scales
        elif self.save_fmt == "real_and_kvcache":
            save_func = self.quant_model.get_save_func()(self.quant_model)
            save_via_model_save_func(
                self.quant_model,
                save_func,
                os.path.join(save_path, "final_quant_checkpoint"),
                prebuilt_state_dict=self._rank0_state_dict,
            )
            self._save_kv_cache_scales(os.path.join(save_path, "final_quant_checkpoint"))

        else:
            print_info("Save format not specified, skip save.")
