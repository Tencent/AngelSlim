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
import json
import os
import shutil

import threadpoolctl as tctl
import torch
import torch.distributed as dist
import torch.nn as nn
from huggingface_hub import save_torch_state_dict
from tqdm import tqdm

from .....utils import decide_device_for_distributed, find_layers, print_info
from ...modules.catcher import Catcher
from ...modules.helper_layer import GPTQQuantLinear
from .gptaq_module import GPTAQModule
from .gptq_module import GPTQModule


def _extract_hidden_states(output):
    """Extract hidden_states tensor from a layer's forward output.

    Some decoder layers return a plain tensor, others return a tuple where
    the first element is hidden_states.  Using ``output[0]`` unconditionally
    is wrong for the plain-tensor case because it strips the batch dimension.
    """
    if isinstance(output, tuple):
        return output[0]
    return output


__all__ = ["GPTQ"]


class GPTQ:
    def __init__(self, model, seq_length=2048, hidden_size=2560, sym=True, actorder=True):
        super(GPTQ, self).__init__()
        self.model = model
        self.modal_type = self.model.modal_type
        self.layers = self.model.get_quant_module()
        self.layers_block_name = self.model.block_name
        self.quant_bits = self.model.quant_config.quant_bit
        self.group_size = self.model.quant_config.quant_algo_info["group_size"]
        self.ignore_layers = self.model.quant_config.quant_algo_info["ignore_layers"]
        self.dequant_to_bf16 = bool(
            self.model.quant_config.quant_algo_info.get("dequant_to_bf16", False)
        )
        self.percdamp = 0.01
        self.sym = sym
        self.actorder = actorder
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.dtype = next(iter(self.layers.parameters())).dtype
        self.quantizers = {}
        self.gptq = {}
        self.quant_algo = self.model.quant_config.quant_algo
        self.native_inp_caches = {}
        self.quant_linear_cls = GPTQQuantLinear

    def _prepare_rotary_emb(self, device):
        model = getattr(self.model, "model", None)
        transformer = getattr(model, "model", None)
        rotary_emb = getattr(transformer, "rotary_emb", None)
        if rotary_emb is None:
            return

        has_meta_buffer = any(
            isinstance(buffer, torch.Tensor) and buffer.is_meta
            for buffer in rotary_emb.buffers(recurse=False)
        )
        if has_meta_buffer:
            print_info(f"Materializing rotary_emb on {device}")
            rotary_emb = rotary_emb.__class__(rotary_emb.config, device=device)
        transformer.rotary_emb = rotary_emb.to(device)

    def _move_layer_to_device(self, layer, device):
        if self._is_distributed_expert_parallel():
            try:
                return layer.to(device)
            except NotImplementedError as e:
                if "Cannot copy out of meta tensor" not in str(e):
                    raise
                print_info(
                    f"Skip moving meta tensors while moving expert-parallel layer to {device}"
                )
                for module in layer.modules():
                    for name, param in list(module._parameters.items()):
                        if param is not None and not param.is_meta:
                            module._parameters[name] = nn.Parameter(
                                param.to(device), requires_grad=param.requires_grad
                            )
                    for name, buffer in list(module._buffers.items()):
                        if isinstance(buffer, torch.Tensor) and not buffer.is_meta:
                            module._buffers[name] = buffer.to(device)
                return layer
        return layer.to(device)

    def get_actorder_prev_names(self, name, subset):
        if not name.endswith("down_proj"):
            return []

        prefix = name[: -len("down_proj")]
        prev_names = [f"{prefix}gate_proj", f"{prefix}up_proj"]
        return [prev_name for prev_name in prev_names if prev_name in subset]

    def reorder_quantizer_output_channels(self, quantizer_name, input_perm):
        if quantizer_name not in self.quantizers:
            return

        scale, zero = self.quantizers[quantizer_name]
        scale_perm = input_perm.to(scale.device)
        zero_perm = input_perm.to(zero.device)
        self.quantizers[quantizer_name] = (
            scale.index_select(0, scale_perm),
            zero.index_select(0, zero_perm),
        )

    def fold_input_permutation_into_prev_layers(
        self,
        layer_idx,
        subset,
        prev_names,
        input_perm,
    ):
        if input_perm is None:
            return

        for prev_name in prev_names:
            prev_layer = subset[prev_name]
            weight_perm = input_perm.to(prev_layer.weight.device)
            prev_layer.weight.data.copy_(prev_layer.weight.data.index_select(0, weight_perm))
            if prev_layer.bias is not None:
                prev_layer.bias.data.copy_(prev_layer.bias.data.index_select(0, weight_perm))

            quantizer_name = f"{self.layers_block_name}.{layer_idx}.{prev_name}"
            self.reorder_quantizer_output_channels(quantizer_name, input_perm)

    @staticmethod
    def _truncate_tensor_dim(tensor, max_len, dim):
        if tensor is None or tensor.dim() == 0:
            return tensor
        dim = dim if dim >= 0 else tensor.dim() + dim
        if dim < 0 or dim >= tensor.dim() or tensor.shape[dim] <= max_len:
            return tensor
        return tensor.narrow(dim, 0, max_len)

    def _align_layer_input(self, hidden_states, kwargs):
        seq_len = hidden_states.shape[1]
        if self.seq_length is not None:
            seq_len = min(seq_len, int(self.seq_length))
        hidden_states = self._truncate_tensor_dim(hidden_states, seq_len, dim=1)

        aligned_kwargs = {}
        for key, value in kwargs.items():
            if key == "position_ids" and isinstance(value, torch.Tensor):
                aligned_kwargs[key] = self._truncate_tensor_dim(value, seq_len, dim=-1)
            elif key == "attention_mask" and isinstance(value, torch.Tensor):
                value = self._truncate_tensor_dim(value, seq_len, dim=-1)
                if value.dim() >= 3:
                    value = self._truncate_tensor_dim(value, seq_len, dim=-2)
                aligned_kwargs[key] = value
            elif key == "position_embeddings" and isinstance(value, tuple):
                aligned_kwargs[key] = tuple(
                    (
                        self._truncate_tensor_dim(tensor, seq_len, dim=1)
                        if isinstance(tensor, torch.Tensor) and tensor.dim() >= 3
                        else (
                            self._truncate_tensor_dim(tensor, seq_len, dim=0)
                            if isinstance(tensor, torch.Tensor)
                            else tensor
                        )
                    )
                    for tensor in value
                )
            else:
                aligned_kwargs[key] = value
        return hidden_states, aligned_kwargs

    def _forward_layer(self, layer, hidden_states, kwargs):
        hidden_states, kwargs = self._align_layer_input(hidden_states, kwargs)
        return _extract_hidden_states(layer(hidden_states=hidden_states, **kwargs))

    @staticmethod
    def _get_expert_idx_from_name(name):
        parts = name.split(".")
        for idx, part in enumerate(parts[:-1]):
            if part == "experts" and idx + 1 < len(parts):
                try:
                    return int(parts[idx + 1])
                except ValueError:
                    return None
        return None

    def _get_local_expert_range(self, layer, subset):
        moe_module = getattr(layer, "mlp", None)
        experts = getattr(moe_module, "experts", None) if moe_module is not None else None

        start = getattr(moe_module, "experts_start_idx", None)
        end = getattr(moe_module, "experts_end_idx", None)
        if start is None and experts is not None:
            start = getattr(experts, "experts_start_idx", None)
        if end is None and experts is not None:
            end = getattr(experts, "experts_end_idx", None)
        if start is not None and end is not None:
            return int(start), int(end)

        if experts is not None and hasattr(experts, "num_experts"):
            num_experts = int(experts.num_experts)
        else:
            expert_ids = [
                expert_idx
                for expert_idx in (self._get_expert_idx_from_name(name) for name in subset)
                if expert_idx is not None
            ]
            if not expert_ids:
                return None
            num_experts = max(expert_ids) + 1

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if num_experts % world_size != 0:
            return None
        n_local_experts = num_experts // world_size
        return rank * n_local_experts, (rank + 1) * n_local_experts

    def _filter_distributed_expert_subset(self, layer, subset):
        if not self._is_distributed_expert_parallel():
            return subset

        local_range = self._get_local_expert_range(layer, subset)
        if local_range is None:
            return subset

        start, end = local_range
        filtered_subset = {}
        skipped = 0
        for name, module in subset.items():
            expert_idx = self._get_expert_idx_from_name(name)
            if expert_idx is not None and not (start <= expert_idx < end):
                skipped += 1
                continue
            filtered_subset[name] = module

        if skipped > 0:
            print_info(
                "Filter non-local expert layers for GPTQ: "
                f"local experts [{start}, {end}), skipped {skipped} layer(s)."
            )
        return filtered_subset

    @torch.no_grad()
    def run(self, dataloader):
        for model_module in self.layers:
            model_module.eval()

        layers = self.layers
        dev = decide_device_for_distributed()

        print_info("dev = :{}".format(dev))

        nsamples = len(dataloader)

        pre_transformer_modules_dict = self.model.get_pre_transformer_modules()
        for _, module in pre_transformer_modules_dict.items():
            module.to(dev)
        self._prepare_rotary_emb(dev)
        layers[0] = layers[0].to(dev)
        layers[0] = Catcher(layers[0], max_seq_length=self.seq_length)
        # get model input in dataloader
        self.model.model_forward(dataloader)

        # Retrieve dynamically captured inputs and per-sample kwargs
        inps = layers[0].captured_inputs
        layer_kwargs_list = layers[0].captured_kwargs
        nsamples = len(inps)
        print_info("captured samples: {}".format(nsamples))

        layers[0] = layers[0].module
        for _, module in pre_transformer_modules_dict.items():
            module.cpu()
        layers[0].cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Move all captured inputs and kwargs to the target device
        inps = [x.to(dev) for x in inps]
        layer_kwargs_list = [
            {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in kw.items()}
            for kw in layer_kwargs_list
        ]

        outs = [torch.zeros_like(x) for x in inps]
        if "gptaq" in self.quant_algo:
            native_inps = [x.clone().detach() for x in inps]
        # begin the gptq process
        print_info("Ready.")

        if not self._is_distributed_expert_parallel():
            layers = layers.cpu()

        for i in range(len(layers)):
            layer = self._move_layer_to_device(layers[i], dev)
            subset = find_layers(layer, layers=self.model.observer_layer_classes)
            subset = self._filter_distributed_expert_subset(layer, subset)
            print_info("subset:{}".format(subset))

            self.gptq = {}
            if "gptaq" in self.quant_algo:
                self.native_inp_caches = {}
            print_info("GPTQMoe start layer {}".format(i))
            for name in subset:
                if any(ignore in name for ignore in self.ignore_layers):
                    continue
                if "gptaq" in self.quant_algo:
                    self.native_inp_caches[name] = []
                    self.gptq[name] = GPTAQModule(subset[name], quant_bits=self.quant_bits)
                else:
                    self.gptq[name] = GPTQModule(subset[name], quant_bits=self.quant_bits)

            def pre_process_fwd_hook(layer_name):
                def tmp(_, inp, out):
                    self.native_inp_caches[layer_name] += [inp[0].data]
                    del inp, out

                return tmp

            def add_batch(layer_name):
                def tmp(_, inp, out):
                    # Some modules return a tuple from forward; extract the tensor
                    out_data = out[0].data if isinstance(out, tuple) else out.data
                    if "gptaq" in self.quant_algo:
                        native_inp = self.native_inp_caches[layer_name].pop(0)
                        self.gptq[layer_name].add_batch(inp[0].data, out_data, native_inp)
                    else:
                        self.gptq[layer_name].add_batch(inp[0].data, out_data)

                return tmp

            if "gptaq" in self.quant_algo:
                native_handles = []
                for name in self.native_inp_caches:
                    native_handles.append(
                        subset[name].register_forward_hook(pre_process_fwd_hook(name))
                    )

                # native hook forward
                for j in range(nsamples):
                    with torch.no_grad():
                        outs[j] = self._forward_layer(layer, native_inps[j], layer_kwargs_list[j])
                native_inps = [x.clone().detach() for x in outs]

                print_info("Native HOOK Step{}".format(j))
                for h in native_handles:
                    h.remove()

            handles = []
            for name in self.gptq:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            # hook forward
            for j in range(nsamples):
                with torch.no_grad():
                    outs[j] = self._forward_layer(layer, inps[j], layer_kwargs_list[j])

            print_info("HOOK Step{}".format(j))
            for h in handles:
                h.remove()

            for name in self.gptq:
                if any(ignore in name for ignore in self.ignore_layers):
                    continue
                if (
                    self._is_distributed_expert_parallel()
                    and self._get_expert_idx_from_name(name) is not None
                    and self.gptq[name].nsamples == 0
                ):
                    print_info(
                        f"Skip {name} because no calibration samples were routed "
                        "to this local expert layer."
                    )
                    self.gptq[name].free()
                    continue
                print_info(f"Quant {name} ,nsamples: {self.gptq[name].nsamples}...")
                prev_names = self.get_actorder_prev_names(name, subset)
                actorder = self.actorder and bool(prev_names)
                scale, zero, input_perm = self.gptq[name].fasterquant(
                    percdamp=self.percdamp,
                    group_size=self.group_size,
                    actorder=actorder,
                    sym=self.sym,
                )
                self.quantizers[f"{self.layers_block_name}.{i}.{name}"] = (
                    scale,
                    zero,
                )
                self.fold_input_permutation_into_prev_layers(
                    i,
                    subset,
                    prev_names,
                    input_perm,
                )
                self.gptq[name].free()

            for j in range(nsamples):
                with torch.no_grad():
                    outs[j] = self._forward_layer(layer, inps[j], layer_kwargs_list[j])

            for name in self.gptq:
                del self.gptq[name].layer

            layers[i] = self._move_layer_to_device(layer, "cpu")
            del layer
            # del gptq
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            inps, outs = outs, inps
            print_info("GPTQ end layer {}\n".format(i))

        del inps, outs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_info("GPTQ done.")

    def _make_quant(
        self,
        module,
        names,
        bits,
        group_size,
    ):
        if isinstance(module, self.quant_linear_cls):
            return

        for name, submodule in module.named_modules():
            if name in names:
                ori_layer_device = next(submodule.parameters()).device

                # Support non-standard Linear modules (e.g. TopKRouter) that lack
                # in_features/out_features by inferring from weight shape
                if hasattr(submodule, "in_features"):
                    in_features = submodule.in_features
                    out_features = submodule.out_features
                else:
                    out_features, in_features = submodule.weight.shape
                bias = getattr(submodule, "bias", None) is not None
                new_layer = self.quant_linear_cls(
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
        layers = find_layers(model, layers=self.model.observer_layer_classes)
        layers = {n: layers[n] for n in quantizers}

        self._make_quant(model, quantizers, bits, group_size)

        qlayers = find_layers(model, [self.quant_linear_cls])

        with tctl.threadpool_limits(limits=1):
            pbar = tqdm(qlayers.keys(), leave=True)
            for name in pbar:
                pbar.set_description(f"Packing {name}...", refresh=True)

                scale, zero = quantizers[name]
                # so far can only pack layer on CPU
                layer_device = qlayers[name].device
                qlayers[name].cpu()
                layers[name], scale, zero = (
                    layers[name].cpu(),
                    scale.cpu(),
                    zero.cpu(),
                )
                qlayers[name].pack(layers[name], scale, zero)
                qlayers[name].to(layer_device)
                del layers[name]
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
        if self.dequant_to_bf16:
            print_info(
                "dequant_to_bf16=True: skip int4 packing, keep fake-quantized bf16 weights."
            )
        else:
            self._convert_llm()
        print_info("convert model done.")

    def _is_distributed_expert_parallel(self):
        return (
            getattr(self.model, "using_multi_nodes", False)
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

    def _collect_local_expert_state_dict(self, state_dict):
        return {
            k: v.cpu()
            for k, v in state_dict.items()
            if ".mlp.experts." in k and not k.endswith(".g_idx")
        }

    def _drop_non_persistent_gptq_buffers(self, state_dict):
        return {k: v for k, v in state_dict.items() if not k.endswith(".g_idx")}

    def _patch_saved_hyv3_config_for_serving(self, save_dir):
        config_path = os.path.join(save_dir, "config.json")
        if not os.path.exists(config_path):
            return

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if config.get("model_type") != "hy_v3":
            return

        rope_parameters = config.get("rope_parameters")
        if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
            config.setdefault("rope_theta", rope_parameters["rope_theta"])

        dtype = config.get("torch_dtype") or config.get("dtype") or "bfloat16"
        if not isinstance(dtype, str):
            dtype = str(dtype).replace("torch.", "")
        config["torch_dtype"] = dtype

        if isinstance(config.get("eos_token_id"), int):
            config["eos_token_id"] = [config["eos_token_id"]]

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")

        generation_config_path = os.path.join(save_dir, "generation_config.json")
        if not os.path.exists(generation_config_path):
            return

        with open(generation_config_path, "r", encoding="utf-8") as f:
            generation_config = json.load(f)

        if isinstance(generation_config.get("eos_token_id"), int):
            generation_config["eos_token_id"] = [generation_config["eos_token_id"]]
            with open(generation_config_path, "w", encoding="utf-8") as f:
                json.dump(generation_config, f, indent=2, ensure_ascii=False)
                f.write("\n")

    def _save_merged_state_dict(self, save_dir, state_dict, shard_size, safetensors):
        state_dict = self._drop_non_persistent_gptq_buffers(state_dict)
        state_dict = self.model.format_state_dict_for_save(state_dict)

        class EmptyModule(nn.Module):
            def __init__(self):
                super(EmptyModule, self).__init__()

            def forward(self, x):
                return x

        if self.dequant_to_bf16:
            if hasattr(self.model.model.config, "quantization_config"):
                try:
                    delattr(self.model.model.config, "quantization_config")
                except Exception:
                    self.model.model.config.quantization_config = None
            self.model.model.config.torch_dtype = "bfloat16"
        else:
            self.model.model.config.quantization_config = {
                "bits": self.quant_bits,
                "checkpoint_format": self.model.quant_config.quant_algo_info.get(
                    "checkpoint_format", "gptq"
                ),
                "desc_act": False,
                "group_size": self.group_size,
                "quant_method": "gptq",
                "static_groups": True,
                "sym": self.sym,
                "true_sequential": True,
            }
        self.model.model.config.save_pretrained(save_dir, state_dict=EmptyModule().state_dict())
        self.model.model.generation_config.save_pretrained(save_dir)

        default_paths = [
            f"{save_dir}/model.safetensors",
            f"{save_dir}/pytorch_model.bin",
        ]
        for path in default_paths:
            if os.path.exists(path):
                os.remove(path)

        save_torch_state_dict(
            state_dict=state_dict,
            save_directory=save_dir,
            max_shard_size=shard_size,
            safe_serialization=safetensors,
            force_contiguous=True,
            shared_tensors_to_discard=self.model.model._tied_weights_keys,
        )
        self.model.model.config.to_json_file(os.path.join(save_dir, "config.json"))
        self._patch_saved_hyv3_config_for_serving(save_dir)

        if self.modal_type == "VLM" and self.model.processor is not None:
            self.model.processor.save_pretrained(save_dir)
        if self.modal_type in ["LLM", "VLM"]:
            self.model.tokenizer.save_pretrained(save_dir)
            source_path = getattr(self.model.model.config, "_name_or_path", None)
            if source_path:
                source_tokenizer_config = os.path.join(source_path, "tokenizer_config.json")
                if os.path.exists(source_tokenizer_config):
                    shutil.copy2(source_tokenizer_config, save_dir)

    def _save_distributed(self, save_dir: str, shard_size="5GB", safetensors=True):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        tmp_dir = os.path.join(save_dir, ".expert_parallel_states")

        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
        dist.barrier()

        self.model.model.cpu()
        state_dict = self.model.model.state_dict()
        local_expert_state_dict = self._collect_local_expert_state_dict(state_dict)

        if rank != 0:
            torch.save(local_expert_state_dict, os.path.join(tmp_dir, f"rank{rank}.pt"))
        dist.barrier()

        if rank == 0:
            merged_state_dict = dict(state_dict)
            for other_rank in range(1, world_size):
                expert_state_path = os.path.join(tmp_dir, f"rank{other_rank}.pt")
                expert_state_dict = torch.load(
                    expert_state_path,
                    map_location="cpu",
                    weights_only=False,
                )
                merged_state_dict.update(expert_state_dict)
            self._save_merged_state_dict(
                save_dir,
                merged_state_dict,
                shard_size=shard_size,
                safetensors=safetensors,
            )
            shutil.rmtree(tmp_dir)
            print_info(f"Merged distributed GPTQ model saved to {save_dir}")
        dist.barrier()

    def save(self, save_dir: str, shard_size="5GB", safetensors=True):
        """save quantized model and configs to local disk"""
        if self._is_distributed_expert_parallel():
            self._save_distributed(
                save_dir,
                shard_size=shard_size,
                safetensors=safetensors,
            )
            return

        os.makedirs(save_dir, exist_ok=True)
        self.model.model.cpu()
        self._save_merged_state_dict(
            save_dir,
            self.model.model.state_dict(),
            shard_size=shard_size,
            safetensors=safetensors,
        )

    def _recurse_setattr(self, module, name, value):
        """A function to recursively set attributes to a module."""
        if "." not in name:
            setattr(module, name, value)
        else:
            name, rest = name.split(".", 1)
            self._recurse_setattr(getattr(module, name), rest, value)
