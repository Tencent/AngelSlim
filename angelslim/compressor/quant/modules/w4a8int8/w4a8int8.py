import gc
import json
import os
import re
from typing import Callable, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file

from .....utils import find_layers, print_info
from ...core import DeepSeekV3PTQSaveMulti, weight_dequant
from ...modules.helper_layer import W4A8Int8QuantLinear
from ..gptq.gptq import GPTQ


class W4A8INT8(GPTQ):
    """GPTQ runner with FP8 -> BF16 pre-processing and W4A8 INT8 linear."""

    def __init__(
        self,
        model,
        seq_length: int = 2048,
        hidden_size: int = 2560,
        sym: bool = True,
        actorder: bool = True,
    ):
        super().__init__(model, seq_length, hidden_size, sym, actorder)
        self.dtype = torch.bfloat16
        self.quant_linear_cls = W4A8Int8QuantLinear

    @torch.no_grad()
    def run(self, dataloader):
        self._prepare_fp8_weights()
        torch.cuda.empty_cache()
        super().run(dataloader)

    @torch.no_grad()
    def _prepare_fp8_weights(self):
        for layer in self.layers:
            subset = find_layers(layer, layers=self.model.observer_layer_classes)
            for sub_layer in subset.values():
                self._maybe_dequant_fp8(sub_layer)
                sub_layer.to("cpu")

    @staticmethod
    def _maybe_dequant_fp8(module: nn.Module):
        if (
            hasattr(module, "weight")
            and module.weight.dtype == torch.float8_e4m3fn
            and hasattr(module, "weight_scale_inv")
        ):
            module.weight = nn.Parameter(
                weight_dequant(module.weight, module.weight_scale_inv),
                requires_grad=False,
            )
            module.weight_scale_inv = None

    def save(self, save_dir: str):
        saver = DeepSeekV3W4A8Int8Save(self.model)
        saver.save(save_dir)


class Int8PerChannelQuantizer:
    """Per-channel symmetric int8 quantizer."""

    @torch.no_grad()
    def quantize(self, tensor: torch.Tensor):
        assert tensor.dtype == torch.bfloat16
        qmax = 127.0
        abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]
        scale = abs_max / qmax
        quantized = torch.round(tensor / scale)
        quantized = torch.clamp(quantized, -qmax, qmax)
        return quantized.to(torch.int8), scale.to(torch.float32)


class TPManager:
    def __init__(self, world_size: int):
        self.world_size = world_size

    @torch.no_grad()
    def gather(self, local: torch.Tensor, dim: int):
        if self.world_size == 1:
            return local.cpu()

        if dim != 0:
            local = local.transpose(0, dim).contiguous()

        shape = list(local.shape)
        shape[0] *= self.world_size

        full = torch.empty(
            shape,
            dtype=local.dtype,
            device=local.device,
        )

        dist.all_gather_into_tensor(full, local)

        if dim != 0:
            full = full.transpose(0, dim).contiguous()

        return full.cpu()


class MoEExpertGather:
    """EP gather using gather_object (CPU only)."""

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def gather(
        self,
        key: str,
        local_tensor: torch.Tensor,
        on_rank0_callback,
    ):
        obj = {
            "key": key,
            "tensor": local_tensor.cpu(),
        }
        gathered = [None] * self.world_size if self.rank == 0 else None

        dist.gather_object(
            obj,
            gathered,
            dst=0,
        )

        if self.rank == 0:
            for item in gathered:
                on_rank0_callback(
                    item["key"],
                    item["tensor"],
                )


class DeepSeekKeyRouter:
    def __init__(self):
        self.dim0_keys = [
            ".q_proj.",
            ".q_b_proj.",
            ".kv_b_proj.",
            ".mlp.gate_proj",
            ".mlp.up_proj",
            ".mlp.shared_experts.gate_proj",
            ".mlp.shared_experts.up_proj",
        ]

        self.dim1_keys = [
            ".o_proj.",
            ".mlp.down_proj",
            ".mlp.shared_experts.down_proj",
        ]

    def layer_id(self, key: str):
        if key.startswith("model.embed_tokens") or key.startswith("lm_head"):
            return -1
        if key.startswith("model.norm"):
            return -2
        m = re.search(r"model\.layers\.(\d+)\.", key)
        return int(m.group(1)) if m else None

    def is_moe_expert(self, key: str):
        return "mlp.experts" in key

    def tp_gather_dim(self, key: str):
        if any(x in key for x in self.dim0_keys):
            return 0
        if any(x in key for x in self.dim1_keys):
            return 1
        return None


class DeepSeekV3W4A8Int8Save(DeepSeekV3PTQSaveMulti):
    """
    DeepSeek R1 PTQ weight saver
    - TP layers: all_gather -> quantize -> save
    - EP (MoE experts): gather_object -> save
    """

    def __init__(self, quant_model, check_scales=False):
        super().__init__(quant_model=quant_model)

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.router = DeepSeekKeyRouter()
        self.quantizer = Int8PerChannelQuantizer()
        self.tp_mgr = TPManager(self.world_size)

    @torch.no_grad()
    def save(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        state = self.quant_model.model.state_dict()
        moe_gather = MoEExpertGather(self.rank, self.world_size)

        safetensors_index: Dict[str, str] = {}
        shard_idx = 1
        n_shards = 63

        def shard_name(i):
            return f"model-{i:05d}-of-{n_shards:05d}.safetensors"

        shard_idx = self._save_embeddings_and_norm(
            state, safetensors_index, save_path, shard_idx, shard_name
        )
        shard_idx = self._save_transformer_layers(
            state, safetensors_index, save_path, shard_idx, shard_name, moe_gather
        )

        if self.rank == 0:
            self._finalize(save_path, safetensors_index, shard_idx, shard_name)

    def _save_embeddings_and_norm(
        self,
        state: Dict[str, torch.Tensor],
        index: Dict[str, str],
        save_path: str,
        shard_idx: int,
        shard_name,
    ) -> int:
        if self.rank == 0:
            out = {}

        device = torch.device(
            f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
        )

        for k in list(state.keys()):
            lid = self.router.layer_id(k)
            if lid not in (-1, -2):
                continue

            if lid == -2:  # model.norm
                if self.rank == 0:
                    full = state[k]
            else:  # embed / lm_head (TP)
                v = state[k].to(device)
                full = self.tp_mgr.gather(v, 0)

            if self.rank == 0:
                out[k] = full
                index[k] = shard_name(shard_idx)

            del state[k]

        if self.rank == 0:
            save_file(out, os.path.join(save_path, shard_name(shard_idx)))
            shard_idx += 1
            del out
            gc.collect()

        if dist.is_initialized():
            dist.barrier()

        return shard_idx

    def _save_transformer_layers(
        self,
        state: Dict[str, torch.Tensor],
        index: Dict[str, str],
        save_path: str,
        shard_idx: int,
        shard_name,
        moe_gather: MoEExpertGather,
    ) -> int:
        device = torch.device(
            f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
        )

        for lid in range(61):
            layer_keys = [k for k in state if self.router.layer_id(k) == lid]

            # ---------------- EP ----------------
            expert_out = {} if self.rank == 0 else None

            def on_moe(k, t):
                expert_out[k] = t

            for k in layer_keys:
                if self.router.is_moe_expert(k):
                    moe_gather.gather(k, state[k], on_moe)
                    del state[k]

            # ---------------- TP / local ----------------
            layer_out = {} if self.rank == 0 else None

            for k in layer_keys:
                if self.router.is_moe_expert(k):
                    continue

                v = state[k]

                # passthrough
                if any(
                    x in k
                    for x in ("layernorm", "gate.weight", "e_score_correction_bias")
                ):
                    if self.rank == 0:
                        layer_out[k] = v.cpu()
                    del state[k]
                    continue

                assert k.endswith("weight"), f"Unexpected param: {k}"

                dim = self.router.tp_gather_dim(k)
                if dim is None:
                    if self.rank == 0 and (
                        "q_a_proj" in k or "kv_a_proj_with_mqa" in k
                    ):
                        q, s = self.quantizer.quantize(v)
                        layer_out[k] = q
                        layer_out[k + "_scale"] = s
                    del state[k]
                    continue

                v = v.to(device)
                full = self.tp_mgr.gather(v, dim)
                del v

                if self.rank == 0:
                    q, s = self.quantizer.quantize(full)
                    layer_out[k] = q
                    layer_out[k + "_scale"] = s

                del state[k]
                del full
                torch.cuda.empty_cache()

            if self.rank == 0:
                if expert_out:
                    layer_out.update(expert_out)

                save_file(layer_out, os.path.join(save_path, shard_name(shard_idx)))

                for k in layer_out:
                    index[k] = shard_name(shard_idx)

                shard_idx += 1
                del layer_out
                gc.collect()
                torch.cuda.empty_cache()

            if dist.is_initialized():
                dist.barrier()

        return shard_idx

    def _finalize(self, save_path, index, shard_idx, shard_name):
        self._save_quantization_config(save_path)
        self._copy_additional_files(self.quant_model.model.ori_model_path, save_path)

        with open(os.path.join(save_path, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": index}, f, indent=2)

        self.save_mtp_int8_from_fp8(
            self.quant_model.model.ori_model_path,
            save_path,
            shard_name(shard_idx),
        )

    def _save_quantization_config(self, save_path):
        quantization_config = {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "quantization_status": "compressed",
            "kv_cache_scheme": None,
            "ignore": ["lm_head"],
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "strategy": "channel",
                        "symmetric": True,
                        "type": "int",
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "strategy": "token",
                        "symmetric": True,
                        "type": "int",
                        "dynamic": True,
                    },
                },
                "group_1": {
                    "targets": [
                        "re:.*(self_attn|shared_experts|mlp\\.gate_proj|mlp\\.up_proj|mlp\\.gate_up_proj|mlp\\.down_proj).*",
                        "re:model\\.layers\\.61.*",
                    ],
                    "weights": {
                        "num_bits": 8,
                        "strategy": "channel",
                        "symmetric": True,
                        "type": "int",
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "strategy": "token",
                        "symmetric": True,
                        "type": "int",
                        "dynamic": True,
                    },
                },
            },
        }

        config = self.quant_model.model.config
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")

        config.update({"quantization_config": quantization_config})
        config.save_pretrained(save_path)

    def _copy_additional_files(self, source_path: str, target_path: str):
        """
        Copy additional files with specific suffixes,
        but NEVER override config.json generated by quantization.
        """
        import os
        import shutil

        os.makedirs(target_path, exist_ok=True)

        VALID_SUFFIX = (".py", ".json")
        SKIP_FILES = {"config.json"}

        for fname in os.listdir(source_path):
            if fname in SKIP_FILES:
                continue
            if not fname.endswith(VALID_SUFFIX):
                continue

            src = os.path.join(source_path, fname)
            dst = os.path.join(target_path, fname)

            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print_info(f"Copied {fname}")

    @torch.no_grad()
    def save_mtp_int8_from_fp8(
        self,
        input_path: str,
        save_path: str,
        shard_name: str,
    ):
        """
        Convert MTP (model.layers.61) fp8 weights into int8 per-channel
        and save as a new safetensors shard.
        """
        weight_map = self._read_weight_map(input_path)

        cache = {}
        out_state = {}
        new_weight_map = {}

        def is_mtp(name: str) -> bool:
            return name.startswith("model.layers.61.")

        def is_scale_inv(name: str) -> bool:
            return name.endswith("_scale_inv")

        def is_quantizable_weight(name: str) -> bool:
            """
            Only real GEMM weights should be quantized.
            """
            if not name.endswith(".weight"):
                return False

            skip_keywords = [
                "norm",
                "embed_tokens",
                "shared_head",
                "e_score_correction_bias",
                "eh_proj",
                "mlp.gate",
            ]
            return not any(k in name for k in skip_keywords)

        for weight_name, src_file in weight_map.items():
            if not is_mtp(weight_name):
                continue

            if is_scale_inv(weight_name):
                continue

            tensor = self._get_tensor_from_safetensor(
                input_path, weight_name, src_file, cache
            )

            if not is_quantizable_weight(weight_name):
                out_state[weight_name] = tensor
                new_weight_map[weight_name] = shard_name
                continue

            # ---------- fp8 weight ----------
            scale_inv_name = weight_name + "_scale_inv"
            assert scale_inv_name in weight_map, f"Missing {scale_inv_name}"

            scale_inv = self._get_tensor_from_safetensor(
                input_path,
                scale_inv_name,
                weight_map[scale_inv_name],
                cache,
            )

            # ---------- fp8 → bf16 (block dequant) ----------
            w_bf16 = weight_dequant(tensor.cuda(), scale_inv.cuda()).cpu()

            # ---------- bf16 → int8 per-channel ----------
            q, scale = self.quantizer.quantize(w_bf16)

            out_state[weight_name] = q
            out_state[weight_name + "_scale"] = scale

            new_weight_map[weight_name] = shard_name
            new_weight_map[weight_name + "_scale"] = shard_name

            del tensor, scale_inv, w_bf16, q, scale

        # ---------- save safetensors ----------
        save_file(out_state, os.path.join(save_path, shard_name))

        # ---------- update index ----------
        index_file = os.path.join(save_path, "model.safetensors.index.json")
        with open(index_file, "r") as f:
            index = json.load(f)

        index["weight_map"].update(new_weight_map)

        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)

        print(f"[Done] Saved MTP int8 shard: {shard_name}")
