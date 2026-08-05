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
from tqdm import tqdm

from ....utils import (
    gathered_param_if_zero3,
    gathered_params_if_zero3,
    is_deepspeed_zero3_enabled,
    print_info,
    set_op_by_name,
    stream_load_scales,
)
from ..modules.quantizer import FP8_E4M3_QMAX, QuantLinear
from .base_plugin import BasePlugin
from .plugin_manager import PluginManager

_QKV_PROJ_MAP = {
    "q_proj": "q",
    "k_proj": "k",
    "v_proj": "v",
}


@torch.no_grad()
def initialize_nvfp4_weight_scale_2(model):
    """Initialize NVFP4 weight global scales from complete weight tensors.

    Under ZeRO-3, ``Quantizer.__init__`` cannot inspect the sharded weight and
    leaves ``scale_2`` uninitialized. Do the data-dependent initialization in
    an explicit gather before the first forward so it never depends on
    DeepSpeed's module-hook timing.
    """
    for name, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue
        quantizer = getattr(module, "weight_quantizer", None)
        if (
            quantizer is None
            or getattr(quantizer, "dtype", None) != "nvfp4"
            or getattr(quantizer, "init", False)
        ):
            continue

        with gathered_param_if_zero3(module.weight):
            if module.weight.numel() == 0:
                raise RuntimeError(f"Cannot initialize NVFP4 scale_2 from empty weight for {name}")
            global_amax = module.weight.detach().abs().amax().float()
            if not torch.isfinite(global_amax):
                raise ValueError(f"Non-finite NVFP4 weight amax for {name}")
            init_val = global_amax.clamp(min=1e-12) / quantizer.nvfp4_max_norm / FP8_E4M3_QMAX
            quantizer.scale_2.copy_(
                init_val.reshape_as(quantizer.scale_2).to(
                    device=quantizer.scale_2.device,
                    dtype=quantizer.scale_2.dtype,
                )
            )
        quantizer.init = True


@PluginManager.plugin("learnable_scale")
class LearnableScalePlugin(BasePlugin):
    def __init__(
        self,
        quant_info=None,
        ignore_layers=None,
        resume_ckpt_dir=None,
        from_ptq_ckpt_dir=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.quant_info = quant_info
        self.ignore_layers = ignore_layers
        self.resume_ckpt_dir = resume_ckpt_dir
        # Optional warm-start from a PTQ "real" checkpoint (only scales are
        # read; base weights stay as loaded by from_pretrained). Required
        # under DeepSpeed ZeRO-3.
        self.from_ptq_ckpt_dir = from_ptq_ckpt_dir
        self.use_weight_quant = self.config.get("use_weight_quant", False)
        self.use_activation_quant = self.config.get("use_activation_quant", False)
        self.fp8_attn = self.config.get("fp8_attn", False)

        # Parse learnable config (boolean switches for each parameter group)
        learnable_cfg = self.config.get("learnable", {})
        self.learn_act_scale = learnable_cfg.get("act_scale", False)
        self.learn_weight_scale = learnable_cfg.get("weight_scale", True)
        self.learn_lwc = learnable_cfg.get("lwc", False)
        self.learn_kv_scale = learnable_cfg.get("kv_scale", False)
        self.learn_norm = learnable_cfg.get("norm", False)

    def _zero3_needs_no_warmstart(self):
        """Return whether configured quantizers need no data calibration.

        FOCUS MXFP/NVFP weight scales are constant-initialized and learned,
        while their effective block scales are recomputed during forward.
        Dynamic or disabled activation quantization likewise needs no static
        calibration checkpoint.
        """
        weight_cfg = self.config.get("weight", {}) or {}
        act_cfg = self.config.get("activation", {}) or {}
        weight_qtype = str(weight_cfg.get("qtype", "")).lower()
        weight_ok = (not self.use_weight_quant) or (
            "mxfp" in weight_qtype or "nvfp" in weight_qtype
        )
        act_ok = (not self.use_activation_quant) or bool(act_cfg.get("dynamic", False))
        return weight_ok and act_ok

    def before_train(self, **kwargs):
        zero3 = is_deepspeed_zero3_enabled()
        if zero3 and not self.from_ptq_ckpt_dir and not self._zero3_needs_no_warmstart():
            raise ValueError(
                "DeepSpeed ZeRO-3 QAT requires `compression.QAT.from_ptq_ckpt` "
                "to warm-start scales (lazy_init via forward is impossible "
                "on sharded weights). This requirement is waived for FOCUS-style "
                "quantizers (MXFP/NVFP weight with constant-initialized learnable "
                "scale + dynamic/no activation quant), which need no calibration."
            )

        # Retrieve KV head count from model config for per-head quantization
        model_config = getattr(self.quant_model.model, "config", None)
        num_kv_heads = getattr(model_config, "num_key_value_heads", -1)
        # Pre-allocate ``act_quantizer.scale`` as a Parameter whenever we
        # plan to fill it from a checkpoint (full resume OR PTQ warm-start
        # OR ZeRO-3 — where lazy_init is impossible).
        act_preallocate = (
            self.resume_ckpt_dir is not None or self.from_ptq_ckpt_dir is not None or zero3
        )
        for name, module in self.quant_model.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if any(ig in name for ig in self.ignore_layers):
                    continue

                qkv_cfg = _get_qkv_config_for_layer(name, self.config)
                # Inject num_heads into qkv_config for per-head granularity
                if qkv_cfg is not None:
                    suffix = name.rsplit(".", 1)[-1]
                    if suffix in ("k_proj", "v_proj"):
                        qkv_cfg["num_heads"] = num_kv_heads
                q_linear = QuantLinear(
                    module,
                    self.config,
                    self.quant_info,
                    self.use_weight_quant,
                    self.use_activation_quant,
                    resume=act_preallocate,
                    qkv_config=qkv_cfg,
                )
                set_op_by_name(self.quant_model.model, name, q_linear)

        if zero3:
            initialize_nvfp4_weight_scale_2(self.quant_model.model)

        # FP8 attention simulation — delegate to model (model-specific override)
        if self.fp8_attn:
            self.quant_model.patch_fp8_attention()

        print_info(self.quant_model.model)

        # Warm-start scales from a previous PTQ "real" checkpoint. Only
        # quantizer Parameters are touched; base Linear weights are NOT
        # overwritten.
        if self.from_ptq_ckpt_dir is not None:
            stream_load_scales(self.quant_model.model, self.from_ptq_ckpt_dir)

        if (
            self.use_activation_quant
            and not q_linear.act_quantizer.dynamic
            and self.resume_ckpt_dir is None
            and not zero3
            and self.from_ptq_ckpt_dir is None
        ):
            self._lazy_init(**kwargs)

        self._apply_learn_strategy()

    def _apply_learn_strategy(self):
        """Set requires_grad on parameters according to ``learnable`` config."""
        model = self.quant_model.model

        # Freeze everything first
        set_quant_parameters(model, requires_grad=False)
        set_weight_parameters(model, requires_grad=False)

        # Selectively enable each parameter group based on boolean switches
        _set_learnable_parameters(
            model,
            act_scale=self.learn_act_scale,
            weight_scale=self.learn_weight_scale,
            kv_scale=self.learn_kv_scale,
            lwc=self.learn_lwc,
        )

        if self.learn_norm:
            _set_norm_parameters(model, requires_grad=True)

        learnable_summary = (
            f"act_scale={self.learn_act_scale}, "
            f"weight_scale={self.learn_weight_scale}, "
            f"kv_scale={self.learn_kv_scale}, "
            f"norm={self.learn_norm}, "
            f"lwc={self.learn_lwc}"
        )
        print_info(
            f"Learnable config ({learnable_summary}): "
            f"{sum(1 for p in model.parameters() if p.requires_grad)} trainable params"
        )

    def after_train(self, skip_weight_bake=False):
        if self.use_weight_quant and not skip_weight_bake:
            quant_inplace(self.quant_model.model)
        if skip_weight_bake:
            _mark_nvfp4_quantizers_initialized(self.quant_model.model)
        set_quant_state(
            self.quant_model.model, weight_quant=False, act_quant=self.use_activation_quant
        )
        if self.use_activation_quant:
            quant_modules = [
                (name, module)
                for name, module in self.quant_model.model.named_modules()
                if isinstance(module, QuantLinear)
            ]
            disabled = [
                name
                for name, module in quant_modules
                if not module.use_act_quant or not hasattr(module, "act_quantizer")
            ]
            if disabled:
                raise RuntimeError(
                    "Activation quantization must remain enabled for FOCUS evaluation; "
                    f"disabled modules: {disabled[:5]}"
                )
            print_info(
                f"Activation quantization enabled for {len(quant_modules)} QuantLinear modules."
            )

    def _lazy_init(self, **kwargs):
        set_quant_state(self.quant_model.model, weight_quant=False, act_quant=True, qkv_quant=True)

        init_samples = self.config.get("lazy_init_samples", 10)
        for i in tqdm(range(init_samples), desc="Lazy init"):
            batch = kwargs["train_dataset"][i]
            inputs = {
                k: torch.tensor(v).unsqueeze(0).to(self.quant_model.model.device)
                for k, v in batch.items()
                if k != "labels"
            }
            with torch.no_grad():
                self.quant_model.model(**inputs)

        for name, module in self.quant_model.model.named_modules():
            if isinstance(module, QuantLinear):
                dtype, device = module.weight.dtype, module.weight.device
                _finalize_quantizer(module.act_quantizer, dtype, device, tag="act")
                if module.use_qkv_quant:
                    _finalize_quantizer(module.qkv_quantizer, dtype, device, tag=f"qkv({name})")

        set_quant_state(self.quant_model.model, weight_quant=self.use_weight_quant, act_quant=True)


def _finalize_quantizer(quantizer, dtype, device, tag=""):
    if quantizer.dynamic:
        return
    quantizer.init = True
    if isinstance(quantizer.scale, torch.nn.Parameter):
        return
    label = getattr(quantizer, "role", tag)
    if not hasattr(quantizer, "overall_scale") or not quantizer.overall_scale:
        scale = torch.tensor(1.0, dtype=dtype, device=device)
        zp = torch.tensor(0.0, dtype=dtype, device=device) if not quantizer.is_sym else None
        quantizer._set_quant_parameters(scale, zp)
        print_info(f"Lazy init ({label}): no calib, init scale=1.0")
    else:
        scale = quantizer.overall_scale[0]
        for s in quantizer.overall_scale[1:]:
            scale = torch.maximum(scale, s)
        zp = None
        if hasattr(quantizer, "overall_zero_point") and not quantizer.is_sym:
            zp = quantizer.overall_zero_point[0]
            for z in quantizer.overall_zero_point[1:]:
                zp = torch.maximum(zp, z)
        quantizer._set_quant_parameters(scale, zp)
        print_info(
            f"Lazy init ({label}) done, scale: {quantizer.scale.max().item():.4f}, samples: {quantizer.calib_count}"  # noqa: E501
        )
        if hasattr(quantizer, "overall_zero_point"):
            del quantizer.overall_scale, quantizer.overall_zero_point, quantizer.calib_count
        else:
            del quantizer.overall_scale, quantizer.calib_count


def set_quant_state(model, weight_quant=False, act_quant=False, qkv_quant=None):
    for module in model.modules():
        if isinstance(module, QuantLinear):
            module.set_quant_state(
                weight_quant=weight_quant, act_quant=act_quant, qkv_quant=qkv_quant
            )


def _mark_nvfp4_quantizers_initialized(model):
    """Preserve loaded NVFP4 ``scale_2`` buffers on the first eval forward."""
    for module in model.modules():
        if not isinstance(module, QuantLinear):
            continue
        for attr in ("weight_quantizer", "act_quantizer", "qkv_quantizer"):
            quantizer = getattr(module, attr, None)
            if getattr(quantizer, "dtype", None) == "nvfp4" and isinstance(
                getattr(quantizer, "scale_2", None), torch.Tensor
            ):
                quantizer.init = True


def set_quant_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        if n.find("scale") > -1 or n.find("zero_point") > -1:
            m.requires_grad = requires_grad
    return iter(params)


def quant_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find("scale") > -1 or n.find("zero_point") > -1:
            params.append(m)
    return iter(params)


def set_weight_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        if n.endswith("weight") and not (n.find("scale") > -1 or n.find("zero_point") > -1):
            m.requires_grad = requires_grad
    return iter(params)


def weight_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.endswith("weight") and not (n.find("scale") > -1 or n.find("zero_point") > -1):
            params.append(m)
    return iter(params)


def trainable_parameters(model):
    params = []
    for _, m in model.named_parameters():
        if m.requires_grad:
            params.append(m)
    return iter(params)


def _set_learnable_parameters(
    model, act_scale=False, weight_scale=False, kv_scale=False, lwc=False
):
    _KV_SUFFIXES = ("k_proj", "v_proj")

    for name, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue

        if act_scale and hasattr(module, "act_quantizer"):
            for pname, param in module.act_quantizer.named_parameters():
                if "scale" in pname or "zero_point" in pname:
                    param.requires_grad = True

        if weight_scale and hasattr(module, "weight_quantizer"):
            for pname, param in module.weight_quantizer.named_parameters():
                if "scale" in pname or "zero_point" in pname:
                    param.requires_grad = True

        if lwc and hasattr(module, "weight_quantizer"):
            for pname, param in module.weight_quantizer.named_parameters():
                if "clip_factor_w_max" in pname or "clip_factor_w_min" in pname:
                    param.requires_grad = True

        if kv_scale and hasattr(module, "qkv_quantizer"):
            suffix = name.rsplit(".", 1)[-1] if "." in name else name
            if suffix in _KV_SUFFIXES:
                for pname, param in module.qkv_quantizer.named_parameters():
                    if "scale" in pname or "zero_point" in pname:
                        param.requires_grad = True


def _set_norm_parameters(model, requires_grad):
    """Enable/disable gradient for norm layer (RMSNorm / LayerNorm) weight parameters."""
    for name, param in model.named_parameters():
        # Match typical norm layer parameter names, e.g.
        #   input_layernorm.weight, post_attention_layernorm.weight, norm.weight
        if "norm" in name and "weight" in name:
            param.requires_grad = requires_grad


def _get_qkv_config_for_layer(name, quant_config):
    suffix = name.rsplit(".", 1)[-1] if "." in name else name
    qkv_key = _QKV_PROJ_MAP.get(suffix)
    if qkv_key is None:
        return None
    qkv_section = quant_config.get(qkv_key)
    if qkv_section is None:
        return None
    return dict(qkv_section)


@torch.no_grad()
def quant_inplace(model):
    for _, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue
        # Quantizer parameters are read-only here, while the weight is
        # modified. DeepSpeed requires ``modifier_rank`` for modified gathered
        # parameters; otherwise it repartitions the old ``ds_tensor`` and
        # silently discards the fake-quantized full weight.
        quantizer_params = list(module.weight_quantizer.parameters(recurse=True))
        with gathered_params_if_zero3(quantizer_params, modifier_rank=None):
            with gathered_param_if_zero3(module.weight, modifier_rank=0):
                module.weight.copy_(module.weight_quantizer(module.weight))
