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

"""GLM-5 (``GlmMoeDsaForCausalLM``) model adapter for AngelSlim PTQ.

Architecture highlights (see ``config.json``):

* MLA (DeepSeek-style latent attention):
      ``q_a_proj``, ``q_b_proj``, ``kv_a_proj_with_mqa``, ``kv_b_proj``, ``o_proj``
  ``kv_b_proj`` MUST be excluded from W8A8 quantization (accuracy regression on
  the latent-to-heads projection is severe; msmodelslim's GLM-5 spec also skips
  it, along with ``wk`` and ``weights_proj`` from the DSA indexer).

* DSA (DeepSeek Sparse Attention) indexer:
      ``self_attn.indexer.wq``, ``self_attn.indexer.wk``,
      ``self_attn.indexer.weights_proj``, ``self_attn.indexer.k_norm``
  These are tiny scoring heads; skip them entirely.

* MoE (78 layers, first 3 dense + 75 sparse, 256 routed + 1 shared expert):
      ``mlp.experts.<idx>.gate_proj / up_proj / down_proj``
      ``mlp.shared_experts.gate_proj / up_proj / down_proj``
      ``mlp.gate.weight``          <- router, DO NOT quantize
      ``mlp.gate.e_score_correction_bias``

* MTP (Multi-Token Prediction) draft blocks live under
  ``model.layers.<N>.mtp_block.*`` (N == num_hidden_layers, i.e. layer 78 in
  the reference config).  Their **regular** linears (q_a_proj / q_b_proj /
  kv_a_proj_with_mqa / o_proj / gate_proj / up_proj / down_proj) are
  quantized alongside the main stack; MTP-internal ``kv_b_proj`` /
  ``indexer.*`` / ``weights_proj`` / ``mlp.gate.`` remain skipped for the
  same reasons as the main stack.
"""

import re

import torch.nn as nn

from ...compressor.quant.core import PTQSaveVllmHF
from ...utils.utils import find_layers
from ..base_model import BaseLLMModel
from ..model_factory import SlimModelFactory


# Sub-module leaf names we WANT to observe / quantize inside every
# transformer block (dense or MoE).  Everything else is auto-skipped
# by ``get_observer_layers``.
_QUANTIZABLE_LEAF_NAMES = (
    # ---- MLA attention linears ----
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",   # kept in the whitelist but pushed to ignore_layers below
    "o_proj",
    # ---- Dense MLP / MoE experts / shared experts ----
    "gate_proj",
    "up_proj",
    "down_proj",
)


# Sub-strings that must NEVER be quantized on GLM-5, regardless of the
# user-supplied ``ignore_layers`` in the YAML.  These are forced-skip.
#
# NOTE: MTP-block sub-modules are NOT force-skipped here.  MTP has its own
# copy of MLA + MoE, so its regular linears (q_a_proj / q_b_proj /
# kv_a_proj_with_mqa / o_proj / gate_proj / up_proj / down_proj) will be
# quantized just like the main stack.  The MTP-internal ``kv_b_proj`` /
# ``indexer.*`` / ``weights_proj`` / ``mlp.gate.`` still get skipped
# because the substrings below match them regardless of where they live.
_FORCED_SKIP_SUBSTRINGS = (
    # MLA latent-to-heads projection: quantizing this collapses accuracy
    "kv_b_proj",
    # DSA indexer: tiny scoring heads, not worth quantizing
    ".indexer.",
    "weights_proj",
    # MoE router (never quantize gate)
    "mlp.gate.",
    # Output head / embeddings
    "lm_head",
    "embed_tokens",
)


@SlimModelFactory.register
class GLM5(BaseLLMModel):
    """AngelSlim model adapter for ``GlmMoeDsaForCausalLM`` (GLM-5)."""

    def __init__(
        self,
        model=None,
        deploy_backend="vllm",
    ):
        super().__init__(
            model=model,
            deploy_backend=deploy_backend,
        )
        self.block_name = "model.layers"

    # ------------------------------------------------------------------
    # Observer selection
    # ------------------------------------------------------------------
    def get_observer_layers(self):
        """Select the set of nn.Linear layers to observe / quantize.

        The base ``PTQ`` pipeline uses this to:
          * install the activation-observer hook on every returned layer,
          * later swap each layer for a ``QDQModule`` at ``convert()`` time.

        Anything NOT in the returned dict is silently kept in the original
        (bf16) precision, which is exactly what we want for
        ``kv_b_proj`` / indexer / router / MTP / lm_head / embed.
        """
        obs_layer_classes = [nn.Linear]
        layers_dict = find_layers(self.model, layers=obs_layer_classes)

        # Start with any user-supplied ignore list, then append our forced
        # skips.  ``skip_layer_names()`` returns a reference to
        # ``quant_config.quant_algo_info['ignore_layers']`` so mutating it
        # here also updates PTQ's downstream views.
        ignore_layers = self.skip_layer_names()

        observer_layers_dict = {}
        for name, module in layers_dict.items():
            # (1) must live inside a transformer block
            if not name.startswith(self.block_name):
                ignore_layers.append(name)
                continue

            # (2) leaf name must be in the whitelist
            leaf = name.split(".")[-1]
            if leaf not in _QUANTIZABLE_LEAF_NAMES:
                ignore_layers.append(name)
                continue

            # (3) hard-skip patterns (kv_b_proj / indexer / router / mtp / …)
            if any(pat in name for pat in _FORCED_SKIP_SUBSTRINGS):
                ignore_layers.append(name)
                continue

            observer_layers_dict[name] = module

        # De-dup and stable-sort the ignore list.
        ignore_layers[:] = sorted(list(set(ignore_layers)))
        self.quant_config.quant_algo_info["ignore_layers"] = ignore_layers

        # Optional user-scoped filter (kept for parity with ``GLM``).
        if self.quant_config.custom_observe_layers_names != "default":
            for custom_observe_name in self.quant_config.custom_observe_layers_names:
                for default_name in list(observer_layers_dict.keys()):
                    if custom_observe_name not in default_name:
                        observer_layers_dict.pop(default_name)

        return observer_layers_dict

    # ------------------------------------------------------------------
    # SmoothQuant mapping — must reflect MLA, not q/k/v_proj
    # ------------------------------------------------------------------
    def get_smooth_mapping_layers(self, smooth_config, mappings=None):
        """Return the ``{norm_layer: (norm, [linear, ...])}`` mapping used by
        SmoothQuant.

        For GLM-5 MLA the two smoothable groups are:

          * ``input_layernorm`` -> [``q_a_proj``, ``kv_a_proj_with_mqa``]
            (the first attention linears that consume the norm output;
            ``q_b_proj`` / ``kv_b_proj`` are behind their own ``q_a_layernorm`` /
            ``kv_a_layernorm`` and are not directly smoothable here.)

          * ``post_attention_layernorm`` -> [``gate_proj``, ``up_proj``]
            for dense MLP and every MoE ``experts.*`` / ``shared_experts.*``
            block.  ``BaseLLMModel.get_smooth_mapping_layers`` walks
            ``named_modules()`` with a longest-common-prefix rule, which
            correctly picks up all expert copies.
        """
        if mappings is None:
            mappings = [
                (["q_a_proj", "kv_a_proj_with_mqa"], "input_layernorm"),
                (["gate_proj", "up_proj"], "post_attention_layernorm"),
            ]
        print(f"[GLM5] smooth mappings={mappings}")
        assert len(mappings) == 2
        assert smooth_config.smooth_first_linears or smooth_config.smooth_last_linears
        return super().get_smooth_mapping_layers(smooth_config, mappings)

    # ------------------------------------------------------------------
    # MoE experts share the same parent module (``mlp.experts``); this
    # collapses per-expert observer keys onto that parent so the saver
    # can fuse expert scales correctly.  Copied verbatim from GLM.
    # ------------------------------------------------------------------
    def get_parent_dict(self, observer_layers_dict):
        parent_mapping = {r"experts\.\d+": "experts"}
        parent_dict = {}
        for layer_name in observer_layers_dict.keys():
            parent_name = layer_name
            for k, v in parent_mapping.items():
                parent_name = re.sub(k, v, layer_name)
            if parent_name != layer_name:
                parent_dict[layer_name] = parent_name
        return parent_dict

    # ------------------------------------------------------------------
    # Fuse qkv / gate-up scales just like the base ``GLM`` adapter.
    # For MLA the natural fuse group is (q_a_proj, kv_a_proj_with_mqa)
    # sharing ``input_layernorm``; keep gate/up fused for MLP.
    # ------------------------------------------------------------------
    def fuse_observer_amax(self, sub_layer, name):
        if "q_a_proj" in name or "kv_a_proj_with_mqa" in name:
            prefix = name.rsplit(".", 1)[0]
            q_name = f"{prefix}.q_a_proj"
            kv_name = f"{prefix}.kv_a_proj_with_mqa"

            weight_scales = []
            act_scales = []
            for key in [q_name, kv_name]:
                if key in self.weight_observer_amax_dict:
                    weight_scales.append(self.weight_observer_amax_dict[key])
                if key in self.input_observer_amax_dict:
                    act_scales.append(self.input_observer_amax_dict[key])
            weight_observer_amax = max(weight_scales) if weight_scales else \
                self.weight_observer_amax_dict[name]
            input_observer_amax = max(act_scales) if act_scales else \
                self.input_observer_amax_dict[name]
        elif "gate_proj" in name or "up_proj" in name:
            prefix = name.rsplit(".", 1)[0]
            gate_name = f"{prefix}.gate_proj"
            up_name = f"{prefix}.up_proj"

            weight_scales = []
            act_scales = []
            for key in [gate_name, up_name]:
                if key in self.weight_observer_amax_dict:
                    weight_scales.append(self.weight_observer_amax_dict[key])
                if key in self.input_observer_amax_dict:
                    act_scales.append(self.input_observer_amax_dict[key])
            weight_observer_amax = max(weight_scales) if weight_scales else \
                self.weight_observer_amax_dict[name]
            input_observer_amax = max(act_scales) if act_scales else \
                self.input_observer_amax_dict[name]
        else:
            weight_observer_amax = self.weight_observer_amax_dict[name]
            input_observer_amax = self.input_observer_amax_dict[name]

        return weight_observer_amax, input_observer_amax

    # ------------------------------------------------------------------
    # Saver — reuse the standard vLLM / HF compressed-tensors saver.
    # ------------------------------------------------------------------
    def get_save_func(self):
        if self.deploy_backend in ["vllm", "huggingface"]:
            return PTQSaveVllmHF
        raise NotImplementedError(
            f"deploy_backend {self.deploy_backend} is not supported for saving."
        )
