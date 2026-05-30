# Copyright 2026 Tencent Inc. All Rights Reserved.
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

from typing import Callable, Unpack

import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import (
    ALL_ATTENTION_FUNCTIONS as CLIP_ALL_ATTENTION_FUNCTIONS,
)
from transformers.models.clip.modeling_clip import CLIPAttention, TransformersKwargs
from transformers.models.clip.modeling_clip import (
    eager_attention_forward as clip_eager_attention_forward,
)
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS as LLAMA_ALL_ATTENTION_FUNCTIONS,
)
from transformers.models.llama.modeling_llama import (
    BaseModelOutputWithPast,
    Cache,
    LlamaAttention,
    LlamaModel,
    apply_rotary_pos_emb,
    create_causal_mask,
)
from transformers.models.llama.modeling_llama import (
    eager_attention_forward as llama_eager_attention_forward,
)

# Import original model components from Transformers
from transformers.models.llava.modeling_llava import (
    LlavaModel,
    LlavaModelOutputWithPast,
)

from angelslim.compressor.token_compressor.factory import compression_strategy_factory

from ..base.cache import PruningCache
from ..base.config import TokenCompressorConfig

# Import AngelSlim base components
from ..base.context import PruningContext
from ..utils.mask_utils import (
    apply_pruning_mask,
    apply_token_merging,
    compensate_decoding_state,
)


class Prunable_CLIPAttention(CLIPAttention):
    def __init__(
        self,
        original_model: nn.Module,
        pruning_config: TokenCompressorConfig,
    ):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_model.__dict__)
        self.pruning_config = pruning_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context: PruningContext | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)

        attention_interface: Callable = clip_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = CLIP_ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # PRUNING START: Capture Vision Q/K States #
        assert context is not None, "PruningContext must be provided for prunable attention."
        if context is not None and hasattr(self, "layer_idx"):
            if self.pruning_config.requirements.needs_vit_q(self.layer_idx):
                context.vit_q[self.layer_idx] = queries
            if self.pruning_config.requirements.needs_vit_k(self.layer_idx):
                context.vit_k[self.layer_idx] = keys
        # PRUNING END #

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Prunable_LlamaAttention(LlamaAttention):
    def __init__(
        self,
        original_model: nn.Module,
        pruning_config: TokenCompressorConfig,
    ):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_model.__dict__)
        self.pruning_config = pruning_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        context: PruningContext | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # PRUNING START: Capture LLM States #
        assert context is not None, "PruningContext must be provided for prunable attention."
        if context is not None and hasattr(self, "layer_idx"):
            if self.pruning_config.requirements.needs_llm_q(self.layer_idx):
                context.llm_q[self.layer_idx] = query_states
            if self.pruning_config.requirements.needs_llm_k(self.layer_idx):
                context.llm_k[self.layer_idx] = key_states
            if self.pruning_config.requirements.feature_map and context.feature_map is None:
                context.feature_map = hidden_states
        # PRUNING END #

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed
            # for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = llama_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = LLAMA_ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=(0.0 if not self.training else self.attention_dropout),
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Prunable_LlamaModel(LlamaModel):
    def __init__(self, original_model: nn.Module, config: TokenCompressorConfig):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_model.__dict__)
        self.compressor_config = config
        self.pruning_fns = {
            stage: (
                compression_strategy_factory(c.strategy),
                c.params,
            )
            for stage, c in config.strategies.items()
        }

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        context: PruningContext | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and not isinstance(past_key_values, PruningCache):
            past_key_values = PruningCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position: torch.Tensor = (
                torch.arange(
                    inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )
                + past_seen_tokens
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        is_prefill = getattr(past_key_values, "is_prefill", True) if past_key_values else True

        # PRUNING START: Global Compression #
        assert context is not None, "PruningContext must be provided for prunable model."
        assert inputs_embeds.shape[0] == 1, "Batch size must be 1 for global pruning."
        if is_prefill:
            if "global" in self.pruning_fns:
                fn, p = self.pruning_fns["global"]
                res = fn(context, **p)
                if isinstance(res, (torch.Tensor, tuple)):
                    update_fn = (
                        apply_pruning_mask
                        if isinstance(res, torch.Tensor)
                        else apply_token_merging
                    )
                    # Synchronize sequence metadata
                    (
                        inputs_embeds,
                        position_ids,
                        _,
                        attention_mask,
                        cache_position,
                    ) = update_fn(
                        inputs_embeds,
                        *(res if isinstance(res, tuple) else [res]),
                        context,
                        position_ids,
                        None,  # No separate text_position_ids in Llama-1.5
                        attention_mask,
                        cache_position,
                        stage_key="global",
                        past_key_values=past_key_values,
                    )
        else:
            # Decoding Coordinate Compensation
            position_ids, cache_position, _ = compensate_decoding_state(
                position_ids,
                cache_position,
                None,
                "global",
                past_key_values,
            )
        # PRUNING END #

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                context=context,
                **kwargs,
            )

            # PRUNING START: Layer-wise Compression #
            if is_prefill:
                if layer_idx in self.pruning_fns:
                    fn, p = self.pruning_fns[layer_idx]
                    res = fn(context, **p)
                    if isinstance(res, (torch.Tensor, tuple)):
                        update_fn = (
                            apply_pruning_mask
                            if isinstance(res, torch.Tensor)
                            else apply_token_merging
                        )
                        (
                            hidden_states,
                            position_ids,
                            _,
                            causal_mask,
                            cache_position,
                        ) = update_fn(
                            hidden_states,
                            *(res if isinstance(res, tuple) else [res]),
                            context,
                            position_ids,
                            None,
                            causal_mask,
                            cache_position,
                            stage_key=layer_idx,
                            past_key_values=past_key_values,
                        )
                        # Slice RoPE for subsequent layers
                        m = res if isinstance(res, torch.Tensor) else res[-1]
                        cos, sin = position_embeddings
                        position_embeddings = (
                            cos[:, m.view(-1), :],
                            sin[:, m.view(-1), :],
                        )
            else:
                position_ids, cache_position, causal_mask = compensate_decoding_state(
                    position_ids,
                    cache_position,
                    causal_mask,
                    layer_idx,
                    past_key_values,
                )
            # PRUNING END #

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Prunable_LlavaModel(LlavaModel):
    def __init__(
        self,
        original_model: nn.Module,
        pruning_config: TokenCompressorConfig,
    ):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_model.__dict__)
        self.pruning_config = pruning_config

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        cache_position: torch.LongTensor | None = None,
        image_sizes: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | LlavaModelOutputWithPast:
        # PRUNING START: Context Init #
        context = PruningContext(
            input_ids=input_ids,
            inputs_embeds=None,
            model_config=self.config,
        )
        context.vit_layer_num = self.config.vision_config.num_hidden_layers
        # PRUNING END #

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
                return_dict=True,
                context=context,
            ).pooler_output
            image_features = torch.cat(image_features, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            special_image_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # PRUNING START: Capturing Image Info #
        context.inputs_embeds = inputs_embeds
        # PRUNING END #

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            context=context,
            **kwargs,
        )

        return LlavaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=(image_features if pixel_values is not None else None),
        )
