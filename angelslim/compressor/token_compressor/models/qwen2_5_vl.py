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

from typing import Any, Callable, Optional, Unpack

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    BaseModelOutputWithPooling,
    FlashAttentionKwargs,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLAttention,
    Qwen2_5_VLModel,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLVisionAttention,
    TransformersKwargs,
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    is_flash_attention_requested,
)

from ..base.cache import PruningCache
from ..base.config import TokenCompressorConfig
from ..base.context import PruningContext
from ..factory import compression_strategy_factory
from ..utils.mask_utils import (
    apply_pruning_mask,
    apply_token_merging,
    compensate_decoding_state,
)


class Prunable_Qwen2_5_VLVisionAttention(Qwen2_5_VLVisionAttention):
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
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        context: Optional[PruningContext] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # PRUNING START #
        if context is not None:
            if self.pruning_config.requirements.needs_vit_q(self.layer_idx):
                context.vit_q[self.layer_idx] = query_states
            if self.pruning_config.requirements.needs_vit_k(self.layer_idx):
                context.vit_k[self.layer_idx] = key_states
        # PRUNING END #

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            # Flash Attention: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Prunable_Qwen2_5_VLAttention(Qwen2_5_VLAttention):
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
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        context: Optional[PruningContext] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            self.config.rope_parameters["mrope_section"],
        )

        # PRUNING START #
        if context is not None:
            if self.pruning_config.requirements.needs_llm_q(self.layer_idx):
                context.llm_q[self.layer_idx] = query_states
            if self.pruning_config.requirements.needs_llm_k(self.layer_idx):
                context.llm_k[self.layer_idx] = key_states
            if self.pruning_config.requirements.feature_map and context.feature_map is None:
                context.feature_map = hidden_states
        # PRUNING END #

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=(0.0 if not self.training else self.attention_dropout),
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,  # pass positions for FA2
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Prunable_Qwen2_5_VisionTransformer(Qwen2_5_VisionTransformerPretrainedModel):
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
        hidden_states,
        grid_thw,
        context: Optional[PruningContext] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=(grid_thw.dtype if torch.jit.is_tracing() else torch.int32),
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            dtype=(grid_thw.dtype if torch.jit.is_tracing() else torch.int32),
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                context=context,
                **kwargs,
            )

        merged_hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        merged_hidden_states = merged_hidden_states[reverse_indices, :]

        # PRUNING START #
        if context is not None:
            context.spatial_merge_size = self.spatial_merge_size
            context.cu_seqlens_full = cu_seqlens
            context.reverse_indices = reverse_indices
            context.window_index = window_index
        # PRUNING END #

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


class Prunable_Qwen2_5_VLTextModel(Qwen2_5_VLTextModel):
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
        use_cache: bool | None = None,
        context: Optional[Any] = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        """
        Forward pass for Prunable Qwen2.5-VL.
        Maintains HF source compatibility
        while integrating post-layer pruning and compensation.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if use_cache and not isinstance(past_key_values, PruningCache):
            past_key_values = PruningCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Initialize 3D position_ids (HF Standard)
        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_seen_tokens
            )
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # Extract text_position_ids for masking/packing (HF Standard)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        is_prefill = getattr(past_key_values, "is_prefill", True) if past_key_values else True

        # PRUNING START: GLOBAL STAGE #
        assert inputs_embeds.shape[0] == 1, "Batch size must be 1 for global pruning."

        # Global compression occurs BEFORE RoPE calculation to allow natural
        # coordinate alignment
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
                    # Global update: truncation for IDs, re-generation for
                    # cache_position
                    (
                        inputs_embeds,
                        position_ids,
                        text_position_ids,
                        attention_mask,
                        cache_position,
                    ) = update_fn(
                        inputs_embeds,
                        *(res if isinstance(res, tuple) else [res]),
                        context,
                        position_ids,
                        text_position_ids,
                        attention_mask,
                        None,
                        stage_key="global",
                        past_key_values=past_key_values,
                    )
        else:
            # Global Decoding Compensation: Offset indices only, mask is not
            # updated here
            text_position_ids, cache_position, _ = compensate_decoding_state(
                text_position_ids,
                None,
                None,
                "global",
                past_key_values,
            )
        # PRUNING END #

        # Initialize causal_mask_mapping (HF Standard)
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
                    **mask_kwargs
                )

        hidden_states = inputs_embeds

        # Calculate RoPE (HF Standard). Pick up updated coordinates if global
        # pruning happened.
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Decoder layers loop
        for layer_idx, decoder_layer in enumerate(self.layers):

            curr_mask = causal_mask_mapping[self.config.layer_types[layer_idx]]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=curr_mask,
                position_embeddings=position_embeddings,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                context=context,
                **kwargs,
            )

            # PRUNING START: POST-LAYER LOGIC #
            if is_prefill:
                # Apply pruning/merging AFTER the current layer call
                if layer_idx in self.pruning_fns:
                    fn, p = self.pruning_fns[layer_idx]
                    res = fn(context, **p)
                    if isinstance(res, (torch.Tensor, tuple)):
                        update_fn = (
                            apply_pruning_mask
                            if isinstance(res, torch.Tensor)
                            else apply_token_merging
                        )
                        # Update sequence with truncation logic
                        (
                            hidden_states,
                            position_ids,
                            text_position_ids,
                            curr_mask,
                            cache_position,
                        ) = update_fn(
                            hidden_states,
                            *(res if isinstance(res, tuple) else [res]),
                            context,
                            position_ids,
                            text_position_ids,
                            curr_mask,
                            None,
                            stage_key=layer_idx,
                            past_key_values=past_key_values,
                        )

                        # Physically slice RoPE cache for all subsequent layers
                        m = res if isinstance(res, torch.Tensor) else res[-1]
                        cos, sin = position_embeddings
                        m = m.view(-1)
                        position_embeddings = (
                            cos[:, :, m, :],
                            sin[:, :, m, :],
                        )
                        causal_mask_mapping[self.config.layer_types[layer_idx]] = curr_mask
            else:
                text_position_ids, cache_position, curr_mask = compensate_decoding_state(
                    text_position_ids,
                    None,
                    curr_mask,
                    layer_idx,
                    past_key_values,
                )
                causal_mask_mapping[self.config.layer_types[layer_idx]] = curr_mask
            # PRUNING END #

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Prunable_Qwen2_5_VLModel(Qwen2_5_VLModel):
    def __init__(
        self,
        original_model: nn.Module,
        pruning_config: TokenCompressorConfig,
    ):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_model.__dict__)
        self.pruning_config = pruning_config

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor`
        of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input videos.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values_videos, grid_thw=video_grid_thw, **kwargs)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        video_embeds = torch.split(vision_outputs.pooler_output, split_sizes)
        vision_outputs.pooler_output = video_embeds

        return vision_outputs

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor`
        of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(vision_outputs.pooler_output, split_sizes)
        vision_outputs.pooler_output = image_embeds

        return vision_outputs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen2_5_VLModelOutputWithPast:

        # PRUNING START #
        context = PruningContext(
            input_ids=input_ids,
            inputs_embeds=None,
            model_config=self.config,
        )
        context.image_grid_thw, context.video_grid_thw = (
            image_grid_thw,
            video_grid_thw,
        )
        context.vit_layer_num = self.config.vision_config.depth
        # PRUNING END #

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(
                pixel_values,
                image_grid_thw,
                context=context,
            ).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(
                pixel_values_videos,
                video_grid_thw,
                context=context,
            ).pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            past_key_values_length = (
                0 if past_key_values is None else past_key_values.get_seq_length()
            )
            if self.rope_deltas is None or past_key_values_length == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                delta = (past_key_values_length + self.rope_deltas).to(inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids = position_ids + delta.to(position_ids.device)

        # PRUNING START #
        context.inputs_embeds = inputs_embeds
        # PRUNING END #

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            context=context,  # Forward the context
            **kwargs,
        )

        return Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
