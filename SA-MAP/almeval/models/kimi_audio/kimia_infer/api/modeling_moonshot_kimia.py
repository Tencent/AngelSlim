# coding=utf-8
# Copyright 2025 The Moonshot AI Team, Qwen Team, and HuggingFace Inc. team. All rights reserved.
#
# The code is based on Qwen2.5-7B, but modified for KimiAudio.
#
# Licensing Information:
# - Code derived from Qwen2.5-7B is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the MIT License.
#
# Apache License, Version 2.0:
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
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""PyTorch KimiAudio model."""

import os
from typing import List, Optional, Tuple, Union
import torch
from torch import nn

import transformers
from packaging import version

assert version.parse(transformers.__version__) >= version.parse("4.34.1")

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import (
    logging,
)
from .configuration_moonshot_kimia import KimiAudioConfig
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2MLP,
    Qwen2PreTrainedModel,
)
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

if version.parse(transformers.__version__) >= version.parse("4.35.0"):
    from transformers.utils import is_flash_attn_2_available as is_flash_attn_available
else:
    from transformers.utils import is_flash_attn_available

if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
else:
    raise RuntimeError("flash attention must be installed")


logger = logging.get_logger(__name__)

prune_method = os.environ.get('method')
remain_token_ratio = float(os.environ.get('remain_token_ratio', 1.0))
threshold = float(os.environ.get('threshold', 0.0))

def _get_unpad_data(padding_mask):
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _upad_input(query_layer, key_layer, value_layer, padding_mask, query_length):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(padding_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
    num_heads = query_layer.shape[2]

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        padding_mask = padding_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
            query_layer, padding_mask
        )

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MoonshotAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: KimiAudioConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self._init_rope()

    def _init_rope(self):

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dime x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        cos = cos[position_ids]
        sin = sin[position_ids]
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # TODO: llama does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            logger.warning_once(
                "The input hidden states seems to be silently casted in float32, this might be related to"
                " the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                " float16."
            )

            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            padding_mask,
            q_len,
            dropout=dropout_rate,
        )

        if input_dtype == torch.float32:
            attn_output = attn_output.to(torch.float32)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        padding_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            padding_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        if padding_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = _upad_input(
                query_states, key_states, value_states, padding_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=True,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=True,
            )

        return attn_output


class MoonshotDecoderLayer(nn.Module):
    def __init__(self, config: KimiAudioConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config

        logger.warning_once("using normal flash attention")
        self.self_attn = MoonshotAttention(config=config)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class VQAdaptor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.kimia_adaptor_input_dim, config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, bias=True),
        )

    def forward(self, x):
        return self.layers(x)


class MoonshotKimiaModel(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`QwenDecoderLayer`]

    Args:
        config: KimiAudioConfig
    """

    config_class = KimiAudioConfig

    def __init__(self, config: KimiAudioConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.kimia_mimo_transformer_from_layer_index = (
            config.kimia_mimo_transformer_from_layer_index
        )

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [MoonshotDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # extra 1B audio transformers
        self.mimo_layers = nn.ModuleList(
            [MoonshotDecoderLayer(config) for _ in range(config.kimia_mimo_layers)]
        )
        self.mimo_norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_whisper_feature = config.use_whisper_feature
        if self.use_whisper_feature:
            self.vq_adaptor = VQAdaptor(config)
        self.kimia_media_begin = config.kimia_media_begin
        self.kimia_media_end = config.kimia_media_end

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        whisper_input_feature: Optional[torch.FloatTensor] = None,
        attentions: Optional[torch.Tensor] = None,
        attn_key: Optional[torch.Tensor] = None,
        is_continuous_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            # shape: batch, seq_len, hidden_size
            input_ids = input_ids.to(torch.cuda.current_device())
            text_input_ids = text_input_ids.to(torch.cuda.current_device())
            audio_emb = self.embed_tokens(input_ids)
            if self.use_whisper_feature and whisper_input_feature is not None:
                if not isinstance(whisper_input_feature, list):
                    whisper_input_feature = whisper_input_feature.squeeze(0)
                    whisper_input_feature = [whisper_input_feature]

                media_start_idx = (input_ids == self.kimia_media_begin).nonzero()
                media_end_idx = (input_ids == self.kimia_media_end).nonzero()
                # shape: batch, seq_len, hidden_size
                whisper_input_dim = whisper_input_feature[0].shape[-1]
                whisper_dtype = whisper_input_feature[0].dtype
                expanded_whisper = (
                    torch.zeros(audio_emb.shape[1], whisper_input_dim)
                    .to(torch.cuda.current_device())
                    .to(whisper_dtype)
                )
                assert (media_end_idx - media_start_idx).sum() - media_start_idx.shape[0] == is_continuous_mask.sum()
                for seg_idx, ((batch_idx, start_idx), (_, end_idx)) in enumerate(zip(
                    media_start_idx, media_end_idx
                )):

                    feat_len = end_idx - (start_idx + 1)
                    whisper_input_feature_i = whisper_input_feature[seg_idx].squeeze(0)
                    expanded_whisper[start_idx + 1 : end_idx, :] = (
                        whisper_input_feature_i[:feat_len, :]
                    )

                expanded_whisper = expanded_whisper.unsqueeze(0)
                whisper_emb = self.vq_adaptor(
                    expanded_whisper.transpose(0, 1)
                ).transpose(0, 1)
                is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())
                whisper_emb = whisper_emb.to(torch.cuda.current_device())
                whisper_emb = whisper_emb * is_continuous_mask[:, :, None]

                if prune_method == 'atome-merge':
                    batch_masks = torch.zeros_like(is_continuous_mask)
                    for i in range(whisper_emb.shape[0]):
                        audio_output_length = is_continuous_mask[i].sum()
                        audio_token_num = max(int(audio_output_length * remain_token_ratio),2)
                        merge_token_num =  audio_output_length - audio_token_num
                        audio_features = whisper_emb[i][is_continuous_mask[i]]
                        audio_normalized = audio_features / audio_features.norm(dim=-1, keepdim=True)
                        cos_similarities = F.cosine_similarity(audio_normalized[::2,:][:audio_normalized[1::2,:].shape[0]], audio_normalized[1::2,:], dim=1) # (seq_len - 1)
                        merge_token_num = min(merge_token_num, cos_similarities.shape[0])
                        merge_indexes = torch.topk(cos_similarities, merge_token_num, dim=-1).indices * 2
                        merge_indexes = merge_indexes.sort().values
                        h = []
                        idx = 0
                        remain_pos = []
                        while idx < audio_output_length:
                            if idx in merge_indexes:
                                h.append((audio_features[idx] + audio_features[idx + 1])/2.0)
                                remain_pos.append(idx)
                                idx += 2
                            else:
                                h.append(audio_features[idx])
                                remain_pos.append(idx)
                                idx += 1
                        audio_features = torch.stack(h)

                        if audio_features.shape[0] > audio_token_num:
                            merge_token_num =  audio_features.shape[0] - audio_token_num
                            audio_features = audio_features[:audio_token_num]
                            while merge_token_num:
                                del remain_pos[-1]
                                merge_token_num -= 1
                            remain_pos_new = remain_pos
                        else:
                            remain_pos_new = remain_pos
                        is_continuous_indices = torch.where(is_continuous_mask[i])[0]
                        whisper_emb[i][is_continuous_indices[remain_pos_new], :] = audio_features
                        batch_masks[i][~is_continuous_mask[i]] = True
                        batch_masks[i][is_continuous_indices[remain_pos_new]] = True
                    # bs=1
                    whisper_emb = whisper_emb[batch_masks].unsqueeze(0)
                    audio_emb = audio_emb[batch_masks].unsqueeze(0)
                    is_continuous_mask = is_continuous_mask[batch_masks].unsqueeze(0)
                    text_input_ids = text_input_ids[batch_masks].unsqueeze(0)

                elif prune_method == 'atome-merge-attention':
                    batch_masks = torch.zeros_like(is_continuous_mask)
                    assert attentions[0].shape[-2] == 4 * is_continuous_mask.sum().item()
                    for i in range(whisper_emb.shape[0]):
                        if type(attentions) is list:
                            attentions = attentions[0]
                        
                        attn_logits_i = attentions[i].max(dim=0).values # (N)
                        if attn_logits_i.size(-1) < attn_logits_i.size(-2):
                            attn_logits_i_a = attn_logits_i[:attn_logits_i.size(-1), :]
                            attn_logits_i_a = attn_logits_i_a.reshape(attn_logits_i.size(-1)//4, 4, attn_logits_i.size(-1)//4, 4).mean(1).sum(-1)
                            importance_a = attn_logits_i_a.mean(0).to(whisper_emb.device)
                            attn_logits_i_b = attn_logits_i[attn_logits_i.size(-1):, :]
                            attn_logits_i_b = attn_logits_i_b[:, :attn_logits_i_b.shape[0]]
                            attn_logits_i_b = attn_logits_i_b.reshape(attn_logits_i_b.shape[0]//4, 4, attn_logits_i_b.shape[1]//4, 4).mean(1).sum(-1)
                            importance_b = attn_logits_i_b.mean(0).to(whisper_emb.device)
                            importance = torch.cat([importance_a, importance_b], dim=0)
                        else:
                            attn_logits_i = attn_logits_i[:, : attn_logits_i.shape[0]]
                            attn_logits_i = attn_logits_i.reshape(attn_logits_i.shape[0]//4, 4, attn_logits_i.shape[1]//4, 4).mean(1).sum(-1)
                            importance = attn_logits_i.mean(0).to(whisper_emb.device)
                        
                        audio_output_length = is_continuous_mask[i].sum()
                        audio_token_num = max(int(audio_output_length * remain_token_ratio),2)
                        merge_token_num =  audio_output_length - audio_token_num
                        audio_features = whisper_emb[i][is_continuous_mask[i]]
                        audio_normalized = audio_features / audio_features.norm(dim=-1, keepdim=True)
                        cos_similarities = F.cosine_similarity(audio_normalized[::2,:][:audio_normalized[1::2,:].shape[0]], audio_normalized[1::2,:], dim=1) # (seq_len - 1)
                        merge_token_num = min(merge_token_num, cos_similarities.shape[0])
                        merge_indexes = torch.topk(cos_similarities, merge_token_num, dim=-1).indices * 2
                        merge_indexes = merge_indexes.sort().values
                        h = []
                        idx = 0
                        remain_pos = []
                        new_importance = []
                        while idx < audio_output_length:
                            if idx in merge_indexes:
                                merged_indexes = [idx, idx + 1]
                                h.append((importance[merged_indexes].unsqueeze(-1).expand_as(audio_features[merged_indexes]) * audio_features[merged_indexes]).sum(0) / importance[merged_indexes].sum())
                                new_importance.append(importance[merged_indexes].mean())
                                remain_pos.append(idx)
                                idx += 2
                            else:
                                h.append(audio_features[idx])
                                new_importance.append(importance[idx].mean())
                                remain_pos.append(idx)
                                idx += 1
                        audio_features = torch.stack(h)
                        if audio_features.shape[0] > audio_token_num:
                            new_importance = torch.stack(new_importance)
                            merge_token_num =  audio_features.shape[0] - audio_token_num
                            audio_normalized = audio_features / audio_features.norm(dim=-1, keepdim=True)
                            cos_similarities = F.cosine_similarity(audio_normalized[::2,:][:audio_normalized[1::2,:].shape[0]], audio_normalized[1::2,:], dim=1) # (seq_len - 1)
                            merge_token_num = min(merge_token_num, cos_similarities.shape[0])
                            merge_indexes = torch.topk(cos_similarities, merge_token_num, dim=-1).indices * 2
                            merge_indexes = merge_indexes.sort().values
                            h = []
                            idx = 0
                            remain_pos_2nd = []
                            while idx < audio_features.shape[0]:
                                if idx in merge_indexes:
                                    merged_indexes = [idx, idx + 1]
                                    h.append((new_importance[merged_indexes].unsqueeze(-1).expand_as(audio_features[merged_indexes]) * audio_features[merged_indexes]).sum(0) / new_importance[merged_indexes].sum())
                                    remain_pos_2nd.append(idx)
                                    idx += 2
                                else:
                                    h.append(audio_features[idx])
                                    remain_pos_2nd.append(idx)
                                    idx += 1
                            audio_features = torch.stack(h)
                            remain_pos_new = list(map(lambda x: x * 2, remain_pos_2nd))
                            if len(remain_pos_new) == audio_token_num + 1:
                                del remain_pos_new[-1]
                                audio_features = audio_features[:-1]
                        else:
                            remain_pos_new = remain_pos

                        is_continuous_indices = torch.where(is_continuous_mask[i])[0]
                        whisper_emb[i][is_continuous_indices[remain_pos_new], :] = audio_features
                        batch_masks[i][~is_continuous_mask[i]] = True
                        batch_masks[i][is_continuous_indices[remain_pos_new]] = True
                    # bs=1
                    whisper_emb = whisper_emb[batch_masks].unsqueeze(0)
                    audio_emb = audio_emb[batch_masks].unsqueeze(0)
                    is_continuous_mask = is_continuous_mask[batch_masks].unsqueeze(0)
                    text_input_ids = text_input_ids[batch_masks].unsqueeze(0)

                elif prune_method == 'atome-merge-attention-dpp':
                    batch_masks = torch.zeros_like(is_continuous_mask)
                    for i in range(whisper_emb.shape[0]):
                        if type(attentions) is list:
                            attentions = attentions[0]
                        audio_output_length = is_continuous_mask[i].sum()
                        audio_token_num = int(remain_token_ratio * audio_output_length)
                        audio_token_num = max(audio_token_num, 2) # T
                        merge_token_num =  int(audio_output_length - audio_token_num)
                        audio_features = whisper_emb[i][is_continuous_mask[i]]
                        audio_normalized = audio_features / audio_features.norm(dim=-1, keepdim=True)
                        cos_similarities = F.cosine_similarity(audio_normalized[::2,:][:audio_normalized[1::2,:].shape[0]], audio_normalized[1::2,:], dim=1) # (seq_len - 1)
                        cos_similarities_th = float(os.environ.get('threshold', 0.0))
                        merge_token_num = min((cos_similarities > cos_similarities_th).sum().item(), merge_token_num)
                        merge_indexes = torch.topk(cos_similarities, merge_token_num, dim=-1).indices * 2
                        merge_indexes = merge_indexes.sort().values

                        attn_logits_i = attentions[i].max(dim=0).values # (N)
                        if attn_logits_i.size(-1) < attn_logits_i.size(-2):
                            attn_logits_i_a = attn_logits_i[:attn_logits_i.size(-1), :]
                            attn_logits_i_a = attn_logits_i_a.reshape(attn_logits_i.size(-1)//4, 4, attn_logits_i.size(-1)//4, 4).mean(1).sum(-1)
                            importance_a = attn_logits_i_a.mean(0).to(whisper_emb.device)
                            attn_logits_i_b = attn_logits_i[attn_logits_i.size(-1):, :]
                            attn_logits_i_b = attn_logits_i_b[:, :attn_logits_i_b.shape[0]]
                            attn_logits_i_b = attn_logits_i_b.reshape(attn_logits_i_b.shape[0]//4, 4, attn_logits_i_b.shape[1]//4, 4).mean(1).sum(-1)
                            importance_b = attn_logits_i_b.mean(0).to(whisper_emb.device)
                            merge_importance = torch.cat([importance_a, importance_b], dim=0)
                        else:
                            attn_logits_i = attn_logits_i[:, : attn_logits_i.shape[0]]
                            attn_logits_i = attn_logits_i.reshape(attn_logits_i.shape[0]//4, 4, attn_logits_i.shape[1]//4, 4).mean(1).sum(-1)
                            merge_importance = attn_logits_i.mean(0).to(whisper_emb.device)

                        attn_logits_i = attn_logits_i.float()
                        attn_logits_i = attentions[i].max(dim=0).values # (N)
                        if attn_logits_i.size(-1) < attn_logits_i.size(-2):
                            attn_logits_i_a = attn_logits_i[:attn_logits_i.size(-1), :]
                            importance_a = torch.diagonal(attn_logits_i_a.reshape(attn_logits_i_a.shape[0]//4, 4, attn_logits_i_a.shape[1]//4, 4).mean(1).sum(-1))
                            attn_logits_i_b = attn_logits_i[attn_logits_i.size(-1):, :]
                            attn_logits_i_b = attn_logits_i_b[:, :attn_logits_i_b.shape[0]]
                            importance_b = torch.diagonal(attn_logits_i_b.reshape(attn_logits_i_b.shape[0]//4, 4, attn_logits_i_b.shape[1]//4, 4).mean(1).sum(-1))
                            prune_attn_logits_i = torch.cat([importance_a, importance_b], dim=0)
                        else:
                            attn_logits_i = attn_logits_i[:, :attn_logits_i.shape[0]]
                            prune_attn_logits_i = torch.diagonal(attn_logits_i.reshape(attn_logits_i.shape[0]//4, 4, attn_logits_i.shape[1]//4, 4).mean(1).sum(-1))

                        prune_importance = []
                        remain_pos = []
                        h = []
                        idx = 0
                        while idx < audio_output_length:
                            if idx in merge_indexes:
                                merged_indexes = [idx, idx + 1]
                                h.append((merge_importance[merged_indexes].unsqueeze(-1).expand_as(audio_features[merged_indexes]) * audio_features[merged_indexes]).sum(0) / merge_importance[merged_indexes].sum())
                                remain_pos.append(merged_indexes[0])
                                prune_importance.append(prune_attn_logits_i[merged_indexes].mean())
                                idx += 2
                            else:
                                h.append(audio_features[idx])
                                remain_pos.append(idx)
                                prune_importance.append(prune_attn_logits_i[idx])
                                idx += 1
                            
                        audio_features = torch.stack(h)
                        is_continuous_indices = torch.where(is_continuous_mask[i])[0]
                        whisper_emb[i][is_continuous_indices[remain_pos], :] = audio_features
                        batch_masks[i][~is_continuous_mask[i]] = True
                        prune_importance = torch.stack(prune_importance)
                        relevance = prune_importance.to(whisper_emb.device)

                        # dpp-pruner
                        new_audio_features_normalized = audio_features / audio_features.norm(dim=-1, keepdim=True) # (N, C)
                        similarity = torch.matmul(new_audio_features_normalized, new_audio_features_normalized.transpose(0, 1))
                        cur_audio_output_length = audio_features.shape[0]
                        kernel = relevance.unsqueeze(1) * similarity * relevance.unsqueeze(0)
                        # [CDPruner] Fast MAP inference of conditional DPP
                        cis = torch.zeros((audio_token_num, cur_audio_output_length), device=audio_features.device)
                        di2s = torch.diagonal(kernel, dim1=0, dim2=1).clone()
                        select_idx = torch.empty(audio_token_num, dtype=torch.long, device=audio_features.device)
                        for index in range(audio_token_num):
                            k = torch.argmax(di2s, dim=-1)
                            select_idx[index] = k
                            eis = (kernel[k] - torch.einsum('t,tn->n', cis[:index, k], cis[:index])) / torch.sqrt(di2s[k])
                            cis[index, :] = eis
                            di2s -= torch.square(eis)
                            di2s[k] = -float('inf')
                        select_mask = torch.zeros(cur_audio_output_length, dtype=torch.bool)
                        select_mask[select_idx] = True
                        batch_masks[i][is_continuous_indices[remain_pos][select_mask]] = True
                        print(f'after prune num: {select_idx.shape}')
                    # bs=1
                    whisper_emb = whisper_emb[batch_masks].unsqueeze(0)
                    audio_emb = audio_emb[batch_masks].unsqueeze(0)
                    is_continuous_mask = is_continuous_mask[batch_masks].unsqueeze(0)
                    text_input_ids = text_input_ids[batch_masks].unsqueeze(0)
                
                elif prune_method == 'vispruner':
                    batch_masks = torch.zeros_like(is_continuous_mask)
                    for i in range(whisper_emb.shape[0]):
                        audio_output_length = is_continuous_mask[i].sum()
                        audio_token_num = max(int(audio_output_length * remain_token_ratio),2) # T
                        important_ratio = 0.5
                        important_token_num = round(audio_token_num * important_ratio) # T_imp = T * r
                        diverse_token_num = audio_token_num - important_token_num # T_div = T * (1 - r)
                        if type(attentions) is list:
                            attentions = attentions[0]
                        attn_logits_i = attentions[i].max(dim=0).values # (N)
                        if attn_logits_i.size(-1) < attn_logits_i.size(-2):
                            attn_logits_i_a = attn_logits_i[:attn_logits_i.size(-1), :]
                            attn_logits_i_a = attn_logits_i_a.reshape(attn_logits_i.size(-1)//4, 4, attn_logits_i.size(-1)//4, 4).mean(1).sum(-1)
                            attn_logits_i_a = attn_logits_i_a.mean(0).to(whisper_emb.device)
                            attn_logits_i_b = attn_logits_i[attn_logits_i.size(-1):, :]
                            attn_logits_i_b = attn_logits_i_b[:, :attn_logits_i_b.shape[0]]
                            attn_logits_i_b = attn_logits_i_b.reshape(attn_logits_i_b.shape[0]//4, 4, attn_logits_i_b.shape[1]//4, 4).mean(1).sum(-1)
                            attn_logits_i_b = attn_logits_i_b.mean(0).to(whisper_emb.device)
                            attn_logits_i = torch.cat([attn_logits_i_a, attn_logits_i_b], dim=0)
                        else:
                            attn_logits_i = attn_logits_i[:, : attn_logits_i.shape[0]]
                            attn_logits_i = attn_logits_i.reshape(attn_logits_i.shape[0]//4, 4, attn_logits_i.shape[1]//4, 4).mean(1).sum(-1)
                            attn_logits_i = attn_logits_i.mean(0).to(whisper_emb.device)

                        token_indices = attn_logits_i.argsort(dim=-1, descending=True) # (N)
                        important_indices = token_indices[:important_token_num] # (T_imp)
                        residual_indices,_ = token_indices[important_token_num:].sort() # (N - T_imp)
                        audio_normalized = whisper_emb[i][is_continuous_mask[i]] / whisper_emb[i][is_continuous_mask[i]].norm(dim=-1, keepdim=True) # (N, C)
                        while diverse_token_num > 0:
                            R = residual_indices.shape[0]
                            r = int(min(4, R - diverse_token_num))
                            if r <= 0:
                                break
                            residual_tokens = audio_normalized[residual_indices] # (R, C)
                            a, b = residual_tokens[::2, :], residual_tokens[1::2, :] # (R // 2, C)
                            scores = a @ b.transpose(-1, -2) # (R // 2, R // 2)
                            scores = scores.max(dim=-1).values # (R // 2)
                            distinct_indices = scores.argsort(dim=-1, descending=True)[r:] # (R // 2 - r)
                            residual_indices = torch.cat([
                                residual_indices[::2][distinct_indices], 
                                residual_indices[1::2]
                            ], dim=-1) # (R - r)
                        if diverse_token_num > 0:
                            selected_indices = torch.cat([important_indices, residual_indices], dim=-1) # (T)
                        else:
                            selected_indices = important_indices # (T)
                        index_mask = torch.zeros_like(is_continuous_mask[i], dtype=torch.bool)
                        index_mask[~is_continuous_mask[i]] = True
                        is_continuous_indices = torch.where(is_continuous_mask[i])[0]
                        index_mask[is_continuous_indices[selected_indices]] = True
                        batch_masks[i] = index_mask
                    # bs=1
                    whisper_emb = whisper_emb[batch_masks].unsqueeze(0)
                    audio_emb = audio_emb[batch_masks].unsqueeze(0)
                    is_continuous_mask = is_continuous_mask[batch_masks].unsqueeze(0)
                    text_input_ids = text_input_ids[batch_masks].unsqueeze(0)

                elif prune_method == "cdpruner":
                    batch_masks = torch.zeros_like(is_continuous_mask)
                    for i in range(whisper_emb.shape[0]):
                        audio_features_normalized = whisper_emb[i][is_continuous_mask[i]] / whisper_emb[i][is_continuous_mask[i]].norm(dim=-1, keepdim=True)
                        audio_features_normalized = audio_features_normalized.float()
                        similarity = torch.matmul(audio_features_normalized, audio_features_normalized.transpose(0, 1))
                        kernel = similarity
                        remain_token_num = int(remain_token_ratio * torch.tensor(audio_features_normalized.shape[0]))
                        cis = torch.zeros((remain_token_num, audio_features_normalized.shape[0]), device=whisper_emb.device)
                        di2s = torch.diagonal(kernel, dim1=0, dim2=1).clone()
                        select_idx = torch.empty(remain_token_num, dtype=torch.long, device=whisper_emb.device)
                        for index in range(remain_token_num):
                            k = torch.argmax(di2s, dim=-1)
                            select_idx[index] = k
                            eis = (kernel[k] - torch.einsum('t,tn->n', cis[:index, k], cis[:index])) / torch.sqrt(di2s[k])
                            cis[index, :] = eis
                            di2s -= torch.square(eis)
                            di2s[k] = -float('inf')
                        is_continuous_indices = torch.where(is_continuous_mask[i])[0]
                        select_mask = torch.zeros(whisper_emb[i].shape[0], dtype=torch.bool)
                        select_mask[is_continuous_indices[select_idx]] = True
                        select_mask[~is_continuous_mask[i]] = True
                        batch_masks[i] = select_mask
                    # bs=1
                    whisper_emb = whisper_emb[batch_masks].unsqueeze(0)
                    audio_emb = audio_emb[batch_masks].unsqueeze(0)
                    is_continuous_mask = is_continuous_mask[batch_masks].unsqueeze(0)
                    text_input_ids = text_input_ids[batch_masks].unsqueeze(0)

                elif prune_method == "cdpruner-attention":
                    batch_masks = torch.zeros_like(is_continuous_mask)
                    assert attentions[0].shape[-2] == 4 * is_continuous_mask.sum().item()
                    for i in range(whisper_emb.shape[0]):
                        audio_features_normalized = whisper_emb[i][is_continuous_mask[i]] / whisper_emb[i][is_continuous_mask[i]].norm(dim=-1, keepdim=True)
                        audio_features_normalized = audio_features_normalized.float()
                        similarity = torch.matmul(audio_features_normalized, audio_features_normalized.transpose(0, 1))
                        if type(attentions) is list:
                            attentions = attentions[0]

                        attn_logits_i = attentions[i].max(dim=0).values # (N)
                        if attn_logits_i.size(-1) < attn_logits_i.size(-2):
                            attn_logits_i_a = attn_logits_i[:attn_logits_i.size(-1), :]
                            importance_a = torch.diagonal(attn_logits_i_a.reshape(attn_logits_i_a.shape[0]//4, 4, attn_logits_i_a.shape[1]//4, 4).mean(1).sum(-1))
                            attn_logits_i_b = attn_logits_i[attn_logits_i.size(-1):, :]
                            attn_logits_i_b = attn_logits_i_b[:, :attn_logits_i_b.shape[0]]
                            importance_b = torch.diagonal(attn_logits_i_b.reshape(attn_logits_i_b.shape[0]//4, 4, attn_logits_i_b.shape[1]//4, 4).mean(1).sum(-1))
                            importance = torch.cat([importance_a, importance_b], dim=0)
                        else:
                            attn_logits_i = attn_logits_i[:, :attn_logits_i.shape[0]]
                            importance = torch.diagonal(attn_logits_i.reshape(attn_logits_i.shape[0]//4, 4, attn_logits_i.shape[1]//4, 4).mean(1).sum(-1))
                        relevance = importance.to(whisper_emb.device)
                        kernel = relevance.unsqueeze(1) * similarity * relevance.unsqueeze(0)
                        # [CDPruner] Fast MAP inference of conditional DPP
                        remain_token_num = int(remain_token_ratio * torch.tensor(audio_features_normalized.shape[0]))
                        cis = torch.zeros((remain_token_num, audio_features_normalized.shape[0]), device=whisper_emb.device)
                        di2s = torch.diagonal(kernel, dim1=0, dim2=1).clone()
                        select_idx = torch.empty(remain_token_num, dtype=torch.long, device=whisper_emb.device)
                        for index in range(remain_token_num):
                            k = torch.argmax(di2s, dim=-1)
                            select_idx[index] = k
                            eis = (kernel[k] - torch.einsum('t,tn->n', cis[:index, k], cis[:index])) / torch.sqrt(di2s[k])
                            cis[index, :] = eis
                            di2s -= torch.square(eis)
                            di2s[k] = -float('inf')
                        is_continuous_indices = torch.where(is_continuous_mask[i])[0]
                        select_mask = torch.zeros(whisper_emb[i].shape[0], dtype=torch.bool)
                        select_mask[is_continuous_indices[select_idx]] = True
                        select_mask[~is_continuous_mask[i]] = True
                        batch_masks[i] = select_mask
                    # bs=1
                    whisper_emb = whisper_emb[batch_masks].unsqueeze(0)
                    audio_emb = audio_emb[batch_masks].unsqueeze(0)
                    is_continuous_mask = is_continuous_mask[batch_masks].unsqueeze(0)
                    text_input_ids = text_input_ids[batch_masks].unsqueeze(0)
                
                elif prune_method == "fastadasp":
                    batch_masks = torch.zeros_like(is_continuous_mask)
                    for i in range(whisper_emb.shape[0]):
                        audio_output_length = is_continuous_mask[i].sum()
                        audio_token_num = int(remain_token_ratio * audio_output_length)
                        audio_token_num = max(audio_token_num, 2) # T
                        merge_token_num =  audio_output_length - audio_token_num
                        audio_features = whisper_emb[i][is_continuous_mask[i]]
                        audio_normalized = audio_features / audio_features.norm(dim=-1, keepdim=True)
                        cos_similarities = F.cosine_similarity(audio_normalized[:-1,:], audio_normalized[1:,:], dim=1) # (seq_len - 1)
                        selected_indices = torch.topk(cos_similarities, merge_token_num, dim=-1).indices
                        selected_indices = selected_indices.sort().values
                        h = []
                        idx = 0
                        remain_pos = []
                        while idx < audio_output_length:
                            merged_indexes = []
                            while idx in selected_indices:
                                merged_indexes.append(idx)
                                idx += 1
                            merged_indexes.append(idx)
                            remain_pos.append(merged_indexes[0])
                            h.append(audio_features[merged_indexes].mean(0))
                            idx+=1
                        audio_features = torch.stack(h)
                        is_continuous_indices = torch.where(is_continuous_mask[i])[0]
                        whisper_emb[i][is_continuous_indices[remain_pos], :] = audio_features
                        batch_masks[i][~is_continuous_mask[i]] = True
                        batch_masks[i][is_continuous_indices[remain_pos]] = True
                    # bs=1
                    whisper_emb = whisper_emb[batch_masks].unsqueeze(0)
                    audio_emb = audio_emb[batch_masks].unsqueeze(0)
                    is_continuous_mask = is_continuous_mask[batch_masks].unsqueeze(0)
                    text_input_ids = text_input_ids[batch_masks].unsqueeze(0)

                elif prune_method == "fastadasp-attention":
                    batch_masks = torch.zeros_like(is_continuous_mask)
                    assert attentions[0].shape[-2] == 4 * is_continuous_mask.sum().item()
                    for i in range(whisper_emb.shape[0]):
                        if type(attentions) is list:
                            attentions = attentions[0]

                        attn_logits_i = attentions[i].max(dim=0).values # (N)
                        if attn_logits_i.size(-1) < attn_logits_i.size(-2):
                            attn_logits_i_a = attn_logits_i[:attn_logits_i.size(-1), :]
                            attn_logits_i_a = attn_logits_i_a.reshape(attn_logits_i.size(-1)//4, 4, attn_logits_i.size(-1)//4, 4).mean(1).sum(-1)
                            importance_a = attn_logits_i_a.mean(0).to(whisper_emb.device)
                            attn_logits_i_b = attn_logits_i[attn_logits_i.size(-1):, :]
                            attn_logits_i_b = attn_logits_i_b[:, :attn_logits_i_b.shape[0]]
                            attn_logits_i_b = attn_logits_i_b.reshape(attn_logits_i_b.shape[0]//4, 4, attn_logits_i_b.shape[1]//4, 4).mean(1).sum(-1)
                            importance_b = attn_logits_i_b.mean(0).to(whisper_emb.device)
                            importance = torch.cat([importance_a, importance_b], dim=0)
                        else:
                            attn_logits_i = attn_logits_i[:, : attn_logits_i.shape[0]]
                            attn_logits_i = attn_logits_i.reshape(attn_logits_i.shape[0]//4, 4, attn_logits_i.shape[1]//4, 4).mean(1).sum(-1)
                            importance = attn_logits_i.mean(0).to(whisper_emb.device)
                        
                        audio_output_length = is_continuous_mask[i].sum()
                        audio_token_num = int(remain_token_ratio * audio_output_length)
                        audio_token_num = max(audio_token_num, 2) # T
                        merge_token_num =  audio_output_length - audio_token_num
                        audio_features = whisper_emb[i][is_continuous_mask[i]]
                        audio_normalized = audio_features / audio_features.norm(dim=-1, keepdim=True)
                        cos_similarities = F.cosine_similarity(audio_normalized[:-1,:], audio_normalized[1:,:], dim=1) # (seq_len - 1)
                        selected_indices = torch.topk(cos_similarities, merge_token_num, dim=-1).indices
                        selected_indices = selected_indices.sort().values
                        h = []
                        idx = 0
                        remain_pos = []
                        while idx < audio_output_length:
                            merged_indexes = []
                            while idx in selected_indices:
                                merged_indexes.append(idx)
                                idx += 1
                            merged_indexes.append(idx)
                            remain_pos.append(merged_indexes[0])
                            h.append((importance[merged_indexes].unsqueeze(-1).expand_as(audio_features[merged_indexes]) * audio_features[merged_indexes]).sum(0) / importance[merged_indexes].sum())
                            idx+=1
                        audio_features = torch.stack(h)
                        is_continuous_indices = torch.where(is_continuous_mask[i])[0]
                        whisper_emb[i][is_continuous_indices[remain_pos], :] = audio_features
                        batch_masks[i][~is_continuous_mask[i]] = True
                        batch_masks[i][is_continuous_indices[remain_pos]] = True
                    # bs=1
                    whisper_emb = whisper_emb[batch_masks].unsqueeze(0)
                    audio_emb = audio_emb[batch_masks].unsqueeze(0)
                    is_continuous_mask = is_continuous_mask[batch_masks].unsqueeze(0)
                    text_input_ids = text_input_ids[batch_masks].unsqueeze(0)
                
                elif prune_method == "fastadasp-attention-dpp":
                    batch_masks = torch.zeros_like(is_continuous_mask)
                    assert attentions[0].shape[-2] == 4 * is_continuous_mask.sum().item()
                    for i in range(whisper_emb.shape[0]):
                        if type(attentions) is list:
                            attentions = attentions[0]
                        audio_output_length = is_continuous_mask[i].sum()
                        audio_token_num = int(remain_token_ratio * audio_output_length)
                        audio_token_num = max(audio_token_num, 2) # T
                        merge_token_num =  int(audio_output_length - audio_token_num)
                        audio_features = whisper_emb[i][is_continuous_mask[i]]
                        audio_normalized = audio_features / audio_features.norm(dim=-1, keepdim=True)
                        cos_similarities = F.cosine_similarity(audio_normalized[:-1,:], audio_normalized[1:,:], dim=1) # (seq_len - 1)
                        cos_similarities_th = float(os.environ.get('threshold', 0.0))
                        merge_token_num = min((cos_similarities > cos_similarities_th).sum().item(), merge_token_num)
                        selected_indices = torch.topk(cos_similarities, merge_token_num, dim=-1).indices
                        selected_indices = selected_indices.sort().values
                        
                        attn_logits_i = attentions[i].max(dim=0).values # (N)
                        if attn_logits_i.size(-1) < attn_logits_i.size(-2):
                            attn_logits_i_a = attn_logits_i[:attn_logits_i.size(-1), :]
                            attn_logits_i_a = attn_logits_i_a.reshape(attn_logits_i.size(-1)//4, 4, attn_logits_i.size(-1)//4, 4).mean(1).sum(-1)
                            importance_a = attn_logits_i_a.mean(0).to(whisper_emb.device)
                            attn_logits_i_b = attn_logits_i[attn_logits_i.size(-1):, :]
                            attn_logits_i_b = attn_logits_i_b[:, :attn_logits_i_b.shape[0]]
                            attn_logits_i_b = attn_logits_i_b.reshape(attn_logits_i_b.shape[0]//4, 4, attn_logits_i_b.shape[1]//4, 4).mean(1).sum(-1)
                            importance_b = attn_logits_i_b.mean(0).to(whisper_emb.device)
                            merge_importance = torch.cat([importance_a, importance_b], dim=0)
                        else:
                            attn_logits_i = attn_logits_i[:, : attn_logits_i.shape[0]]
                            attn_logits_i = attn_logits_i.reshape(attn_logits_i.shape[0]//4, 4, attn_logits_i.shape[1]//4, 4).mean(1).sum(-1)
                            merge_importance = attn_logits_i.mean(0).to(whisper_emb.device)

                        attn_logits_i = attn_logits_i.float()
                        attn_logits_i = attentions[i].max(dim=0).values # (N)
                        if attn_logits_i.size(-1) < attn_logits_i.size(-2):
                            attn_logits_i_a = attn_logits_i[:attn_logits_i.size(-1), :]
                            importance_a = torch.diagonal(attn_logits_i_a.reshape(attn_logits_i_a.shape[0]//4, 4, attn_logits_i_a.shape[1]//4, 4).mean(1).sum(-1))

                            attn_logits_i_b = attn_logits_i[attn_logits_i.size(-1):, :]
                            attn_logits_i_b = attn_logits_i_b[:, :attn_logits_i_b.shape[0]]
                            importance_b = torch.diagonal(attn_logits_i_b.reshape(attn_logits_i_b.shape[0]//4, 4, attn_logits_i_b.shape[1]//4, 4).mean(1).sum(-1))
                            prune_attn_logits_i = torch.cat([importance_a, importance_b], dim=0)
                        else:
                            attn_logits_i = attn_logits_i[:, :attn_logits_i.shape[0]]
                            prune_attn_logits_i = torch.diagonal(attn_logits_i.reshape(attn_logits_i.shape[0]//4, 4, attn_logits_i.shape[1]//4, 4).mean(1).sum(-1))

                        prune_importance = []
                        remain_pos = []
                        h = []
                        idx = 0
                        while idx < audio_output_length:
                            merged_indexes = []
                            while idx in selected_indices:
                                merged_indexes.append(idx)
                                idx += 1
                            merged_indexes.append(idx)
                            remain_pos.append(merged_indexes[0])
                            h.append((merge_importance[merged_indexes].unsqueeze(-1).expand_as(audio_features[merged_indexes]) * audio_features[merged_indexes]).sum(0) / merge_importance[merged_indexes].sum())
                            prune_importance.append(prune_attn_logits_i[merged_indexes].mean())
                            idx+=1
                        audio_features = torch.stack(h)
                        is_continuous_indices = torch.where(is_continuous_mask[i])[0]
                        whisper_emb[i][is_continuous_indices[remain_pos], :] = audio_features
                        batch_masks[i][~is_continuous_mask[i]] = True
                        
                        prune_importance = torch.stack(prune_importance)
                        relevance = prune_importance.to(whisper_emb.device)
                        # dpp-pruner
                        new_audio_features_normalized = audio_features / audio_features.norm(dim=-1, keepdim=True) # (N, C)
                        new_audio_features_normalized = new_audio_features_normalized.float()
                        similarity = torch.matmul(new_audio_features_normalized, new_audio_features_normalized.transpose(0, 1))
                        cur_audio_output_length = audio_features.shape[0]
                        
                        kernel = relevance.unsqueeze(1) * similarity * relevance.unsqueeze(0)
                        # [CDPruner] Fast MAP inference of conditional DPP
                        cis = torch.zeros((audio_token_num, cur_audio_output_length), device=audio_features.device)
                        di2s = torch.diagonal(kernel, dim1=0, dim2=1).clone()
                        select_idx = torch.empty(audio_token_num, dtype=torch.long, device=audio_features.device)
                        for index in range(audio_token_num):
                            k = torch.argmax(di2s, dim=-1)
                            select_idx[index] = k
                            eis = (kernel[k] - torch.einsum('t,tn->n', cis[:index, k], cis[:index])) / torch.sqrt(di2s[k])
                            cis[index, :] = eis
                            di2s -= torch.square(eis)
                            di2s[k] = -float('inf')
                        select_mask = torch.zeros(cur_audio_output_length, dtype=torch.bool)
                        select_mask[select_idx] = True
                        batch_masks[i][is_continuous_indices[remain_pos][select_mask]] = True
                    # bs=1
                    whisper_emb = whisper_emb[batch_masks].unsqueeze(0)
                    audio_emb = audio_emb[batch_masks].unsqueeze(0)
                    is_continuous_mask = is_continuous_mask[batch_masks].unsqueeze(0)
                    text_input_ids = text_input_ids[batch_masks].unsqueeze(0)
                
                elif prune_method == "SA-MAP":
                    batch_masks = torch.zeros_like(is_continuous_mask)
                    assert attentions[0].shape[-2] == 4 * is_continuous_mask.sum().item()
                    for i in range(whisper_emb.shape[0]):
                        if type(attentions) is list:
                            attentions = attentions[0]
                        
                        audio_output_length = is_continuous_mask[i].sum()
                        audio_token_num = int(remain_token_ratio * audio_output_length)
                        audio_token_num = max(audio_token_num, 2) # T
                        audio_features = whisper_emb[i][is_continuous_mask[i]]
                        audio_normalized = audio_features / audio_features.norm(dim=-1, keepdim=True)
                        audio_normalized = audio_normalized.float()
                        similarity = torch.matmul(audio_normalized, audio_normalized.transpose(0, 1))

                        attn_logits_i = attentions[i].max(dim=0).values # (N)
                        if attn_logits_i.size(-1) < attn_logits_i.size(-2):
                            attn_logits_i_a = attn_logits_i[:attn_logits_i.size(-1), :]
                            attn_logits_i_a = attn_logits_i_a.reshape(attn_logits_i.size(-1)//4, 4, attn_logits_i.size(-1)//4, 4).mean(1).sum(-1)
                            importance_a = attn_logits_i_a.mean(0).to(whisper_emb.device)
                            attn_logits_i_b = attn_logits_i[attn_logits_i.size(-1):, :]
                            attn_logits_i_b = attn_logits_i_b[:, :attn_logits_i_b.shape[0]]
                            attn_logits_i_b = attn_logits_i_b.reshape(attn_logits_i_b.shape[0]//4, 4, attn_logits_i_b.shape[1]//4, 4).mean(1).sum(-1)
                            importance_b = attn_logits_i_b.mean(0).to(whisper_emb.device)
                            merge_importance = torch.cat([importance_a, importance_b], dim=0)
                        else:
                            attn_logits_i = attn_logits_i[:, : attn_logits_i.shape[0]]
                            attn_logits_i = attn_logits_i.reshape(attn_logits_i.shape[0]//4, 4, attn_logits_i.shape[1]//4, 4).mean(1).sum(-1)
                            merge_importance = attn_logits_i.mean(0).to(whisper_emb.device)

                        attn_logits_i = attn_logits_i.float()
                        attn_logits_i = attentions[i].max(dim=0).values # (N)
                        if attn_logits_i.size(-1) < attn_logits_i.size(-2):
                            attn_logits_i_a = attn_logits_i[:attn_logits_i.size(-1), :]
                            importance_a = torch.diagonal(attn_logits_i_a.reshape(attn_logits_i_a.shape[0]//4, 4, attn_logits_i_a.shape[1]//4, 4).mean(1).sum(-1))
                            attn_logits_i_b = attn_logits_i[attn_logits_i.size(-1):, :]
                            attn_logits_i_b = attn_logits_i_b[:, :attn_logits_i_b.shape[0]]
                            importance_b = torch.diagonal(attn_logits_i_b.reshape(attn_logits_i_b.shape[0]//4, 4, attn_logits_i_b.shape[1]//4, 4).mean(1).sum(-1))
                            prune_attn_logits_i = torch.cat([importance_a, importance_b], dim=0)
                        else:
                            attn_logits_i = attn_logits_i[:, :attn_logits_i.shape[0]]
                            prune_attn_logits_i = torch.diagonal(attn_logits_i.reshape(attn_logits_i.shape[0]//4, 4, attn_logits_i.shape[1]//4, 4).mean(1).sum(-1))
                            
                        prune_importance = []
                        idx = 0
                        h = []
                        merged_indexes = []
                        remain_pos = []
                        merged_groups = []
                        group_sim_score = {}
                        group_position = 0
                        while idx < audio_output_length:
                            merged_indexes.append(idx)
                            if idx == audio_output_length-1:
                                h.append((merge_importance[merged_indexes].unsqueeze(-1).expand_as(audio_features[merged_indexes]) * audio_features[merged_indexes]).sum(0) / merge_importance[merged_indexes].sum())
                                prune_importance.append(prune_attn_logits_i[merged_indexes].mean())
                                remain_pos.append(merged_indexes[0])
                                merged_groups.append(merged_indexes)
                                break
                            while idx+1 < audio_output_length:
                                if similarity[merged_indexes][:,idx+1].mean() >= threshold:
                                    merged_indexes.append(idx+1)
                                    if str(group_position) not in group_sim_score.keys():
                                        group_sim_score[str(group_position)] = [similarity[merged_indexes][:,idx+1].mean()]
                                    else:
                                        group_sim_score[str(group_position)].append(similarity[merged_indexes][:,idx+1].mean())
                                    idx+=1
                                    if idx == audio_output_length-1:
                                        h.append((merge_importance[merged_indexes].unsqueeze(-1).expand_as(audio_features[merged_indexes]) * audio_features[merged_indexes]).sum(0) / merge_importance[merged_indexes].sum())
                                        prune_importance.append(prune_attn_logits_i[merged_indexes].mean())
                                        remain_pos.append(merged_indexes[0])
                                        merged_groups.append(merged_indexes)
                                else:
                                    h.append((merge_importance[merged_indexes].unsqueeze(-1).expand_as(audio_features[merged_indexes]) * audio_features[merged_indexes]).sum(0) / merge_importance[merged_indexes].sum())
                                    prune_importance.append(prune_attn_logits_i[merged_indexes].mean())
                                    remain_pos.append(merged_indexes[0])
                                    merged_groups.append(merged_indexes)
                                    group_position += 1
                                    merged_indexes = []
                                    break
                            idx+=1
                        
                        if len(h) == audio_token_num:
                            audio_features = torch.stack(h)
                            is_continuous_indices = torch.where(is_continuous_mask[i])[0]
                            whisper_emb[i][is_continuous_indices[remain_pos], :] = audio_features
                            batch_masks[i][~is_continuous_mask[i]] = True
                            batch_masks[i][is_continuous_indices[remain_pos]] = True
                        elif len(h) < audio_token_num:
                            need_release = audio_token_num - len(h)
                            scores = []
                            '''
                            group_sim_score {"1": [0.85]
                                            "2": [0.9, 0.93]}
                            merged_groups [
                                                [0]
                                                [1,2]
                                                [3,4,5]
                                            ]
                            '''
                            for key in group_sim_score.keys():
                                scores.extend(group_sim_score[key])
                            assert len(set(scores)) == len(scores)
                            need_release_score = sorted(scores)[:need_release]
                            h = []
                            remain_pos = []
                            for group_idx, group in enumerate(merged_groups):
                                if len(group) == 1:
                                    h.append(audio_features[group].mean(0))
                                    remain_pos.append(group[0])
                                else:
                                    new_group = []
                                    mask = torch.ones(len(group), dtype=bool, device=audio_features.device)
                                    for idx, sim in enumerate(group_sim_score[str(group_idx)]):
                                        if sim in need_release_score:
                                            mask[idx+1] = False
                                            new_group.append(group[idx+1])
                                    ori_group = torch.tensor(group, device=audio_features.device)
                                    h.append((merge_importance[ori_group[mask].tolist()].unsqueeze(-1).expand_as(audio_features[ori_group[mask].tolist()]) * audio_features[ori_group[mask].tolist()]).sum(0) / merge_importance[ori_group[mask].tolist()].sum())
                                    remain_pos.append(ori_group[mask].tolist()[0])
                                    for item in new_group:
                                        h.append(audio_features[item])
                                        remain_pos.append(item)
                            audio_features = torch.stack(h)
                            is_continuous_indices = torch.where(is_continuous_mask[i])[0]
                            whisper_emb[i][is_continuous_indices[remain_pos], :] = audio_features
                            batch_masks[i][~is_continuous_mask[i]] = True
                            batch_masks[i][is_continuous_indices[remain_pos]] = True
                        else:
                            audio_features = torch.stack(h)
                            is_continuous_indices = torch.where(is_continuous_mask[i])[0]
                            whisper_emb[i][is_continuous_indices[remain_pos], :] = audio_features
                            batch_masks[i][~is_continuous_mask[i]] = True
                            prune_importance = torch.stack(prune_importance)
                            relevance = prune_importance.to(whisper_emb.device)
                            new_audio_features_normalized = audio_features / audio_features.norm(dim=-1, keepdim=True) # (N, C)
                            new_audio_features_normalized = new_audio_features_normalized.float()
                            similarity = torch.matmul(new_audio_features_normalized, new_audio_features_normalized.transpose(0, 1))
                            cur_audio_output_length = audio_features.shape[0]
                            kernel = relevance.unsqueeze(1) * similarity * relevance.unsqueeze(0)
                            # [CDPruner] Fast MAP inference of conditional DPP
                            cis = torch.zeros((audio_token_num, cur_audio_output_length), device=audio_features.device)
                            di2s = torch.diagonal(kernel, dim1=0, dim2=1).clone()
                            select_idx = torch.empty(audio_token_num, dtype=torch.long, device=audio_features.device)
                            for index in range(audio_token_num):
                                k = torch.argmax(di2s, dim=-1)
                                select_idx[index] = k
                                eis = (kernel[k] - torch.einsum('t,tn->n', cis[:index, k], cis[:index])) / torch.sqrt(di2s[k])
                                cis[index, :] = eis
                                di2s -= torch.square(eis)
                                di2s[k] = -float('inf')
                            assert not torch.isnan(cis).any()
                            select_mask = torch.zeros(cur_audio_output_length, dtype=torch.bool)
                            select_mask[select_idx] = True
                            batch_masks[i][is_continuous_indices[remain_pos][select_mask]] = True
                    # bs=1
                    whisper_emb = whisper_emb[batch_masks].unsqueeze(0)
                    audio_emb = audio_emb[batch_masks].unsqueeze(0)
                    is_continuous_mask = is_continuous_mask[batch_masks].unsqueeze(0)
                    text_input_ids = text_input_ids[batch_masks].unsqueeze(0)

                elif prune_method == "visionzip":
                    contextual_num_ratio = 0.05
                    dominant_num_ratio = remain_token_ratio - contextual_num_ratio
                    batch_masks = torch.zeros_like(is_continuous_mask)
                    assert attentions[0].shape[-2] == 4 * is_continuous_mask.sum().item()
                    for i in range(whisper_emb.shape[0]):
                        if type(attentions) is list:
                            attentions = attentions[0]
                        if type(attn_key) is list:
                            attn_key = attn_key[0].unsqueeze(0).to(is_continuous_mask.device)

                        attn_logits_i = attentions[i].max(dim=0).values # (N)
                        if attn_logits_i.size(-1) < attn_logits_i.size(-2):
                            attn_logits_i_a = attn_logits_i[:attn_logits_i.size(-1), :]
                            attn_logits_i_a = attn_logits_i_a.reshape(attn_logits_i.size(-1)//4, 4, attn_logits_i.size(-1)//4, 4).mean(1).sum(-1)
                            attn_logits_i_a = attn_logits_i_a.mean(0).to(whisper_emb.device)
                            attn_logits_i_b = attn_logits_i[attn_logits_i.size(-1):, :]
                            attn_logits_i_b = attn_logits_i_b[:, :attn_logits_i_b.shape[0]]
                            attn_logits_i_b = attn_logits_i_b.reshape(attn_logits_i_b.shape[0]//4, 4, attn_logits_i_b.shape[1]//4, 4).mean(1).sum(-1)
                            attn_logits_i_b = attn_logits_i_b.mean(0).to(whisper_emb.device)
                            attn_logits_i = torch.cat([attn_logits_i_a, attn_logits_i_b], dim=0)
                        else:
                            attn_logits_i = attn_logits_i[:, : attn_logits_i.shape[0]]
                            attn_logits_i = attn_logits_i.reshape(attn_logits_i.shape[0]//4, 4, attn_logits_i.shape[1]//4, 4).mean(1).sum(-1)
                            attn_logits_i = attn_logits_i.mean(0).to(whisper_emb.device)

                        audio_output_length = is_continuous_mask[i].sum()
                        dominant_num = int(dominant_num_ratio * audio_output_length)
                        contextual_num =  max(int(contextual_num_ratio * audio_output_length),1)
                        topk_indices = torch.topk(attn_logits_i, dominant_num)[1]
                        mask = torch.zeros_like(attn_logits_i, dtype=torch.bool)
                        mask[topk_indices] = True
                        contextual_mask = ~mask

                        ## Contextual Tokens Merging
                        attn_key = attn_key.mean(dim=1).unsqueeze(1) # [1, 1, 116, 64]
                        attn_key = attn_key.view(attn_key.shape[0], attn_key.shape[1], attn_key.shape[2]//4, 4, attn_key.shape[-1]).mean(dim=3) # [1, 1, 29, 64]
                        metric_filtered = attn_key[i, :, contextual_mask] # [1, contextual_len, 64]
                        # metric_filtered = attn_key[i, contextual_mask].unsqueeze(0) # [1, contextual_len, 1280]
                        metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True) # [1, contextual_len, 64]
                        del metric_filtered
                        step = max(1, metric_normalized.shape[1] // contextual_num)
                        target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]
                        target_tokens = metric_normalized[:, target_indices, :]
                        tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :]
                        similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
                        assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num, dtype=attn_logits_i.dtype, device=metric_normalized.device)
                        assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
                        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
                        
                        select_mask = torch.zeros_like(is_continuous_mask[i], dtype=torch.bool)
                        select_mask[~is_continuous_mask[i]] = True
                        is_continuous_indices = torch.where(is_continuous_mask[i])[0]
                        select_mask[is_continuous_indices[topk_indices]] = True
                        false_pos = (~select_mask).nonzero(as_tuple=True)[0]
                        select_mask[false_pos[target_indices]] = True
                        batch_masks[i] = select_mask

                        contextual_mask = contextual_mask.to(whisper_emb.device)
                        hidden_states_filtered = whisper_emb[i, is_continuous_indices[contextual_mask]].unsqueeze(0) # [contextual_len, 4096]
                        target_indices = target_indices.to(hidden_states_filtered.device)
                        hidden_to_merge = hidden_states_filtered[:, ~torch.isin(torch.arange(hidden_states_filtered.shape[1], device=hidden_states_filtered.device), target_indices), :]
                        assign_one_hot = assign_one_hot.to(hidden_to_merge.device)
                        counts = counts.to(hidden_to_merge.device)
                        try:
                            aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts # [1, 7, 4096]
                        except:
                            aggregated_hidden = torch.bmm(assign_one_hot.unsqueeze(0).transpose(1, 2), hidden_to_merge) / counts # [1, 7, 4096]
                        target_hidden = hidden_states_filtered[:, target_indices, :]
                        contextual_tokens = target_hidden + aggregated_hidden
                        whisper_emb[i, false_pos[target_indices]] = contextual_tokens.squeeze(0)
                    # bs=1
                    whisper_emb = whisper_emb[batch_masks].unsqueeze(0)
                    audio_emb = audio_emb[batch_masks].unsqueeze(0)
                    is_continuous_mask = is_continuous_mask[batch_masks].unsqueeze(0)
                    text_input_ids = text_input_ids[batch_masks].unsqueeze(0)
                    del attn_key
                    torch.cuda.empty_cache()
                
                
                encoder_input_addwith_discrete_token = (
                    audio_emb + whisper_emb
                ) * torch.sqrt(
                    torch.tensor(
                        2.0, dtype=whisper_emb.dtype, device=torch.cuda.current_device()
                    )
                )
                audio_emb = (
                    audio_emb * (~is_continuous_mask[:, :, None])
                    + encoder_input_addwith_discrete_token
                    * is_continuous_mask[:, :, None]
                )

            if text_input_ids is not None and text_input_ids.sum() != 0:
                inputs_embeds = audio_emb + self.embed_tokens(text_input_ids)
            else:
                inputs_embeds = audio_emb

        # embed positions
        # TODO kill attention_mask for prefill
        padding_mask = attention_mask

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

            hidden_states = layer_outputs[0]
            if idx == self.kimia_mimo_transformer_from_layer_index:
                mimo_hidden_states = hidden_states.clone()

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # apply audio transformer layers
        for idx, decoder_layer in enumerate(self.mimo_layers):
            if output_hidden_states:
                all_hidden_states += (mimo_hidden_states,)

            past_key_value = (
                past_key_values[idx + len(self.layers)]
                if past_key_values is not None
                else None
            )
            layer_outputs = decoder_layer(
                mimo_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

            mimo_hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        mimo_hidden_states = self.mimo_norm(mimo_hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (mimo_hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    mimo_hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=(hidden_states, mimo_hidden_states),
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MoonshotKimiaForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight", "mimo_output.weight"]
    config_class = KimiAudioConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MoonshotKimiaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mimo_output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        whisper_input_feature: Optional[torch.FloatTensor] = None,
        attentions: Optional[torch.Tensor] = None,
        attn_key: Optional[torch.Tensor] = None,
        is_continuous_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        generation_mode: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=whisper_input_feature,
            attentions=attentions,
            attn_key=attn_key,
            is_continuous_mask=is_continuous_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if return_dict:
            hidden_states, mimo_hidden_states = (
                outputs.last_hidden_state[0],
                outputs.last_hidden_state[1],
            )
        else:
            hidden_states, mimo_hidden_states = outputs[0], outputs[1]

        text_logits = self.lm_head(hidden_states)
        audio_logits = self.mimo_output(mimo_hidden_states)

        if not return_dict:
            output = (audio_logits, text_logits) + outputs[2:]
            return output
        return CausalLMOutputWithPast(
            loss=None,
            logits=(audio_logits, text_logits),
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
