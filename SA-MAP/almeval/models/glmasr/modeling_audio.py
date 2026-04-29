from typing import Any, Optional, Tuple

import torch
from torch import Tensor, nn
from transformers import WhisperConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperEncoderLayer, WhisperFlashAttention2
from transformers.utils import logging
from torch.nn.functional import scaled_dot_product_attention

logger = logging.get_logger(__name__)


class RotaryEmbedding:
    def __init__(self, dim, rope_ratio=1, original_impl=False):
        super().__init__()
        self.dim = dim
        self.original_impl = original_impl
        self.rope_ratio = rope_ratio

    def forward_impl(
        self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        base = base * self.rope_ratio
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    @torch.no_grad()
    def get_emb(self, max_seq_len, dtype, device):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=dtype, device=device,
        )


def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


class WhisperRoPEFlashAttn(WhisperFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _reshape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # WhisperFlashAttention2 attention does not support output_attentions
        if output_attentions:
            logger.warning_once("WhisperFlashAttention2 attention does not support output_attentions, "
                                "manually calculating attention weights.")

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, q_len, _ = hidden_states.size()

        # get query proj
        assert not is_cross_attention, "Cross-attention not supported"
        key_states = self._reshape(self.k_proj(hidden_states), -1, bsz)
        query_states = self._reshape(self.q_proj(hidden_states), -1, bsz)
        if rotary_pos_emb is not None:
            query_states, key_states = [apply_rotary_pos_emb(
                i.transpose(1, 2),
                rotary_pos_emb,
            ).transpose(1, 2) for i in (query_states, key_states)]
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        value_states = self._reshape(self.v_proj(hidden_states), -1, bsz)
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0].transpose(1, 2), key_states], dim=1)
            value_states = torch.cat([past_key_value[1].transpose(1, 2), value_states], dim=1)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states.transpose(1, 2), value_states.transpose(1, 2))

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = scaled_dot_product_attention(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal,
        ).transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        else:
            attn_weights = (query_states.transpose(1, 2) * self.scaling) @ key_states.permute(0, 2, 3, 1)
            if self.is_causal:
                causal_mask = torch.triu(
                    torch.ones(q_len, q_len, device=attn_weights.device), diagonal=1,
                ).unsqueeze(0).unsqueeze(0) * -1e9
                attn_weights = attn_weights + causal_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        return attn_output, attn_weights, past_key_value, key_states


class WhisperSpecialEncoderLayer(WhisperEncoderLayer):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.self_attn = WhisperRoPEFlashAttn(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[Tensor, Any]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, kv_cache, attn_key = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            rotary_pos_emb=rotary_pos_emb,
            position_ids=position_ids,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states, kv_cache)

        if output_attentions:
            outputs += (attn_weights,)

        outputs += (attn_key,)

        return outputs

class WhisperSpecialEncoder(WhisperEncoder):
    def __init__(
        self,
        config: WhisperConfig,
        use_rope=False,
        rope_ratio=1,
    ):
        super().__init__(config)
        self.use_rope = use_rope
        self.layers = nn.ModuleList(
            [WhisperSpecialEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        if use_rope:
            self.rotary_embedding = RotaryEmbedding(
                config.hidden_size // config.encoder_attention_heads // 2,
                rope_ratio,
            )

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        position_ids=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
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
        # use_cache = use_cache if use_cache is not None else self.config.use_cache

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        if self.use_rope:
            rotary_embs = self.rotary_embedding.get_emb(
                inputs_embeds.shape[1],
                inputs_embeds.dtype,
                inputs_embeds.device,
            )
            if position_ids is not None:
                rotary_embs = rotary_embs[position_ids]
            else:
                rotary_embs = rotary_embs[None]
            hidden_states = inputs_embeds
        else:
            rotary_embs = None
            if position_ids is not None:
                # wrap tail, those are usually paddings to avoid inter-sample conv interfering
                max_l = self.embed_positions.weight.shape[0]
                if position_ids.max() >= max_l:
                    print("Pos id max", position_ids.max(), "wrapping")
                embed_pos = self.embed_positions.weight[position_ids % max_l]
            else:
                embed_pos = self.embed_positions.weight[:inputs_embeds.shape[1]]
            hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                        rotary_embs,
                        position_ids,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        output_attentions=output_attentions,
                        rotary_pos_emb=rotary_embs,
                        position_ids=position_ids,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        attn_key = layer_outputs[3]

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            ), attn_key
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        ), attn_key
