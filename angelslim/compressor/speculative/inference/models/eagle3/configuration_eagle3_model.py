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

from transformers.configuration_utils import PretrainedConfig


class Eagle3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Eagle3Model`].
    It is used to instantiate an Eagle3 model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the
    defaults will yield a configuration similar to the base Eagle3 model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control
    the model outputs. Read the documentation from [`PretrainedConfig`] for more
    information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Eagle3 model. Defines the number of different tokens
            that can be represented by the `inputs_ids` passed when
            calling [`Eagle3Model`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations in the transformer layers.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the feed-forward network (FFN) representations.
        num_hidden_layers (`int`, *optional*, defaults to 1):
            Number of transformer layers in the model.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each transformer layer.
        num_key_value_heads (`int`, *optional*):
            Number of key and value heads for grouped query attention (GQA). If equal
            to `num_attention_heads`, uses multi-head attention (MHA). If set to 1,
            uses multi-query attention (MQA). Otherwise uses GQA. Defaults to
            `num_attention_heads`if not specified.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Tensor parallelism rank used during pretraining.
            See [parallelism docs](
            https://huggingface.co/docs/transformers/parallelism)
            for more details.
        hidden_act (`str` or `function`, *optional*, defaults to "silu"):
            Non-linear activation function used in the feed-forward network.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            Maximum sequence length the model can process.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon value for RMS layer normalization.
        use_cache (`bool`, *optional*, defaults to True):
            Whether to cache key/value attention states for faster decoding.
        pad_token_id (`int`, *optional*):
            Token ID used for padding sequences.
        bos_token_id (`int`, *optional*, defaults to 1):
            Token ID used for beginning of sequence.
        eos_token_id (`int`, *optional*, defaults to 2):
            Token ID used for end of sequence.
        tie_word_embeddings (`bool`, *optional*, defaults to False):
            Whether to tie input and output embeddings.
        rope_scaling (`Dict`, *optional*):
            Configuration for RoPE (Rotary Position Embedding) scaling. Supports:
            - `{"type": "linear", "factor": float}` for linear scaling
            - `{"type": "dynamic", "factor": float}` for dynamic scaling
            Scaling factor must be > 1.0.
    ```"""

    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_scaling=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, "
                f"`name` and `factor`, got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s name field must be one of ['linear', 'dynamic'], "
                f"got {rope_scaling_type}"
            )
        if (
            rope_scaling_factor is None
            or not isinstance(rope_scaling_factor, float)
            or rope_scaling_factor <= 1.0
        ):
            raise ValueError(
                f"`rope_scaling`'s factor field must be an float > 1, "
                f"got {rope_scaling_factor}"
            )
