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


__all__ = ["linear_mapping", "norm_mapping"]


linear_mapping = dict(
    embedding="embed_tokens",
    attn="self_attn",
    attn_q="q_proj",
    attn_k="k_proj",
    attn_v="v_proj",
    attn_o="o_proj",
    mlp_in=["up_proj", "gate_proj"],
    mlp_out=["down_proj"],
    lm_head="lm_head",
)

# Each entry is (to_linear_list, to_norm),
# matching get_rotation_mapping_layers norm_mapping format.
# Longest-prefix matching is used to support MoE experts.
norm_mapping = [
    (["q_proj", "k_proj", "v_proj"], "input_layernorm"),
    (["up_proj", "gate_proj"], "post_attention_layernorm"),
    (["lm_head"], "norm"),
]
