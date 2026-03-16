# adopted from llm-compressor/src/llmcompressor/modifiers/transform/spinquant/mappings.py

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

__all__ = ["linear_mapping", "norm_mapping"]


class SpinQuantMapping(BaseModel):
    """
    SpinQuant needs to know the entire architecture of the model,
    as R1, R2, R3, and R4 rotations need to be applied to specific
    layers (https://arxiv.org/pdf/2405.16406 Fig. 1).

    :param embedding: name suffix of embedding layer
    :param attn: name suffix of attention block in decoder layer
    :param attn_q: name suffix of q_proj layer in attention block
    :param attn_k: name suffix of k_proj layer in attention block
    :param attn_v: name suffix of v_proj layer in attention block
    :param attn_o: name suffix of o_proj layer in attention block
    :param attn_head_dim: head_dim of the attention module, needed
        because R2 needs to be applied "head-wisely" to v_proj and
        o_proj
    :param mlp_in: list of name suffixes for the mlp blocks that
        receive the input to the MLP block, usually up_proj and gate_proj
    :param mlp_out: list of name suffixes for the mlp blocks that
        constitute the output of the MLP block, usually down_proj
    """

    embedding: str

    attn: str
    attn_q: str
    attn_k: str
    attn_v: str
    attn_o: str
    attn_head_dim: Optional[int] = Field(default=None)

    mlp_in: List[str]  # up_proj, gate_proj
    mlp_out: List[str]  # down_proj

    lm_head: str

    @field_validator("mlp_in", "mlp_out", mode="before")
    def cast_to_list(cls, value):
        if isinstance(value, str):
            return [value]

        return value


linear_mapping = SpinQuantMapping(
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
