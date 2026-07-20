"""Build an mcore GPTModel from a (TransformerConfig, ModelMeta).

The builder is model-agnostic and uses a local, non-TE layer specification.
"""

from __future__ import annotations

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig

from angelslim.compressor.mcore_qad.mcore.patches import apply_patches
from angelslim.compressor.mcore_qad.models.base import ModelMeta

apply_patches()


def enable_input_require_grads(model) -> None:
    """HF-style: make the input word-embedding output require grad.

    Activation checkpointing decides whether its recomputed output requires grad from
    whether its INPUTS do. With a frozen backbone (only quantizer scales train), the
    embedding output otherwise has no grad and the checkpointed graph detaches. Hooking
    the input embedding gives the recompute a gradient entry point at the very front,
    WITHOUT touching any parameter's frozen state (mirrors transformers'
    PreTrainedModel.enable_input_require_grads). The teacher's no-grad pass is unaffected.
    """
    emb = getattr(model, "embedding", None)
    if emb is None:  # not the pre-process stage
        return
    word_emb = getattr(emb, "word_embeddings", emb)  # innermost input embedding

    def _make_require_grad(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)

    word_emb.register_forward_hook(_make_require_grad)


def _use_te_core_attention(block_spec) -> None:
    """Swap the local DotProductAttention for TEDotProductAttention (enables CP),
    while keeping local Column/RowParallelLinear so quant injection still works."""
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    for layer_spec in block_spec.layer_specs:
        sa = getattr(layer_spec.submodules, "self_attention", None)
        if sa is not None and hasattr(sa, "submodules"):
            sa.submodules.core_attention = TEDotProductAttention


def build_gpt_model(
    cfg: TransformerConfig,
    meta: ModelMeta,
    *,
    pre_process: bool = True,
    post_process: bool = True,
    te_core_attention: bool = False,
    vp_stage=None,
) -> GPTModel:
    """Construct a non-TE (local) GPTModel from the given config + meta.

    ``te_core_attention`` swaps only the attention core to TE's (required for CP);
    all linears stay local so weight parametrization / quant still applies.
    """
    block_spec = get_gpt_decoder_block_spec(cfg, use_transformer_engine=False, vp_stage=vp_stage)
    if te_core_attention:
        _use_te_core_attention(block_spec)
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=block_spec,
        vocab_size=meta.vocab_size,
        max_sequence_length=meta.max_sequence_length,
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type="rope",
        rotary_base=int(meta.rotary_base),
        share_embeddings_and_output_weights=meta.tie_embeddings,
        vp_stage=vp_stage,
    )
    return model
