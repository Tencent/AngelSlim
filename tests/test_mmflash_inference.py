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

"""MMFlash speculative decoding on a hybrid target (Gated DeltaNet + full attention).

A tiny random Qwen3.5 model is the target. Rolling the linear-attention states back to the
accepted prefix has to keep speculative decoding lossless: greedy outputs equal the
target's autoregressive outputs whatever the draft proposes. No weights are downloaded.
"""

import importlib.util

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5ForCausalLM"):
    pytest.skip("needs a transformers version with Qwen3.5", allow_module_level=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu" and any(importlib.util.find_spec(m) for m in ("fla", "causal_conv1d")):
    # transformers then binds these GPU kernels for the Gated DeltaNet layers
    pytest.skip("the installed linear-attention kernels need a GPU", allow_module_level=True)

from transformers import (  # noqa: E402
    DynamicCache,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    Qwen3Config,
)

from angelslim.compressor.speculative.inference import mmflash  # noqa: E402
from angelslim.compressor.speculative.inference.mmflash import (  # noqa: E402
    MMFlashGenerator,
    _LinearAttentionRollback,
    crop_attention_cache,
)
from angelslim.compressor.speculative.train.models.draft.qwen_dflash import (  # noqa: E402
    QwenDFlashDraftModel,
)

VOCAB, HIDDEN, BLOCK = 96, 64, 5
LAYER_TYPES = ["linear_attention", "linear_attention", "full_attention"] * 2
# On GPU, flash-linear-attention computes a multi-token block (chunk kernel) and a single
# token (recurrent kernel) with different rounding, about 1e-3 on these logits; a wrong
# rollback changes them by about 5.
TOL = 1e-4 if DEVICE == "cpu" else 1e-2


def _target():
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=len(LAYER_TYPES),
        layer_types=LAYER_TYPES,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    )
    model = Qwen3_5ForCausalLM(config)
    # With the default init the random model keeps repeating one token, whatever its state.
    # Larger weights make its outputs depend on the state, so a wrong rollback shows up.
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.2)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, std=1.0)
    return model.to(DEVICE).float().eval()


def _draft():
    torch.manual_seed(1)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1024,
    )
    config.block_size = BLOCK
    config.num_target_layers = len(LAYER_TYPES)
    config.dflash_config = {"mask_token_id": VOCAB - 1, "target_layer_ids": [1, 3]}
    return QwenDFlashDraftModel(config).to(DEVICE).float().eval()


def _prompt(length=23, seed=2):
    ids = torch.randint(0, VOCAB - 1, (1, length), generator=torch.Generator().manual_seed(seed))
    return {"input_ids": ids.to(DEVICE)}


class _OracleGenerator(MMFlashGenerator):
    """Proposes the reference continuation with one random draft position corrupted.

    Some blocks stay intact, so every acceptance length from 1 to the block size occurs.
    """

    def __init__(self, target, draft, reference, seed=3):
        super().__init__(target, draft)
        self.reference = reference[0]
        self.rng = torch.Generator().manual_seed(seed)

    def propose(self, anchor, target_hidden, draft_cache, start, block_size):
        super().propose(anchor, target_hidden, draft_cache, start, block_size)  # draft cache
        drafts = torch.zeros(block_size - 1, dtype=torch.long, device=anchor.device)
        ref = self.reference[start + 1 : start + block_size]
        drafts[: ref.numel()] = ref
        wrong = int(torch.randint(0, block_size, (1,), generator=self.rng))
        if wrong < block_size - 1:
            drafts[wrong] = (drafts[wrong] + 1) % (VOCAB - 1)
        return drafts[None]


def test_crop_attention_cache_leaves_linear_layers_alone():
    target = _target()
    out = target(**_prompt(), use_cache=True)
    cache = out.past_key_values
    crop_attention_cache(cache, 10)
    for i, kind in enumerate(LAYER_TYPES):
        layer = cache.layers[i]
        if kind == "full_attention":
            assert layer.keys.shape[-2] == 10 and layer.values.shape[-2] == 10
        else:
            assert getattr(layer, "keys", None) is None


@pytest.mark.parametrize("recompute", [True, False], ids=["recompute", "replay"])
@pytest.mark.parametrize("num_tokens", range(1, BLOCK))
def test_rollback_matches_feeding_only_the_accepted_prefix(num_tokens, recompute):
    target = _target()
    prompt = _prompt()["input_ids"]
    block = torch.randint(0, VOCAB - 1, (1, BLOCK), generator=torch.Generator().manual_seed(4))
    block = block.to(DEVICE)
    generator = MMFlashGenerator(target)
    length = prompt.shape[1]

    def positions(n):
        return torch.arange(length, length + n, device=DEVICE)[None]

    with torch.inference_mode():
        # rejected suffix: verify the whole block, then roll back to its first num_tokens
        cache = target(input_ids=prompt, use_cache=True).past_key_values
        with _LinearAttentionRollback(generator.linear_attn, recompute) as rollback:
            rollback.save(cache)
            target(input_ids=block, position_ids=positions(BLOCK), past_key_values=cache)
            rollback.stop()
            crop_attention_cache(cache, length + num_tokens)
            rollback.rollback(num_tokens, BLOCK)
        # reference: feed only the accepted prefix
        ref = target(input_ids=prompt, use_cache=True).past_key_values
        target(
            input_ids=block[:, :num_tokens],
            position_ids=positions(num_tokens),
            past_key_values=ref,
        )
        nxt = block[:, -1:]
        pos = torch.tensor([[length + num_tokens]], device=DEVICE)
        a = target(input_ids=nxt, position_ids=pos, past_key_values=cache).logits
        b = target(input_ids=nxt, position_ids=pos, past_key_values=ref).logits
    torch.testing.assert_close(a, b, rtol=TOL, atol=TOL)


def test_qwen35_layers_use_the_state_recompute():
    assert _LinearAttentionRollback(MMFlashGenerator(_target()).linear_attn).recompute


@pytest.mark.parametrize("recompute", [True, False], ids=["recompute", "replay"])
def test_speculative_decoding_is_lossless_for_every_acceptance_length(recompute, monkeypatch):
    if not recompute:
        monkeypatch.setattr(mmflash, "_is_gated_deltanet", lambda module: False)
    target, draft = _target(), _draft()
    inputs = _prompt()
    ar = MMFlashGenerator(target).generate(inputs, max_new_tokens=80, block_size=1)
    oracle = _OracleGenerator(target, draft, ar.output_ids)
    spec = oracle.generate(inputs, max_new_tokens=80, block_size=BLOCK)
    assert torch.equal(spec.output_ids, ar.output_ids)
    assert set(spec.acceptance_lengths) == set(range(1, BLOCK + 1))
    assert sum(spec.acceptance_lengths) >= 80 - 1


def test_draft_model_path_is_lossless_and_stops():
    target, draft = _target(), _draft()
    inputs = _prompt(seed=5)
    ar = MMFlashGenerator(target).generate(inputs, max_new_tokens=40, block_size=1)
    stop = int(ar.output_ids[0, inputs["input_ids"].shape[1] + 20])
    generator = MMFlashGenerator(target, draft)
    full = generator.generate(inputs, max_new_tokens=40)
    assert torch.equal(full.output_ids, ar.output_ids)
    assert len(full.acceptance_lengths) > 0
    # generation ends at the first stop token, which is kept
    stopped = generator.generate(inputs, max_new_tokens=40, stop_token_ids=[stop])
    reply = stopped.output_ids[0, inputs["input_ids"].shape[1] :]
    assert int(reply[-1]) == stop and (reply[:-1] != stop).all()
    assert torch.equal(reply, ar.output_ids[0, inputs["input_ids"].shape[1] :][: reply.numel()])


def test_draft_cache_holds_only_committed_context():
    target, draft = _target(), _draft()
    generator = MMFlashGenerator(target, draft)
    inputs = _prompt()
    out = generator.generate(inputs, max_new_tokens=12)
    assert out.num_output_tokens == 12
    cache = DynamicCache()
    with torch.inference_mode():
        feats = torch.zeros(1, 7, 2 * HIDDEN, device=DEVICE)
        anchor = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
        generator.propose(anchor, feats, cache, start=7, block_size=BLOCK)
    assert cache.get_seq_length() == 7
