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

"""MMFlash speculative decoding for vision-language models (transformers backend).

MMFlash (https://github.com/fscdc/MMFlash) drafts keep the DFlash block-diffusion
architecture, so they load with
:class:`~angelslim.compressor.speculative.train.models.draft.qwen_dflash.QwenDFlashDraftModel`.
The draft reads hidden states of a few target layers and proposes a block of tokens in
one forward pass; the target verifies the block in one forward pass and keeps the longest
matching prefix plus its own next token.

On top of the text-only DFlash loop (``tools/dflash_benchmark.py``), a vision-language
target needs:

* Multimodal prefill. The processor outputs (``pixel_values``, ``image_grid_thw``,
  ``mm_token_type_ids``, ...) go into the first target forward. Models with multimodal
  RoPE (the Qwen-VL family) store the position offset of the prompt in
  ``model.rope_deltas``; every verify block is placed at ``cached length + offset``.
  The draft uses plain sequence positions.
* Hybrid targets. Qwen3.5 interleaves Gated DeltaNet (linear-attention) layers with full
  attention. A rejected draft suffix has to be removed from their convolution and
  recurrent states, which ``Cache.crop`` cannot do. The states are saved before each
  verify forward and the outputs of the layers' input projections are recorded during it;
  after acceptance the states of the accepted prefix are recomputed in one batched pass
  over all layers. Other linear-attention modules are restored and re-run on the accepted
  prefix of their recorded inputs. Both are exact: in a causal model the inputs at the
  accepted positions do not depend on the rejected tokens. Full-attention layers are
  cropped to the accepted length.

The loop relies only on forward pre-hooks and on the cache layers' ``keys`` / ``values`` and
``conv_states`` / ``recurrent_states`` attributes, which transformers 5.8 through 5.23
share. Block size 1 runs the target alone through the same loop, which is the
autoregressive baseline.
"""

import contextlib
import functools
import time
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DynamicCache

from ..train.models.draft.qwen_dflash import extract_context_feature, sample

__all__ = ["MMFlashGenerator", "crop_attention_cache"]

_STATE_NAMES = ("conv_states", "recurrent_states")


def _now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def crop_attention_cache(cache, length: int) -> None:
    """Keep the first ``length`` positions of every attention layer of ``cache``.

    Linear-attention layers have no sequence axis and are left alone. ``Cache.crop`` is not
    used because it raises on linear-attention layers and its argument changed meaning
    (absolute length vs. tokens to remove) within transformers 5.x.
    """
    for layer in cache.layers:
        keys = getattr(layer, "keys", None)
        if torch.is_tensor(keys) and keys.dim() == 4 and keys.shape[-2] > length:
            layer.keys = keys[..., :length, :]
            layer.values = layer.values[..., :length, :]


def _linear_attention_modules(model: nn.Module) -> List[nn.Module]:
    """Token mixers of the linear-attention decoder layers (``layer.linear_attn``)."""
    found = []
    for module in model.modules():
        mixer = getattr(module, "linear_attn", None)
        if isinstance(mixer, nn.Module) and isinstance(getattr(mixer, "layer_idx", None), int):
            found.append(mixer)
    return sorted(found, key=lambda m: m.layer_idx)


def _prefix(value, num_tokens: int, block_len: int):
    """``value[:, :num_tokens]`` for tensors laid out ``[batch, block_len, ...]``."""
    if torch.is_tensor(value) and value.dim() >= 2 and value.shape[1] == block_len:
        return value[:, :num_tokens]
    return value


def _read_state(layer, name: str) -> torch.Tensor:
    states = getattr(layer, name)
    return states[0] if isinstance(states, dict) else states


def _write_state(layer, name: str, value: torch.Tensor) -> None:
    states = getattr(layer, name)
    current = states[0] if isinstance(states, dict) else states
    if torch.is_tensor(current) and current.shape == value.shape:
        current.copy_(value)  # in place: keeps static cache addresses valid
    elif isinstance(states, dict):
        states[0] = value
    else:
        setattr(layer, name, value)


# attributes of a Qwen3.5 Gated DeltaNet layer that the state recomputation reads
_GDN_ATTRS = (
    "in_proj_qkv",
    "in_proj_b",
    "in_proj_a",
    "conv1d",
    "A_log",
    "dt_bias",
    "key_dim",
    "value_dim",
    "num_k_heads",
    "num_v_heads",
    "head_k_dim",
    "head_v_dim",
    "conv_kernel_size",
    "activation",
)
_GDN_PROJECTIONS = ("in_proj_qkv", "in_proj_b", "in_proj_a")


def _is_gated_deltanet(module: nn.Module) -> bool:
    return all(hasattr(module, a) for a in _GDN_ATTRS) and module.activation == "silu"


def _can_recompute(modules: List[nn.Module]) -> bool:
    """Qwen3.5 Gated DeltaNet layers of one shape, whose states are recomputed as a batch."""
    if not modules or not all(_is_gated_deltanet(m) for m in modules):
        return False
    shapes = {
        (m.key_dim, m.value_dim, m.num_k_heads, m.num_v_heads, m.conv_kernel_size)
        + (m.conv1d.bias is None,)
        for m in modules
    }
    return len(shapes) == 1


def _gated_delta_rule_state(k, v, g, beta, state):
    """Final state of the gated delta rule (transformers' reference recurrence for Qwen3.5).

    ``k`` [N, T, H, Dk] (l2-normalized here), ``v`` [N, T, H, Dv], ``g`` and ``beta``
    [N, T, H], ``state`` [N, H, Dk, Dv]; computed in float32.
    """
    k = k.float()
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    v, g, beta, state = v.float(), g.float(), beta.float(), state.float()
    for t in range(k.shape[1]):
        state = state * g[:, t, :, None, None].exp()
        delta = (v[:, t] - (state * k[:, t, :, :, None]).sum(-2)) * beta[:, t, :, None]
        state = state + k[:, t, :, :, None] * delta[:, :, None, :]
    return state


def _fla_fused_recurrent():
    try:
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    except ImportError:
        return None
    return fused_recurrent_gated_delta_rule


class _LinearAttentionRollback:
    """Rolls the linear-attention layers of a hybrid target back to a prefix of a block.

    For Qwen3.5 Gated DeltaNet layers, the outputs of their input projections are recorded
    during the verify forward; after acceptance, the convolution and recurrent states of the
    accepted prefix are recomputed from the saved states in one batched pass over all layers.
    Other linear-attention modules get their states restored and are re-run on the accepted
    prefix of the inputs they received.
    """

    def __init__(self, modules: Iterable[nn.Module], recompute: Optional[bool] = None):
        self.modules = list(modules)
        if recompute is None:
            recompute = _can_recompute(self.modules)
        self.recompute = recompute
        self.recording = False
        self.calls: Dict = {}
        self.saved: List[tuple] = []
        self._handles = []
        if recompute:
            self.conv_weight = torch.stack([m.conv1d.weight.squeeze(1) for m in self.modules])
            self.conv_weight = self.conv_weight.float()  # [layers, channels, kernel]
            self.conv_bias = None
            if self.modules[0].conv1d.bias is not None:
                self.conv_bias = torch.stack([m.conv1d.bias for m in self.modules]).float()
            self.decay = torch.stack([m.A_log.float().exp() for m in self.modules])
            self.dt_bias = torch.stack([m.dt_bias.float() for m in self.modules])
            self.kernel = _fla_fused_recurrent()

    def __enter__(self):
        if self.recompute:
            for i, module in enumerate(self.modules):
                for name in _GDN_PROJECTIONS:
                    hook = functools.partial(self._record_output, (i, name))
                    self._handles.append(getattr(module, name).register_forward_hook(hook))
        else:
            self._handles = [
                m.register_forward_pre_hook(self._record_inputs, with_kwargs=True)
                for m in self.modules
            ]
        return self

    def __exit__(self, *exc_info):
        for handle in self._handles:
            handle.remove()
        self._handles, self.calls, self.saved = [], {}, []
        self.recording = False

    def _record_output(self, key, module, args, output):
        if self.recording:
            self.calls[key] = output

    def _record_inputs(self, module, args, kwargs):
        if self.recording:
            self.calls[module.layer_idx] = (args, kwargs)

    def save(self, cache) -> None:
        """Copy every linear-attention state and record what the next forward computes."""
        self.layers = [cache.layers[m.layer_idx] for m in self.modules]
        if self.recompute:
            self.saved_conv = torch.stack([_read_state(x, "conv_states") for x in self.layers])
            self.saved_recurrent = torch.stack(
                [_read_state(x, "recurrent_states") for x in self.layers]
            )
        else:
            self.saved = []
            for layer in self.layers:
                for name in _STATE_NAMES:
                    states = getattr(layer, name, None)
                    if torch.is_tensor(states):  # one tensor per layer (transformers <= 5.8)
                        self.saved.append((layer, name, None, states.clone()))
                    elif isinstance(states, dict):  # {state index: tensor}
                        for key, state in states.items():
                            if torch.is_tensor(state):
                                self.saved.append((layer, name, key, state.clone()))
        self.calls = {}
        self.recording = True

    def stop(self) -> None:
        self.recording = False

    def rollback(self, num_tokens: int, block_len: int) -> None:
        """Set the states to what they are after the first ``num_tokens`` tokens of the block."""
        if self.recompute:
            self._recompute(num_tokens)
            return
        for layer, name, key, saved in self.saved:
            states = getattr(layer, name)
            current = states if key is None else states.get(key)
            if torch.is_tensor(current) and current.shape == saved.shape:
                current.copy_(saved)
            elif key is None:
                setattr(layer, name, saved)
            else:
                states[key] = saved
        for module in self.modules:
            args, kwargs = self.calls[module.layer_idx]
            module(
                *(_prefix(a, num_tokens, block_len) for a in args),
                **{k: _prefix(v, num_tokens, block_len) for k, v in kwargs.items()},
            )

    def _recompute(self, n: int) -> None:
        m = self.modules[0]
        num_layers, width = len(self.modules), m.conv_kernel_size
        qkv, b, a = (
            torch.stack([self.calls[(i, name)][:, :n] for i in range(num_layers)])
            for name in _GDN_PROJECTIONS
        )  # [layers, batch, n, channels]
        batch = qkv.shape[1]
        # causal depthwise convolution over the saved window followed by the accepted inputs
        window = torch.cat([self.saved_conv, qkv.transpose(-1, -2).to(self.saved_conv)], dim=-1)
        x, w = window.float(), self.conv_weight[:, None]
        conv = sum(x[..., j + 1 : j + 1 + n] * w[..., j : j + 1] for j in range(width))
        if self.conv_bias is not None:
            conv = conv + self.conv_bias[:, None, :, None]
        mixed = F.silu(conv).to(qkv.dtype).transpose(-1, -2)  # [layers, batch, n, channels]
        q, k, v = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        rows = num_layers * batch
        q = q.reshape(rows, n, m.num_k_heads, m.head_k_dim)
        k = k.reshape(rows, n, m.num_k_heads, m.head_k_dim)
        v = v.reshape(rows, n, m.num_v_heads, m.head_v_dim)
        if m.num_v_heads > m.num_k_heads:
            q = q.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
            k = k.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
        beta = b.sigmoid().reshape(rows, n, m.num_v_heads)
        g = -self.decay[:, None, None] * F.softplus(a.float() + self.dt_bias[:, None, None])
        g = g.reshape(rows, n, m.num_v_heads)
        state = self.saved_recurrent.flatten(0, 1)
        if self.kernel is not None and k.is_cuda:
            _, state = self.kernel(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                g=g,
                beta=beta.contiguous(),
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            state = _gated_delta_rule_state(k, v, g, beta, state)
        state = state.reshape(num_layers, batch, *state.shape[1:])
        for i, layer in enumerate(self.layers):
            _write_state(layer, "conv_states", window[i, ..., -width:])
            _write_state(layer, "recurrent_states", state[i])


class MMFlashGenerator:
    """Batch-size-1 speculative decoding of a vision-language target with an MMFlash draft.

    Args:
        target: the target model, e.g.
            ``AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-4B")``.
        draft: a ``QwenDFlashDraftModel`` (MMFlash and DFlash checkpoints share this
            architecture), or ``None`` for autoregressive decoding only.

    Example::

        generator = MMFlashGenerator(target, draft)
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                                               tokenize=True, return_dict=True,
                                               return_tensors="pt")
        out = generator.generate(inputs, max_new_tokens=1024, stop_token_ids=[eos])
        text = processor.decode(out.output_ids[0, out.num_input_tokens:])
    """

    def __init__(self, target: nn.Module, draft: Optional[nn.Module] = None):
        layer_types = getattr(target.config.get_text_config(), "layer_types", None) or []
        if "sliding_attention" in layer_types:
            # a sliding-window cache keeps only the last window of keys, so it cannot be
            # cropped back to an accepted length
            raise NotImplementedError("targets with sliding-window attention are not supported")
        self.target = target
        self.draft = draft
        self.embed_tokens = target.get_input_embeddings()
        self.lm_head = target.get_output_embeddings()
        self.linear_attn = _linear_attention_modules(target)
        base = getattr(target, "model", None)
        # Qwen-VL style models keep the multimodal RoPE offset of the current prompt here
        self.rope_owner = base if hasattr(base, "rope_deltas") else None

    def _target_positions(self, start: int, length: int, offset: int, device) -> torch.Tensor:
        positions = torch.arange(start + offset, start + offset + length, device=device)[None]
        if self.rope_owner is not None:
            # multimodal RoPE: the temporal / height / width rows coincide for text tokens
            positions = positions[None].expand(3, 1, length)
        return positions

    @torch.inference_mode()
    def propose(
        self,
        anchor: torch.Tensor,
        target_hidden: torch.Tensor,
        draft_cache: DynamicCache,
        start: int,
        block_size: int,
    ) -> torch.Tensor:
        """Greedy draft tokens ``[1, block_size - 1]`` for the block starting with ``anchor``.

        ``target_hidden`` holds the target features of the tokens the draft cache does not
        contain yet; ``start`` is the sequence position of ``anchor``.
        """
        draft = self.draft
        block = torch.full(
            (1, block_size), draft.mask_token_id, dtype=torch.long, device=anchor.device
        )
        block[:, :1] = anchor
        position_ids = torch.arange(
            draft_cache.get_seq_length(), start + block_size, device=anchor.device
        )[None]
        hidden = draft(
            target_hidden=target_hidden,
            noise_embedding=self.embed_tokens(block),
            position_ids=position_ids,
            past_key_values=draft_cache,
            use_cache=True,
            is_causal=False,
        )
        crop_attention_cache(draft_cache, start)  # keep the context, drop the block
        return sample(self.lm_head(hidden[:, 1:]))

    @torch.inference_mode()
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int = 2048,
        stop_token_ids: Optional[Iterable[int]] = None,
        temperature: float = 0.0,
        block_size: Optional[int] = None,
    ) -> SimpleNamespace:
        """Generate the reply to one prompt.

        Args:
            inputs: processor outputs for one prompt (``input_ids`` of shape ``[1, L]`` and the
                image inputs, if any).
            max_new_tokens: generation budget.
            stop_token_ids: generation ends after the first of these tokens (kept in the output).
            temperature: 0 for greedy decoding. Otherwise the target samples and a draft token
                is accepted when it equals the target's sample, which keeps the target's
                output distribution.
            block_size: tokens per verify step. Defaults to the draft's block size; 1 runs the
                target alone (autoregressive decoding).

        Returns:
            ``SimpleNamespace`` with ``output_ids`` (prompt and reply), ``num_input_tokens``,
            ``num_output_tokens``, ``time_to_first_token``, ``decode_time``,
            ``time_per_output_token`` and ``acceptance_lengths`` (tokens committed per step).
        """
        target, draft = self.target, self.draft
        if block_size is None:
            block_size = draft.block_size if draft is not None else 1
        spec = block_size > 1
        if spec and draft is None:
            raise ValueError("block_size > 1 needs a draft model")
        device = self.embed_tokens.weight.device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        if input_ids.shape[0] != 1:
            raise ValueError("MMFlashGenerator decodes one prompt at a time")
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        stop = torch.tensor(sorted(set(stop_token_ids or [])), dtype=torch.long, device=device)
        output_ids = torch.zeros((1, max_length + block_size), dtype=torch.long, device=device)
        output_ids[:, :num_input_tokens] = input_ids

        def hits_stop(tokens: torch.Tensor) -> bool:
            return stop.numel() > 0 and bool(torch.isin(tokens, stop).any())

        # ---- prefill
        if self.rope_owner is not None:
            self.rope_owner.rope_deltas = None  # recomputed for this prompt
        prefill_start = _now(device)
        out = target(**inputs, use_cache=True, logits_to_keep=1, output_hidden_states=spec)
        cache = out.past_key_values
        output_ids[:, num_input_tokens] = sample(out.logits, temperature)[:, -1]
        offset = 0
        if self.rope_owner is not None and self.rope_owner.rope_deltas is not None:
            offset = int(self.rope_owner.rope_deltas.view(-1)[0])
        if spec:
            target_hidden = extract_context_feature(out.hidden_states, draft.target_layer_ids)
            draft_cache = DynamicCache()
        time_to_first_token = _now(device) - prefill_start

        # ---- decode: output_ids[:, start] is the newest token, not in the target cache yet
        start = num_input_tokens
        acceptance_lengths = []
        done = hits_stop(output_ids[0, start : start + 1])
        rollback = None
        if spec and self.linear_attn:
            rollback = _LinearAttentionRollback(self.linear_attn)
        decode_start = _now(device)
        with rollback if rollback is not None else contextlib.nullcontext():
            while not done and start < max_length:
                block = output_ids[:, start : start + block_size].clone()
                if spec:
                    block[:, 1:] = self.propose(
                        block[:, :1], target_hidden, draft_cache, start, block_size
                    )
                    if not acceptance_lengths:
                        # the first draft pass reads the whole prompt; count it as prefill
                        decode_start = _now(device)
                if rollback is not None:
                    rollback.save(cache)
                out = target(
                    input_ids=block,
                    position_ids=self._target_positions(start, block_size, offset, device),
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=spec,
                )
                if rollback is not None:
                    rollback.stop()
                posterior = sample(out.logits, temperature)
                accepted = 0
                if spec:
                    accepted = int((block[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum())
                num_tokens = accepted + 1  # the block's first token plus the accepted drafts
                output_ids[:, start : start + num_tokens] = block[:, :num_tokens]
                output_ids[:, start + num_tokens] = posterior[:, accepted]
                new_tokens = output_ids[0, start + 1 : start + num_tokens + 1]
                start += num_tokens
                acceptance_lengths.append(num_tokens)
                crop_attention_cache(cache, start)
                if rollback is not None and num_tokens < block_size:
                    rollback.rollback(num_tokens, block_size)
                if spec:
                    target_hidden = extract_context_feature(
                        out.hidden_states, draft.target_layer_ids
                    )[:, :num_tokens]
                done = hits_stop(new_tokens)
        decode_time = _now(device) - decode_start

        generated = output_ids[0, num_input_tokens : min(start + 1, max_length)]
        if stop.numel() > 0:
            stops = torch.isin(generated, stop).nonzero()
            if stops.numel() > 0:
                generated = generated[: int(stops[0]) + 1]
        num_output_tokens = generated.numel()
        return SimpleNamespace(
            output_ids=torch.cat([input_ids, generated[None]], dim=1),
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            time_to_first_token=time_to_first_token,
            decode_time=decode_time,
            time_per_output_token=decode_time / max(num_output_tokens, 1),
            acceptance_lengths=acceptance_lengths,
        )
