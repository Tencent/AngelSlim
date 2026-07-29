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

"""Attention-forward patching for sparse algorithms.

Applies a :class:`SparsityAlgorithm` to a model by replacing the
``forward`` of each sparsable attention module via the LIFO push/pop API on
``BaseLLMModel``. Enforces the runtime guards (TP/PP/multi-node hard-fail,
CUDA-graph/vLLM refusal) and the FP8-attn composition guard.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from ._base import IncompatibleConfigError, SparsityAlgorithm
from ._distributed import detect_cuda_graph_or_compile, detect_tp, detect_world_size
from ._layers import resolve_sparsable_layers
from ._modal import resolve_modal

if TYPE_CHECKING:
    from transformers import PreTrainedModel

# Sentinel attribute marking a forward callable as sparse-patched (idempotency).
_SPARSE_MARKER = "_angelslim_sparse_patched"


def _guard_runtime(slim_model, hf_model: "PreTrainedModel") -> None:
    """Hard-fail on unsupported distributed / compiled runtimes."""
    ws = detect_world_size()
    if ws > 1:
        raise IncompatibleConfigError(
            f"Sparsity is single-rank, single-node only in this release (detected "
            f"WORLD_SIZE={ws}). Multi-node sparse is not yet supported; run "
            f"`python tools/run.py -c <yaml>` without --multi-nodes."
        )
    tp = detect_tp(hf_model)
    if tp > 1:
        raise IncompatibleConfigError(
            f"Sparse attention is not supported under tensor/pipeline "
            f"parallelism (detected degree={tp}). This release supports "
            f"single-rank only."
        )
    reason = detect_cuda_graph_or_compile(hf_model)
    if reason:
        raise IncompatibleConfigError(f"Sparse attention runtime guard: {reason}")


def _guard_fp8_attn_collision(slim_model) -> None:
    """Refuse if a QAT/legacy fp8_attn patch is already on the attn forwards.

    Two production patch paths must be caught:

    * **Legacy direct-assign** (the only one wired today): ``qwen.py``'s
      ``patch_fp8_attention`` and ``hunyuan_v3_moe``'s KV-observer install a
      patched ``attn.forward`` by direct assignment and record the original in
      ``slim_model._original_attn_forwards`` (via ``setdefault`` /
      ``dict[...] = ...``). That dict is populated **only** by these
      quant-attention patches — the sparse path uses a *separate*
      ``_attn_forward_patches`` registry (base_model.py) — so a NON-EMPTY
      ``_original_attn_forwards`` is the load-bearing signal that an fp8/legacy
      attn patch is live. (The previous guard additionally required a
      ``_fp8_attn_patched`` flag that nothing ever sets, so it could never
      fire.)
    * **Future label**: once fp8-attn is migrated onto ``push_attn_forward``
      it will carry a ``"qwen_fp8"`` label; check that too, forward-compatibly.
    """
    labels = (
        slim_model.attn_forward_labels() if hasattr(slim_model, "attn_forward_labels") else set()
    )
    if "qwen_fp8" in labels or getattr(slim_model, "fp8_attn_active", False):
        raise IncompatibleConfigError(
            "QAT plugin_config.quant_config.fp8_attn=true patches attn.forward; "
            "sparse cannot compose with it. Set fp8_attn=false in your "
            "QAT YAML, or remove compression.sparsity from this run."
        )
    # The populated legacy capture dict is the real signal: qwen.py /
    # hunyuan_v3_moe record originals here when they patch attn.forward for
    # fp8-attn / KV observers. Sparse refuses to stack on top of that.
    legacy = getattr(slim_model, "_original_attn_forwards", {}) or {}
    if legacy:
        raise IncompatibleConfigError(
            f"A legacy attention patch is already active "
            f"({len(legacy)} module(s) captured in _original_attn_forwards — "
            f"e.g. QAT fp8_attn or a KV-cache observer); sparse attention does "
            f"not compose with it. Disable fp8_attn / remove the quant "
            f"step, or drop compression.sparsity from this run."
        )


# Module class names produced by AngelSlim's quantizers (PTQ / QAT / weight-only).
# A model carrying any of these has already been quantized — sparse refuses to
# operate on it (sparse does NOT compose with quantization).
_QUANT_MODULE_NAMES = frozenset(
    {
        "QDQModule",
        "QDQSingleModule",
        "NVFP4QDQModule",
        "MoEQDQModule",
        "QLinear",
        "GPTQQuantLinear",
        "WQLinearGEMM",
        "W4A8Int8QuantLinear",
    }
)
# Resilience: also treat any class whose name contains one of these fragments as
# quantized, so a new quantizer module added upstream is caught by default.
_QUANT_NAME_FRAGMENTS = ("QDQ", "QuantLinear", "WQLinear")


def _is_quant_module(module) -> bool:
    cls = type(module).__name__
    if cls in _QUANT_MODULE_NAMES:
        return True
    return any(frag in cls for frag in _QUANT_NAME_FRAGMENTS)


def _guard_quantization_collision(slim_model, hf_model: "PreTrainedModel") -> None:
    """Refuse to sparsify a model that has been (or is being) quantized.

    In the current stage sparse attention is
    a **standalone** transform — it does NOT combine with quantization (PTQ /
    QAT / FP8 / INT4 / AWQ / GPTQ / NVFP4). Doing sparsity must mean doing *only*
    sparsity. The FP8-attn case is caught earlier by
    :func:`_guard_fp8_attn_collision` (it patches ``attn.forward``); this guard
    catches the broader weight-quantization case, which otherwise proceeds
    silently and ships a model labelled sparse-on-quantized that no downstream
    runtime supports.

    Signals (any one triggers the hard-fail):
      1. ``slim_model.quantized`` truthy (the wrapper's own flag), or
      2. ``hf_model.config.quantization_config`` set — a checkpoint that was
         *saved* quantized (GPTQ / AWQ / compressed-tensors / fp8 / bnb) and
         **reloaded**; its modules may be plain ``nn.Linear`` look-alikes that
         the name-based walk below cannot catch, so the config flag is the only
         reliable signal for the reload case, or
      3. any submodule is an AngelSlim quantized module (QDQModule / GPTQ /
         AWQ / NVFP4 / W4A8 / ... — see ``_is_quant_module``) — the
         conversion-time case.
    """
    if getattr(slim_model, "quantized", False):
        raise IncompatibleConfigError(
            "This model is already quantized (slim_model.quantized=True); sparse "
            "attention does not combine with quantization. Run sparsity on "
            "the unquantized model, in its own run — do not put a quantization "
            "method and Sparsity in the same compression pipeline."
        )
    qcfg = getattr(getattr(hf_model, "config", None), "quantization_config", None)
    if qcfg:
        qname = (
            getattr(qcfg, "quant_method", None)
            or (qcfg.get("quant_method") if isinstance(qcfg, dict) else None)
            or type(qcfg).__name__
        )
        raise IncompatibleConfigError(
            f"This checkpoint was saved with a quantization_config "
            f"({qname!r}); it is a reloaded quantized model. Sparse attention "
            f"does not combine with quantization — load the unquantized "
            f"checkpoint and run sparsity on it, in its own run."
        )
    for name, module in hf_model.named_modules():
        if _is_quant_module(module):
            raise IncompatibleConfigError(
                f"Model contains a quantized module "
                f"{type(module).__name__!r} at {name!r}; sparse attention does "
                f"not combine with quantization. Do sparsity on the "
                f"unquantized model, in its own run — do not put a quantization "
                f"method (PTQ/QAT/FP8/INT4/AWQ/GPTQ) and Sparsity in the same "
                f"compression pipeline."
            )


def _check_model_type(algo: SparsityAlgorithm, hf_model: "PreTrainedModel") -> None:
    """Validate model_type against the algorithm's compatibility set + modality."""
    model_type = getattr(hf_model.config, "model_type", None)
    compat = algo.traits.compatible_model_types
    if compat is not None and model_type not in compat:
        raise IncompatibleConfigError(
            f"Algorithm {type(algo).__name__} supports model_type "
            f"{sorted(compat)}, but this model is {model_type!r}."
        )
    algo_modal = algo.traits.model_modal
    if algo_modal != "any":
        actual = resolve_modal(hf_model)
        if algo_modal != actual:
            raise IncompatibleConfigError(
                f"Algorithm {type(algo).__name__} declares model_modal="
                f"{algo_modal!r} but model resolves to {actual!r}."
            )


def _guard_sliding_window(layers) -> None:
    """Refuse to sparsify layers whose attention uses a sliding window.

    The sparse prefill paths select keys from the raw Q/K with a full-causal
    assumption — they carry NO window parameter. Patching a sliding-window layer
    would silently run it window-less (attending to keys outside the window),
    changing the model's semantics. Until windowed sparse prefill exists, this is
    a hard precondition rather than a silent miscompute. Detected via the per-
    module ``sliding_window`` HF sets on sliding-attention layers, or a
    ``layer_types[i] == "sliding_attention"`` entry.
    """
    offenders = []
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        win = getattr(attn, "sliding_window", None)
        ltype = None
        cfg = getattr(attn, "config", None)
        idx = getattr(attn, "layer_idx", None)
        types = getattr(cfg, "layer_types", None) if cfg is not None else None
        if types is not None and idx is not None and 0 <= idx < len(types):
            ltype = types[idx]
        if (win is not None and win) or ltype == "sliding_attention":
            offenders.append(idx if idx is not None else len(offenders))
    if offenders:
        raise IncompatibleConfigError(
            "Sparse attention does not support sliding-window layers: the sparse "
            "prefill ignores the window and would attend outside it. Offending "
            f"layer_idx(es): {offenders}. Disable sliding window "
            "(config.use_sliding_window=False) or exclude these layers."
        )


def apply_sparsity_patch(slim_model, algo: SparsityAlgorithm):
    """Patch every sparsable attention layer of ``slim_model.model`` with ``algo``.

    ``slim_model`` is the AngelSlim ``BaseLLMModel`` wrapper (provides the LIFO
    push/pop API); ``slim_model.model`` is the HF model.

    Returns the list of patched attention modules (for unpatch). Idempotent:
    layers already carrying a sparse-marked forward are skipped.
    """
    hf_model = slim_model.model

    _guard_runtime(slim_model, hf_model)
    _guard_fp8_attn_collision(slim_model)
    _guard_quantization_collision(slim_model, hf_model)
    _check_model_type(algo, hf_model)

    algo.setup(hf_model)

    layers = resolve_sparsable_layers(hf_model)
    if not layers:
        raise IncompatibleConfigError(
            "No full-attention layers found; sparse cannot patch this model."
        )
    _guard_sliding_window(layers)

    patched = []
    try:
        for idx, layer in enumerate(layers):
            attn = layer.self_attn
            # Prefer the module's OWN global layer_idx (HF sets it at
            # construction). resolve_sparsable_layers may return a *filtered*
            # subset (e.g. Qwen3.5 keeps only full-attention layers, with global
            # indices 0,4,8,...), so the loop counter `idx` is NOT the global
            # index — KV-cache writes / best_pattern reads use the global one.
            # Only fall back to `idx` when the module genuinely has no layer_idx
            # (tiny synthetic models), and warn so it is not mistaken for real.
            if getattr(attn, "layer_idx", None) is None:
                warnings.warn(
                    f"attn module at filtered position {idx} has no layer_idx; "
                    f"falling back to the filtered index. On a real interleaved "
                    f"model this would be wrong — the module should carry its "
                    f"global layer_idx.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                attn.layer_idx = idx
            if getattr(attn.forward, _SPARSE_MARKER, False):
                continue  # idempotent
            new_forward = algo.build_attn_forward(attn, hf_model)
            try:
                setattr(new_forward, _SPARSE_MARKER, True)
            except AttributeError:
                # bound methods can't take attributes; wrap.
                new_forward = _mark(new_forward)
            slim_model.push_attn_forward("sparse", attn, new_forward)
            patched.append(attn)
    except Exception:
        # The patch must be ATOMIC. If build_attn_forward (or anything in
        # the loop) throws partway through, pop every patch pushed so far so the
        # model is left in its pristine state — otherwise the orphaned patches
        # have no handle (unpatch only sees what we return) and leak.
        for attn in reversed(patched):
            try:
                slim_model.pop_attn_forward(attn, expected_label="sparse")
            except RuntimeError:
                pass
        raise
    return patched


def _mark(fn):
    """Wrap a bound method so we can attach the sparse marker (idempotency)."""

    def _wrapped(*args, **kwargs):
        return fn(*args, **kwargs)

    setattr(_wrapped, _SPARSE_MARKER, True)
    return _wrapped


def unpatch_sparsity(slim_model, patched_modules):
    """Restore the original forward on each previously-patched attention module.

    A label MISMATCH is a real patch-ownership violation (something is
    patched over sparse and was not removed first) — leaving the model silently
    half-patched hides a bug, so we collect mismatches and RAISE after attempting
    every restore. A "no patch on stack" is benign (already popped / restored)
    and is ignored. (Was: mismatch only warned — too quiet for a correctness
    hazard.)
    """
    violations = []
    for attn in patched_modules:
        try:
            slim_model.pop_attn_forward(attn, expected_label="sparse")
        except RuntimeError as e:
            if "mismatch" in str(e).lower():
                violations.append(f"{type(attn).__name__}: {e}")
            # else: "no patch on stack" — benign, ignore.
    if violations:
        raise RuntimeError(
            "unpatch_sparsity: patch-ownership violation on "
            f"{len(violations)} module(s) — another attention patch is stacked "
            "above sparse and was not removed first; the model may be left "
            "partially patched. Details:\n  - " + "\n  - ".join(violations)
        )
