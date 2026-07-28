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

"""Framework-core acceptance (merged): layer resolution, convert idempotence,
and the model-surface regression guard.

Three former standalone files folded into one, with no change to what runs:

  * `resolve_sparsable_layers` filters non-self_attn layers, parametrized

    over (qwen3, qwen3_moe, qwen3_5, qwen3_5_moe). Re-exports the already-green
    folded functions from `test_stem_integration` (single source of truth) and
    adds the Appendix-B parametrization.
  * `Sparsity.convert` is idempotent and unpatch restores the original

    forward across two full cycles. Re-exports the folded idempotent+unpatch test
    and adds the explicit two-cycle assertion.
  * — every model's `from_pretrained` binds the kwargs `Engine.prepare_model`

    forwards (incl. attn_implementation); the engine gates it; the sparse registry
    import does not pull the quant stack; lazy compressor registration is complete.
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import sys
import warnings

import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import CUDA  # noqa: E402
from _harness import run_all  # noqa: E402
from _harness import FakeSlim as _FakeSlim  # noqa: E402
from _harness import Skip as _Skip  # noqa: E402

# integration suite (folded there when it landed); re-exporting keeps this file
# from carrying a drift-prone copy.
from test_stem_integration import (  # noqa: E402,F401
    _tiny_qwen3,
    test_resolve_layers_error_paths,
    test_resolve_layers_prefers_language_model_tower,
    test_resolve_sparsable_layers_filters_linear_attention,
    test_resolve_sparsable_layers_qwen3,
    test_sparsity_convert_idempotent_and_unpatch,
)

# Single source of truth: the / coverage is DEFINED in the Stem


# ===========================================================================
# — resolve_sparsable_layers parametrized over model_types


# ===========================================================================
def _model_with_layer_types(layer_types):
    """A minimal decoder whose layers carry self_attn only on full-attention
    positions — mirrors the interleaved Qwen3.5 shape without loading weights.

    ``layer_types[i] == "full_attention"`` -> the layer has ``self_attn`` (and is
    sparsable); anything else (``"linear_attention"``) carries ``linear_attn``
    instead and must be filtered out by ``resolve_sparsable_layers``.
    """

    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Linear(2, 2)

    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = nn.Linear(2, 2)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [Full() if t == "full_attention" else Linear() for t in layer_types]
            )

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    return Outer()


def test_resolve_sparsable_layers_parametrized_over_model_types():
    """The Appendix-B parametrization: 4 model_types, the contract each.


    qwen3 / qwen3_moe are uniform full-attention (every layer sparsable); qwen3_5
    / qwen3_5_moe interleave linear_attention on the `(i+1) % 4 != 0` positions
    (only the full-attention layers are sparsable). All four must yield > 0
    sparsable layers.
    """
    from angelslim.compressor.sparsity._layers import (
        resolve_layers,
        resolve_sparsable_layers,
    )

    n = 8
    full = ["full_attention"] * n
    # Qwen3.5 rule: layer i is full_attention iff (i+1) % 4 == 0.
    interleaved = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n)]
    cases = {
        "qwen3": (full, n),
        "qwen3_moe": (full, n),
        "qwen3_5": (interleaved, n // 4),
        "qwen3_5_moe": (interleaved, n // 4),
    }
    for model_type, (layer_types, expected_sparsable) in cases.items():
        m = _model_with_layer_types(layer_types)
        assert len(resolve_layers(m)) == n, f"{model_type}: resolve_layers"
        sparsable = resolve_sparsable_layers(m)
        assert len(sparsable) == expected_sparsable, (
            f"{model_type}: expected {expected_sparsable} sparsable, " f"got {len(sparsable)}"
        )
        assert len(sparsable) > 0, f"{model_type}: no sparsable layers"
        assert all(
            hasattr(layer, "self_attn") for layer in sparsable
        ), f"{model_type}: a non-self_attn layer leaked into the sparsable set"


# ===========================================================================
# — convert idempotent + two-cycle unpatch


# ===========================================================================
def test_convert_unpatch_two_full_cycles_restore_original():
    """Appendix-B explicit clause: convert→unpatch→convert→unpatch must complete
    two full cycles with the original forward restored each time (the sentinel
    does not leave the module half-patched after a cycle)."""
    if not CUDA:
        raise _Skip("CUDA required")
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.sparsity import Sparsity
    from angelslim.utils.config_parser import SparsityConfig

    m = _tiny_qwen3(num_layers=4)
    slim = _FakeSlim(m)
    sp_cfg = SparsityConfig(
        name="stem",
        attn_kwargs={
            "backend": "torch",
            "block_size": 128,
            "layer_keep_ratios": [1.0, 1.0, 0.2, 0.2],
        },
    )
    comp = type("C", (), {"sparsity": sp_cfg})()

    # Capture the pristine per-module forwards before any patching.
    attns = [layer.self_attn for layer in m.model.layers]
    original_forwards = [a.forward for a in attns]

    for cycle in range(2):
        sparse = Sparsity(slim, {"compress_config": comp, "global_config": None})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sparse.convert()
        assert sparse._patched, f"cycle {cycle}: convert did not patch"
        assert len(sparse._patched_modules) == 4, f"cycle {cycle}: wrong patch count"
        # Idempotent: a second convert() adds no extra patches.
        sparse.convert()
        assert len(sparse._patched_modules) == 4, f"cycle {cycle}: not idempotent"

        sparse.unpatch()
        assert not sparse._patched, f"cycle {cycle}: still patched after unpatch"
        assert slim.attn_forward_labels() == set(), f"cycle {cycle}: labels leaked"
        # Original forward restored exactly.
        for a, orig in zip(attns, original_forwards):
            assert a.forward == orig, f"cycle {cycle}: forward not restored"


# ===========================================================================
# — model from_pretrained kwargs regression surface

# ===========================================================================
_ENGINE_FORWARDS = dict(
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    use_cache=False,
    using_multi_nodes=False,
)


def _model_classes_overriding_from_pretrained():
    """Import angelslim.models.* and yield (modname, class) for every class that
    defines its own from_pretrained (i.e. not just the inherited one)."""
    import angelslim.models as models_pkg

    found = []
    for mod in pkgutil.walk_packages(models_pkg.__path__, models_pkg.__name__ + "."):
        name = mod.name
        try:
            m = importlib.import_module(name)
        except Exception:
            # Some model modules need optional deps; skip import failures here —
            # the bind test only covers importable model classes.
            continue
        for _, obj in vars(m).items():
            if (
                inspect.isclass(obj)
                and obj.__module__ == name
                and "from_pretrained" in obj.__dict__
                and callable(obj.__dict__["from_pretrained"])
            ):
                found.append((name, obj))
    return found


def _can_bind(func, **extra):
    sig = inspect.signature(func)
    try:
        # bind as an unbound function: self placeholder + model_path positional
        sig.bind(None, "model_path", **_ENGINE_FORWARDS, **extra)
        return True, None
    except TypeError as e:
        return False, str(e)


def test_all_model_from_pretrained_accept_engine_default_kwargs():
    """Default case: Engine passes no attn_implementation (gated). Every override
    must bind the base kwarg set."""
    classes = _model_classes_overriding_from_pretrained()
    assert classes, "no model classes discovered — import/path problem"
    failures = []
    for modname, kls in classes:
        # Omni has a different signature (use_audio_in_video, attn_implementation);
        # it is reached by the Omni branch, not the LLM/VLM/Audio branch, so it
        # uses a different kwarg set. Skip it here; it has its own branch.
        if ".omni." in modname:
            continue
        ok, err = _can_bind(kls.from_pretrained)
        if not ok:
            failures.append(f"{kls.__name__} ({modname}): {err}")
    assert not failures, (
        "model from_pretrained cannot bind Engine.prepare_model's default kwargs "
        "(regression surface \n - " + "\n - ".join(failures)
    )


def test_all_model_from_pretrained_accept_attn_implementation():
    """Non-default case: a sparse compress run passes
    attn_implementation=flash_attention_2. Every LLM/VLM/Audio override must bind
    it (via an explicit param or **kwargs) — this is the exact regression."""

    classes = _model_classes_overriding_from_pretrained()
    assert classes, "no model classes discovered — import/path problem"
    failures = []
    for modname, kls in classes:
        if ".omni." in modname:
            continue
        ok, err = _can_bind(kls.from_pretrained, attn_implementation="flash_attention_2")
        if not ok:
            failures.append(f"{kls.__name__} ({modname}): {err}")
    assert not failures, (
        "model from_pretrained rejects attn_implementation — this is the "
        "regression (Engine.prepare_model forwards it on sparse/q3 runs):\n  - "
        + "\n  - ".join(failures)
    )


def test_engine_prepare_model_gates_attn_implementation():
    """The engine call site must NOT pass attn_implementation when it is
    'default' (so non-sparse PTQ/QAT loads are byte-identical). Asserted by
    source inspection of the gating construct."""
    import angelslim.engine as engine_mod

    src = inspect.getsource(engine_mod.Engine.prepare_model)
    # The fix introduces a gated dict; assert the gate exists (not an
    # unconditional attn_implementation=attn_implementation in this branch).
    assert 'attn_implementation in (None, "default")' in src or (
        '"attn_implementation": attn_implementation' in src
    ), "Engine.prepare_model no longer gates attn_implementation guard lost)"


def test_sparse_registry_import_does_not_pull_quant_stack():
    """Importing the sparse registry must NOT drag in the quant/GPTQ eager chain
    (threadpoolctl etc.) or the engine. A fresh interpreter imports only the
    sparse registry and asserts no `gptq` / `threadpoolctl` / `angelslim.engine`
    module was loaded as a side effect — the lazy-import decoupling."""
    import subprocess

    code = (
        "import sys; import angelslim.compressor.sparsity.registry as r;"
        "loaded = set(sys.modules);"
        "bad = [m for m in loaded if 'gptq' in m or m == 'threadpoolctl' "
        "or m == 'angelslim.engine'];"
        "assert not bad, ('sparse registry import pulled: %r' % bad);"
        "from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry;"
        "print('OK')"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, (
        f"sparse registry import is still coupled to the quant stack:\n"
        f"stdout={res.stdout}\nstderr={res.stderr}"
    )


def test_engine_compressor_registration_is_lazy_but_complete():
    """ensure_compressors_registered() must register the full set so the engine
    path is unchanged, even though package import is now lazy."""
    import subprocess

    code = (
        "from angelslim.compressor import CompressorFactory, "
        "ensure_compressors_registered;"
        "ensure_compressors_registered();"
        "avail = set(CompressorFactory.get_available_compressor());"
        "need = {'Distill', 'PTQ', 'QAD', 'QAT', 'Sparsity'};"
        "assert need <= avail, ('missing: %r' % (need - avail));"
        "print('OK')"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, (
        f"lazy compressor registration is incomplete:\n"
        f"stdout={res.stdout}\nstderr={res.stderr}"
    )


if __name__ == "__main__":
    sys.exit(1 if run_all(globals(), f"CUDA={CUDA}") else 0)
