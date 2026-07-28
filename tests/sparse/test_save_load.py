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

"""save/load round-trip — the *genuine* Engine -> InferEngine path.

Until this slice, sparse save/load was only exercised by calling
``Sparsity.save()`` + ``Qwen3ForCausalLM.from_pretrained()`` + a *manual*
re-patch (in the Stem suite). That bypasses the layer where silent corruption
hides: the real
``Engine.prepare_model -> prepare_compressor -> run -> convert -> save`` compress
pipeline and the real ``InferEngine.from_pretrained`` load pipeline (which writes
/ reads ``angelslim_config.json`` and *auto-re-applies* the sparse patch via the
production ``BaseLLMModel`` LIFO API). This file drives that real path.

Key design (matches the established weights policy):
  * (save emits weights+tokenizer), (compression_config survives JSON), (patch
    auto-re-applies on load), (attn_implementation survives) are all
    **structural** — they assert files / config fields / the
    presence of the sparse marker, never weight-dependent numerics. So the
    default tests run the full Engine round-trip on a **tiny synthetic Qwen3 +
    the real tokenizer** (fast, no 16 GB, runs anywhere CUDA-or-CPU).
  * Only the *semantic* check (the reloaded sparse model still answers "Paris")
    needs real weights, so ``test_engine_e2e_save_load_real_weights`` is gated
    behind ``SPARSE_E2E_SAVE=1`` (mirrors Stem's ``STEM_REAL_SAVE``) and loads
    the real Qwen3-8B.

The synthetic round-trip still needs a real *tokenizer* (a loadable directory
must carry tokenizer files), so the default tests skip — not fail — when the
Qwen3-8B checkpoint directory is absent (tokenizer is read from there; it is
cheap, no GPU, no 16 GB weight load).

Runnable without pytest via the ``__main__`` block, like the sibling suites.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings

import torch

# shared scaffolding (single source of truth) — env constants, Skip, the

# coverage-floored runner, and the availability predicates. (This suite's local
# ``_run_all`` was MISSING the coverage floor; the shared runner fixes that.)

# ``_SCRATCH`` (the big-FS tempdir root) is resolved by _harness (migration-proof).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import real_weights_available  # noqa: E402
from _harness import CUDA, HEAD_DIM  # noqa: E402
from _harness import REAL_W_QWEN3_8B as _REAL_W  # noqa: E402
from _harness import SCRATCH_ROOT as _SCRATCH  # noqa: E402
from _harness import Skip as _Skip  # noqa: E402
from _harness import SkipReason as _SR  # noqa: E402
from _harness import record_capability as _cap  # noqa: E402
from _harness import run_all, tokenizer_available  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — build a loadable tiny source dir, write a sparse YAML, drive Engine.
# ---------------------------------------------------------------------------
def _make_tiny_source(dirpath: str) -> None:
    """Save a tiny random-init Qwen3 + the real tokenizer to ``dirpath``.

    The result is a directory ``AutoModelForCausalLM.from_pretrained`` /
    ``AutoTokenizer.from_pretrained`` can load — i.e. a faithful stand-in for a
    real checkpoint dir, but a few MB instead of 16 GB. Random weights are fine:
    every test here is structural (files / config / marker), not numeric.

    """
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=151936,  # match Qwen3 tokenizer vocab so the tokenizer fits
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    m = Qwen3ForCausalLM(cfg)
    m.save_pretrained(dirpath)
    AutoTokenizer.from_pretrained(_REAL_W).save_pretrained(dirpath)


def _write_sparse_yaml(
    path: str,
    *,
    model_path: str,
    save_path: str,
    attn_impl: str,
    variant: str = "a_shape",
    allow_pseudo: bool = True,
) -> None:
    import yaml as _yaml

    doc = {
        "global": {"save_path": save_path, "deploy_backend": "huggingface"},
        "model": {
            "name": "Qwen",
            "model_path": model_path,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "use_cache": False,
            "torch_dtype": "bfloat16" if CUDA else "float32",
            "device_map": "cuda" if CUDA else "cpu",
            "attn_implementation": attn_impl,
        },
        "compression": {
            "name": "Sparsity",
            "sparsity": {
                "name": variant,
                "allow_pseudo_sparse": allow_pseudo,
                "attn_kwargs": {"n_init": 128, "n_local": 3968},
            },
        },
    }
    with open(path, "w") as f:
        _yaml.safe_dump(doc, f)


def _run_engine_compress(yaml_path: str):
    """Mirror tools/run.py::run() for a sparse YAML; return (config, save_dir)."""
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register sparse algorithms
    from angelslim.engine import Engine
    from angelslim.utils.config_parser import SlimConfigParser

    config = SlimConfigParser().parse(yaml_path)
    mc, gc, cc = config.model_config, config.global_config, config.compression_config

    engine = Engine()
    engine.prepare_model(
        model_name=mc.name,
        model_path=mc.model_path,
        torch_dtype=mc.torch_dtype,
        device_map=mc.device_map,
        trust_remote_code=mc.trust_remote_code,
        low_cpu_mem_usage=mc.low_cpu_mem_usage,
        use_cache=mc.use_cache,
        cache_dir=mc.cache_dir,
        attn_implementation=mc.attn_implementation,
        deploy_backend=gc.deploy_backend,
    )
    engine.prepare_compressor(
        compress_name=cc.name,
        compress_config=cc,
        global_config=gc,
        transform_config=config.transform_config,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.run()  # no-op for sparse
        engine.convert()  # applies the patch
        engine.save(gc.save_path, config)
    return engine, gc.save_path


# ===========================================================================
# — compression_config (incl. sparsity sub-config) survives JSON


# ===========================================================================
def test_load_recovers_compression_config():
    """Pure serialization: a Sparsity FullConfig -> asdict -> json -> parse keeps
    the compression_config and its sparsity sub-config (no weights / no CUDA)."""
    from dataclasses import asdict

    from angelslim.utils.config_parser import (
        CompressionConfig,
        FullConfig,
        ModelConfig,
        SparsityConfig,
        parse_json_full_config,
    )

    cfg = FullConfig(
        model_config=ModelConfig(
            name="Qwen", model_path="x", attn_implementation="flash_attention_2"
        ),
        compression_config=CompressionConfig(
            name="Sparsity",
            sparsity=SparsityConfig(
                name="a_shape",
                attn_kwargs={"n_init": 128, "n_local": 3968},
                allow_pseudo_sparse=False,
            ),
        ),
    )
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "angelslim_config.json")
        with open(p, "w") as f:
            json.dump(asdict(cfg), f)
        out = parse_json_full_config(p)

    assert out.compression_config is not None, "compression_config dropped "

    assert "Sparsity" in out.compression_config.name
    assert out.compression_config.sparsity is not None
    assert out.compression_config.sparsity.name == "a_shape"
    assert out.compression_config.sparsity.attn_kwargs["n_local"] == 3968
    # at the serialization layer: attn_implementation survives the round trip.

    assert out.model_config.attn_implementation == "flash_attention_2"


# ===========================================================================
# (fix) — Engine.prepare_model forwards attn_implementation to the model


# ===========================================================================
def test_prepare_model_forwards_attn_implementation():
    """The LLM/VLM branch of Engine.prepare_model must forward attn_implementation
    (the Omni branch already did). Without the forward, a sparse compress run
    loads SDPA despite the q3 fa2 default. Tested with 'eager' so it runs on CPU
    too (no flash_attn dependency)."""
    if not tokenizer_available():
        raise _Skip("real tokenizer (Qwen3-8B dir) unavailable")
    from angelslim.engine import Engine

    with tempfile.TemporaryDirectory(dir=_SCRATCH) as src:
        _make_tiny_source(src)
        engine = Engine()
        engine.prepare_model(
            model_name="Qwen",
            model_path=src,
            torch_dtype="float32",
            device_map="cpu",
            attn_implementation="eager",
            deploy_backend="huggingface",
        )
        impl = engine.slim_model.model.config._attn_implementation
    assert impl == "eager", f"prepare_model dropped attn_implementation: {impl!r}"


# ===========================================================================
# — Engine.save emits weights + tokenizer + angelslim_config.json


# ===========================================================================
def test_save_emits_weights_and_tokenizer():
    """Drive the real compress pipeline to save; assert the saved directory is a
    loadable HF dir (weights + config + tokenizer) AND carries
    angelslim_config.json with the Sparsity compression_config."""

    if not tokenizer_available():
        raise _Skip("real tokenizer (Qwen3-8B dir) unavailable")

    with tempfile.TemporaryDirectory(dir=_SCRATCH) as src, tempfile.TemporaryDirectory(
        dir=_SCRATCH
    ) as out:
        _make_tiny_source(src)
        yaml_path = os.path.join(src, "_sparse.yaml")
        _write_sparse_yaml(
            yaml_path,
            model_path=src,
            save_path=out,
            attn_impl="eager",
            allow_pseudo=True,
        )
        _run_engine_compress(yaml_path)

        files = set(os.listdir(out))
        has_weights = (
            any(f.endswith(".safetensors") for f in files)
            or "model.safetensors.index.json" in files
        )
        assert has_weights, f"no weights saved: {files}"

        assert "config.json" in files, f"no HF config.json: {files}"
        assert any(f.startswith("tokenizer") for f in files), f"no tokenizer: {files}"
        assert "angelslim_config.json" in files, f"no angelslim_config.json: {files}"

        with open(os.path.join(out, "angelslim_config.json")) as f:
            saved = json.load(f)
        comp = saved.get("compression_config")
        assert comp is not None and "Sparsity" in comp["name"], comp
        assert comp["sparsity"]["name"] == "a_shape", comp["sparsity"]


# ===========================================================================
# — InferEngine.from_pretrained auto-re-applies the sparse patch

# — and preserves the saved attn_implementation on the live model


# ===========================================================================
def test_load_re_applies_sparse_patch():
    """Full genuine round trip on synthetic weights: compress+save, then
    InferEngine.from_pretrained — assert the sparse patch auto-re-applied (the
    marker is present on a patched attn.forward) and the saved

    attn_implementation is honored on the reloaded model."""
    if not tokenizer_available():
        raise _Skip("real tokenizer (Qwen3-8B dir) unavailable")
    import angelslim.compressor.sparsity.algorithms  # noqa: F401
    from angelslim.compressor.sparsity._layers import resolve_sparsable_layers
    from angelslim.compressor.sparsity.patcher import _SPARSE_MARKER
    from angelslim.engine import InferEngine

    # Use a CPU-runnable attn impl so this default test needs no flash_attn/GPU.
    impl = "eager"

    with tempfile.TemporaryDirectory(dir=_SCRATCH) as src, tempfile.TemporaryDirectory(
        dir=_SCRATCH
    ) as out:
        _make_tiny_source(src)
        yaml_path = os.path.join(src, "_sparse.yaml")
        _write_sparse_yaml(
            yaml_path,
            model_path=src,
            save_path=out,
            attn_impl=impl,
            allow_pseudo=True,
        )
        _run_engine_compress(yaml_path)

        infer = InferEngine()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            infer.from_pretrained(
                out,
                torch_dtype="float32",
                device_map="cpu",
            )
        model = infer.slim_model.model

        # at least one full-attention layer carries a sparse-marked forward.

        marked = [
            layer
            for layer in resolve_sparsable_layers(model)
            if getattr(layer.self_attn.forward, _SPARSE_MARKER, False)
        ]
        assert marked, "sparse patch did NOT auto-re-apply on load "

        # the live reloaded model honors the saved attn_implementation.

        assert model.config._attn_implementation == impl, (
            f"attn_implementation not preserved on load: " f"{model.config._attn_implementation!r}"
        )
        assert hasattr(infer, "_sparse_compressor"), "no sparse compressor cached"


# ===========================================================================
# — attn_implementation=flash_attention_2 preserved on a CUDA load


# ===========================================================================
def test_attn_impl_preserved():
    """The q3 default fa2 specifically survives the round trip on a real CUDA
    load (the CPU test above uses eager). Skips without CUDA / flash_attn."""
    if not CUDA:
        raise _Skip("CUDA required for flash_attention_2")
    if not tokenizer_available():
        raise _Skip("real tokenizer (Qwen3-8B dir) unavailable")
    try:
        import flash_attn  # noqa: F401
    except Exception:
        raise _Skip("flash_attn not installed")
    import angelslim.compressor.sparsity.algorithms  # noqa: F401
    from angelslim.engine import InferEngine

    with tempfile.TemporaryDirectory(dir=_SCRATCH) as src, tempfile.TemporaryDirectory(
        dir=_SCRATCH
    ) as out:
        _make_tiny_source(src)
        yaml_path = os.path.join(src, "_sparse.yaml")
        _write_sparse_yaml(
            yaml_path,
            model_path=src,
            save_path=out,
            attn_impl="flash_attention_2",
            allow_pseudo=True,
        )
        _run_engine_compress(yaml_path)

        infer = InferEngine()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            infer.from_pretrained(out)
        impl = infer.slim_model.model.config._attn_implementation
    assert impl == "flash_attention_2", f"fa2 not preserved on load: {impl!r}"


# ===========================================================================
# Semantic e2e (GATED) — reloaded REAL sparse model still answers correctly
# ===========================================================================
def test_engine_e2e_save_load_real_weights():
    """The headline: the genuine CLI-equivalent path on REAL Qwen3-8B.

    Engine.prepare_model -> prepare_compressor -> run -> convert -> save, then
    InferEngine.from_pretrained -> generate. Asserts the reloaded sparse model
    (a) re-applied the patch, (b) honors fa2, (c) still answers "Paris". Heavy
    (~16 GB written + two loads), so gated behind SPARSE_E2E_SAVE=1."""
    if os.environ.get("SPARSE_E2E_SAVE") != "1":
        raise _Skip("set SPARSE_E2E_SAVE=1 to run the heavy real-weights e2e")
    if not (CUDA and real_weights_available()):
        raise _Skip("CUDA + real Qwen3-8B weights required")
    try:
        import flash_attn  # noqa: F401
    except Exception:
        raise _Skip("flash_attn not installed")
    import angelslim.compressor.sparsity.algorithms  # noqa: F401
    from angelslim.compressor.sparsity._layers import resolve_sparsable_layers
    from angelslim.compressor.sparsity.patcher import _SPARSE_MARKER
    from angelslim.engine import InferEngine

    with tempfile.TemporaryDirectory(dir=_SCRATCH) as out:
        yaml_path = os.path.join(out, "_sparse.yaml")
        _write_sparse_yaml(
            yaml_path,
            model_path=_REAL_W,
            save_path=out,
            attn_impl="flash_attention_2",
            variant="a_shape",
            allow_pseudo=False,
        )
        engine, _ = _run_engine_compress(yaml_path)
        # Free the compress-time model before reloading a second copy.
        del engine
        torch.cuda.empty_cache()

        infer = InferEngine()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            infer.from_pretrained(out)
        model = infer.slim_model.model

        marked = [
            layer
            for layer in resolve_sparsable_layers(model)
            if getattr(layer.self_attn.forward, _SPARSE_MARKER, False)
        ]
        assert marked, "sparse patch did NOT auto-re-apply on real load "

        assert model.config._attn_implementation == "flash_attention_2"

        tok = infer.slim_model.tokenizer
        enc = tok("The capital of France is", return_tensors="pt").to(model.device)
        n_in = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=8, do_sample=False)
        text = tok.decode(gen[0, n_in:])

    del infer, model
    torch.cuda.empty_cache()
    assert "Paris" in text, f"reloaded real sparse model lost the answer: {text!r}"


# ---------------------------------------------------------------------------


# ===========================================================================
# / — minference pattern_path save->reload round-trip, OFF-CWD,

# YAML-dir relative resolution, and missing-pattern hard-fail.

# (merged from the former test_pattern_path_roundtrip.py)
# ===========================================================================
# Tiny model config — also the pattern fingerprint must match these exactly.
_N_LAYERS = 4
_N_HEADS = 4
_N_KV = 2
_HIDDEN = 256
_MAXPOS = 4096
_ROPE_THETA = 1000000.0


def _pp_make_tiny_source(dirpath):
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=151936,
        hidden_size=_HIDDEN,
        intermediate_size=512,
        num_hidden_layers=_N_LAYERS,
        num_attention_heads=_N_HEADS,
        num_key_value_heads=_N_KV,
        head_dim=HEAD_DIM,
        max_position_embeddings=_MAXPOS,
        rope_theta=_ROPE_THETA,
        tie_word_embeddings=True,
    )
    Qwen3ForCausalLM(cfg).save_pretrained(dirpath)
    AutoTokenizer.from_pretrained(_REAL_W).save_pretrained(dirpath)
    return cfg


def _write_pattern_json(path):
    """A minimal, schema-v1, fingerprint-matching minference pattern."""
    doc = {
        "schema_version": 1,
        "model_fingerprint": {
            "model_type": "qwen3",
            "rope_theta": _ROPE_THETA,
            "max_position_embeddings": _MAXPOS,
            "num_attention_heads": _N_HEADS,
            "num_key_value_heads": _N_KV,
            "hidden_size": _HIDDEN,
            "num_hidden_layers": _N_LAYERS,
        },
        # best_pattern: per layer → per head → ["vertical_and_slash", v, s]
        "best_pattern": {
            str(layer): {str(head): ["vertical_and_slash", 64, 256] for head in range(_N_HEADS)}
            for layer in range(_N_LAYERS)
        },
    }
    with open(path, "w") as f:
        json.dump(doc, f)


def _write_minference_yaml(path, *, model_path, save_path, pattern_path):
    import yaml as _yaml

    doc = {
        "global": {"save_path": save_path, "deploy_backend": "huggingface"},
        "model": {
            "name": "Qwen",
            "model_path": model_path,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "use_cache": False,
            "torch_dtype": "bfloat16" if CUDA else "float32",
            "device_map": "cuda" if CUDA else "cpu",
            "attn_implementation": "eager",
        },
        "compression": {
            "name": "Sparsity",
            "sparsity": {
                "name": "minference",
                "allow_pseudo_sparse": True,  # reference path, no fused kernel
                "attn_kwargs": {"pattern_path": pattern_path},
            },
        },
    }
    with open(path, "w") as f:
        _yaml.safe_dump(doc, f)


def test_minference_pattern_path_offcwd_roundtrip():
    if not tokenizer_available():
        raise _Skip("tokenizer checkpoint dir unavailable", _SR.NO_TOKENIZER)
    from angelslim.engine import Engine, InferEngine
    from angelslim.utils.config_parser import SlimConfigParser

    with tempfile.TemporaryDirectory(dir=_SCRATCH) as tmp:
        src = os.path.join(tmp, "src")
        _pp_make_tiny_source(src)
        pattern_path = os.path.join(tmp, "searched_pattern.json")
        _write_pattern_json(pattern_path)
        save_path = os.path.join(tmp, "saved")
        yaml_path = os.path.join(tmp, "minference.yaml")
        _write_minference_yaml(
            yaml_path, model_path=src, save_path=save_path, pattern_path=pattern_path
        )

        # Compress + save via the genuine Engine pipeline.
        cfg = SlimConfigParser().parse(yaml_path)
        eng = Engine()
        eng.prepare_model(
            model_name=cfg.model_config.name,
            model_path=src,
            torch_dtype=cfg.model_config.torch_dtype,
            device_map=cfg.model_config.device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False,
            deploy_backend="huggingface",
            attn_implementation="eager",
        )
        eng.prepare_compressor(
            compress_name=cfg.compression_config.name,
            compress_config=cfg.compression_config,
            global_config=cfg.global_config,
        )
        eng.run()
        eng.convert()
        eng.save(save_path, cfg)

        # The pattern must have been copied next to the checkpoint.
        copied = os.path.join(save_path, "sparse_patterns", "searched_pattern.json")
        assert os.path.isfile(
            copied
        ), f"save() did not copy the pattern into sparse_patterns/: {copied}"
        # The serialized config must record a RELATIVE path (sanitized model_path).
        with open(os.path.join(save_path, "angelslim_config.json")) as f:
            saved_cfg = json.load(f)
        assert saved_cfg["model_config"]["model_path"] == "Base Model Path", (
            "precondition: Engine.save sanitizes model_path (this is what made "
            "off-CWD resolution hard)"
        )

        # Reload from an UNRELATED CWD — the off-CWD scenario is about.

        prev_cwd = os.getcwd()
        elsewhere = os.path.join(tmp, "elsewhere")
        os.makedirs(elsewhere, exist_ok=True)
        os.chdir(elsewhere)
        try:
            ie = InferEngine()
            # Must NOT raise IncompatibleConfigError("pattern_path does not exist").
            ie.from_pretrained(model_path=save_path)
        finally:
            os.chdir(prev_cwd)

        # The reloaded sparse compressor resolved the pattern to the real file
        # under the checkpoint (not CWD).
        sparse = getattr(ie, "_sparse_compressor", None)
        assert sparse is not None, "reload did not create a sparse compressor"
        resolved = getattr(sparse, "_pattern_path", None)
        assert resolved and os.path.isfile(
            resolved
        ), f"pattern_path did not resolve on off-CWD reload: {resolved!r}"
        assert os.path.realpath(resolved) == os.path.realpath(
            copied
        ), f"resolved to the wrong file: {resolved!r} != {copied!r}"
        _cap("pattern_path_offcwd")


def test_pattern_path_resolves_against_yaml_dir_not_cwd():
    """A relative ``pattern_path`` in a YAML must resolve against the YAML's own
    directory (``global_config.config_dir``), NOT the process CWD, so
    ``tools/run.py -c /elsewhere/foo.yaml`` works from any directory.

    Weightless: drives ``Sparsity._resolve_pattern_path`` directly with a
    ``global_config`` carrying ``config_dir``. The pattern file exists ONLY under
    the config dir; CWD is an unrelated empty dir. Correct resolution requires
    the config_dir branch."""
    import types

    from angelslim.compressor.sparsity.sparsity import Sparsity

    with tempfile.TemporaryDirectory(dir=_SCRATCH) as tmp:
        config_dir = os.path.join(tmp, "cfgdir")
        os.makedirs(config_dir, exist_ok=True)
        # The pattern lives next to the (hypothetical) YAML, under config_dir.
        rel = os.path.join("patterns", "p.json")
        abs_in_cfgdir = os.path.join(config_dir, rel)
        os.makedirs(os.path.dirname(abs_in_cfgdir), exist_ok=True)
        with open(abs_in_cfgdir, "w") as f:
            f.write("{}")

        gc = types.SimpleNamespace(model_path=None, save_path=None, config_dir=config_dir)
        sp = Sparsity.__new__(Sparsity)
        sp.config = {"global_config": gc}

        # Run from an UNRELATED cwd where the relative path does NOT exist.
        prev_cwd = os.getcwd()
        elsewhere = os.path.join(tmp, "elsewhere")
        os.makedirs(elsewhere, exist_ok=True)
        os.chdir(elsewhere)
        try:
            resolved = sp._resolve_pattern_path(rel)
        finally:
            os.chdir(prev_cwd)

        assert resolved and os.path.isfile(
            resolved
        ), f"relative pattern_path did not resolve against the YAML dir: {resolved!r}"
        assert os.path.realpath(resolved) == os.path.realpath(
            abs_in_cfgdir
        ), f"resolved to the wrong file (CWD instead of config_dir?): {resolved!r}"
        _cap("pattern_path_yaml_dir")


def test_save_hardfails_when_pattern_path_missing():
    """if a pattern_path was configured but the file is gone at save time,

    Sparsity.save() must HARD-FAIL — not silently skip the rewrite and leave the
    original (absolute, internal) path in the serialized config (which would both
    leak an internal path and break reload). Construct a Sparsity compressor with a
    _pattern_path pointing at a now-deleted file and assert save() raises, and that
    no angelslim config leaks the absolute path."""
    from angelslim.compressor.sparsity.sparsity import Sparsity

    with tempfile.TemporaryDirectory(dir=_SCRATCH) as tmp:
        # A pattern path that does NOT exist (the scenario: configured but

        # missing at save time). Use an absolute, internal-looking path.
        missing = os.path.join(tmp, "secret_internal_dir", "searched_pattern.json")
        save_path = os.path.join(tmp, "saved")

        # Build a minimal Sparsity instance without running compression: we only
        # exercise save()'s pattern-path branch. __new__ + set the two attributes
        # save() touches before the pattern block (slim_model.model / tokenizer).
        sp = Sparsity.__new__(Sparsity)
        sp._pattern_path = missing

        class _FakeHF:
            def save_pretrained(self, p):
                os.makedirs(p, exist_ok=True)

        class _FakeSlim:
            model = _FakeHF()
            tokenizer = None

        sp.slim_model = _FakeSlim()

        raised = None
        try:
            sp.save(save_path)
        except FileNotFoundError as e:
            raised = e
        assert raised is not None, (
            "save() must hard-fail when a configured pattern_path is missing; it "
            "silently skipped (would leak the original absolute path + break reload)"
        )
        # The error must name the missing path (actionable) ...
        assert missing in str(
            raised
        ), f"hard-fail message should name the missing pattern_path; got: {raised}"
        # ... and crucially must NOT have written a config that leaks the path.
        cfg_json = os.path.join(save_path, "angelslim_config.json")
        if os.path.isfile(cfg_json):
            with open(cfg_json) as f:
                body = f.read()
            assert missing not in body, (
                "save() leaked the internal absolute pattern_path into "
                "angelslim_config.json despite the missing file"
            )
        _cap("pattern_path_missing_hardfail")


if __name__ == "__main__":
    sys.exit(1 if run_all(globals(), f"CUDA={CUDA}") else 0)
