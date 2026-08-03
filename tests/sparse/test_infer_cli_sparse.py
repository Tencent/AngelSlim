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

"""tools/infer.py sparse-CLI path lock.

The user-facing inference entrypoint had two integration breaks the rest of the
suite stepped around (every other e2e test calls ``model.generate`` directly):

  1. the ``-c <sparse.yaml>`` config path called ``prepare_model`` +
     ``prepare_compressor`` but NOT ``run()/convert()`` — so ``compression.name:
     Sparsity`` loaded a DENSE model and silently ignored the sparse config (the
     attention forward was never patched);
  2. ``InferEngine.generate`` called ``self.slim_model.generate(...)``, but
     ``BaseLLMModel`` has no ``generate`` and its ``__getattr__`` hard-raises
     (no proxy to the HF model) — so generate crashed with AttributeError for
     EVERY model, sparse or not.

This file drives the REAL ``tools/infer.py::infer`` function (not a hand-rolled
shortcut) on both code paths:

  * config path  — ``infer(config, args)`` with a sparse YAML; asserts the model
    is actually sparse-patched (marker present on attn.forward) AND generate
    returns tokens.
  * saved path   — save a sparse model, then ``infer(None, args(model_path=...))``
    → ``InferEngine.from_pretrained`` re-applies the patch and generate runs.

Uses a tiny synthetic Qwen3 + the real tokenizer (structural: asserts the patch
fired and generate runs, not weight-dependent numerics) with
``allow_pseudo_sparse`` so it runs without the fused kernel. Skips (never bare
PASS) when the tokenizer checkpoint dir is absent.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import CUDA, HEAD_DIM  # noqa: E402
from _harness import REAL_W_QWEN3_8B as _REAL_W  # noqa: E402
from _harness import SCRATCH_ROOT as _SCRATCH  # noqa: E402
from _harness import Skip as _Skip  # noqa: E402
from _harness import SkipReason as _SR  # noqa: E402
from _harness import record_capability as _cap  # noqa: E402
from _harness import run_all, tokenizer_available  # noqa: E402

# tools/ on sys.path so we exercise the real infer() entrypoint.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "tools"))

_SPARSE_MARKER = "_angelslim_sparse_patched"


def _make_tiny_source(dirpath):
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=151936,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    Qwen3ForCausalLM(cfg).save_pretrained(dirpath)
    AutoTokenizer.from_pretrained(_REAL_W).save_pretrained(dirpath)


def _write_sparse_yaml(path, *, model_path, save_path):
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
                "name": "a_shape",
                "allow_pseudo_sparse": True,  # no fused kernel needed
                "attn_kwargs": {"n_init": 8, "n_local": 16},
            },
        },
    }
    with open(path, "w") as f:
        _yaml.safe_dump(doc, f)


def _args(
    config_path=None,
    model_path=None,
    prompt="The capital of France is",
    output_file=None,
    print_full=False,
):
    import argparse

    return argparse.Namespace(
        config=config_path,
        model_path=model_path,
        input_prompt=prompt,
        save_path="./output/",
        output_file=output_file,
        print_full=print_full,
    )


def _any_attn_patched(engine) -> bool:
    """True if at least one self_attn.forward carries the sparse marker."""
    model = engine.slim_model.model
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", [])
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and getattr(attn.forward, _SPARSE_MARKER, False):
            return True
    return False


# ===========================================================================
# config path: infer(-c sparse.yaml) must PATCH and generate
# ===========================================================================
def test_infer_cli_config_path_patches_and_generates():
    if not tokenizer_available():
        raise _Skip("tokenizer checkpoint dir unavailable", _SR.NO_TOKENIZER)
    import infer as infer_cli  # tools/infer.py

    from angelslim.utils.config_parser import SlimConfigParser

    with tempfile.TemporaryDirectory(dir=_SCRATCH) as tmp:
        src = os.path.join(tmp, "src")
        _make_tiny_source(src)
        yaml_path = os.path.join(tmp, "a_shape.yaml")
        _write_sparse_yaml(yaml_path, model_path=src, save_path=os.path.join(tmp, "out"))

        config = SlimConfigParser().parse(yaml_path)
        args = _args(config_path=yaml_path)

        # Drive the REAL entrypoint. infer() builds the InferEngine, prepares the
        # model, the compressor, then run()+convert() (the fix) and generate().
        # We capture the engine by monkeypatching generate to record + run.
        from angelslim.engine import InferEngine

        captured = {}
        orig_generate = InferEngine.generate

        def _spy_generate(self, prompt, **kw):
            captured["engine"] = self
            captured["patched"] = _any_attn_patched(self)
            return orig_generate(self, prompt, **kw)

        InferEngine.generate = _spy_generate
        try:
            infer_cli.infer(config, args)
        finally:
            InferEngine.generate = orig_generate

        assert "engine" in captured, "infer() never reached generate()"
        assert captured["patched"], (
            "config path did NOT sparse-patch the model — prepare_compressor "
            "without convert() loads a dense model"
        )
        _cap("cli_sparse_config")


# ===========================================================================
# infer() must DECODE and surface generation output (not silently discard it)
# ===========================================================================
def test_infer_cli_emits_decoded_output():
    """The CLI must produce observable generation output. Regression guard for
    the `_ = slim_engine.generate(...)` bug where infer.py ran the model but
    discarded the ids — community users saw "ran, no output". Drives the REAL
    infer() and asserts (a) it returns decoded text, (b) it prints a
    '=== generated ===' block to stdout, (c) --output-file is written."""
    if not tokenizer_available():
        raise _Skip("tokenizer checkpoint dir unavailable", _SR.NO_TOKENIZER)
    import contextlib
    import io

    import infer as infer_cli  # tools/infer.py

    from angelslim.utils.config_parser import SlimConfigParser

    with tempfile.TemporaryDirectory(dir=_SCRATCH) as tmp:
        src = os.path.join(tmp, "src")
        _make_tiny_source(src)
        yaml_path = os.path.join(tmp, "a_shape.yaml")
        _write_sparse_yaml(yaml_path, model_path=src, save_path=os.path.join(tmp, "out"))
        out_file = os.path.join(tmp, "gen.txt")

        config = SlimConfigParser().parse(yaml_path)
        args = _args(config_path=yaml_path, output_file=out_file)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            returned = infer_cli.infer(config, args)
        printed = buf.getvalue()

        # (a) infer() returns the decoded text (not None / not token ids).
        assert isinstance(
            returned, str
        ), f"infer() should return decoded text, got {type(returned)}"
        # (b) a generation block was printed to stdout.
        assert (
            "=== generated ===" in printed
        ), "infer() did not print a generation block — output was discarded"
        # (c) --output-file was written with the same text.
        assert os.path.isfile(out_file), "infer() did not honor --output-file"
        with open(out_file) as f:
            assert f.read() == returned, "output-file content != returned text"
        _cap("cli_sparse_output")


# ===========================================================================
# saved path: infer(--model-path saved_sparse) re-patches and generates
# ===========================================================================
def test_infer_cli_saved_path_repatches_and_generates():
    if not tokenizer_available():
        raise _Skip("tokenizer checkpoint dir unavailable", _SR.NO_TOKENIZER)
    import infer as infer_cli

    from angelslim.engine import Engine, InferEngine
    from angelslim.utils.config_parser import SlimConfigParser

    with tempfile.TemporaryDirectory(dir=_SCRATCH) as tmp:
        src = os.path.join(tmp, "src")
        _make_tiny_source(src)
        save_path = os.path.join(tmp, "saved")
        yaml_path = os.path.join(tmp, "a_shape.yaml")
        _write_sparse_yaml(yaml_path, model_path=src, save_path=save_path)

        # Build + save a sparse model via the genuine Engine compress pipeline.
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
        eng.save(save_path, cfg)  # config arg → writes angelslim_config.json

        # Now drive infer.py's saved path: config=None, model_path=save_path.
        args = _args(model_path=save_path)
        captured = {}
        orig_generate = InferEngine.generate

        def _spy_generate(self, prompt, **kw):
            captured["patched"] = _any_attn_patched(self)
            return orig_generate(self, prompt, **kw)

        InferEngine.generate = _spy_generate
        try:
            infer_cli.infer(None, args)
        finally:
            InferEngine.generate = orig_generate

        assert "patched" in captured, "infer() never reached generate() on saved path"
        assert captured["patched"], "saved path did NOT re-apply the sparse patch on load"
        _cap("cli_sparse_saved")


# ===========================================================================
# Non-Sparsity configs must NOT auto-run calibration/convert from infer.py
# ===========================================================================
def test_infer_cli_non_sparsity_does_not_autoconvert():
    """A PTQ/QAT/Distill config through tools/infer.py must NOT call run()/
    convert() — those need a dataset and mutate the model; an inference
    entrypoint auto-running them is a CLI regression. Only Sparsity auto-patches.

    Hermetic: stub prepare_model/prepare_compressor/generate and spy on
    run()/convert(); assert they fire for a Sparsity config but not for a PTQ
    one. No weights/tokenizer needed."""
    import types

    import infer as infer_cli  # tools/infer.py

    from angelslim.engine import InferEngine

    def _make_cfg(compress_name):
        ns = types.SimpleNamespace
        return ns(
            model_config=ns(
                name="Qwen",
                model_path="/nonexistent",
                torch_dtype="float32",
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=False,
                cache_dir=None,
                attn_implementation="eager",
            ),
            compression_config=ns(name=compress_name),
            global_config=ns(deploy_backend="huggingface"),
            infer_config=None,
        )

    calls = {"run": 0, "convert": 0}
    o_prep_model = InferEngine.prepare_model
    o_prep_comp = InferEngine.prepare_compressor
    o_run = InferEngine.run
    o_convert = InferEngine.convert
    o_gen = InferEngine.generate
    InferEngine.prepare_model = lambda self, **kw: None
    InferEngine.prepare_compressor = lambda self, **kw: None
    InferEngine.run = lambda self, *a, **kw: calls.__setitem__("run", calls["run"] + 1)
    InferEngine.convert = lambda self, *a, **kw: calls.__setitem__("convert", calls["convert"] + 1)
    InferEngine.generate = lambda self, prompt, **kw: None
    try:
        # PTQ config: run/convert must NOT be called.
        infer_cli.infer(_make_cfg("PTQ"), _args(config_path="x.yaml"))
        assert calls == {
            "run": 0,
            "convert": 0,
        }, f"non-Sparsity config auto-ran calibration/convert: {calls}"
        # Sparsity config: run/convert MUST be called (the patch path).
        infer_cli.infer(_make_cfg("Sparsity"), _args(config_path="x.yaml"))
        assert calls == {"run": 1, "convert": 1}, f"Sparsity config did not auto-patch: {calls}"
    finally:
        InferEngine.prepare_model = o_prep_model
        InferEngine.prepare_compressor = o_prep_comp
        InferEngine.run = o_run
        InferEngine.convert = o_convert
        InferEngine.generate = o_gen


if __name__ == "__main__":
    sys.exit(1 if run_all(globals(), f"CUDA={CUDA}") else 0)
