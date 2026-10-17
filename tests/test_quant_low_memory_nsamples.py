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

"""The low-memory capture buffer must hold every sample, not one per batch.

``INT8.low_memory_run`` and ``AWQ.run`` allocate ``self.inps`` with
``nsamples`` rows and then copy the inputs captured by ``Catcher`` into it:

    for idx in range(min(len(captured_inputs), self.inps.shape[0])):
        self.inps[idx, : inp.shape[1], :].copy_(inp[0])

``min(...)`` makes the buffer the silent ceiling on how much data the
calibration sees. Sizing it from ``len(dataloader)`` - one row per *batch* -
therefore keeps only that many samples and discards the rest, so int8/AWQ
low-memory calibration ran on a fraction of the corpus while reporting
success. The buffer has to be sized from the number of *samples*.

These tests drive the real methods with a stubbed model, following the
approach used by ``tests/test_dataloader.py``: the modules are loaded
individually with their third-party imports stubbed, so no weights or GPU are
needed.
"""

import importlib.util
import os
import sys
import types

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INT8_PATH = os.path.join(
    _REPO_ROOT, "angelslim", "compressor", "quant", "modules", "int8", "int8.py"
)
_AWQ_PATH = os.path.join(
    _REPO_ROOT, "angelslim", "compressor", "quant", "modules", "awq", "awq.py"
)

BATCHES = 3
BATCH_SIZE = 8
SAMPLES = BATCHES * BATCH_SIZE


class _FakeDataloader:
    """Only the two attributes the allocation reads."""

    def __init__(self, n_batches=BATCHES, batch_size=BATCH_SIZE):
        self.n_batches = n_batches
        self.batch_size = batch_size

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return iter([])


class _StopAfterCapture(Exception):
    """Ends the run at the point where the captured inputs have been copied.

    Both methods carry on into the full quantization pass, which needs a real
    model. Everything under test happens before that, so the stand-in layer
    raises as soon as the code unwraps ``Catcher``.
    """


class _CapturingLayer:
    """Stands in for ``Catcher``: records what the model forward pushes in."""

    def __init__(self, *args, **kwargs):
        self.captured_inputs = []
        self.captured_kwargs = []

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self

    @property
    def module(self):
        raise _StopAfterCapture


class _FakeEmbed:
    def to(self, *_args, **_kwargs):
        return self


class _FakeInner:
    def __init__(self):
        self.embed_tokens = _FakeEmbed()


class _FakeModel:
    """A model whose forward captures one tensor per sample in the batch."""

    def __init__(self, seq_length, hidden_size, layers):
        self.model = types.SimpleNamespace(model=_FakeInner())
        self._seq_length = seq_length
        self._hidden_size = hidden_size
        self._layers = layers

    def model_forward(self, dataloader):
        # ``low_memory_run`` swaps layers[0] for a Catcher before calling this,
        # and the model drives whatever instance is there now.
        layer = self._layers[0]
        for _ in range(len(dataloader)):
            for _ in range(dataloader.batch_size):
                # A distinct value per sample so the assertions can tell a copied
                # row from an untouched one.
                n = len(layer.captured_inputs) + 1
                layer.captured_inputs.append(
                    torch.full((1, self._seq_length, self._hidden_size), float(n))
                )

    def get_pre_transformer_modules(self):
        return {}


def _install_stubs():
    """Make the two modules importable without the rest of AngelSlim."""
    for name in (
        "angelslim",
        "angelslim.compressor",
        "angelslim.compressor.quant",
        "angelslim.compressor.quant.modules",
        "angelslim.compressor.quant.modules.int8",
        "angelslim.compressor.quant.modules.awq",
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod

    utils = types.ModuleType("angelslim.utils")
    utils.get_best_device = lambda: "cpu"
    utils.print_info = lambda *a, **k: None
    utils.find_layers = lambda *a, **k: {}
    utils.set_op_by_name = lambda *a, **k: None
    sys.modules["angelslim.utils"] = utils

    catcher = types.ModuleType("angelslim.compressor.quant.modules.catcher")
    catcher.Catcher = _CapturingLayer
    sys.modules["angelslim.compressor.quant.modules.catcher"] = catcher

    core = types.ModuleType("angelslim.compressor.quant.core")
    core.pseudo_quantize_tensor = lambda t, *a, **k: t
    core.weight_dequant = lambda t, *a, **k: t
    sys.modules["angelslim.compressor.quant.core"] = core

    helper = types.ModuleType("angelslim.compressor.quant.modules.helper_layer")
    helper.WQLinearGEMM = type("WQLinearGEMM", (), {})
    sys.modules["angelslim.compressor.quant.modules.helper_layer"] = helper

    for leaf in ("auto_clip", "auto_scale"):
        mod = types.ModuleType(f"angelslim.compressor.quant.modules.awq.{leaf}")
        setattr(
            mod, "AutoLayerClip" if leaf == "auto_clip" else "AutoLayerScale", type("X", (), {})
        )
        sys.modules[f"angelslim.compressor.quant.modules.awq.{leaf}"] = mod

    for name in ("huggingface_hub", "tqdm"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.save_torch_state_dict = lambda *a, **k: None
            mod.tqdm = lambda x, *a, **k: x
            sys.modules[name] = mod


def _load(path, module_name):
    _install_stubs()
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_SEQ, _HIDDEN = 4, 8


def _int8_instance(module):
    op = module.INT8.__new__(module.INT8)
    op.low_memory = True
    op.seq_length = _SEQ
    op.hidden_size = _HIDDEN
    op.dtype = torch.float32
    layer = _CapturingLayer()
    op.layers = [layer]
    op.model = _FakeModel(_SEQ, _HIDDEN, op.layers)
    return op, op.layers


def test_int8_buffer_holds_one_row_per_sample():
    module = _load(_INT8_PATH, "angelslim.compressor.quant.modules.int8.int8")
    op, _layers = _int8_instance(module)

    dl = _FakeDataloader()
    with pytest.raises(_StopAfterCapture):
        op.low_memory_run(dl)

    assert len(op.layers[0].captured_inputs) == SAMPLES
    assert op.inps.shape[0] == SAMPLES
    # Every captured sample must have landed in the buffer: a row stays zero
    # only if the copy loop was capped below the number of captures.
    assert torch.count_nonzero(op.inps) == SAMPLES * _SEQ * _HIDDEN


def test_int8_buffer_is_not_sized_by_batch_count():
    module = _load(_INT8_PATH, "angelslim.compressor.quant.modules.int8.int8")
    op, _ = _int8_instance(module)
    with pytest.raises(_StopAfterCapture):
        op.low_memory_run(_FakeDataloader())
    assert (
        op.inps.shape[0] != BATCHES
    ), "the buffer must be sized for every sample, not one row per batch"


def test_awq_sizes_its_capture_buffer_for_every_sample():
    module = _load(_AWQ_PATH, "angelslim.compressor.quant.modules.awq.awq")
    op = module.AWQ.__new__(module.AWQ)
    op.seq_length = _SEQ
    op.hidden_size = _HIDDEN
    op.dtype = torch.float32
    layer = _CapturingLayer()
    op.layers = [layer]
    op.model = _FakeModel(_SEQ, _HIDDEN, op.layers)

    recorded = {}

    def _forward(_dataloader):
        # The AWQ search that follows needs a real model; stop here, having
        # already observed the one thing under test.
        recorded["shape"] = tuple(op.inps.shape)
        raise _StopAfterCapture

    op.model.model_forward = _forward

    with pytest.raises(_StopAfterCapture):
        op.run(_FakeDataloader())

    assert recorded["shape"][0] == SAMPLES
