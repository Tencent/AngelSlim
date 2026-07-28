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

"""Lock tests for the shared test harness (``_harness.py``) — the anti-drift
guarantee.

WHY THIS EXISTS: the helpers in ``_harness.py`` used to be copy-pasted across 6
suites and silently drifted — 2 copies of the patch double dropped the
``expected_label`` mismatch check and stubbed ``attn_forward_labels`` to
``set()`` (so the patch contract was untested there), and 2 of 7
``_run_all`` copies omitted the coverage floor. Centralizing fixed it; THIS
file makes the centralized contract *executable* so a future edit that loosens
``FakeSlim`` or ``run_all`` fails here.

Each of these is a "guard-liveness" test: it asserts the
guard REALLY fires on the bad case, not merely that the good case passes.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import FakeSlim, Skip, run_all  # noqa: E402


def _run_all_quiet(ns):
    """Invoke run_all on a throwaway namespace, swallowing its stdout.

    The meta-tests below deliberately drive run_all with failing / all-skip
    fixtures; their PASS/FAIL/=== lines must NOT leak to the real console (the
    test gate greps for those tokens to detect failures)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return run_all(ns)


class _Attn:
    """Stand-in attention module. ``forward`` is an INSTANCE attribute holding a
    distinct callable (mirroring how the production patcher assigns
    ``attn.forward = fwd``), so identity comparison is stable — unlike a bound
    class method, which yields a fresh object on every access."""

    def __init__(self):
        # A per-instance closure: a unique, identity-stable callable.
        self.forward = lambda *a, **k: "orig"


# ---------------------------------------------------------------------------
# FakeSlim — LIFO push/pop contract (the drifted copies skipped these)
# ---------------------------------------------------------------------------
def test_fakeslim_push_then_pop_restores_original():
    slim = FakeSlim(model=None)
    attn = _Attn()
    orig = attn.forward

    def patched():
        return "patched"

    slim.push_attn_forward("sparse", attn, patched)
    assert attn.forward is patched
    assert slim.attn_forward_labels() == {"sparse"}, "labels must be reported, not stubbed"

    slim.pop_attn_forward(attn, expected_label="sparse")
    assert attn.forward is orig, "pop must restore the captured original forward"
    assert slim.attn_forward_labels() == set()


def test_fakeslim_pop_label_mismatch_raises():
    """The drifted copies IGNORED expected_label — this asserts the real LIFO
    discipline: popping with the wrong label is a hard error."""
    slim = FakeSlim(model=None)
    attn = _Attn()
    slim.push_attn_forward("sparse", attn, lambda: None)
    try:
        slim.pop_attn_forward(attn, expected_label="qwen_fp8")
        raise AssertionError("expected RuntimeError on label mismatch")
    except RuntimeError as e:
        assert "mismatch" in str(e).lower()
    # The patch must still be on the stack (a failed pop does not remove it).
    assert slim.attn_forward_labels() == {"sparse"}


def test_fakeslim_pop_empty_stack_raises():
    """The drifted copies silently RETURNED on an empty stack — this asserts the
    real contract: popping when nothing is patched is a hard error."""
    slim = FakeSlim(model=None)
    attn = _Attn()
    try:
        slim.pop_attn_forward(attn)
        raise AssertionError("expected RuntimeError popping an empty stack")
    except RuntimeError as e:
        assert "no patch" in str(e).lower()


def test_fakeslim_double_push_same_module_raises():
    """Per-module SINGLE SLOT: pushing a second patch onto an
    already-patched module must hard-fail. This replaces the old "LIFO stack of
    two patches on one module" test — sparse never stacks, so the contract is
    'no stacking', enforced at the API layer."""
    slim = FakeSlim(model=None)
    attn = _Attn()
    orig = attn.forward
    slim.push_attn_forward("a", attn, lambda: "A")
    try:
        slim.push_attn_forward("b", attn, lambda: "B")
        raise AssertionError("expected RuntimeError on double-push to same module")
    except RuntimeError as e:
        assert "stack" in str(e).lower() or "already" in str(e).lower()
    # The first patch is intact and unaffected by the refused second push.
    assert slim.attn_forward_labels() == {"a"}
    assert attn.forward() == "A"
    slim.pop_attn_forward(attn, expected_label="a")
    assert attn.forward is orig


def test_fakeslim_two_modules_independent_stacks():
    slim = FakeSlim(model=None)
    a, b = _Attn(), _Attn()
    oa, ob = a.forward, b.forward
    slim.push_attn_forward("sparse", a, lambda: None)
    slim.push_attn_forward("sparse", b, lambda: None)
    slim.pop_attn_forward(a, expected_label="sparse")
    assert a.forward is oa
    assert b.forward is not ob, "popping a must not touch b"
    slim.pop_attn_forward(b, expected_label="sparse")
    assert b.forward is ob


# ---------------------------------------------------------------------------
# run_all — coverage floor (all-skip / nothing-ran must NOT read green)
# ---------------------------------------------------------------------------
def test_run_all_all_skipped_is_failure():
    """A suite where every test raises Skip returns a NON-zero code — the
    coverage floor. (The two suites that omitted this floor are why it was added.)"""

    def test_only_skips():
        raise Skip("no hardware")

    ns = {"test_only_skips": test_only_skips}
    rc = _run_all_quiet(ns)
    assert rc != 0, "all-skipped run must be treated as FAIL (coverage floor)"


def test_run_all_passes_when_something_ran():
    def test_trivial_pass():
        assert True

    rc = _run_all_quiet({"test_trivial_pass": test_trivial_pass})
    assert rc == 0


def test_run_all_failure_propagates():
    def test_boom():
        raise AssertionError("boom")

    def test_ok():
        assert True

    rc = _run_all_quiet({"test_boom": test_boom, "test_ok": test_ok})
    assert rc != 0, "a failing test must make run_all return non-zero"


if __name__ == "__main__":
    sys.exit(1 if run_all(globals()) else 0)
