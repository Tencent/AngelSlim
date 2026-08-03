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

"""The `[sparse]` extra pins transformers to `[5.8, 6.0)`.

The CI lane runs `pip install -e .[sparse]` and asserts
`transformers.__version__` in `[5.8, 6.0)`. The actual `pip install` is the CI
lane's job (it owns a clean env); doing a live reinstall inside a test would
mutate the shared dev env and is not hermetic. So this test binds the same
CONTRACT without the destructive install:

  1. the transformers currently importable in this env satisfies the sparse
     floor+ceiling `>=5.8,<6.0` (i.e. the env the sparse suites actually run in
     is a valid `[sparse]` env — what the CI assertion ultimately checks);
  2. the `[sparse]` extra in setup.py resolves to requirements_sparse.txt, which
     DECLARES `transformers>=5.8.0,<6.0` (so `pip install .[sparse]` would
     enforce it for a fresh env);
  3. the MAIN requirements keep the `<6.0` ceiling (the 6.0 removal of
     `config._attn_implementation` breaks every path, sparse or not).

(1) is the runtime fact; (2)+(3) are the static guarantee that a from-scratch
install reproduces it. Together they are the parseable, non-destructive form of
the CI assertion. Pure text + one import; no GPU, no network.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# tests/ -> repo root (this file is at tests/test_install_extras.py).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_version(v: str) -> tuple[int, int]:
    """(major, minor) from a version string like '5.9.0' / '5.9.0.dev0'."""
    m = re.match(r"^(\d+)\.(\d+)", v)
    assert m, f"unparseable transformers version: {v!r}"
    return int(m.group(1)), int(m.group(2))


def _within_5_8_to_6_0(v: str) -> bool:
    major, minor = _parse_version(v)
    # >= 5.8 and < 6.0
    return major == 5 and minor >= 8


def test_installed_transformers_in_sparse_range():
    """(1) The transformers actually importable here satisfies [5.8, 6.0)."""
    import transformers

    v = transformers.__version__
    assert _within_5_8_to_6_0(v), (
        f"transformers {v} is outside the sparse extra's [5.8, 6.0) — the env "
        f"running the sparse suites is not a valid [sparse] env."
    )


def test_sparse_extra_declares_5_8_floor_and_6_0_ceiling():
    """(2) setup.py's [sparse] extra resolves to a requirements file pinning
    transformers>=5.8,<6.0."""
    setup_src = open(os.path.join(_REPO, "setup.py")).read()
    # The extra must point the sparse key at requirements_sparse.txt.
    assert re.search(
        r'"sparse":\s*get_requirements\(\s*["\']requirements/requirements_sparse\.txt["\']',
        setup_src,
    ), "setup.py [sparse] extra does not load requirements/requirements_sparse.txt"

    req = open(os.path.join(_REPO, "requirements/requirements_sparse.txt")).read()
    line = next(
        (ln.strip() for ln in req.splitlines() if ln.strip().startswith("transformers")),
        None,
    )
    assert line is not None, "requirements_sparse.txt has no transformers pin"
    assert (
        ">=5.8" in line and "<6.0" in line
    ), f"sparse transformers pin must be >=5.8,<6.0; got {line!r}"


def test_main_requirements_keep_6_0_ceiling():
    """(3) The main requirements keep the <6.0 ceiling that protects every path
    from the 6.0 removal of config._attn_implementation."""
    req = open(os.path.join(_REPO, "requirements/requirements.txt")).read()
    line = next(
        (ln.strip() for ln in req.splitlines() if ln.strip().startswith("transformers")),
        None,
    )
    assert line is not None, "main requirements.txt has no transformers pin"
    assert "<6.0" in line, f"main transformers pin lost the <6.0 ceiling: {line!r}"


def test_manifest_ships_every_requirements_file_setup_reads():
    """sdist guard: every ``requirements/*.txt`` that setup.py reads at BUILD
    time (via get_requirements) must be covered by a MANIFEST.in directive, or
    it is absent from the sdist and `pip install <sdist>.tar.gz` dies with
    ``FileNotFoundError: requirements/...`` before any dependency resolves.

    Pure text cross-check of setup.py against MANIFEST.in — the real end-to-end
    `python setup.py sdist && pip install dist/*.tar.gz` lives in CI. This binds
    the contract that the two files cannot drift apart again.
    """
    setup_src = open(os.path.join(_REPO, "setup.py")).read()
    # Every path passed to get_requirements("requirements/<name>.txt").
    needed = set(re.findall(r'get_requirements\(\s*["\'](requirements/[^"\']+)["\']', setup_src))
    assert needed, "no get_requirements(...) calls found in setup.py — pattern drift"

    manifest = open(os.path.join(_REPO, "MANIFEST.in")).read()
    # A `graft requirements` (recursive include of the dir) covers all of them;
    # otherwise each path needs its own include / recursive-include line.
    grafts_dir = re.search(r"^\s*graft\s+requirements\s*$", manifest, re.MULTILINE)
    missing = []
    for path in sorted(needed):
        if grafts_dir:
            continue
        # accept an explicit `include requirements/<name>.txt` or a
        # `recursive-include requirements ...` that would match it.
        if re.search(rf"^\s*include\s+{re.escape(path)}\s*$", manifest, re.MULTILINE):
            continue
        if re.search(r"^\s*recursive-include\s+requirements\b", manifest, re.MULTILINE):
            continue
        missing.append(path)
    assert not missing, (
        "MANIFEST.in does not ship these requirements files that setup.py reads "
        f"at build time (sdist install will FileNotFoundError): {missing}. "
        "Add `graft requirements` to MANIFEST.in."
    )


def test_setup_version_is_non_git_tolerant():
    """`setup.py --version` must yield a PEP 440-valid version even when git is
    unavailable (the exact situation when pip's PEP 517 backend re-runs setup.py
    from an extracted, non-git sdist tree). The old git-branch/git-tag logic put
    `fatal: not a git repository...` into the version string, which raised
    packaging.version.InvalidVersion and broke `pip install <sdist>.tar.gz`.

    Hermetic: runs setup.py with an empty PATH (no `git` binary resolvable) in a
    subprocess, so it deterministically exercises the non-git path without
    building a real sdist. The wheel/sdist end-to-end build lives in CI.
    """
    import subprocess

    env = dict(os.environ)
    # Strip PATH so the `git` executable cannot be found -> the version logic
    # must degrade gracefully rather than embed git's error text.
    env["PATH"] = ""
    res = subprocess.run(
        [sys.executable, "setup.py", "--version"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, (
        f"setup.py --version crashed without git (sdist install would break):\n"
        f"stdout={res.stdout}\nstderr={res.stderr}"
    )
    version = res.stdout.strip().splitlines()[-1].strip() if res.stdout.strip() else ""
    assert (
        "fatal" not in version and "not a git repository" not in version
    ), f"setup.py embedded git's error into the version: {version!r}"
    # Must be acceptable to packaging (what setuptools/pip enforce).
    try:
        from packaging.version import Version

        Version(version)
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"setup.py version {version!r} is not PEP 440-valid: {e}")


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"=== {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
