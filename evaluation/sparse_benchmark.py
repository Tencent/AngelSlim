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

"""H20 sparse-attention prefill benchmark.

Measures **prefill latency** of dense FlashAttention-2 vs each sparse variant on
a single H20, on REAL Qwen3-8B (the reference config: qwen3, head_dim 128,
so the real minference vertical_slash kernel + a_shape/tri_shape streaming kernel
all run). Reports p50/p90 latency, speedup-vs-dense, and the crossover length.

This is the FIRST performance measurement of the sparse subsystem — everything
prior verified correctness (sparse ~= dense). It supplies measured perf numbers
(previously upstream-paper/derived only), provenance, the lower-bound, and an
H20 dense baseline.

It is a standalone tool — it does NOT modify production code; it calls the same
``apply_sparsity_patch`` / ``unpatch_sparsity`` path the tests use.

IMPORTANT (cluster constraint): GPUs run a util-keepalive program that
must be KILLED before measuring (it pins util to 100%) and RESTARTED after (the
cluster releases idle cards after ~3h). The orchestration is in the sibling
``run_sparse_benchmark.sh``; this script only measures. Run it through that
wrapper, never bare, unless you have already killed the keepalive yourself.

Usage:
  python evaluation/sparse_benchmark.py \
      --model /.../weights/Qwen3-8B \
      --seq-lens 4096,16384,32768,65536,131072 \
      --variants a_shape,tri_shape,minference \
      --warmup 5 --measure 30 \
      --out /.../sparse_bench_qwen3_8b.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import warnings


def _log(msg: str) -> None:
    print(msg, flush=True)


def _device_sync():
    import torch

    torch.cuda.synchronize()


def _time_prefill(model, ids, warmup: int, measure: int) -> dict:
    """Return latency stats (ms) for a single prefill forward over ``ids``.

    Only the prefill (full-sequence forward, ``use_cache`` irrelevant for the
    compute we care about) is timed — that is where sparse attention acts.
    """
    import torch

    # Warmup (also triggers Triton autotune / kernel JIT on first call).
    for _ in range(warmup):
        with torch.no_grad():
            model(ids)
    _device_sync()

    samples = []
    for _ in range(measure):
        _device_sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(ids)
        _device_sync()
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    n = len(samples)
    p50 = statistics.median(samples)
    p90 = samples[min(n - 1, int(round(0.9 * (n - 1))))]
    return {
        "p50_ms": round(p50, 3),
        "p90_ms": round(p90, 3),
        "min_ms": round(samples[0], 3),
        "max_ms": round(samples[-1], 3),
        "n": n,
    }


def _variant_kwargs(variant: str, seq_len: int) -> dict:
    """Reasonable sparse budgets per variant.

    a_shape/tri_shape: a sink + local window that is genuinely sparse for long
    sequences (n_init=64 sink, n_local ~ 1/4 of the sequence capped). minference:
    its own per-head vertical/slash estimation (dense fallback budgets when no
    searched pattern is present — the in-repo default state).

    flexprefill / xattention / flashprefill: the shipped-YAML default operating
    points (the realistic sparse setting each algorithm is configured for, NOT a
    keep-all dense degenerate). On Qwen3-8B (head_dim 128) all three run their
    REAL kernel path (block_sparse_attn / pure-Triton), which is exactly the
    speed we want to characterize:
      * flexprefill: gamma=0.9 mass-coverage, tau=0.1 (pure-Triton kernel).
      * xattention: threshold=0.9 coverage, stride=8 antidiagonal estimate,
        block_size=128 (block_sparse_attn). chunk_size=2048 per upstream.
      * flashprefill: alpha=0.2 max-based dynamic threshold + sink 256 + window
        512 + last_n_block_full 2 (block_sparse_attn; alpha>0 so the real
        block-sparse path, not the alpha=0 flash-dense shortcut).
    """
    if variant == "a_shape":
        return {"n_init": 64, "n_local": max(256, seq_len // 4)}
    if variant == "tri_shape":
        return {"n_init": 64, "n_local": max(256, seq_len // 4), "n_last": 100}
    if variant == "minference":
        return {}  # vertical_and_slash with the in-repo dense-fallback pattern
    if variant == "flexprefill":
        return {"gamma": 0.9, "tau": 0.1, "block_size": 128}
    if variant == "xattention":
        return {"stride": 8, "norm": 1, "threshold": 0.9, "block_size": 128, "chunk_size": 2048}
    if variant == "flashprefill":
        return {
            "alpha": 0.2,
            "block_size": 128,
            "sink": 256,
            "window": 512,
            "last_n_block_full": 2,
        }
    raise ValueError(f"unknown variant {variant!r}")


def _patch(model, variant: str, attn_kwargs: dict):
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register algorithms
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    sys.path.insert(0, _harness_dir())
    from _harness import FakeSlim  # type: ignore

    slim = FakeSlim(model)
    algo = SparsityAlgorithmRegistry.create(variant, attn_kwargs=dict(attn_kwargs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    return slim, patched


def _harness_dir() -> str:
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "tests", "sparse")


def _env_provenance() -> dict:
    """Provenance stamped into the perf JSON so a number is traceable to the
    exact code + stack + hardware that produced it."""
    import subprocess

    def _pkg(name):
        try:
            import importlib.metadata as md

            return md.version(name)
        except Exception:  # noqa: BLE001
            return None

    sha = None
    dirty = None
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        # Honesty: a run started with uncommitted changes (e.g.
        # the driver itself not yet committed) would otherwise stamp a git_sha
        # that does NOT contain the code that produced the number. Record whether
        # the tree was dirty so the provenance can't silently lie.
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = bool(status)
        if dirty:
            sha = f"{sha}-dirty"
    except Exception:  # noqa: BLE001
        pass

    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "torch": _pkg("torch"),
        "transformers": _pkg("transformers"),
        "triton": _pkg("triton"),
        "flash_attn": _pkg("flash-attn") or _pkg("flash_attn"),
        "gpu": gpu,
    }


class UnpatchFailure(RuntimeError):
    """Raised when ``unpatch_sparsity`` fails. Semantically FATAL to a measurement
    sweep: an unpatch failure leaves the shared model object in an undefined,
    partially-patched state (sparse holds at most one patch per module with
    atomic rollback — a failed rollback means ownership is broken), so any
    SUBSEQUENT dense/sparse measurement on the same model object is untrustworthy.
    Callers must abort the sweep (or reload a fresh model) rather than continue.
    A recorded perf/accuracy number measured
    on a dirty model is worse than a missing one."""


def sanitize_model_id(model_path: str) -> str:
    """Map a local checkpoint path to its public HF identifier for the published
    JSON, so a recorded result never leaks an internal filesystem path.
    Falls back to the basename for unknown checkpoints (still no abs path).
    Run-time code keeps using the real ``--model`` path; only the recorded
    provenance string is sanitized.
    """
    import os

    base = os.path.basename(os.path.normpath(model_path))
    _KNOWN = {
        "Qwen3-8B": "Qwen/Qwen3-8B",
        "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
        "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
        "Qwen3.5-35B-A3B": "Qwen/Qwen3.5-35B-A3B",
    }
    return _KNOWN.get(base, base)


# ---------------------------------------------------------------------------
# Accuracy-benchmark helpers (shared with evaluation/longbench_lmeval.py). The perf
# benchmark above uses the 2-arg ``_variant_kwargs``; the accuracy drivers need
# the full variant set + an ``allow_pseudo_sparse`` flag + the kernel-path probe.
# Kept here so the accuracy driver depends only on this module (no cross-import
# between drivers).
# ---------------------------------------------------------------------------

# Default accuracy sweep: dense + every registered sparse algorithm. Kept as a
# single named constant so a smoke test can assert it equals `dense` + the live
# registry — i.e. no algorithm can silently fall out of the default run.
DEFAULT_VARIANTS = (
    "dense,a_shape,tri_shape,minference,flexprefill," "xattention,flashprefill,vecattention,stem"
)


def _accuracy_variant_kwargs(variant: str, seq_len: int, allow_pseudo_sparse: bool) -> dict:
    """Per-variant sparse config for the accuracy drivers. Reuses the perf
    budgets where they exist; adds the two the perf sweep doesn't cover (stem,
    vecattention) at their shipped operating points. The dense baseline takes no
    kwargs (it is never patched)."""
    if variant == "stem":
        # block_sparse_attn may be unbuilt; allow_pseudo_sparse routes to the
        # torch reference rather than hard-failing.
        kw = {"backend": "torch"}
    elif variant == "vecattention":
        # head_dim 128 -> real vllm_flash_attn kernel path (kernels_available 128).
        kw = {"threshold": 0.1, "block_size_q": 64, "block_size_k": 16}
    else:
        kw = _variant_kwargs(variant, seq_len)
    kw["allow_pseudo_sparse"] = allow_pseudo_sparse
    return kw


def _kernel_path(variant: str, head_dim: int) -> str:
    """Probe whether ``variant`` will run its REAL kernel or the torch reference
    on this head_dim, so the recorded number is never ambiguous.

    Every branch gates on the algorithm's OWN ``kernels_available(head_dim)`` —
    the same predicate the prefill dispatch uses to pick kernel-vs-reference — so
    the recorded ``kernel_path`` cannot disagree with what actually ran.
    """
    try:
        if variant == "dense":
            return "dense"
        if variant == "stem":
            import importlib

            importlib.import_module("torch")  # block_sparse_attn links libtorch
            importlib.import_module("block_sparse_attn")
            return "real"
        if variant in ("a_shape", "tri_shape", "minference"):
            from angelslim.compressor.sparsity.algorithms.minference.kernels import (
                kernels_available,
            )

            return "real" if kernels_available(variant, head_dim) else "reference"
        if variant == "flexprefill":
            from angelslim.compressor.sparsity.algorithms.flexprefill.kernels import (
                kernels_available,
            )

            return "real" if kernels_available(head_dim) else "reference"
        if variant == "flashprefill":
            from angelslim.compressor.sparsity.algorithms.flashprefill.kernels_check import (
                kernels_available,
            )

            return "real" if kernels_available(head_dim) else "reference"
        if variant == "vecattention":
            from angelslim.compressor.sparsity.algorithms.vecattention.kernels_check import (
                kernels_available,
            )

            return "real" if kernels_available(head_dim) else "reference"
        if variant == "xattention":
            from angelslim.compressor.sparsity.algorithms.xattention.kernels import (
                kernels_available,
            )

            return "real" if kernels_available(head_dim) else "reference"
    except Exception:  # noqa: BLE001
        return "reference"
    return "reference"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a real Qwen3 checkpoint")
    ap.add_argument("--seq-lens", default="4096,16384,32768,65536,131072")
    ap.add_argument("--variants", default="a_shape,tri_shape,minference")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--measure", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--allow-pseudo-sparse",
        action="store_true",
        help="permit the torch reference when a kernel is unavailable "
        "(default: hard-fail, so we measure the REAL kernel only)",
    )
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="exit 0 even if some (seq_len, variant) cells errored / "
        "OOM'd / are missing. Default is fail-on-error: any "
        "missing or errored cell, dense OOM, or unpatch failure "
        "makes the run exit non-zero (so CI / perf/README cannot "
        "be fed a half-broken matrix as 'measured-on-target').",
    )
    ap.add_argument(
        "--write-incomplete",
        action="store_true",
        help="persist the JSON even when the matrix is incomplete "
        "(default: a broken sweep does NOT overwrite the "
        "published --out JSON; the existing file is left intact).",
    )
    args = ap.parse_args()

    import torch
    from transformers import Qwen3ForCausalLM

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    seq_lens = [int(s) for s in args.seq_lens.split(",") if s.strip()]
    variants = [v for v in args.variants.split(",") if v.strip()]

    assert torch.cuda.is_available(), "CUDA required"
    dev_name = torch.cuda.get_device_name(0)
    _log(f"[bench] device={dev_name} | model={args.model}")
    _log(
        f"[bench] seq_lens={seq_lens} variants={variants} "
        f"warmup={args.warmup} measure={args.measure} bs={args.batch_size}"
    )

    # Single load, reused across all measurements (one device — the reference config).
    _log("[bench] loading model (fa2, bf16, cuda:0)…")
    model = (
        Qwen3ForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        .to("cuda:0")
        .eval()
    )
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )
    _log(
        f"[bench] loaded: {model.config.model_type} "
        f"layers={model.config.num_hidden_layers} head_dim={head_dim}"
    )

    results = {
        "device": dev_name,
        "model": sanitize_model_id(args.model),
        "model_type": model.config.model_type,
        "num_hidden_layers": model.config.num_hidden_layers,
        "head_dim": head_dim,
        "warmup": args.warmup,
        "measure": args.measure,
        "batch_size": args.batch_size,
        "env": _env_provenance(),  # git sha + torch/transformers/triton + GPU
        "rows": [],  # one per (seq_len, mode)
    }

    torch.manual_seed(0)
    problems = []  # (seq_len, mode, reason) — anything that makes the matrix incomplete
    model_dirty = False  # set if an unpatch fails (model no longer trustworthy)
    for seq_len in seq_lens:
        if model_dirty:
            break
        ids = torch.randint(0, 151000, (args.batch_size, seq_len), device="cuda:0")

        # 1) dense FA-2 baseline
        try:
            dense_stats = _time_prefill(model, ids, args.warmup, args.measure)
            dense_p50 = dense_stats["p50_ms"]
            results["rows"].append(
                {
                    "seq_len": seq_len,
                    "mode": "dense_fa2",
                    **dense_stats,
                    "speedup_vs_dense": 1.0,
                }
            )
            _log(
                f"[bench] L={seq_len:>7} dense_fa2     "
                f"p50={dense_p50:8.2f}ms p90={dense_stats['p90_ms']:8.2f}ms"
            )
        except torch.cuda.OutOfMemoryError:
            _log(f"[bench] L={seq_len:>7} dense_fa2     OOM — skipping this length")
            # dense OOM means every variant at this length is also unmeasured.
            problems.append((seq_len, "dense_fa2", "OOM"))
            for variant in variants:
                problems.append((seq_len, variant, "skipped (dense OOM)"))
            torch.cuda.empty_cache()
            continue

        # 2) each sparse variant
        for variant in variants:
            attn_kwargs = _variant_kwargs(variant, seq_len)
            attn_kwargs["allow_pseudo_sparse"] = args.allow_pseudo_sparse
            slim = patched = None
            try:
                slim, patched = _patch(model, variant, attn_kwargs)
                stats = _time_prefill(model, ids, args.warmup, args.measure)
                speedup = round(dense_p50 / stats["p50_ms"], 3) if stats["p50_ms"] else 0.0
                results["rows"].append(
                    {
                        "seq_len": seq_len,
                        "mode": variant,
                        **stats,
                        "speedup_vs_dense": speedup,
                    }
                )
                _log(
                    f"[bench] L={seq_len:>7} {variant:<13} "
                    f"p50={stats['p50_ms']:8.2f}ms p90={stats['p90_ms']:8.2f}ms "
                    f"speedup={speedup:5.2f}x"
                )
            except Exception as e:  # noqa: BLE001 — record + continue, but it IS a problem
                results["rows"].append(
                    {
                        "seq_len": seq_len,
                        "mode": variant,
                        "error": repr(e)[:200],
                    }
                )
                problems.append((seq_len, variant, f"error: {repr(e)[:120]}"))
                _log(f"[bench] L={seq_len:>7} {variant:<13} ERROR: {repr(e)[:160]}")
            finally:
                if slim is not None and patched is not None:
                    try:
                        unpatch_sparsity(slim, patched)
                    except Exception as ue:  # noqa: BLE001
                        # An unpatch failure leaves the shared model in an
                        # undefined partially-patched state — EVERY subsequent
                        # measurement (dense or sparse) on it is untrustworthy.
                        # Mark dirty + abort the sweep rather than record numbers
                        # measured on a corrupted model.
                        model_dirty = True
                        problems.append(
                            (
                                seq_len,
                                variant,
                                f"unpatch failed (FATAL, sweep aborted): {repr(ue)[:90]}",
                            )
                        )
                        _log(
                            f"[bench] L={seq_len:>7} {variant:<13} UNPATCH FAILED — "
                            f"model is now dirty; ABORTING sweep: {repr(ue)[:120]}"
                        )
                gc.collect()
                torch.cuda.empty_cache()
            if model_dirty:
                break

        del ids
        gc.collect()
        torch.cuda.empty_cache()

    # ---- markdown summary table ----
    _log("\n===== SPEEDUP vs FA-2 dense (p50) =====")
    by_len = {}
    for r in results["rows"]:
        by_len.setdefault(r["seq_len"], {})[r["mode"]] = r
    header = "| seq_len | " + " | ".join(["dense_fa2 (ms)"] + variants) + " |"
    _log(header)
    _log("|" + "---|" * (len(variants) + 2))
    for L in seq_lens:
        row = by_len.get(L, {})
        d = row.get("dense_fa2", {})
        cells = [f"{d.get('p50_ms', '—')}"]
        for v in variants:
            rv = row.get(v, {})
            if "error" in rv:
                cells.append("ERR")
            else:
                cells.append(f"{rv.get('speedup_vs_dense', '—')}x")
        _log(f"| {L} | " + " | ".join(str(c) for c in cells) + " |")

    # Decide completeness BEFORE writing, so a broken sweep cannot overwrite a
    # published perf JSON. The matrix is "complete" iff no problems.
    expected = len(seq_lens) * (1 + len(variants))  # dense + each variant per length
    measured = sum(1 for r in results["rows"] if "error" not in r)
    complete = not problems
    results["complete"] = complete
    results["expected_cells"] = expected
    results["measured_cells"] = measured
    # Surface a sweep aborted by a dirty model. Rows recorded BEFORE the
    # failing unpatch are valid (measured on a clean model); the abort prevents
    # any row measured AFTER corruption. A consumer must treat model_dirty=true as
    # "this matrix was truncated by model corruption", never as a clean partial.
    results["model_dirty"] = model_dirty
    results["problems"] = [
        {"seq_len": sl, "mode": mode, "reason": reason} for sl, mode, reason in problems
    ]

    if problems:
        _log(
            f"\n[bench] INCOMPLETE: {len(problems)} problem(s); "
            f"{measured}/{expected} cells measured:"
        )
        for sl, mode, reason in problems:
            _log(f"  - L={sl} {mode}: {reason}")

    # Atomic write — only when complete, OR when the user explicitly opts in to
    # writing a partial result (--write-incomplete). Default: a broken sweep
    # leaves the existing published JSON untouched. Write to a temp file in the
    # same dir then os.replace (atomic on POSIX).
    if args.out and (complete or args.write_incomplete):
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, args.out)
        tag = "complete" if complete else "INCOMPLETE (--write-incomplete)"
        _log(f"\n[bench] wrote {args.out} [{tag}]")
    elif args.out:
        _log(
            f"\n[bench] NOT writing {args.out}: matrix incomplete and "
            f"--write-incomplete not set (published JSON left untouched)."
        )

    if problems:
        if args.allow_incomplete:
            _log("[bench] --allow-incomplete set; exiting 0 despite problems.")
            return 0
        _log(
            "[bench] fail-on-error (default): exiting non-zero. "
            "Pass --allow-incomplete to override (and --write-incomplete to "
            "persist the partial JSON)."
        )
        return 1
    _log(f"\n[bench] complete: {measured}/{expected} cells measured, no errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
