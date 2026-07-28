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

"""Merge per-variant lm-eval LongBench JSONs (one per GPU) into one result.

The parallel orchestrator (``run_longbench_lmeval_parallel.sh``) runs each variant
in its own process, each writing a single-variant JSON. This merges them into a
unified multi-variant result: shared top-level metadata, a combined ``variants``
map, recomputed ``delta_vs_dense`` on the overall score, and a
``complete``/``problems`` accounting over the requested variant set.

A stale per-variant JSON (a hard-killed variant leaving a prior run's file) is
rejected: every part must carry the SAME ``env.git_sha`` as the first part.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True, help="dir with <variant>.json files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", required=True, help="comma-separated expected set")
    args = ap.parse_args()

    expected = [v for v in args.variants.split(",") if v.strip()]
    merged = None
    variants_map: dict = {}
    problems: list = []
    seed_sha = None  # git sha of the first part; all parts must agree

    for v in expected:
        path = os.path.join(args.workdir, f"{v}.json")
        if not os.path.isfile(path):
            problems.append({"variant": v, "reason": "missing per-variant JSON"})
            continue
        with open(path, encoding="utf-8") as f:
            part = json.load(f)
        # Reject a stale per-variant JSON: every part must share the run's git_sha.
        part_sha = part.get("env", {}).get("git_sha")
        if seed_sha is None and part_sha is not None:
            seed_sha = part_sha
        if part_sha is not None and seed_sha is not None and part_sha != seed_sha:
            problems.append(
                {
                    "variant": v,
                    "reason": f"git_sha {part_sha} != run sha {seed_sha} "
                    f"(stale per-variant JSON?)",
                }
            )
            continue
        # First successful part seeds the shared top-level metadata.
        if merged is None:
            merged = {
                k: part[k]
                for k in (
                    "task",
                    "protocol",
                    "harness",
                    "truncation",
                    "greedy",
                    "device",
                    "model",
                    "model_type",
                    "head_dim",
                    "max_len",
                    "dataset",
                    "env",
                )
                if k in part
            }
        variants_map.update(part.get("variants", {}))
        problems.extend(part.get("problems", []))

    if merged is None:
        print("[merge] ERROR: no per-variant JSON could be read", file=sys.stderr)
        return 1

    # Recompute delta-vs-dense on the overall score across the unified map.
    dense = variants_map.get("dense", {}).get("overall")
    if dense is not None:
        for v, blk in variants_map.items():
            if v != "dense" and blk.get("overall") is not None:
                blk["delta_vs_dense"] = round(blk["overall"] - dense, 2)

    merged["variants"] = variants_map
    merged["problems"] = problems
    merged["complete"] = len(problems) == 0 and all(v in variants_map for v in expected)

    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp, args.out)

    print(
        f"[merge] wrote {args.out} (complete={merged['complete']}, "
        f"variants={sorted(variants_map)}, problems={len(problems)})"
    )
    for v in expected:
        blk = variants_map.get(v)
        if blk and blk.get("overall") is not None:
            d = f" (Δ{blk['delta_vs_dense']:+.2f})" if "delta_vs_dense" in blk else ""
            print(
                f"  {v:14s} overall={blk['overall']:.2f}{d} "
                f"kernel={blk.get('kernel_path', '?')}"
            )
        elif blk:
            print(f"  {v:14s} overall=None kernel={blk.get('kernel_path', '?')}")
    return 0 if merged["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
