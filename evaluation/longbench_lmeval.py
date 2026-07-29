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

"""LongBench accuracy benchmark via the **native lm-evaluation-harness**.

We drive EleutherAI's lm-eval `longbench` task group directly, so the numbers
line up with the lm-eval LongBench leaderboard rather than a re-implementation.
The sparse-vs-dense comparison: the dense model is loaded once, each variant is
patched in place (the SAME ``apply_sparsity_patch`` runtime path as the perf
benchmark ``evaluation/sparse_benchmark.py``), wrapped in lm-eval's
``HFLM(pretrained=<model object>)``, and scored by
``lm_eval.simple_evaluate(tasks=["longbench"])``.

GPL DEPENDENCY — NOT in requirements, install manually for this script only:
    pip install jieba fuzzywuzzy python-Levenshtein
lm-eval's LongBench ``metrics.py`` hard-imports ``jieba`` (zh tokenizer) and
``fuzzywuzzy`` (code edit-sim), both GPL. The project's license policy keeps GPL
out of the shipped deps, so this driver's GPL deps are an opt-in eval-only
install, never declared in requirements*.txt and never imported by the sparse
runtime.

Protocol notes (these are lm-eval's native choices):
  * **Task set** = the full ``longbench`` group = 21 tasks (English + the 5
    Chinese tasks: dureader/vcsum/multifieldqa_zh/passage_retrieval_zh/lsht).
    The zh tasks need jieba (installed above).
  * **Truncation** = lm-eval HFLM default = LEFT-truncation to ``--max-len``
    (keep the tail, drop the head); recorded in the result JSON as
    ``truncation: left``.
  * **Greedy** decoding (lm-eval coerces the task's ``do_sample=False`` +
    ``temperature=1`` to ``temperature=0``).
  * **trec** classification metric is commented out in lm-eval's ``utils.py``;
    it falls back to QA-F1 there. Recorded as-is (native lm-eval behavior).

Dataset: lm-eval pulls ``Xnhyacinth/LongBench`` (a HF mirror of THUDM/LongBench).
Pre-cache it once (the run is then offline-capable); on a network behind a proxy,
export ``https_proxy``/``http_proxy`` before the first fetch.

Variants reuse the sparse injection helpers from ``sparse_benchmark.py``:
``_patch`` (FakeSlim + registry + apply_sparsity_patch), ``unpatch_sparsity``,
``_accuracy_variant_kwargs``, ``_kernel_path``. ``dense`` = the unpatched FA2
baseline. Each variant records ``kernel_path`` so a torch-reference number is
never mistaken for a kernel number.

Canonical entry point: ``evaluation/run_longbench_lmeval.sh`` (1-GPU) /
``evaluation/run_longbench_lmeval_parallel.sh`` (registry-derived multi-GPU fan-out).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Reuse the proven perf/accuracy helpers verbatim (no duplication).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparse_benchmark import (  # noqa: E402
    DEFAULT_VARIANTS,
    UnpatchFailure,
    _accuracy_variant_kwargs,
    _env_provenance,
    _kernel_path,
    _log,
    _patch,
    sanitize_model_id,
)

# The full lm-eval ``longbench`` group expands to these task-type subgroups; we
# read the group result + each leaf task + each subgroup rollup out of the
# lm-eval results dict. Leaf task names are prefixed ``longbench_`` by lm-eval.
LONGBENCH_GROUP = "longbench"
SUBGROUPS = (
    "longbench_single",
    "longbench_multi",
    "longbench_summarization",
    "longbench_fewshot",
    "longbench_synthetic",
    "longbench_code",
)


def _balanced_device_map(model_path: str, n_gpu: int, reserve_gpu0: int) -> dict:
    """Explicit decoder-layer→GPU map that gives GPU0 ``reserve_gpu0`` FEWER layers.

    accelerate's ``device_map="auto"`` packs GPU0 (which also holds embed_tokens and
    sees the full-sequence activations of the first layers) to the brim, so the dense
    variant's full-attention activations OOM there while the sparse variants fit. This
    builds a map that co-locates embed_tokens + norm + lm_head sensibly and distributes
    the N decoder layers so GPU0 carries fewer of them — leaving activation headroom.

    Module naming (``model.embed_tokens`` / ``model.layers.<i>`` / ``model.norm`` /
    ``lm_head``) is the standard Llama-like layout shared by Qwen3 / Qwen3.5 / Hy3.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path)
    n_layers = cfg.num_hidden_layers
    # Weight each GPU: GPU0 gets a reduced share (its budget minus reserve_gpu0),
    # the rest split the remainder as evenly as possible.
    base = n_layers // n_gpu
    gpu0_layers = max(1, base - reserve_gpu0)
    remaining = n_layers - gpu0_layers
    others = n_gpu - 1
    per_other = remaining // others
    extra = remaining - per_other * others  # spread the leftover onto the last GPUs

    dm = {"model.embed_tokens": 0}
    idx = 0
    for g in range(n_gpu):
        count = gpu0_layers if g == 0 else per_other + (1 if g >= n_gpu - extra else 0)
        for _ in range(count):
            if idx < n_layers:
                dm[f"model.layers.{idx}"] = g
                idx += 1
    # Any rounding leftover → last GPU.
    while idx < n_layers:
        dm[f"model.layers.{idx}"] = n_gpu - 1
        idx += 1
    # Final norm + lm_head live with the last decoder layer's device.
    last_dev = dm[f"model.layers.{n_layers - 1}"]
    dm["model.norm"] = last_dev
    dm["lm_head"] = last_dev
    if hasattr(cfg, "tie_word_embeddings") and cfg.tie_word_embeddings:
        # tied lm_head shares embed_tokens' storage → keep it on GPU0 to avoid a copy.
        dm["lm_head"] = 0
    return dm


def _score_of(entry: dict):
    """lm-eval stores the headline metric as ``score,none`` (group rollups) or a
    task's primary metric. Return the float score, or None if absent."""
    if not isinstance(entry, dict):
        return None
    for k in ("score,none", "score", "acc,none", "acc"):
        if k in entry:
            return round(float(entry[k]) * (100.0 if entry[k] <= 1.0 else 1.0), 4)
    return None


def _collect(results: dict) -> dict:
    """Pull the group overall + per-subgroup + per-leaf-task scores out of the
    lm-eval ``results`` dict into a compact, JSON-stable block."""
    res = results.get("results", {})
    overall = _score_of(res.get(LONGBENCH_GROUP, {}))
    by_type = {}
    for sg in SUBGROUPS:
        s = _score_of(res.get(sg, {}))
        if s is not None:
            by_type[sg.replace("longbench_", "")] = s
    by_task = {}
    for name, entry in res.items():
        if name.startswith("longbench_") and name not in SUBGROUPS:
            s = _score_of(entry)
            if s is not None:
                by_task[name.replace("longbench_", "")] = s
    # When the run did not include the group rollup (e.g. a few named leaf tasks
    # for a smoke), fall back to the unweighted mean of the leaf-task scores so a
    # subset run still yields an ``overall``. The full ``longbench`` group run
    # gets ``overall`` from lm-eval's own group aggregation above.
    if overall is None and by_task:
        overall = round(sum(by_task.values()) / len(by_task), 4)
    return {"overall": overall, "by_type": by_type, "by_task": by_task}


def _run_variant(model, tokenizer, variant, args, head_dim):
    """Patch (unless dense), wrap in HFLM, run lm-eval, unpatch. Returns
    (result_block, kernel_path)."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    kpath = _kernel_path(variant, head_dim)
    slim = patched = None
    if variant != "dense":
        attn_kwargs = _accuracy_variant_kwargs(variant, args.max_len, args.allow_pseudo_sparse)
        slim, patched = _patch(model, variant, attn_kwargs)

    try:
        # When the model was sharded across GPUs (device_map=auto), it is already
        # placed; HFLM must NOT try to move it. Pin HFLM's input/entry device to the
        # embedding's device (accelerate's hooks route each layer's compute to its
        # own shard from there). For single-GPU loads this is just cuda:0.
        entry_device = str(model.device)
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_len,
            truncation=True,
            device=entry_device,
        )
        tasks = [t for t in args.tasks.split(",") if t.strip()] or [LONGBENCH_GROUP]
        limit = args.limit or None
        t0 = time.time()
        out = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=0,
            limit=limit,
            task_manager=TaskManager(),
            bootstrap_iters=0,
        )
        wall = round(time.time() - t0, 1)
    finally:
        if patched is not None:
            try:
                unpatch_sparsity(slim, patched)
            except Exception as ue:  # noqa: BLE001
                raise UnpatchFailure(f"unpatch failed after variant {variant!r}: {ue!r}") from ue

    block = _collect(out)
    block["kernel_path"] = kpath
    block["wall_s"] = wall
    return block, kpath


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        required=True,
        help="path to a real causal-LM checkpoint (Qwen3 / Qwen3.5 / "
        "Hunyuan-V3 — resolved via AutoModelForCausalLM)",
    )
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument(
        "--tasks",
        default="",
        help="comma list of lm-eval tasks (default: the full " "'longbench' group = 21 tasks)",
    )
    ap.add_argument(
        "--max-len",
        type=int,
        default=32768,
        help="HFLM max_length; inputs longer are LEFT-truncated",
    )
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="evaluate only the first N items per task (smoke); 0=all",
    )
    ap.add_argument(
        "--device-map",
        default="cuda:0",
        help="HF device placement. 'cuda:0' (default) loads on a single "
        "GPU; 'auto' shards decoder layers across all visible GPUs "
        "(required for checkpoints too large for one GPU, e.g. the "
        "~600 GB Hunyuan-V3).",
    )
    ap.add_argument(
        "--attn-impl",
        default="flash_attention_2",
        help="attn_implementation for the base model load "
        "(flash_attention_2 default; eager for archs without FA2).",
    )
    ap.add_argument(
        "--max-memory-per-gpu",
        default="",
        help="with --device-map=auto, cap per-GPU weight budget (e.g. '72GiB') so "
        "device_map spreads layers evenly and leaves headroom for dense full-"
        "attention activations + KV cache. Empty = let accelerate decide (can "
        "over-pack GPU0 and OOM the dense variant on a very large model).",
    )
    ap.add_argument(
        "--shard-reserve-gpu0",
        type=int,
        default=0,
        help="with --device-map=auto, place this many FEWER decoder layers on GPU0 "
        "(which also holds embeddings + first-layer full-sequence activations), via "
        "an explicit balanced device_map. Use when the dense variant OOMs on GPU0 "
        "while sparse variants fit (e.g. 5 for Hunyuan-V3 at 32K). Takes precedence "
        "over --max-memory-per-gpu.",
    )
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--allow-pseudo-sparse",
        action="store_true",
        help="permit the torch reference when a kernel is unavailable",
    )
    ap.add_argument(
        "--allow-incomplete", action="store_true", help="exit 0 even if some variant errored"
    )
    ap.add_argument(
        "--write-incomplete", action="store_true", help="persist the JSON even when incomplete"
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    variants = [v for v in args.variants.split(",") if v.strip()]
    assert torch.cuda.is_available(), "CUDA required"
    dev_name = torch.cuda.get_device_name(0)

    _log(f"[lb-lmeval] device={dev_name} | model={args.model}")
    _log(f"[lb-lmeval] variants={variants}")
    _log(
        f"[lb-lmeval] tasks={args.tasks or LONGBENCH_GROUP} "
        f"max_len={args.max_len} (LEFT-trunc) limit={args.limit or 'all'}"
    )

    # Load via AutoModelForCausalLM so any registered causal LM resolves (Qwen3 /
    # Qwen3.5 / Hunyuan-V3). --device-map=auto shards a checkpoint too large for one
    # GPU across all visible devices (sets model.hf_device_map); the default
    # 'cuda:0' keeps the original single-GPU placement for the 8B/9B models.
    sharded = args.device_map == "auto"
    device_map_arg = args.device_map
    if sharded and args.shard_reserve_gpu0:
        # accelerate's `auto` packs GPU0 with embed_tokens + its share of decoder
        # layers to the brim; the dense variant's full-attention activations (~16 GB
        # at 32K) then have nowhere to land on GPU0 and it OOMs while sparse fits.
        # (max_memory is silently ignored by this transformers' safetensors loader.)
        # So build an explicit layer→GPU map that gives GPU0 FEWER decoder layers —
        # embed_tokens + a reduced layer count — leaving headroom for activations.
        device_map_arg = _balanced_device_map(
            args.model, torch.cuda.device_count(), args.shard_reserve_gpu0
        )
        _log(
            f"[lb-lmeval] balanced device_map: GPU0 gets {args.shard_reserve_gpu0} "
            f"fewer layers to leave activation headroom for the dense variant"
        )
    _log(f"[lb-lmeval] loading model ({args.attn_impl}, bf16, device_map={args.device_map})…")
    from_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
        device_map=device_map_arg,
        low_cpu_mem_usage=True,
    )
    # Cap the per-GPU WEIGHT budget so device_map=auto spreads decoder layers evenly
    # and reserves headroom for the dense variant's full-attention activations + KV
    # cache (auto otherwise packs GPU0 to the brim → dense OOMs while sparse fits).
    if sharded and args.max_memory_per_gpu and not args.shard_reserve_gpu0:
        n_gpu = torch.cuda.device_count()
        from_kwargs["max_memory"] = {i: args.max_memory_per_gpu for i in range(n_gpu)}
        _log(f"[lb-lmeval] max_memory={args.max_memory_per_gpu}/GPU across {n_gpu} GPUs")
    model = AutoModelForCausalLM.from_pretrained(args.model, **from_kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )
    shard_note = ""
    if sharded:
        gpus = len({str(d) for d in model.hf_device_map.values() if str(d) != "cpu"})
        shard_note = f" (sharded: {gpus} GPUs)"
    _log(
        f"[lb-lmeval] loaded: {model.config.model_type} "
        f"layers={model.config.num_hidden_layers} head_dim={head_dim}{shard_note}"
    )

    results = {
        "task": "longbench_v1_lmeval",
        "protocol": "lm_eval_native",
        "harness": "lm-evaluation-harness",
        "truncation": "left",  # lm-eval HFLM default (NOT THUDM middle-trunc)
        "greedy": True,
        "device": dev_name,
        "model": sanitize_model_id(args.model),
        "model_type": model.config.model_type,
        "head_dim": head_dim,
        "max_len": args.max_len,
        "dataset": {"id": "Xnhyacinth/LongBench", "group": args.tasks or LONGBENCH_GROUP},
        "env": _env_provenance(),
        "variants": {},
        "problems": [],
    }

    import importlib.metadata as _md

    try:
        results["env"]["lm_eval"] = _md.version("lm_eval")
    except Exception:  # noqa: BLE001
        pass

    fatal = False
    for variant in variants:
        _log(f"[lb-lmeval] === variant: {variant} ===")
        try:
            block, kpath = _run_variant(model, tokenizer, variant, args, head_dim)
            results["variants"][variant] = block
            ov = block.get("overall")
            _log(
                f"[lb-lmeval] {variant}: overall={ov} kernel={kpath} " f"({block.get('wall_s')}s)"
            )
        except UnpatchFailure as uf:
            # Fatal: the shared model object is now dirty — abort the sweep.
            results["problems"].append({"variant": variant, "error": repr(uf), "fatal": True})
            fatal = True
            _log(f"[lb-lmeval] FATAL {variant}: {uf!r} — aborting sweep")
            break
        except Exception as e:  # noqa: BLE001
            results["problems"].append({"variant": variant, "error": repr(e)})
            _log(f"[lb-lmeval] ERROR {variant}: {e!r}")

    # delta_vs_dense on the overall score
    dense = results["variants"].get("dense", {}).get("overall")
    if dense is not None:
        for v, blk in results["variants"].items():
            if v != "dense" and blk.get("overall") is not None:
                blk["delta_vs_dense"] = round(blk["overall"] - dense, 2)

    complete = (
        (not fatal) and (not results["problems"]) and (set(results["variants"]) >= set(variants))
    )
    results["complete"] = complete

    # ----- report -----
    _log("\n[lb-lmeval] ===== summary (overall, Δ vs dense) =====")
    for v in variants:
        blk = results["variants"].get(v)
        if not blk:
            _log(f"  {v:<13} (missing)")
            continue
        d = f" (Δ{blk['delta_vs_dense']:+.2f})" if "delta_vs_dense" in blk else ""
        _log(f"  {v:<13} {blk.get('overall')}{d}  [{blk.get('kernel_path')}]")

    # ----- atomic write -----
    if args.out and (complete or args.write_incomplete):
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, args.out)  # atomic on POSIX
        _log(f"[lb-lmeval] wrote {args.out} (complete={complete})")
    elif args.out:
        _log(
            f"[lb-lmeval] NOT writing {args.out} — incomplete "
            f"(use --write-incomplete to force)"
        )

    if not complete and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
