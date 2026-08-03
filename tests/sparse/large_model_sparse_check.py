"""Validate sparse on LARGER Qwen3 / Qwen3.5 models (MoE + bigger),
device_map-sharded across multiple GPUs.

This is a heavy verification run (loads 57-67 GB checkpoints), NOT part of the
routine test gate. It proves the claim "sparse supports larger
Qwen3/3.5, scaled via accelerate device_map" on real weights:

  * Qwen3-30B-A3B  — model_type qwen3_moe, Qwen3MoeForCausalLM, 48 layers,
    head_dim 128 -> the REAL minference vertical_slash kernel runs. Covers the
    MoE layer-resolution path AND a model materially bigger than 8B.
  * Qwen3.5-35B-A3B text tower — model_type qwen3_5_moe_text, 40 layers
    (10 full_attention + 30 linear_attention), gated attention, head_dim 256
    -> minference routes to the reference; a_shape/tri_shape use the streaming
    kernel. Covers Qwen3.5 gated + MoE + bigger.

For each: load sharded across >=2 GPUs, confirm the shard actually spans GPUs,
patch with a sparse algorithm, assert sharded-sparse ~= that same sharded
model's own dense (per-position top-1 agreement on a coherent prompt), and
assert unpatch restores the model byte-exactly.

Run:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    "$ANGELSLIM_HOME"/miniconda3/envs/angelslim/bin/python \
    tests/sparse/large_model_sparse_check.py
  (ANGELSLIM_HOME defaults to the repo's grandparent dir; weights are resolved
  migration-proof via _harness, so a cluster remount needs no edit here.)
"""

from __future__ import annotations

import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Migration-proof weights root: reuse the harness resolver ($ANGELSLIM_HOME ->
# derive-from-file -> probe known bases) so a cluster remount needs no edits.
from _harness import _WEIGHTS as WEIGHTS  # noqa: E402

QWEN3_30B = os.path.join(WEIGHTS, "Qwen3-30B-A3B")
QWEN35_35B = os.path.join(WEIGHTS, "Qwen3.5-35B-A3B")


def _ckpt_present(path):
    import glob

    return os.path.isdir(path) and bool(glob.glob(os.path.join(path, "*.safetensors")))


def _patch_real(model, variant, attn_kwargs):
    from _harness import FakeSlim

    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    slim = FakeSlim(model)
    algo = SparsityAlgorithmRegistry.create(variant, attn_kwargs=dict(attn_kwargs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    return slim, patched


def _shard_spans_gpus(model):
    used = {str(d) for d in model.hf_device_map.values()}
    return len({u for u in used if u != "cpu"}) >= 2


def _coherent_ids(tok, n=960):
    para = (
        "The history of science is the study of the development of science "
        "and scientific knowledge, including both natural and social "
        "sciences. "
    ) * 60
    return tok(para, return_tensors="pt")["input_ids"][:, :n]


def _run_one(tag, load_fn, variant, attn_kwargs):
    print(f"\n===== {tag} | variant={variant} =====", flush=True)
    if torch.cuda.device_count() < 2:
        print("  SKIP: needs >= 2 GPUs")
        return None
    from transformers import AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    model, path = load_fn()
    try:
        assert _shard_spans_gpus(
            model
        ), f"device_map did not shard across GPUs: {model.hf_device_map}"
        ndev = len({str(d) for d in model.hf_device_map.values() if str(d) != "cpu"})
        print(
            f"  loaded; sharded across {ndev} GPUs; "
            f"layers={model.config.num_hidden_layers if hasattr(model.config,'num_hidden_layers') else '?'}",  # noqa: E231, E501
            flush=True,
        )
        tok = AutoTokenizer.from_pretrained(path)
        in_dev = next(model.parameters()).device
        ids = _coherent_ids(tok).to(in_dev)
        with torch.no_grad():
            dense = model(ids).logits.float().cpu()

        slim, patched = _patch_real(model, variant, attn_kwargs)
        try:
            print(f"  patched {len(patched)} full-attention layer(s)", flush=True)
            assert len(patched) >= 1, "no layers patched"
            with torch.no_grad():
                sp = model(ids).logits.float().cpu()
        finally:
            unpatch_sparsity(slim, patched)

        assert torch.isfinite(sp).all(), "sparse produced NaN/inf"
        agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
        with torch.no_grad():
            restored = model(ids).logits.float().cpu()
        exact = torch.equal(restored, dense)
        print(
            f"  top-1 agreement vs dense = {agree:.4f} | unpatch byte-exact = {exact}", flush=True
        )
        assert agree > 0.95, f"top-1 agreement {agree:.3f} too low"
        assert exact, "unpatch did not restore byte-exactly"
        print(f"  {tag} PASS", flush=True)
        return True
    finally:
        del model
        torch.cuda.empty_cache()


def load_qwen3_30b():
    from transformers import Qwen3MoeForCausalLM

    m = Qwen3MoeForCausalLM.from_pretrained(
        QWEN3_30B,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()
    return m, QWEN3_30B


def load_qwen35_35b_text():
    # AutoModelForCausalLM gives the text tower (model_type qwen3_5_moe_text).
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        QWEN35_35B,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()
    return m, QWEN35_35B


def main():
    results = {}
    if _ckpt_present(QWEN3_30B):
        # Qwen3-30B-A3B: head_dim 128 -> real minference kernel; also a_shape.
        results["Qwen3-30B-A3B/minference"] = _run_one(
            "Qwen3-30B-A3B (qwen3_moe, hd128, REAL kernel)",
            load_qwen3_30b,
            "minference",
            {},
        )
        results["Qwen3-30B-A3B/a_shape"] = _run_one(
            "Qwen3-30B-A3B (qwen3_moe, a_shape)",
            load_qwen3_30b,
            "a_shape",
            {"n_init": 64, "n_local": 512},
        )
    else:
        print(f"SKIP Qwen3-30B-A3B: checkpoint not present at {QWEN3_30B}")

    if _ckpt_present(QWEN35_35B):
        # Qwen3.5-35B-A3B text tower: gated, hd256 -> a_shape streaming kernel.
        results["Qwen3.5-35B-A3B/a_shape"] = _run_one(
            "Qwen3.5-35B-A3B (qwen3_5_moe_text, gated, hd256, a_shape)",
            load_qwen35_35b_text,
            "a_shape",
            {"n_init": 64, "n_local": 512},
        )
        # minference on hd256 -> reference fallback (needs allow_pseudo_sparse).
        results["Qwen3.5-35B-A3B/minference-ref"] = _run_one(
            "Qwen3.5-35B-A3B (qwen3_5_moe_text, minference reference)",
            load_qwen35_35b_text,
            "minference",
            {"allow_pseudo_sparse": True},
        )
    else:
        print(f"SKIP Qwen3.5-35B-A3B: checkpoint not present at {QWEN35_35B}")

    print("\n===== SUMMARY =====")
    rc = 0
    n_pass = n_skip = n_fail = 0
    for k, v in results.items():
        status = "PASS" if v else ("SKIP" if v is None else "FAIL")
        print(f"  {k}: {status}")
        if v is True:
            n_pass += 1
        elif v is False:
            n_fail += 1
            rc = 1
        else:
            n_skip += 1
    print(
        f"\n  ran={n_pass} pass, {n_fail} fail, {n_skip} skip "
        f"(of {len(results)} requested case(s))"
    )
    # Coverage floor: if the user RAN this script but nothing actually
    # executed (0 pass + 0 fail — all skipped for missing GPUs/checkpoints), that
    # is NOT a success. Exit non-zero so a bypassed multi-GPU/MoE check can't
    # masquerade as green. (Mirrors run_all's coverage floor for the suites.)
    if n_pass == 0 and n_fail == 0:
        print(
            "  -> 0 cases actually ran (need >=2 GPUs + the large checkpoints); "
            "treating as FAIL, not a vacuous pass."
        )
        rc = 2
    sys.exit(rc)


if __name__ == "__main__":
    main()
