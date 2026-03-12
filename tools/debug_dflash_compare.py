#!/usr/bin/env python3
"""
debug_dflash_compare.py
=======================
Compare SpecForge vs AngelSlim DFlash pipeline outputs on the SAME sample.

Run (single GPU):
    cd /cfs_cloud_code/jiebinzhang/AngelSlim
    PYTHONPATH=/cfs_cloud_code/jiebinzhang/AngelSlim:/cfs_cloud_code/jiebinzhang/SpecForge \
    python tools/debug_dflash_compare.py \
        --jsonl /cfs_cloud_code/jiebinzhang/SpecForge/cache/dataset/regen_qwen3_4b.jsonl \
        --target_model /apdcephfs_gy5_303770945/share_303770945/jiebin/hf_models/Qwen/Qwen3-4B \
        --draft_config  /cfs_cloud_code/jiebinzhang/AngelSlim/configs/qwen3_dflash.json \
        --sample_idx 0       # which line in the JSONL to use
        --seed 42

Checks performed (in order):
  [1] input_ids         — are tokenized sequences identical?
  [2] loss_mask         — are loss masks identical?
  [3] hidden_states     — are target model hidden states identical?
  [4] anchor_positions  — are sampled anchor positions identical? (fixed seed)
  [5] noise_embedding   — are noise embeddings identical?
  [6] final loss/acc    — are the computed loss and accuracy values close?
"""

import argparse
import json
import sys
import types

import torch
import torch.nn.functional as F

# ─── silence transformers info logs ─────────────────────────────────────────
import transformers
transformers.logging.set_verbosity_error()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEP = "=" * 72


def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def check_close(name, a, b, rtol=1e-3, atol=1e-3):
    if a is None or b is None:
        print(f"  [{name}] SKIP (one side is None)")
        return
    a = a.float()
    b = b.float()
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH:  {a.shape}  vs  {b.shape}  ← ✗")
        return
    max_diff = (a - b).abs().max().item()
    mean_diff = (a - b).abs().mean().item()
    match = torch.allclose(a, b, rtol=rtol, atol=atol)
    status = "✓" if match else "✗ MISMATCH"
    print(f"  [{name}] max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  {status}")


def check_equal(name, a, b):
    if a is None or b is None:
        print(f"  [{name}] SKIP (one side is None)")
        return
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH: {a.shape} vs {b.shape}  ✗")
        return
    match = torch.equal(a, b)
    status = "✓" if match else "✗ MISMATCH"
    diffcount = (a != b).sum().item() if not match else 0
    print(f"  [{name}] equal={match}  diff_tokens={diffcount}  {status}")


# ─── Args ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--target_model", required=True)
    p.add_argument("--draft_config", required=True)
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=3072)
    return p.parse_args()


# ─── [Step 1] Load one raw sample ────────────────────────────────────────────
def load_raw_sample(jsonl_path, idx):
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"JSONL has fewer than {idx + 1} lines")


# ─── [Step 2a] SpecForge tokenisation ────────────────────────────────────────
def specforge_tokenize(sample, tokenizer, max_length):
    """Reproduce SpecForge's GeneralParser (chat_template='qwen') exactly.

    Uses importlib.util to load specforge/data/parse.py and template.py
    DIRECTLY (bypassing specforge/__init__.py which pulls in sglang / yunchang).
    """
    import importlib.util, types

    SF_ROOT = "/cfs_cloud_code/jiebinzhang/SpecForge/specforge"

    def _load(rel_path, module_name, deps=None):
        """Load a .py file as a module, optionally injecting pre-loaded deps."""
        spec = importlib.util.spec_from_file_location(
            module_name, f"{SF_ROOT}/{rel_path}"
        )
        mod = importlib.util.module_from_spec(spec)
        if deps:
            for k, v in deps.items():
                sys.modules[k] = v
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod

    # Load template.py first (no internal deps beyond pydantic)
    tmpl_mod  = _load("data/template.py", "specforge.data.template")
    # Load parse.py — it imports from .template, which is already in sys.modules
    parse_mod = _load("data/parse.py",    "specforge.data.parse",
                      deps={"specforge.data.template": tmpl_mod})

    template = tmpl_mod.TEMPLATE_REGISTRY.get("qwen")   # uses "qwen" (not "qwen3")
    parser   = parse_mod.GeneralParser(tokenizer, template)

    conversations = sample["conversations"]
    input_ids, loss_mask = parser.parse(conversations, max_length)
    return input_ids, loss_mask   # both 1-D tensors


# ─── [Step 2b] AngelSlim tokenisation ────────────────────────────────────────
def angelslim_tokenize(sample, tokenizer, max_length):
    """Reproduce AngelSlim's OnlineLLMDatasetBuilder._process_single_conversation."""
    from angelslim.compressor.speculative.train.data.chat_templates import (
        ChatTemplateType,
        template_manager,
    )
    from angelslim.compressor.speculative.train.data.dataset_builder.online_dataset_builder import (
        OnlineLLMDatasetBuilder,
    )

    # Instantiate the concrete LLM builder with qwen3 template
    builder = OnlineLLMDatasetBuilder(
        tokenizer=tokenizer,
        max_length=max_length,
        shuffle_seed=42,
        chat_template_type=ChatTemplateType.QWEN3,
        display=False,
    )

    result = builder._process_single_conversation(sample["conversations"])
    if result is None:
        raise RuntimeError("AngelSlim tokeniser returned None for this sample")

    input_ids = result["input_ids"].squeeze(0)   # → 1-D
    loss_mask  = result["loss_mask"].squeeze(0)   # → 1-D
    return input_ids, loss_mask


# ─── [Step 3] Target model hidden states ─────────────────────────────────────
def get_hidden_states(model, input_ids, target_layer_ids):
    """Run target model, extract 5-layer hidden states concat."""
    ids = input_ids.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)
    # hidden_states[0] = embed; [i+1] = layer i
    selected = [out.hidden_states[lid + 1] for lid in target_layer_ids]
    return torch.cat(selected, dim=-1)  # [1, S, D*5]


# ─── [Step 4] Anchor sampling (with fixed seed) ───────────────────────────────
def sample_anchors_deterministic(loss_mask_1d, num_anchors, block_size, seed):
    """Reproduce _sample_anchor_positions with a fixed torch seed."""
    torch.manual_seed(seed)
    loss_mask = loss_mask_1d.unsqueeze(0).float()  # [1, S]
    seq_len = loss_mask.shape[1]
    max_anchor = max(seq_len - block_size, 0)

    valid = loss_mask[:, : max_anchor + 1] > 0.5
    valid_counts = valid.sum(dim=1)
    max_valid = int(valid_counts.max().item())
    if max_valid <= 1:
        return None, None
    max_n = min(num_anchors, max_valid - 1)

    indices = torch.arange(max_anchor + 1).unsqueeze(0)
    masked_indices = torch.where(valid, indices, torch.tensor(seq_len + 1))
    random_vals = torch.rand(1, max_anchor + 1)
    random_vals = torch.where(valid, random_vals, torch.tensor(2.0))
    _, sorted_idx = random_vals.sort(dim=1)
    gathered = torch.gather(masked_indices, 1, sorted_idx)
    anchors = gathered[:, :max_n].sort(dim=1).values
    keep_mask = torch.arange(max_n).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(max=max_n)
    anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors))
    return anchors, keep_mask  # both [1, max_n]


# ─── [Step 5] Noise embedding ─────────────────────────────────────────────────
def build_noise_embed(embed_tokens, input_ids_1d, anchors, keep_mask,
                      block_size, mask_token_id):
    """Reproduce _create_noise_embed."""
    input_ids = input_ids_1d.unsqueeze(0)  # [1, S]
    bsz, seq_len = input_ids.shape
    n = anchors.shape[1]
    bs = block_size

    noise_ids = torch.full((bsz, n * bs), mask_token_id, dtype=torch.long)
    block_starts = (torch.arange(n) * bs).unsqueeze(0)
    valid_anchors = anchors.clamp(0, seq_len - 1)
    anchor_tokens = torch.gather(input_ids, 1, valid_anchors)
    flat_batch_idx = torch.zeros(bsz, n, dtype=torch.long)
    noise_ids[flat_batch_idx, block_starts] = torch.where(
        keep_mask, anchor_tokens,
        torch.tensor(mask_token_id, dtype=torch.long)
    )
    with torch.no_grad():
        return embed_tokens(noise_ids.to(DEVICE))  # [1, N*bs, D]


# ─── Detailed tokenisation diff ──────────────────────────────────────────────
def _print_token_diff(tokenizer, sf_ids, sf_lm, as_ids, as_lm):
    """Print full decoded text + per-token alignment table for both sides."""

    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    # ── 1. Full decoded text ──────────────────────────────────────────────────
    sf_text = tokenizer.decode(sf_ids.tolist(), skip_special_tokens=False)
    as_text = tokenizer.decode(as_ids.tolist(), skip_special_tokens=False)

    print(f"\n{BOLD}── SpecForge decoded text ({len(sf_ids)} tokens) ──{RESET}")
    print(repr(sf_text[:600]) + ("…" if len(sf_text) > 600 else ""))
    print(f"\n{BOLD}── AngelSlim decoded text ({len(as_ids)} tokens) ──{RESET}")
    print(repr(as_text[:600]) + ("…" if len(as_text) > 600 else ""))

    # ── 2. First divergence point ─────────────────────────────────────────────
    min_len = min(len(sf_ids), len(as_ids))
    first_diff = None
    for i in range(min_len):
        if sf_ids[i] != as_ids[i]:
            first_diff = i
            break

    if first_diff is None and len(sf_ids) == len(as_ids):
        print(f"\n  {GREEN}✓ Sequences are IDENTICAL{RESET}")
        # Still show loss_mask diffs
        lm_diff = (sf_lm != as_lm).nonzero(as_tuple=True)[0]
        if len(lm_diff):
            print(f"\n  loss_mask differs at {len(lm_diff)} positions:")
            for pos in lm_diff[:20]:
                tok = tokenizer.decode([sf_ids[pos].item()], skip_special_tokens=False)
                print(f"    pos={pos.item():5d}  SF={sf_lm[pos].item()}  AS={as_lm[pos].item()}"
                      f"  tok={repr(tok)}")
        return

    # ── 3. Per-token table around first divergence ────────────────────────────
    ctx = 8  # tokens to show before/after divergence
    if first_diff is not None:
        print(f"\n  {RED}First divergence at position {first_diff}{RESET}")
        start = max(0, first_diff - ctx)
        end_sf = min(len(sf_ids), first_diff + ctx * 3)
        end_as = min(len(as_ids), first_diff + ctx * 3)
        end = max(end_sf, end_as)

        print(f"\n  {'pos':>5}  {'SF_id':>8}  {'SF_tok':<20}  {'SF_mask':>7}"
              f"  {'AS_id':>8}  {'AS_tok':<20}  {'AS_mask':>7}")
        print("  " + "-" * 78)

        for i in range(start, end):
            sf_id  = sf_ids[i].item() if i < len(sf_ids) else None
            as_id  = as_ids[i].item() if i < len(as_ids) else None
            sf_tok = (tokenizer.decode([sf_id], skip_special_tokens=False)
                      if sf_id is not None else "---")
            as_tok = (tokenizer.decode([as_id], skip_special_tokens=False)
                      if as_id is not None else "---")
            sf_m   = sf_lm[i].item() if i < len(sf_lm) else "-"
            as_m   = as_lm[i].item() if i < len(as_lm) else "-"
            marker = RED + " ←" + RESET if sf_id != as_id else ""
            print(f"  {i:>5}  {str(sf_id):>8}  {repr(sf_tok):<20}  {str(sf_m):>7}"
                  f"  {str(as_id):>8}  {repr(as_tok):<20}  {str(as_m):>7}{marker}")

    # ── 4. Unique special tokens in each side ─────────────────────────────────
    sf_set = set(sf_ids.tolist())
    as_set = set(as_ids.tolist())
    only_sf = sf_set - as_set
    only_as = as_set - sf_set
    if only_sf:
        decoded = [(tid, tokenizer.decode([tid], skip_special_tokens=False))
                   for tid in sorted(only_sf)]
        print(f"\n  Token IDs only in SpecForge: {decoded}")
    if only_as:
        decoded = [(tid, tokenizer.decode([tid], skip_special_tokens=False))
                   for tid in sorted(only_as)]
        print(f"  Token IDs only in AngelSlim: {decoded}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    hdr("Loading tokenizer & models")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")

    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    ).eval()
    print(f"  Target model loaded on {DEVICE}")

    # Load draft config to get target_layer_ids / mask_token_id / block_size
    import json as _json
    with open(args.draft_config) as f:
        dcfg = _json.load(f)
    dflash_cfg    = dcfg.get("dflash_config", {})
    target_layer_ids = dflash_cfg.get("target_layer_ids", [1, 9, 17, 25, 33])
    mask_token_id    = dflash_cfg.get("mask_token_id", 151669)
    block_size       = dcfg.get("block_size", 16)
    num_anchors      = dcfg.get("num_anchors", 512)
    print(f"  target_layer_ids = {target_layer_ids}")
    print(f"  mask_token_id    = {mask_token_id}")
    print(f"  block_size       = {block_size},  num_anchors = {num_anchors}")

    # Raw sample
    hdr(f"[0] Raw sample (idx={args.sample_idx})")
    sample = load_raw_sample(args.jsonl, args.sample_idx)
    n_turns = len(sample["conversations"])
    print(f"  Conversations: {n_turns} turns")
    print(f"  First 80 chars of first turn: {str(sample['conversations'][0])[:80]}")

    # ── [1] Tokenisation ──────────────────────────────────────────────────────
    hdr("[1] Tokenisation")
    sf_ids, sf_lm  = specforge_tokenize(sample, tokenizer, args.max_length)
    as_ids, as_lm  = angelslim_tokenize(sample, tokenizer, args.max_length)

    print(f"\n  SpecForge  → input_ids shape={sf_ids.shape}  loss_mask_sum={sf_lm.sum().item()}")
    print(f"  AngelSlim  → input_ids shape={as_ids.shape}  loss_mask_sum={as_lm.sum().item()}")

    check_equal("[1a] input_ids", sf_ids, as_ids)
    check_equal("[1b] loss_mask", sf_lm,  as_lm)

    # ── Detailed diff: decode both sequences and print side-by-side
    _print_token_diff(tokenizer, sf_ids, sf_lm, as_ids, as_lm)

    # Use AngelSlim ids for downstream checks (or sf_ids — pick the common one)
    # We use sf_ids below so both sides start from SpecForge's tokenisation.
    # To test AngelSlim separately change to as_ids.
    common_ids = sf_ids   # change to as_ids to test AngelSlim tokenisation path
    common_lm  = sf_lm

    # ── [2] Target model hidden states ───────────────────────────────────────
    hdr("[2] Target model hidden states")
    # Both sides use the same target model, so hidden_states should be identical
    # if input_ids match. We run it once and reuse.
    hs = get_hidden_states(target_model, common_ids, target_layer_ids)
    print(f"  hidden_states shape = {hs.shape}  dtype = {hs.dtype}")
    print(f"  hidden_states stats: min={hs.min().item():.4f}  max={hs.max().item():.4f}  mean={hs.mean().item():.6f}")

    # If input_ids matched, hidden_states are identical by construction.
    # Recompute with as_ids if different:
    if not torch.equal(sf_ids, as_ids):
        hs_as = get_hidden_states(target_model, as_ids, target_layer_ids)
        check_close("[2] hs(SF_ids) vs hs(AS_ids)", hs[0], hs_as[0])
    else:
        print("  [2] hidden_states identical (input_ids are the same) ✓")

    # ── [3] Anchor sampling ──────────────────────────────────────────────────
    hdr("[3] Anchor sampling (seed fixed)")
    anchors_sf, keep_sf = sample_anchors_deterministic(
        common_lm, num_anchors, block_size, args.seed
    )
    # AngelSlim path (same algorithm, should be identical for same seed)
    anchors_as, keep_as = sample_anchors_deterministic(
        as_lm, num_anchors, block_size, args.seed
    )

    if anchors_sf is None:
        print("  No valid anchors for SpecForge loss_mask!")
    else:
        print(f"  anchors shape = {anchors_sf.shape}  keep_mask_sum = {keep_sf.sum().item()}")
        check_equal("[3a] anchors", anchors_sf, anchors_as)
        check_equal("[3b] keep_mask", keep_sf, keep_as)

    # ── [4] Noise embedding ──────────────────────────────────────────────────
    hdr("[4] Noise embedding")
    embed_tokens = target_model.model.embed_tokens

    if anchors_sf is not None:
        ne_sf = build_noise_embed(embed_tokens, common_ids, anchors_sf, keep_sf,
                                  block_size, mask_token_id)
        ne_as = build_noise_embed(embed_tokens, as_ids,     anchors_as, keep_as,
                                  block_size, mask_token_id)
        print(f"  noise_embedding shape = {ne_sf.shape}  dtype = {ne_sf.dtype}")
        check_close("[4] noise_embedding SF vs AS", ne_sf[0], ne_as[0])
    else:
        print("  Skipped (no valid anchors)")

    # ── [5] Draft model forward + loss ───────────────────────────────────────
    hdr("[5] Draft model forward+loss")
    sys.path.insert(0, "/cfs_cloud_code/jiebinzhang/AngelSlim")
    from angelslim.compressor.speculative import DraftModelConfig, create_draft_model
    from angelslim.compressor.speculative.train.models.draft.online_dflash_model import (
        OnlineDFlashModel,
    )
    from angelslim.compressor.speculative.train.trainer.online_dflash_trainer import (
        TargetEmbeddingsAndHead,
    )

    draft_cfg = DraftModelConfig.from_file(args.draft_config)
    draft_cfg._attn_implementation = "flex_attention"  # or "eager" for simpler debug
    draft_model = create_draft_model(draft_cfg).to(DEVICE).eval()
    print(f"  Draft model params: {sum(p.numel() for p in draft_model.parameters()):,}")

    comps = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model, device="cuda", trust_remote_code=True
    )
    wrapper = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=comps.lm_head,
        target_embed_tokens=comps.embed_tokens,
        block_size=block_size,
        mask_token_id=mask_token_id,
        attention_backend="flex_attention",
        num_anchors=num_anchors,
        loss_decay_gamma=7.0,
    ).to(DEVICE)

    # Fix seed so anchor sampling is deterministic for both sides
    ids_gpu = common_ids.unsqueeze(0).to(DEVICE)
    hs_gpu  = hs.to(DEVICE)
    lm_gpu  = common_lm.unsqueeze(0).float().to(DEVICE)

    # --- AngelSlim: call _compute_dflash_loss_and_accuracy (new refactored API) ---
    try:
        from angelslim.compressor.speculative.train.trainer.online_dflash_trainer import (
            OnlineDFlashTrainer,
        )

        # Build a minimal trainer-like object: we don't need the full HF Trainer,
        # just call the method directly on an instance that has dflash_model set.
        class _MinimalTrainer:
            """Minimal shell to call _compute_dflash_loss_and_accuracy."""
            def __init__(self, dflash_model):
                self.dflash_model = dflash_model

        _MinimalTrainer._compute_dflash_loss_and_accuracy = (
            OnlineDFlashTrainer._compute_dflash_loss_and_accuracy
        )

        torch.manual_seed(args.seed)
        trainer_shell = _MinimalTrainer(wrapper)
        loss_as, acc_as = trainer_shell._compute_dflash_loss_and_accuracy(
            input_ids=ids_gpu,
            hidden_states=hs_gpu,
            loss_mask=lm_gpu,
        )
        print(f"\n  AngelSlim  → loss={loss_as.item():.6f}  accuracy={acc_as.item():.6f}")
    except Exception as e:
        import traceback
        print(f"\n  AngelSlim  → ERROR: {e}")
        traceback.print_exc()
        loss_as = acc_as = None

    # SpecForge forward (reproduce directly from specforge/core/dflash.py)
    # Use importlib to avoid triggering specforge/__init__ → yunchang chain
    try:
        import importlib.util

        SF_ROOT = "/cfs_cloud_code/jiebinzhang/SpecForge/specforge"

        def _sf_load(rel_path, name, deps=None):
            spec = importlib.util.spec_from_file_location(name, f"{SF_ROOT}/{rel_path}")
            mod  = importlib.util.module_from_spec(spec)
            if deps:
                for k, v in deps.items():
                    sys.modules[k] = v
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod

        # specforge/modeling/draft/dflash.py (needed by core/dflash.py)
        sf_draft_mod = _sf_load(
            "modeling/draft/dflash.py",
            "specforge.modeling.draft.dflash",
        )
        # specforge/core/dflash.py
        sf_core_mod = _sf_load(
            "core/dflash.py",
            "specforge.core.dflash",
            deps={"specforge.modeling.draft.dflash": sf_draft_mod},
        )
        SFOnlineDFlashModel = sf_core_mod.OnlineDFlashModel

        sf_wrapper = SFOnlineDFlashModel(
            draft_model=draft_model,           # same weights
            target_lm_head=comps.lm_head,
            target_embed_tokens=comps.embed_tokens,
            block_size=block_size,
            mask_token_id=mask_token_id,
            attention_backend="flex_attention",
            num_anchors=num_anchors,
            loss_decay_gamma=7.0,
        ).to(DEVICE)

        torch.manual_seed(args.seed)
        loss_sf, acc_sf = sf_wrapper(input_ids=ids_gpu, hidden_states=hs_gpu, loss_mask=lm_gpu)
        print(f"  SpecForge  → loss={loss_sf.item():.6f}  accuracy={acc_sf.item():.6f}")
    except Exception as e:
        import traceback
        print(f"  SpecForge  → ERROR: {e}")
        traceback.print_exc()
        loss_sf = acc_sf = None

    if loss_as is not None and loss_sf is not None:
        loss_diff = abs(loss_as.item() - loss_sf.item())
        acc_diff  = abs(acc_as.item()  - acc_sf.item())
        print(f"\n  Δ loss = {loss_diff:.2e}   Δ accuracy = {acc_diff:.2e}")
        if loss_diff < 1e-3 and acc_diff < 1e-3:
            print("  [5] ✓  Loss and accuracy match!")
        else:
            print("  [5] ✗  Loss / accuracy mismatch – inspect intermediate values above.")

    hdr("Summary")
    checks = [
        ("[1a] input_ids",   torch.equal(sf_ids, as_ids) if sf_ids.shape == as_ids.shape else False),
        ("[1b] loss_mask",   torch.equal(sf_lm,  as_lm)  if sf_lm.shape  == as_lm.shape  else False),
        ("[3a] anchors",     (anchors_sf is not None and torch.equal(anchors_sf, anchors_as))),
        ("[5]  loss",        (loss_as is not None and loss_sf is not None and abs(loss_as.item() - loss_sf.item()) < 1e-3)),
    ]
    for name, ok in checks:
        print(f"  {'✓' if ok else '✗'}  {name}")


if __name__ == "__main__":
    main()
