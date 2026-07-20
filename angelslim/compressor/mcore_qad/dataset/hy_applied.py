"""Assistant-only dataloader for Hy3 pre-rendered `applied_message` jsonl.

Each line is ``{"applied_message": "<full chat-templated text>"}`` carrying Hy3 special
markers. The markers in the file use a per-record placeholder suffix (e.g.
``<｜hy_Assistant:6124c78e｜>``) instead of the tokenizer's real ``:opensource`` suffix, so
they would otherwise shred into subwords; we normalize the placeholder to the tokenizer's
suffix (read off ``eos_token``) so they encode to the true special-token ids.

Loss is on the ASSISTANT REPLY TEXT ONLY: labels are the token id for positions inside an
assistant answer (the visible text after ``</think>`` up to and including ``<｜hy_eos｜>``),
and -100 everywhere else -- user / system / think / tool-call spans are ignored. Output is
fixed [n_batches, batch, seq] (ids, labels) pairs, deterministic so all model-parallel ranks
see identical tokens (the trainer's DP stride selects distinct batches per DP rank).
"""

from __future__ import annotations

import json
from typing import List, Tuple

import torch


def _suffix(tokenizer) -> str:
    """The tokenizer's real special-token suffix, e.g. ':opensource' from its eos_token."""
    core = tokenizer.eos_token.rsplit("｜>", 1)[0]  # '<｜hy_eos:opensource'
    return core[core.rindex(":") :]  # ':opensource'


def _markers(tokenizer, suffix: str) -> dict:
    cid = tokenizer.convert_tokens_to_ids
    return {
        "assistant": cid(f"<｜hy_Assistant{suffix}｜>"),
        "think_end": cid(f"</think{suffix}>"),
        "toolcalls": cid(f"<tool_calls{suffix}>"),
        "eos": tokenizer.eos_token_id,
    }


def _normalize(message: str, suffix: str) -> str:
    """Replace the record's placeholder marker suffix with the tokenizer's real one."""
    if not message.startswith("<｜hy_begin_of_sentence:"):
        return message
    placeholder = message.split("<｜hy_begin_of_sentence:", 1)[1].split("｜>", 1)[0]
    return message.replace(f":{placeholder}", suffix)


def _assistant_labels(ids: List[int], mk: dict) -> List[int]:
    """labels[i] = ids[i] on assistant-reply positions (post-</think> text + eos), else -100."""
    labels = [-100] * len(ids)
    collecting = False
    for i, t in enumerate(ids):
        if t == mk["think_end"]:  # reply text starts after </think>
            collecting = True
        elif not collecting:
            continue
        elif t == mk["eos"]:  # learn to stop, then close the span
            labels[i] = t
            collecting = False
        elif t in (mk["toolcalls"], mk["assistant"]):  # tool-call / next turn -> not reply text
            collecting = False
        else:
            labels[i] = t
    return labels


def load_hy_applied_batches(
    path: str, tokenizer, seq_len: int, batch_size: int, n_batches: int, device
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    suffix = _suffix(tokenizer)
    mk = _markers(tokenizer, suffix)
    need = n_batches * batch_size
    id_rows: List[List[int]] = []
    label_rows: List[List[int]] = []
    with open(path) as f:
        for line in f:
            if len(id_rows) >= need:
                break
            try:
                msg = json.loads(line)["applied_message"]
            except (json.JSONDecodeError, KeyError):
                continue
            ids = tokenizer(_normalize(msg, suffix), add_special_tokens=False).input_ids[:seq_len]
            labels = _assistant_labels(ids, mk)
            if not any(x != -100 for x in labels):  # no assistant reply -> nothing to learn
                continue
            pad = [pad_id] * (seq_len - len(ids))
            id_rows.append(ids + pad)
            label_rows.append(labels + [-100] * len(pad))
    if not id_rows:
        raise RuntimeError(f"no usable assistant spans parsed from {path}")
    if len(id_rows) < need:  # small dataset -> tile to fill the pool
        reps = need // len(id_rows) + 1
        id_rows, label_rows = (id_rows * reps)[:need], (label_rows * reps)[:need]
    ids = torch.tensor(id_rows, dtype=torch.long, device=device).view(
        n_batches, batch_size, seq_len
    )
    labels = torch.tensor(label_rows, dtype=torch.long, device=device).view(
        n_batches, batch_size, seq_len
    )
    return list(zip(ids, labels))
