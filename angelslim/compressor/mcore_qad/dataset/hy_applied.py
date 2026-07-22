"""Assistant-only dataloader for Hy3 pre-rendered `applied_message` jsonl.

Each line is ``{"applied_message": "<full chat-templated text>"}`` carrying Hy3 special
markers. The markers in the file use a per-record placeholder suffix (e.g.
``<｜hy_Assistant:6124c78e｜>``) instead of the tokenizer's real ``:opensource`` suffix, so
they would otherwise shred into subwords; we normalize the placeholder to the tokenizer's
suffix (read off ``eos_token``) so they encode to the true special-token ids.

Loss is on every token emitted by the model inside an assistant turn: reasoning, visible
answer, tool calls, and the terminating ``<｜hy_eos｜>``. Prompt/system/user text and external
tool responses are ignored. Output is fixed [n_batches, batch, seq] (ids, labels) pairs,
deterministic so all model-parallel ranks see identical tokens (the trainer's DP stride
selects distinct batches per DP rank).
"""

from __future__ import annotations

import json
from pathlib import Path
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
        "tool_response": cid(f"<tool_response{suffix}>"),
        "tool_responses": cid(f"<tool_responses{suffix}>"),
        "eos": tokenizer.eos_token_id,
    }


def _normalize(message: str, suffix: str) -> str:
    """Replace the record's placeholder marker suffix with the tokenizer's real one."""
    if not message.startswith("<｜hy_begin_of_sentence:"):
        return message
    placeholder = message.split("<｜hy_begin_of_sentence:", 1)[1].split("｜>", 1)[0]
    return message.replace(f":{placeholder}", suffix)


def _assistant_labels(ids: List[int], mk: dict) -> List[int]:
    """Label model-emitted assistant tokens; mask prompts and external tool responses."""
    labels = [-100] * len(ids)
    collecting = False
    for i, t in enumerate(ids):
        if t == mk["assistant"]:  # role marker is prompt; generation starts after it
            collecting = True
            continue
        if t in (mk["tool_response"], mk["tool_responses"]):
            collecting = False
            continue
        if not collecting:
            continue
        labels[i] = t
        if t == mk["eos"]:  # learn to stop, then external tool feedback may follow
            collecting = False
    return labels


def build_hy_applied_cache(path: str, tokenizer, seq_len: int) -> dict:
    """Tokenize every usable JSONL row once and return CPU tensors."""
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    suffix = _suffix(tokenizer)
    mk = _markers(tokenizer, suffix)
    id_rows: List[List[int]] = []
    label_rows: List[List[int]] = []
    source_indices: List[int] = []

    with open(path) as f:
        for source_index, line in enumerate(f):
            try:
                msg = json.loads(line)["applied_message"]
            except (json.JSONDecodeError, KeyError):
                continue
            ids = tokenizer(_normalize(msg, suffix), add_special_tokens=False).input_ids[:seq_len]
            labels = _assistant_labels(ids, mk)
            if not any(x != -100 for x in labels):  # no generated assistant span -> skip
                continue
            pad = seq_len - len(ids)
            id_rows.append(ids + [pad_id] * pad)
            label_rows.append(labels + [-100] * pad)
            source_indices.append(source_index)
    if not id_rows:
        raise RuntimeError(f"no usable assistant spans parsed from {path}")
    return {
        "input_ids": torch.tensor(id_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "source_indices": torch.tensor(source_indices, dtype=torch.long),
        "seq_len": seq_len,
        "packed": False,
    }


def load_hy_applied_batches(
    path: str, tokenizer, seq_len: int, batch_size: int, n_batches: int, device
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    cache_path = Path(f"{path}.seq{seq_len}.pt")
    if cache_path.is_file():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache.get("seq_len") != seq_len:
            raise ValueError(f"token cache seq_len mismatch: {cache_path}")
        if cache.get("packed", True):
            raise ValueError(f"sample packing must be disabled: {cache_path}")
    else:
        cache = build_hy_applied_cache(path, tokenizer, seq_len)
    rows, label_rows = cache["input_ids"], cache["labels"]
    need = n_batches * batch_size
    indices = torch.arange(need) % rows.shape[0]
    ids = rows[indices].to(device).view(n_batches, batch_size, seq_len)
    labels = label_rows[indices].to(device).view(n_batches, batch_size, seq_len)
    return list(zip(ids, labels))
