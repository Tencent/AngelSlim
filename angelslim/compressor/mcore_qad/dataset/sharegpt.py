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

"""ShareGPT dataloader (chat template) for QAD/LM training.

Reads a ShareGPT jsonl (either {"messages":[{role,content}]} or
{"conversations":[{from,value}]}), renders each conversation with the model's chat
template, tokenizes, and packs into fixed [batch, seq] batches as (ids, labels) pairs
(labels = ids with padding set to -100). Deterministic order so all TP/PP ranks see
identical tokens (required for correctness).
"""

from __future__ import annotations

import json
from typing import List

import torch


def _to_messages(rec: dict) -> list:
    if "messages" in rec:
        return rec["messages"]
    role = {"human": "user", "gpt": "assistant", "system": "system"}
    return [
        {"role": role.get(t.get("from"), "user"), "content": t.get("value", "")}
        for t in rec.get("conversations", [])
    ]


def load_sharegpt_batches(
    path: str, tokenizer, seq_len: int, batch_size: int, n_batches: int, device
):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    seqs: List[List[int]] = []
    need = n_batches * batch_size
    with open(path) as f:
        for line in f:
            if len(seqs) >= need:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = _to_messages(rec)
            if not msgs:
                continue
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            ids = tokenizer(text, add_special_tokens=False).input_ids[:seq_len]
            if len(ids) < seq_len:
                ids = ids + [pad_id] * (seq_len - len(ids))
            seqs.append(ids)
    if not seqs:
        raise RuntimeError("no usable sequences parsed from data")
    if len(seqs) < need:  # small dataset -> tile to fill the pool
        seqs = (seqs * (need // len(seqs) + 1))[:need]
    ids = torch.tensor(seqs[:need], dtype=torch.long, device=device).view(
        n_batches, batch_size, seq_len
    )
    labels = ids.clone()
    labels[ids == pad_id] = -100  # ignore padding in the loss
    return list(zip(ids, labels))
