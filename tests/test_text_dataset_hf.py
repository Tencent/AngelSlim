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

"""Exercise HF text packing with CPU tensors and an offline tokenizer."""

import pytest
import torch
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from angelslim.data import text_dataset
from angelslim.data.base_dataset import BaseDataset


@pytest.fixture
def load_text_dataset(monkeypatch):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1, "b": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    processor = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")

    def load(texts, num_samples=-1):
        dataset = DatasetDict({"train": Dataset.from_dict({"text": texts})})
        monkeypatch.setattr(text_dataset, "load_dataset", lambda *args: dataset)
        return text_dataset.TextDataset("offline-text-dataset", processor, num_samples=num_samples)

    return load


@pytest.mark.parametrize("num_samples", [-1, 0, 256])
def test_short_rows_produce_one_partial_block(load_text_dataset, num_samples):
    dataset = load_text_dataset(["a a", "b b"], num_samples)

    assert len(dataset) == 1
    assert dataset[0]["input_ids"].tolist() == [[1, 1, 2, 2]]


@pytest.mark.parametrize("num_samples,expected", [(-1, 3), (0, 3), (1, 1), (2, 2), (256, 3)])
def test_sample_limit_counts_packed_blocks(load_text_dataset, num_samples, expected):
    dataset = load_text_dataset(["a " * 6144], num_samples)

    assert len(dataset) == expected
    assert all(item["input_ids"].shape == (1, 2048) for item in dataset)


def test_short_tail_after_full_blocks_is_still_dropped(load_text_dataset):
    dataset = load_text_dataset(["a " * 2048, "b b"])

    assert len(dataset) == 1
    assert dataset[0]["input_ids"].tolist() == [[1] * 2048]


@pytest.mark.parametrize("texts", [[], [""], ["", ""]])
def test_empty_token_stream_produces_no_samples(load_text_dataset, texts):
    assert len(load_text_dataset(texts)) == 0


def test_attention_mask_batches_like_input_ids(load_text_dataset):
    dataset = load_text_dataset(["a " * 2048, "b " * 2048], num_samples=2)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=BaseDataset.collate_fn)))

    assert batch["input_ids"].shape == (2, 2048)
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert torch.equal(batch["attention_mask"], torch.ones_like(batch["input_ids"]))
    assert torch.equal(batch["labels"], batch["input_ids"])
    assert batch["labels"].data_ptr() != batch["input_ids"].data_ptr()
    assert batch["input_ids"][0].tolist() == [1] * 2048
    assert batch["input_ids"][1].tolist() == [2] * 2048
