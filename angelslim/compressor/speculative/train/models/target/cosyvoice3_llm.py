# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li, Qihua, Shengqiang Li)
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
# Modified from https://github.com/FunAudioLLM/CosyVoice for AngelSlim project

import os
from typing import Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from transformers import AutoTokenizer

from ....inference.models.eagle3.target.modeling_qwen2_kv import Qwen2ForCausalLM

IGNORE_ID = -1


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def get_qwen_tokenizer(
    token_path: str, skip_special_tokens: bool, version: str = "cosyvoice3"
):
    if version == "cosyvoice3":
        return CosyVoice3Tokenizer(
            token_path=token_path, skip_special_tokens=skip_special_tokens
        )
    else:
        raise ValueError


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    def forward(
        self, xs: torch.Tensor, xs_lens: torch.Tensor, output_hidden_states: bool
    ):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=output_hidden_states,
        )

        return outs, masks.unsqueeze(1)


class CosyVoice3Tokenizer:
    def __init__(self, token_path, skip_special_tokens=True):
        # NOTE: non-chat model, all these special tokens keep randomly initialized.
        # fmt: off
        # flake8: noqa
        special_tokens = {
            'eos_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'additional_special_tokens': [
                '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]',
                "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]", "<|endofsystem|>",
                "[AA]", "[AA0]", "[AA1]", "[AA2]", "[AE]", "[AE0]", "[AE1]", "[AE2]", "[AH]", "[AH0]", "[AH1]", "[AH2]",
                "[AO]", "[AO0]", "[AO1]", "[AO2]", "[AW]", "[AW0]", "[AW1]", "[AW2]", "[AY]", "[AY0]", "[AY1]", "[AY2]",
                "[B]", "[CH]", "[D]", "[DH]", "[EH]", "[EH0]", "[EH1]", "[EH2]", "[ER]", "[ER0]", "[ER1]", "[ER2]", "[EY]",
                "[EY0]", "[EY1]", "[EY2]", "[F]", "[G]", "[HH]", "[IH]", "[IH0]", "[IH1]", "[IH2]", "[IY]", "[IY0]", "[IY1]",
                "[IY2]", "[JH]", "[K]", "[L]", "[M]", "[N]", "[NG]", "[OW]", "[OW0]", "[OW1]", "[OW2]", "[OY]", "[OY0]",
                "[OY1]", "[OY2]", "[P]", "[R]", "[S]", "[SH]", "[T]", "[TH]", "[UH]", "[UH0]", "[UH1]", "[UH2]", "[UW]",
                "[UW0]", "[UW1]", "[UW2]", "[V]", "[W]", "[Y]", "[Z]", "[ZH]",
                "[a]", "[ai]", "[an]", "[ang]", "[ao]", "[b]", "[c]", "[ch]", "[d]", "[e]", "[ei]", "[en]", "[eng]", "[f]",
                "[g]", "[h]", "[i]", "[ian]", "[in]", "[ing]", "[iu]", "[ià]", "[iàn]", "[iàng]", "[iào]", "[iá]", "[ián]",
                "[iáng]", "[iáo]", "[iè]", "[ié]", "[iòng]", "[ióng]", "[iù]", "[iú]", "[iā]", "[iān]", "[iāng]", "[iāo]",
                "[iē]", "[iě]", "[iōng]", "[iū]", "[iǎ]", "[iǎn]", "[iǎng]", "[iǎo]", "[iǒng]", "[iǔ]", "[j]", "[k]", "[l]",
                "[m]", "[n]", "[o]", "[ong]", "[ou]", "[p]", "[q]", "[r]", "[s]", "[sh]", "[t]", "[u]", "[uang]", "[ue]",
                "[un]", "[uo]", "[uà]", "[uài]", "[uàn]", "[uàng]", "[uá]", "[uái]", "[uán]", "[uáng]", "[uè]", "[ué]", "[uì]",
                "[uí]", "[uò]", "[uó]", "[uā]", "[uāi]", "[uān]", "[uāng]", "[uē]", "[uě]", "[uī]", "[uō]", "[uǎ]", "[uǎi]",
                "[uǎn]", "[uǎng]", "[uǐ]", "[uǒ]", "[vè]", "[w]", "[x]", "[y]", "[z]", "[zh]", "[à]", "[ài]", "[àn]", "[àng]",
                "[ào]", "[á]", "[ái]", "[án]", "[áng]", "[áo]", "[è]", "[èi]", "[èn]", "[èng]", "[èr]", "[é]", "[éi]", "[én]",
                "[éng]", "[ér]", "[ì]", "[ìn]", "[ìng]", "[í]", "[ín]", "[íng]", "[ò]", "[òng]", "[òu]", "[ó]", "[óng]", "[óu]",
                "[ù]", "[ùn]", "[ú]", "[ún]", "[ā]", "[āi]", "[ān]", "[āng]", "[āo]", "[ē]", "[ēi]", "[ēn]", "[ēng]", "[ě]",
                "[ěi]", "[ěn]", "[ěng]", "[ěr]", "[ī]", "[īn]", "[īng]", "[ō]", "[ōng]", "[ōu]", "[ū]", "[ūn]", "[ǎ]", "[ǎi]",
                "[ǎn]", "[ǎng]", "[ǎo]", "[ǐ]", "[ǐn]", "[ǐng]", "[ǒ]", "[ǒng]", "[ǒu]", "[ǔ]", "[ǔn]", "[ǘ]", "[ǚ]", "[ǜ]"
            ]
        }
        # fmt: on
        self.special_tokens = special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        self.tokenizer.add_special_tokens(special_tokens)
        self.skip_special_tokens = skip_special_tokens

    def encode(self, text, **kwargs):
        tokens = self.tokenizer([text], return_tensors="pt")
        tokens = tokens["input_ids"][0].cpu().tolist()
        return tokens

    def decode(self, tokens):
        tokens = torch.tensor(tokens, dtype=torch.int64)
        text = self.tokenizer.batch_decode(
            [tokens], skip_special_tokens=self.skip_special_tokens
        )[0]
        return text


class CosyVoice3LM(torch.nn.Module):
    def __init__(
        self,
        model_path,
        llm_input_size: int,
        llm_output_size: int,
        speech_token_size: int,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        # build speech token language model related modules
        self.sos = speech_token_size + 0
        self.eos_token = speech_token_size + 1
        self.task_id = speech_token_size + 2
        self.fill_token = speech_token_size + 3

        self.llm = Qwen2Encoder(os.path.join(model_path, "CosyVoice-BlankEN"))
        self.llm_decoder = nn.Linear(
            llm_output_size, speech_token_size + 200, bias=False
        )

        # [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(
            speech_token_size + 200, llm_input_size
        )
        self.stop_token_ids = [speech_token_size + i for i in range(200)]

        # tokenizer
        self.tokenizer = get_qwen_tokenizer(
            os.path.join(model_path, "CosyVoice-BlankEN"), skip_special_tokens=True
        )

    def forward(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        speech_token: torch.Tensor,
        speech_token_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        output_hidden_states: bool = False,
        **kwargs,
    ):
        device = text.device
        text_token = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text_emb = self.llm.model.model.embed_tokens(text_token)

        # concat llm_input
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(
                1, 0, self.llm_input_size, dtype=text_emb.dtype
            ).to(device)
        speech_token_emb = self.speech_embedding(speech_token)

        # prepare llm_input/target
        lm_input, lm_input_len, loss_mask = self.prepare_lm_input_target(
            sos_emb,
            text_token,
            text_emb,
            text_len,
            task_id_emb,
            prompt_speech_token,
            prompt_speech_token_emb,
            prompt_speech_token_len,
            speech_token,
            speech_token_emb,
            speech_token_len,
        )

        # run lm forward
        outputs, lm_output_mask = self.llm(
            lm_input, lm_input_len.to(device), output_hidden_states
        )
        lm_output = outputs.hidden_states[-1]
        logits = self.llm_decoder(lm_output)
        hidden_states = torch.cat(outputs.hidden_states[:-1], dim=-1)
        return hidden_states, logits, lm_input, loss_mask, lm_output_mask

    def prepare_lm_input_target(
        self,
        sos_emb,
        text_token,
        text_emb,
        text_len,
        task_id_emb,
        prompt_speech_token,
        prompt_speech_token_emb,
        prompt_speech_token_len,
        speech_token,
        speech_token_emb,
        speech_token_len,
    ):
        lm_target, lm_input = [], []
        text_token = unpad_sequence(text_token, text_len.cpu(), batch_first=True)
        text_emb = unpad_sequence(text_emb, text_len.cpu(), batch_first=True)
        prompt_speech_token = unpad_sequence(
            prompt_speech_token, prompt_speech_token_len.cpu(), batch_first=True
        )
        prompt_speech_token_emb = unpad_sequence(
            prompt_speech_token_emb, prompt_speech_token_len.cpu(), batch_first=True
        )
        speech_token = unpad_sequence(
            speech_token, speech_token_len.cpu(), batch_first=True
        )
        speech_token_emb = unpad_sequence(
            speech_token_emb, speech_token_len.cpu(), batch_first=True
        )
        for i in range(len(text_token)):
            this_lm_target = torch.tensor(
                [IGNORE_ID] * (1 + text_len[i] + prompt_speech_token_len[i])
                + speech_token[i].tolist()
                + [self.eos_token]
            )
            this_lm_input = torch.concat(
                [
                    sos_emb.squeeze(dim=0),
                    text_emb[i],
                    task_id_emb.squeeze(dim=0),
                    prompt_speech_token_emb[i],
                    speech_token_emb[i],
                ],
                dim=0,
            )
            lm_input.append(this_lm_input)
            lm_target.append(this_lm_target)
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID)
        loss_mask = torch.ones_like(lm_target, device=lm_target.device)
        loss_mask = loss_mask.masked_fill(lm_target == IGNORE_ID, 0)
        return lm_input, lm_input_len, loss_mask
