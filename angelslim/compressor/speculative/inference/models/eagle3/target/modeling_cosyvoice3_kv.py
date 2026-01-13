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

import functools
import os
import re
from functools import partial
from typing import Any, Callable, Generator, List, Optional

import numpy as np
import regex
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig

from .......utils.lazy_imports import (
    inflect,
    librosa,
    onnxruntime,
    torchaudio,
    wetext,
    whisper,
)
from .modeling_qwen2_kv import Qwen2ForCausalLM

IGNORE_ID = -1
# cosyvoice3 fixed params
use_ttsfrd = False
sample_rate = 24000
llm_input_size = 896
llm_output_size = 896
spk_embed_dim = 192
token_frame_rate = 25
token_mel_ratio = 2
# stream related params
chunk_size = 25  # streaming inference chunk size, in token
num_decoding_left_chunks = (
    -1
)  # streaming inference flow decoder left chunk size, <0 means use all left chunks


# Repetition Aware Sampling in VALL-E 2
def ras_sampling(
    weighted_scores,
    decoded_tokens,
    sampling,
    top_p=0.8,
    top_k=25,
    win_size=10,
    tau_r=0.1,
):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (
        (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids)
        .sum()
        .item()
    )
    if rep_num >= win_size * tau_r:
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(
        descending=True, stable=True
    )
    for i in range(len(sorted_idx)):
        # sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)].item()
    return top_ids


def random_sampling(weighted_scores, decoded_tokens, sampling):
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True).item()
    return top_ids


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


mel_basis = {}
hann_window = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa.filters.mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]+")


# whether contain chinese character
def contains_chinese(text):
    return bool(chinese_char_pattern.search(text))


# replace special symbol
def replace_corner_mark(text):
    text = text.replace("²", "平方")
    text = text.replace("³", "立方")
    return text


# remove meaningless symbol
def remove_bracket(text):
    text = text.replace("（", "").replace("）", "")
    text = text.replace("【", "").replace("】", "")
    text = text.replace("`", "").replace("`", "")
    text = text.replace("——", " ")
    return text


# spell Arabic numerals
def spell_out_number(text: str, inflect_parser):
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st:i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return "".join(new_text)


def split_paragraph(
    text: str,
    tokenize,
    lang="zh",
    token_max_n=80,
    token_min_n=60,
    merge_len=20,
    comma_split=False,
):
    def calc_utt_length(_text: str):
        if lang == "zh":
            return len(_text)
        else:
            return len(tokenize(_text))

    def should_merge(_text: str):
        if lang == "zh":
            return len(_text) < merge_len
        else:
            return len(tokenize(_text)) < merge_len

    if lang == "zh":
        pounc = ["。", "？", "！", "；", "：", "、", ".", "?", "!", ";"]
    else:
        pounc = [".", "?", "!", ";", ":"]
    if comma_split:
        pounc.extend(["，", ","])

    if text[-1] not in pounc:
        if lang == "zh":
            text += "。"
        else:
            text += "."

    st = 0
    utts = []
    for i, c in enumerate(text):
        if c in pounc:
            if len(text[st:i]) > 0:
                utts.append(text[st:i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', "”"]:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1

    final_utts = []
    cur_utt = ""
    for utt in utts:
        if (
            calc_utt_length(cur_utt + utt) > token_max_n
            and calc_utt_length(cur_utt) > token_min_n
        ):
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)

    return final_utts


# remove blank between chinese character
def replace_blank(text: str):
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if (text[i + 1].isascii() and text[i + 1] != " ") and (
                text[i - 1].isascii() and text[i - 1] != " "
            ):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def is_only_punctuation(text):
    # Regular expression: Match strings that consist only of punctuation marks or are empty.
    punctuation_pattern = r"^[\p{P}\p{S}]*$"
    return bool(regex.fullmatch(punctuation_pattern, text))


def load_wav(wav, target_sr, min_sr=16000):
    speech, sample_rate = torchaudio.load(wav, backend="soundfile")
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert (
            sample_rate >= min_sr
        ), "wav sample rate {} must be greater than {}".format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )(speech)
    return speech


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(
            pretrain_path, attn_implementation="eager"
        )

    def forward_one_step(
        self,
        xs,
        masks=None,
        past_key_values=None,
        position_ids=None,
        output_hidden_states=False,
        return_hidden_states=False,
    ):
        if masks is not None:
            input_masks = masks[:, -1, :]
        else:
            input_masks = None
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        xs = outs.hidden_states[-1]

        if return_hidden_states:
            return xs, outs["hidden_states"][:-1]
        return xs


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


class CosyVoiceFrontEnd:

    def __init__(
        self,
        get_tokenizer: Callable,
        feat_extractor: Callable,
        campplus_model: str,
        speech_tokenizer_model: str,
        spk2info: str = "",
        allowed_special: str = "all",
    ):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option, providers=["CPUExecutionProvider"]
        )
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_model,
            sess_options=option,
            providers=[
                (
                    "CUDAExecutionProvider"
                    if torch.cuda.is_available()
                    else "CPUExecutionProvider"
                )
            ],
        )
        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device)
        else:
            self.spk2info = {}
        self.allowed_special = allowed_special
        self.zh_tn_model = wetext.Normalizer(remove_erhua=False)
        self.en_tn_model = wetext.Normalizer()
        self.inflect_parser = inflect.engine()

    def text_normalize(self, text, split=True, text_frontend=True):
        if isinstance(text, Generator):
            print("get tts_text generator, will skip text_normalize!")
            return [text]
        # NOTE skip text_frontend when ssml symbol in text
        if "<|" in text and "|>" in text:
            text_frontend = False
        if text_frontend is False or text == "":
            return [text] if split is True else text
        text = text.strip()
        if contains_chinese(text):
            text = self.zh_tn_model.normalize(text)
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "。")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub(r"[，,、]+$", "。", text)
            texts = list(
                split_paragraph(
                    text,
                    partial(
                        self.tokenizer.encode, allowed_special=self.allowed_special
                    ),
                    "zh",
                    token_max_n=80,
                    token_min_n=60,
                    merge_len=20,
                    comma_split=False,
                )
            )
        else:
            text = self.en_tn_model.normalize(text)
            text = spell_out_number(text, self.inflect_parser)
            texts = list(
                split_paragraph(
                    text,
                    partial(
                        self.tokenizer.encode, allowed_special=self.allowed_special
                    ),
                    "en",
                    token_max_n=80,
                    token_min_n=60,
                    merge_len=20,
                    comma_split=False,
                )
            )
        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split is True else text

    def _extract_text_token(self, text):
        if isinstance(text, Generator):
            print("get tts_text generator, will return _extract_text_token_generator!")
            # NOTE add a dummy text_token_len for compatibility
            return self._extract_text_token_generator(text), torch.tensor(
                [0], dtype=torch.int32
            ).to(self.device)
        else:
            text_token = self.tokenizer.encode(
                text, allowed_special=self.allowed_special
            )
            text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
            text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(
                self.device
            )
            return text_token, text_token_len

    def _extract_text_token_generator(self, text_generator):
        for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i : i + 1]

    def _extract_speech_token(self, prompt_wav):
        speech = load_wav(prompt_wav, 16000)
        assert (
            speech.shape[1] / 16000 <= 30
        ), "do not support extract speech token for audio longer than 30s"
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = (
            self.speech_tokenizer_session.run(
                None,
                {
                    self.speech_tokenizer_session.get_inputs()[0]
                    .name: feat.detach()
                    .cpu()
                    .numpy(),
                    self.speech_tokenizer_session.get_inputs()[1].name: np.array(
                        [feat.shape[2]], dtype=np.int32
                    ),
                },
            )[0]
            .flatten()
            .tolist()
        )
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(
            self.device
        )
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, prompt_wav):
        speech = load_wav(prompt_wav, 16000)
        feat = torchaudio.compliance.kaldi.fbank(
            speech, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = (
            self.campplus_session.run(
                None,
                {
                    self.campplus_session.get_inputs()[0]
                    .name: feat.unsqueeze(dim=0)
                    .cpu()
                    .numpy()
                },
            )[0]
            .flatten()
            .tolist()
        )
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, prompt_wav):
        speech = load_wav(prompt_wav, sample_rate)
        speech_feat = (
            self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        )
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(
            self.device
        )
        return speech_feat, speech_feat_len

    def frontend_zero_shot(
        self, tts_text, prompt_text, prompt_wav, resample_rate, zero_shot_spk_id
    ):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        if zero_shot_spk_id == "":
            prompt_text_token, prompt_text_token_len = self._extract_text_token(
                prompt_text
            )
            speech_feat, speech_feat_len = self._extract_speech_feat(prompt_wav)
            speech_token, speech_token_len = self._extract_speech_token(prompt_wav)
            if resample_rate == 24000:
                # cosyvoice2, force speech_feat % speech_token = 2
                token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
                speech_feat, speech_feat_len[:] = (
                    speech_feat[:, : 2 * token_len],
                    2 * token_len,
                )
                speech_token, speech_token_len[:] = (
                    speech_token[:, :token_len],
                    token_len,
                )
            embedding = self._extract_spk_embedding(prompt_wav)
            model_input = {
                "prompt_text": prompt_text_token,
                "prompt_text_len": prompt_text_token_len,
                "llm_prompt_speech_token": speech_token,
                "llm_prompt_speech_token_len": speech_token_len,
                "flow_prompt_speech_token": speech_token,
                "flow_prompt_speech_token_len": speech_token_len,
                "prompt_speech_feat": speech_feat,
                "prompt_speech_feat_len": speech_feat_len,
                "llm_embedding": embedding,
                "flow_embedding": embedding,
            }
        else:
            model_input = self.spk2info[zero_shot_spk_id]
        model_input["text"] = tts_text_token
        model_input["text_len"] = tts_text_token_len
        return model_input


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
        # 2. build speech token language model related modules
        self.sos = speech_token_size + 0
        self.sos_id = torch.tensor([self.sos])
        self.eos_token = speech_token_size + 1
        self.task_id = speech_token_size + 2
        self.task_token = torch.tensor([self.task_id])
        self.fill_token = speech_token_size + 3

        self.llm = Qwen2Encoder(os.path.join(model_path, "CosyVoice-BlankEN"))
        self.llm_decoder = nn.Linear(
            llm_output_size, speech_token_size + 200, bias=False
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(
            speech_token_size + 200, llm_input_size
        )

        # 4. sampling method
        self.sampling = functools.partial(
            ras_sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1
        )

        self.stop_token_ids = [speech_token_size + i for i in range(200)]

    def sampling_ids(
        self,
        weighted_scores: torch.Tensor,
        decoded_tokens: List,
        sampling: int = 25,
        ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (top_ids < self.speech_token_size):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError(
                    "sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!".format(
                        max_trials
                    )
                )
        return top_ids

    @torch.inference_mode()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
    ) -> List[int]:
        if inputs_embeds is None:
            inputs_embeds = self.speech_embedding.weight[
                input_ids.squeeze(0).tolist()
            ].unsqueeze(0)
        # prefill
        y_pred, hidden_states = self.llm.forward_one_step(
            inputs_embeds,
            masks=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            output_hidden_states=True,
            return_hidden_states=True,
        )
        logp = self.llm_decoder(y_pred).log_softmax(dim=-1)

        outputs = {"hidden_states": hidden_states}

        return outputs, logp


class CosyVoice3Model:
    def __init__(
        self,
        llm: torch.nn.Module,
        flow: Optional[torch.nn.Module],
        hift: Optional[torch.nn.Module],
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = llm
        self.flow = flow
        self.hift = hift

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(
            torch.load(llm_model, map_location=self.device), strict=True
        )
        self.llm.to(self.device).eval()
        if self.flow is not None:
            self.flow.load_state_dict(
                torch.load(flow_model, map_location=self.device), strict=True
            )
            self.flow.to(self.device).eval()
        if self.hift is not None:
            # in case hift_model is a hifigan model
            hift_state_dict = {
                k.replace("generator.", ""): v
                for k, v in torch.load(hift_model, map_location=self.device).items()
            }
            self.hift.load_state_dict(hift_state_dict, strict=True)
            self.hift.to(self.device).eval()

    def token2wav(
        self,
        token,
        prompt_token,
        prompt_feat,
        embedding,
        token_offset,
        uuid,
        stream=False,
        finalize=False,
        speed=1.0,
    ):
        tts_mel, _ = self.flow.inference(
            token=token.to(self.device, dtype=torch.int32),
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
            prompt_token=prompt_token.to(self.device),
            prompt_token_len=torch.tensor(
                [prompt_token.shape[1]], dtype=torch.int32
            ).to(self.device),
            prompt_feat=prompt_feat.to(self.device),
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(
                self.device
            ),
            embedding=embedding.to(self.device),
            streaming=stream,
            finalize=finalize,
        )
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio :]
        if speed != 1.0:
            assert (
                token_offset == 0 and finalize is True
            ), "speed change only support non-stream inference mode"
            tts_mel = F.interpolate(
                tts_mel, size=int(tts_mel.shape[2] / speed), mode="linear"
            )
        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
        return tts_speech


class CosyVoice3:

    def __init__(self, model_dir, generate_audio=False):
        self.config = PretrainedConfig.from_pretrained(
            os.path.join(model_dir, "CosyVoice-BlankEN")
        )
        self.config.model_type = "cosyvoice3"
        self.config.txt_tokenizer_path = os.path.join(model_dir, "CosyVoice-BlankEN")
        self.dtype = self.config.torch_dtype

        self.model_dir = model_dir
        self.frontend = CosyVoiceFrontEnd(
            partial(
                get_qwen_tokenizer,
                token_path=self.config.txt_tokenizer_path,
                skip_special_tokens=True,
            ),
            partial(
                mel_spectrogram,
                n_fft=1920,
                num_mels=80,
                sampling_rate=sample_rate,
                hop_size=480,
                win_size=1920,
                fmin=0,
                fmax=None,
                center=False,
            ),
            os.path.join(model_dir, "campplus.onnx"),
            os.path.join(model_dir, "speech_tokenizer_v3.onnx"),
            os.path.join(model_dir, "spk2info.pt"),
            allowed_special="all",
        )
        self.sample_rate = sample_rate
        llm = CosyVoice3LM(
            model_dir,
            llm_input_size=llm_input_size,
            llm_output_size=llm_output_size,
            speech_token_size=6561,
        )

        llm_path, flow_path, hift_path = os.path.join(model_dir, "llm.pt"), "", ""
        flow, hift = None, None
        self.generate_audio = generate_audio
        if self.generate_audio:
            from cosyvoice.flow.DiT.dit import DiT
            from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
            from cosyvoice.flow.flow_matching import CausalConditionalCFM
            from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
            from cosyvoice.hifigan.generator import CausalHiFTGenerator
            from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer
            from omegaconf import DictConfig

            pre_lookahead_layer = PreLookaheadLayer(
                in_channels=80, channels=1024, pre_lookahead_len=3
            )
            config_dict = {
                "sigma_min": 1e-06,
                "solver": "euler",
                "t_scheduler": "cosine",
                "training_cfg_rate": 0.2,
                "inference_cfg_rate": 0.7,
                "reg_loss_type": "l1",
            }
            cfm_params = DictConfig(content=config_dict)
            estimator = DiT(
                dim=1024,
                depth=22,
                heads=16,
                dim_head=64,
                ff_mult=2,
                mel_dim=80,
                mu_dim=80,
                spk_dim=80,
                out_channels=80,
                static_chunk_size=chunk_size * token_mel_ratio,
                num_decoding_left_chunks=num_decoding_left_chunks,
            )
            decoder = CausalConditionalCFM(
                in_channels=240,
                n_spks=1,
                spk_emb_dim=80,
                cfm_params=cfm_params,
                estimator=estimator,
            )
            flow = CausalMaskedDiffWithDiT(
                input_size=80,
                output_size=80,
                spk_embed_dim=spk_embed_dim,
                output_type="mel",
                vocab_size=6561,
                input_frame_rate=token_frame_rate,
                only_mask_loss=True,
                token_mel_ratio=token_mel_ratio,
                pre_lookahead_len=3,
                pre_lookahead_layer=pre_lookahead_layer,
                decoder=decoder,
            )
            f0_predictor = CausalConvRNNF0Predictor(
                num_class=1, in_channels=80, cond_channels=512
            )
            hift = CausalHiFTGenerator(
                in_channels=80,
                base_channels=512,
                nb_harmonics=8,
                sampling_rate=sample_rate,
                nsf_alpha=0.1,
                nsf_sigma=0.003,
                nsf_voiced_threshold=10,
                upsample_rates=[8, 5, 3],
                upsample_kernel_sizes=[16, 11, 7],
                istft_params={"n_fft": 16, "hop_len": 4},
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                source_resblock_kernel_sizes=[7, 7, 11],
                source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                lrelu_slope=0.1,
                audio_limit=0.99,
                conv_pre_look_right=4,
                f0_predictor=f0_predictor,
            )

            flow_path, hift_path = os.path.join(model_dir, "flow.pt"), os.path.join(
                model_dir, "hift.pt"
            )

        self.model = CosyVoice3Model(llm, flow, hift)
        self.model.load(llm_path, flow_path, hift_path)
