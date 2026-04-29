import re
import unicodedata

import editdistance as ed
import zhconv

from .cn_tn import TextNorm
from .whisper_normalizer.basic import BasicTextNormalizer
from .whisper_normalizer.english import EnglishTextNormalizer

english_normalizer = EnglishTextNormalizer()
chinese_normalizer = TextNorm(
    to_banjiao=False,
    to_upper=False,
    to_lower=False,
    remove_fillers=False,
    remove_erhua=False,
    check_chars=False,
    remove_space=False,
    cc_mode='',
)
basic_normalizer = BasicTextNormalizer()


class EvaluationTokenizer:
    """A generic evaluation-time tokenizer, which leverages built-in tokenizers
    in sacreBLEU (https://github.com/mjpost/sacrebleu). It additionally provides
    lowercasing, punctuation removal and character tokenization, which are
    applied after sacreBLEU tokenization.

    Args:
        tokenizer_type (str): the type of sacreBLEU tokenizer to apply.
        lowercase (bool): lowercase the text.
        punctuation_removal (bool): remove punctuation (based on unicode
        category) from text.
        character_tokenization (bool): tokenize the text to characters.
    """

    SPACE = chr(32)
    SPACE_ESCAPE = chr(9601)
    # ALL_TOKENIZER_TYPES=ChoiceEnum(["none", "13a", "intl", "zh", "ja-mecab"])

    def __init__(
        self,
        tokenizer_type: str = '13a',
        lowercase: bool = False,
        punctuation_removal: bool = False,
        character_tokenization: bool = False,
    ):
        from sacrebleu.tokenizers import TOKENIZERS

        assert tokenizer_type in TOKENIZERS, f'{tokenizer_type}, {TOKENIZERS}'
        self.lowercase = lowercase
        self.punctuation_removal = punctuation_removal
        self.character_tokenization = character_tokenization
        self.tokenizer = TOKENIZERS[tokenizer_type]

    @classmethod
    def remove_punctuation(cls, sent: str):
        """Remove punctuation based on Unicode category."""
        return cls.SPACE.join(
            t
            for t in sent.split(cls.SPACE)
            if not all(unicodedata.category(c)[0] == 'P' for c in t)
        )

    def tokenize(self, sent: str):
        tokenized = self.tokenizer()(sent)

        if self.punctuation_removal:
            tokenized = self.remove_punctuation(tokenized)

        if self.character_tokenization:
            tokenized = self.SPACE.join(
                list(tokenized.replace(self.SPACE, self.SPACE_ESCAPE))
            )

        if self.lowercase:
            tokenized = tokenized.lower()

        return tokenized


PUNCS = '!,.?;:'


def remove_sp(text, language):
    gt = re.sub(r'<\|.*?\|>', ' ', text)
    gt = re.sub(r'\s+', r' ', gt)  # 将文本中的连续空格替换为单个空格
    gt = re.sub(f' ?([{PUNCS}])', r'\1', gt)
    gt = gt.lstrip(' ')
    if language == 'zh':
        gt = re.sub(r'\s+', r'', gt)
    return gt


def compute_wer(refs, hyps, language, return_details=False):
    distance = 0
    ref_length = 0
    tokenizer = EvaluationTokenizer(
        tokenizer_type='zh', # zh separates out Chinese characters and tokenizes the non-Chinese parts using 13a tokenizer, https://github.com/mjpost/sacrebleu?tab=readme-ov-file#bleu
        lowercase=True,
        punctuation_removal=True,
        character_tokenization=False)
    details = []
    for i in range(len(refs)):
        ref = refs[i]
        pred = hyps[i]
        if language in ['yue']:
            ref = zhconv.convert(ref, 'zh-cn')
            pred = zhconv.convert(pred, 'zh-cn')
        if language in ['en']:
            ref = english_normalizer(ref)
            pred = english_normalizer(pred)
        if language in ['zh']:
            ref = chinese_normalizer(ref)
            pred = chinese_normalizer(pred)
        else:
            ref = basic_normalizer(ref)
            pred = basic_normalizer(pred)
        ref_items = tokenizer.tokenize(ref).split()
        pred_items = tokenizer.tokenize(pred).split()
        # if language in ['zh', 'yue']:
        #     ref_items = [x for x in ''.join(ref_items)]
        #     pred_items = [x for x in ''.join(pred_items)]
        if i < 5:
            print(f'ref: {ref}')
            print(f'pred: {pred}')
            print(f'ref_items:\n{ref_items}\n{len(ref_items)}\n{ref_items[0]}')
            print(
                f'pred_items:\n{pred_items}\n{len(ref_items)}\n{ref_items[0]}')
        distance += ed.eval(ref_items, pred_items)
        ref_length += len(ref_items)
        details.append(ed.eval(ref_items, pred_items))
    if return_details:
        return distance/ref_length, details
    else:
        return distance/ref_length
