# Copied from https://github.com/MoonshotAI/Kimi-Audio-Evalkit/blob/master/almeval/datasets/ds_asr.py

import pandas as pd
from loguru import logger

from ..metrics.wer import compute_wer, remove_sp
from .base import AudioBaseDataset


class ASRDataset(AudioBaseDataset):
    INTERACTIVE = 'Audio-analysis'
    AUDIO_TYPE = 'Speech'
    TASK = 'ASR'
    LANG = None

    def __init__(self, subset=None):
        super().__init__(subset=subset)
        self.meta['lang'] = self.LANG

    def evaluate(self, eval_file, dump_judge=True, method=None):
        metrics = self.evaluate_asr_from_file(eval_file, dump_judge=dump_judge)
        logger.info(
            f'evaluating result of {self.DATASET_NAME}, directly based on the qwen2-audio implementation (https://github.com/QwenLM/Qwen2-Audio/blob/595360e82b5839c1507492ec83cae5bda6d5c7d4/eval_audio/evaluate_asr.py#L165-L269) and method parameter is ignored.'
            )
        model_name = self.get_model_name(eval_file)
        result = self.format_performance(
            model_name=model_name, performance=metrics, eval_method=method)
        return result

    def evaluate_asr_from_file(self, eval_file, dump_judge=False):
        '''Evaluate ASR from the eval_file (.jsonl).
        This is based on the qwen2-audio (https://github.com/QwenLM/Qwen2-Audio/blob/595360e82b5839c1507492ec83cae5bda6d5c7d4/eval_audio/evaluate_asr.py#L165-L269).

        Args:
            eval_file (str): Path to the eval_file (.jsonl).
            dump_judge (bool, optional): Whether to dump the judge result to the save_file (`save_file=eval_file.replace('.jsonl', '_wer_details.jsonl')`). Defaults to False.

        Returns:
            dict: Task result for different data subsets.
        '''
        assert self.LANG is not None, 'Please specify the language of the dataset'
        metrics = {}
        lang = self.LANG
        df = pd.read_json(eval_file, lines=True)
        for subset, group in df.groupby('subset'):
            if self.subset is not None and subset != self.subset:
                continue

            valid_mask = group['prediction'].astype(str) != 'null'
            gt = group.loc[valid_mask, 'answer'].astype(str).to_list()
            pred = group.loc[valid_mask, 'prediction'].astype(str).to_list()
            pred = [remove_sp(x, lang) for x in pred]
            gt = [remove_sp(x, lang) for x in gt]
            task_wer, details = compute_wer(
                gt, pred, lang, return_details=True)
            task_result = {
                'wer': round(task_wer*100, 2),
                'total': len(pred)
            }
            metrics[subset] = task_result

            if 'wer_details' not in df.columns:
                df['wer_details'] = None
            # Only update the rows for current subset
            group_indices = group[valid_mask].index
            df.loc[group_indices, 'wer_details'] = details
        if dump_judge:
            # dump the judge result to the eval_file
            save_file = eval_file.replace(
                '.jsonl', '_wer_details.jsonl')
            df.to_json(save_file, orient='records',
                       lines=True, force_ascii=False)
        return metrics


class LibriSpeech(ASRDataset):
    """
    LibriSpeech dataset
    """
    DATASET_NAME = 'LibriSpeech'
    LANG = 'en'
    DATASET_SERIES = 'LibriSpeech'


class Fleurs_zh(ASRDataset):
    """
    Fleurs dataset
    """
    DATASET_NAME = 'Fleurs-zh'
    LANG = 'zh'
    DATASET_SERIES = 'Fleurs'


class Fleurs_en(ASRDataset):
    """
    Fleurs dataset
    """
    DATASET_NAME = 'Fleurs-en'
    LANG = 'en'
    DATASET_SERIES = 'Fleurs'


class Aishell1(ASRDataset):
    """
    Aishell1 dataset
    """
    DATASET_NAME = 'AISHELL-1'
    LANG = 'zh'
    DATASET_SERIES = 'AISHELL-1'


class Aishell2(ASRDataset):
    """
    Aishell2 dataset
    """
    DATASET_NAME = 'AISHELL-2'
    LANG = 'zh'
    DATASET_SERIES = 'AISHELL-2'


class WenetSpeech(ASRDataset):
    """
    WenetSpeech dataset
    """
    DATASET_NAME = 'WenetSpeech'
    LANG = 'zh'
    DATASET_SERIES = 'WenetSpeech'
