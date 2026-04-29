# Copied from https://github.com/MoonshotAI/Kimi-Audio-Evalkit/blob/master/almeval/datasets/base.py

import datetime
import os
from abc import abstractmethod

import jsonlines
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset

from ..utils.config_manager import ConfigManager


class AudioBaseDataset(Dataset):
    """
    Datasets need to be properly classified and defined for subsequent processing and evaluation
    - INTERACTIVE: Dataset interaction type, options: [Audio-QA, Audio-analysis]
        - Audio-QA: Audio input, text output, model answers questions about the audio
        - Audio-analysis: Audio input + text input, model analyzes audio according to text requirements, outputs text results
    - TASK: Dataset task type, options:
        - ASR: The answer field of this dataset is the text of the audio, evaluating error rate
        - Open-Ended: This dataset has no standard answer field, evaluating if generated content is reasonable
        - Closed-Ended: This dataset has standard answer field, evaluating if generated content matches, including:
            - MQA: Select one from multiple choices, check if model output matches correct answer, metric: accuracy
            - Ref-QA: Check if model answer matches reference answer, metric: accuracy

    - DATASET_SERIES: Some sub-datasets are part of a larger dataset, used to categorize datasets with the same format
    - AUDIO_TYPE: Audio type, Speech, Music, AudioEvent

    Each item in the dataset must contain at least the following fields:
        "index": int, # unique identifier for a piece of data
        "audio_path": str | list[str], # audio location
        "question": str, # question or instruction for the audio, e.g., "Please transcribe the audio content into text", set to empty if not needed
        "answer": str, # ground truth answer, set to empty if not needed (e.g., for Open-QA)
        "subset": str, # subset name, sometimes a dataset can be split into several subsets, which will be evaluated separately and reported independently.
    If you don't have subsets, use the dataset name
    If INTERACTIVE is Audio-QA, an additional "audio_content" field is required to store the text content of the audio for evaluation model reference
    If meta (dict) field is included, the information in meta will be included and returned with msg during dataset iteration

    """

    ALLOWED_INTERACTIVE = ['Audio-QA', 'Audio-analysis']
    ALLOWED_TASKS = ['ASR', 'Open-Ended', 'MQA', 'Ref-QA']
    ALLOWED_AUDIO_TYPES = ['Speech', 'Music', 'AudioEvent', 'Any']

    INTERACTIVE = None
    TASK = None
    AUDIO_TYPE = None
    DATASET_SERIES = None
    DATASET_NAME = None

    EXCLUDE = False  # set to True if you want to exclude this dataset from the evaluation

    def __init__(self, subset=None):
        assert self.INTERACTIVE in self.ALLOWED_INTERACTIVE
        assert self.TASK in self.ALLOWED_TASKS
        assert self.AUDIO_TYPE in self.ALLOWED_AUDIO_TYPES
        assert self.DATASET_SERIES is not None
        assert self.DATASET_NAME is not None
        self.subset = subset

        self.config_manager = ConfigManager()
        self.dataset_root = self.config_manager.get_dataset_root()
        self.dataset_file = self.config_manager.get_dataset_path(
            self.DATASET_NAME)

        self.ok = False
        if self.dataset_file is None:
            # auto find dataset path from dataset_root
            if self.dataset_root is None:
                logger.error(
                    f'file or dataset_path for dataset {self.DATASET_NAME} is not set, '
                    'please set either one in config.yaml')
                self.ok = False
                return None
            else:
                # try to find dataset file.
                possible_locations = [
                    os.path.join(self.dataset_root, self.DATASET_NAME,
                                 f'{self.DATASET_NAME}.jsonl'),
                    os.path.join(self.dataset_root, self.DATASET_SERIES,
                                 f'{self.DATASET_NAME}.jsonl'),
                    os.path.join(self.dataset_root, self.DATASET_SERIES,
                                 self.DATASET_NAME, f'{self.DATASET_NAME}.jsonl'),
                ]
                for location in possible_locations:
                    if os.path.exists(location):
                        self.dataset_file = location
                        break
                else:
                    logger.error(
                        f'Cannot find dataset {self.DATASET_NAME}.jsonl in {self.dataset_root}, it should be in one of the following locations: {possible_locations}')
                    self.ok = False
                    return None
        else:
            if not os.path.exists(self.dataset_file):
                logger.error(
                    f'Dataset {self.DATASET_NAME}.jsonl not found in {self.dataset_file}')
                self.ok = False
                return None

        self.data = self.load_data(self.dataset_file)
        self.demo = False
        assert isinstance(self.data, list)
        self.meta = {
            'task': self.TASK,
            'interactive': self.INTERACTIVE,
            'audio_type': self.AUDIO_TYPE,
            'dataset_series': self.DATASET_SERIES,
            'dataset_name': self.DATASET_NAME,
        }
        self.post_build()
        self.ok = True

    def set_demo_mode(self, demo=True):
        self.demo = demo
        self.data = self.data[:10]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.build_prompt(idx)

    # Given the dataset name, return the dataset as a pandas dataframe, can override
    def load_data(self, dataset):
        with jsonlines.open(dataset) as reader:
            data = [line for line in reader]
            return data

    # Post built hook, will be called after the dataset is built, can override
    def post_build(self):
        return None

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, idx: int | str) -> dict:
        """Construct a user conversation from a data source. Currently, we only consider single-turn conversations.

        The user's text may be single, but may require multiple audio inputs, e.g., "Which audio is louder?"
            {
                'audio': [audio_path],
                'text': question,
            }
        The question can also be empty.
        Sometimes, the model needs additional meta information to decide how to construct the prompt, so we return the model's meta information as well.
        - meta adds subset information to indicate which subset this msg belongs to
        - if there is a meta field in msg, it will be merged into msg['meta']
        """
        if isinstance(idx, str) and idx.isdigit():
            item = self.data[int(idx)]
        if isinstance(idx, int):
            item = self.data[idx]

        if 'audio_path' in item:
            audio_path = item['audio_path']

        if isinstance(audio_path, list):
            for i, p in enumerate(audio_path):
                assert os.path.exists(p), f'Audio file not found: {p}'
        else:
            assert os.path.exists(
                audio_path), f'Audio file not found: {audio_path}'

        question = item['question']

        msg = {'index': item['index'], 'audio': [],
               'text': question, 'meta': self.meta}

        if isinstance(audio_path, list):
            msg['audio'].extend(audio_path)
        elif isinstance(audio_path, str):
            msg['audio'].append(audio_path)

        # 还可以加一个meta信息，默认meta是subset
        if 'subset' in item:
            msg['meta']['subset'] = item['subset']
        if 'meta' in item:
            msg['meta'].update(item['meta'])
        return msg

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    @abstractmethod
    def evaluate(self, eval_file, dump_judge=False, method='default'):
        """
        evaluate performance based on result jsonl file.
        if dump_judge=True, will dump the judge result to this file.

        The jduge result will be a copy of eval_file, with additional fields: judge_result
        """
        pass

    def format_performance(self, model_name, performance, eval_method='null'):
        result = {
            'task': self.TASK,
            'dataset': self.DATASET_NAME,
            'model': model_name,
            'date': str(datetime.datetime.now()),
            'performance': performance,
            'eval_method': eval_method,
        }
        return result

    def get_model_name(self, eval_file):
        return os.path.basename(eval_file).split(self.DATASET_NAME)[0][:-1]

    def get_LLM_query(self, group: pd.DataFrame):
        """Get the input (question) for LLM evaluation
           - For Audio-QA, the evaluation question is the prompt, which is the audio's caption
           - For Audio-analysis, the evaluation question is the question field
        """
        if self.INTERACTIVE == 'Audio-QA':
            question = group['audio_content'].astype(str).to_list()
        elif self.INTERACTIVE == 'Audio-analysis':
            question = group['question'].astype(str).to_list()
        else:
            raise ValueError(f'Unsupported interactive: {self.INTERACTIVE}')
        return question

    @staticmethod
    def collect_acc(results: list, origin_df) -> dict:
        correct = 0
        invalid = 0
        judge_result = []
        for i, res in results:
            org_item = origin_df.iloc[i].to_dict()
            org_item['judge_result'] = res
            judge_result.append(org_item)
            llm_res = res.strip().split('\n')[-1].strip().lower()
            if 'yes' in llm_res:
                correct += 1
            elif 'no' in llm_res:
                pass
            else:
                invalid += 1
        n_samples = len(origin_df)
        task_result = {
            'acc': round((correct / (n_samples - invalid)) * 100, 2),
            'valid': n_samples - invalid,
            'total': n_samples,
            'correct': correct
        }

        return task_result, judge_result
