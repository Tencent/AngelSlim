import inspect

from loguru import logger

from . import ds_asr, ds_mqa
from .base import AudioBaseDataset
from .ds_asr import ASRDataset
from .ds_mqa import AudioMQADataset

EXCLUDED_CLASSES = [
    'AudioBaseDataset',
    'ASRDataset',
    'AudioMQADataset',
]


def get_subclasses(base_class, module):
    """Get all subclasses of base_class in module"""
    return [
        cls for name, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, base_class) and cls.__name__ not in EXCLUDED_CLASSES and cls.EXCLUDE is False
    ]


# Automatically get all dataset classes
ASR_DATASETS = get_subclasses(ASRDataset, ds_asr)
MQA_DATASETS = get_subclasses(AudioMQADataset, ds_mqa)

# remove translation datasets
ALL_DATSETS = ASR_DATASETS + MQA_DATASETS
ALL_DATASETS = {ds.DATASET_NAME: ds for ds in ALL_DATSETS}


def build_dataset(name=None, subset=None) -> AudioBaseDataset:
    datasets = []
    if name == 'all' or name is None:
        datasets = [ALL_DATASETS[k]() for k in ALL_DATASETS]
    elif name == 'asr':
        datasets = [DS() for DS in ASR_DATASETS]
    elif name == 'mqa':
        datasets = [DS() for DS in MQA_DATASETS]

    if len(datasets) != 0:
        valid_datasets = [ds for ds in datasets if ds.ok]
        logger.info(
            f'Trying to build {len(valid_datasets)} datasets for {name}, {len(valid_datasets)} successfully built')
        return valid_datasets

    else:
        if name not in ALL_DATASETS:
            raise ValueError(
                f"Dataset {name} not found, all supported datasets: {ALL_DATASETS.keys()}, or 'all', 'asr', 'mqa'")
        dataset = ALL_DATASETS[name](subset=subset)
        if dataset.ok:
            logger.info(
                f'Trying to build dataset {name}, {dataset} successfully built')
            return dataset
        else:
            raise ValueError(f'Dataset {name} build failed')
