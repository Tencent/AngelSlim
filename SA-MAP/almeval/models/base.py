# Copied from https://github.com/MoonshotAI/Kimi-Audio-Evalkit/blob/master/almeval/models/base.py

from abc import abstractmethod

import librosa
import torch
from loguru import logger


class BaseModel:

    NAME = None

    @abstractmethod
    def generate_inner(self, msg: dict) -> (str, str):
        raise NotImplementedError

    @staticmethod
    def check_audio_legal(audio_path: str | list[str], max_duration: float = 60) -> bool:
        """by default, we discard audio longer than 60s. subclasses can override this method (depends on model requirements)
        """
        if isinstance(audio_path, str):
            duration = librosa.get_duration(path=audio_path)
            if duration > max_duration or duration < 0.1:
                return False
        else:
            for path in audio_path:
                duration = librosa.get_duration(path=path)
                if duration > max_duration or duration < 0.1:
                    return False
        return True

    @torch.inference_mode()
    def __call__(self, msg: dict) -> str:
        if not self.check_audio_legal(msg['audio']):
            logger.warning(
                f'dataset: {msg["meta"]["dataset_name"]}, audio: {msg["audio"]}, duration exceeds 60s limit, skipping this sample')
            return msg['text'], None
        return self.generate_inner(msg)
