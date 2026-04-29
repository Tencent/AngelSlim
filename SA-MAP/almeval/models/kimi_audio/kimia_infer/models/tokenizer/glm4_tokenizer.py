import torch
import librosa
import os

from transformers import WhisperFeatureExtractor
from .glm4.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from .glm4.speech_tokenizer.utils import extract_speech_token
from torch import nn


class Glm4Tokenizer(nn.Module):
    def __init__(self, tokenizer_path):
        super().__init__()
        self.whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

    def tokenize(self, speech=None, audio_path=None, sr=16000):
        if audio_path:
            audio, sr = librosa.load(audio_path, sr=16000)
            audio = torch.tensor(audio).unsqueeze(0)
            audio_info = (audio, sr)
        else:
            assert speech is not None
            assert sr
            if isinstance(speech, list):
                speech = torch.tensor(speech).unsqueeze(0)
            if len(speech.shape) == 1:
                speech = speech.unsqueeze(0)
            audio_info = (speech, sr)

        audio_tokens = extract_speech_token(
            self.whisper_model, self.feature_extractor, [audio_info]
        )[0]
        audio_tokens = torch.tensor(audio_tokens).unsqueeze(0)
        return audio_tokens
