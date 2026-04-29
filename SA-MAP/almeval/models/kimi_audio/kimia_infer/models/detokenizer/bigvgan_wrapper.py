import os
import json
import logging

import librosa
import torch

from .vocoder.bigvgan import BigVGAN
from .vocoder.utils import get_melspec, AttrDict, load_checkpoint

logger = logging.getLogger(__name__)


class BigVGANWrapper:
    def __init__(
        self, vocoder: BigVGAN, device: torch.device, h: AttrDict, dtype=None
    ) -> None:
        self.vocoder = vocoder.to(device)
        if dtype is not None:
            self.vocoder = self.vocoder.to(dtype)
        self.vocoder = self.vocoder.eval()
        self.device = device
        self.h = h

    def to_dtype(self, dtype):
        self.vocoder = self.vocoder.to(dtype)

    def extract_mel_from_wav(self, wav_path=None, wav_data=None):
        """
        params:
            wav_path: str, path of the wav, should be 24k
            wav_data: torch.tensor or numpy array, shape [T], wav data, should be 24k
        return:
            mel: [T, num_mels], torch.tensor
        """
        if wav_data is None:
            wav_data, _ = librosa.load(wav_path, sr=self.h["sampling_rate"])

        wav_data = torch.tensor(wav_data).unsqueeze(0)

        mel = get_melspec(
            y=wav_data,
            n_fft=self.h["n_fft"],
            num_mels=self.h["num_mels"],
            sampling_rate=self.h["sampling_rate"],
            hop_size=self.h["hop_size"],
            win_size=self.h["win_size"],
            fmin=self.h["fmin"],
            fmax=self.h["fmax"],
        )
        return mel.squeeze(0).transpose(0, 1)

    @torch.inference_mode()
    def extract_mel_from_wav_batch(self, wav_data):
        """
        params:
            wav_data: torch.tensor or numpy array, shape [Batch, T], wav data, should be 24k
        return:
            mel: [Batch, T, num_mels], torch.tensor
        """

        wav_data = torch.tensor(wav_data)

        mel = get_melspec(
            wav=wav_data,
            n_fft=self.h["n_fft"],
            num_mels=self.h["num_mels"],
            sampling_rate=self.h["sampling_rate"],
            hop_size=self.h["hop_size"],
            win_size=self.h["win_size"],
            fmin=self.h["fmin"],
            fmax=self.h["fmax"],
        )
        return mel.transpose(1, 2)

    def decode_mel(self, mel):
        """
        params:
            mel: [T, num_mels], torch.tensor
        return:
            wav: [1, T], torch.tensor
        """
        mel = mel.transpose(0, 1).unsqueeze(0).to(self.device)
        wav = self.vocoder(mel)
        return wav.squeeze(0)

    def decode_mel_batch(self, mel):
        """
        params:
            mel: [B, T, num_mels], torch.tensor
        return:
            wav: [B, 1, T], torch.tensor
        """
        mel = mel.transpose(1, 2).to(self.device)
        wav = self.vocoder(mel)
        return wav

    @classmethod
    def from_pretrained(cls, model_config, ckpt_path, device):
        with open(model_config) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        vocoder = BigVGAN(h, True)
        state_dict_g = load_checkpoint(ckpt_path, "cpu")
        vocoder.load_state_dict(state_dict_g["generator"])

        logger.info(">>> Load vocoder from {}".format(ckpt_path))
        return cls(vocoder, device, h)
