import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from subprocess import CalledProcessError, run, Popen, PIPE
import os
from functools import lru_cache
from typing import Optional, Union
from .modeling_whisper import WhisperModel

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 120
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk


def load_bytesio_audio(content, sr: int = SAMPLE_RATE):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        "pipe:",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "pipe:",
    ]
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=-1)
    out, _ = p.communicate(input=content)
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", file, "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = 128) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    with np.load(
        os.path.join(os.path.dirname(__file__), "mel_filters.npz")  # todo
        # os.path.join("assets", "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 128,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class WhisperEncoder(nn.Module):
    def __init__(
        self, model_path, mel_batch_size=40, unfreeze_online_whisper_model=False
    ):
        super().__init__()
        self.speech_encoder = WhisperModel.from_pretrained(model_path).encoder
        self.unfreeze_online_whisper_model = unfreeze_online_whisper_model
        if not self.unfreeze_online_whisper_model:
            self.speech_encoder.eval()
        self.mel_batch_size = mel_batch_size
        # self.speech_encoder.to(torch.cuda.current_device())
        # self.speech_encoder.half()

    @torch.no_grad()
    def tokenize_waveform(self, audio, kimia_whisper_clip_silence=False):
        if isinstance(audio, torch.Tensor):
            audio = audio[0]
            audio = audio.cpu().numpy()
        time_step = 0
        audios = []
        while time_step * 16000 < audio.shape[0]:
        # if time_step * 16000 < audio.shape[0]:
            audio_segment = audio[time_step * 16000 : (time_step + 30) * 16000]
            audios.append(audio_segment)
            time_step += 30

        final_audio_embedding = []
        final_attentions = []
        final_attn_key = []

        for audio_segment in audios:
            # import pdb; pdb.set_trace()
            assert audio_segment.shape[0] <= 480000
            L = audio_segment.shape[0]
            token_len = (L - 1) // (
                160 * 8
            ) + 1  # to match huggingface logic, with use attention mask to control the length and the slice with mask[:, ::160], also match the glm4 12.5 logic

            pad_audio = pad_or_trim(audio_segment.flatten())
            mel = log_mel_spectrogram(pad_audio)  # torch.Size([80, 3000])
            assert mel.shape[1] == 3000
            if kimia_whisper_clip_silence:
                input_seq_lens_list = [token_len * 4]
                input_seq_lens = torch.LongTensor(input_seq_lens_list).to(
                    torch.cuda.current_device()
                )
                audio_embedding = self.speech_encoder(
                    mel.unsqueeze(0).to(torch.cuda.current_device()).to(torch.bfloat16),
                    return_dict=True,
                    input_seq_lens=input_seq_lens,
                ).last_hidden_state
            else:
                outputs, attn_key = self.speech_encoder(
                    mel.unsqueeze(0).to(torch.cuda.current_device()).to(torch.bfloat16),
                    return_dict=True,
                    output_attentions=True,
                )
                attentions = outputs.attentions[-1]
                audio_embedding = outputs.last_hidden_state
                # audio_embedding: [1, 3000, 1280]
                audio_embedding = audio_embedding[:, : token_len * 4, :].cpu()
                attentions = attentions[:, :, : token_len * 4, :].cpu()
                attn_key = attn_key[:, : token_len * 4, :].cpu()
            final_audio_embedding.append(audio_embedding)
            final_attentions.append(attentions)
            final_attn_key.append(attn_key)

        final_audio_embedding = torch.cat(final_audio_embedding, dim=1)
        final_attentions = torch.cat(final_attentions, dim=2)
        final_attn_key = torch.cat(final_attn_key, dim=1)
        return final_audio_embedding, final_attentions, final_attn_key
