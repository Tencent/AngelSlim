from librosa.filters import mel as librosa_mel_fn
import torch
import os

mel_basis_cache = {}
hann_window_cache = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def get_melspec(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int = None,
    center: bool = False,
) -> torch.Tensor:
    """
    Calculate the mel spectrogram of an input signal.
    This function uses slaney norm for the librosa mel filterbank (using librosa.filters.mel) and uses Hann window for STFT (using torch.stft).

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT size.
        num_mels (int): Number of mel bins.
        sampling_rate (int): Sampling rate of the input signal.
        hop_size (int): Hop size for STFT.
        win_size (int): Window size for STFT.
        fmin (int): Minimum frequency for mel filterbank.
        fmax (int): Maximum frequency for mel filterbank. If None, defaults to half the sampling rate (fmax = sr / 2.0) inside librosa_mel_fn
        center (bool): Whether to pad the input to center the frames. Default is False.

    Returns:
        torch.Tensor: Mel spectrogram.
    """
    if torch.min(y) < -1.0:
        print(f"[WARNING] Min value of input waveform signal is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"[WARNING] Max value of input waveform signal is {torch.max(y)}")

    device = y.device
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
        hann_window_cache[key] = torch.hann_window(win_size).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = spectral_normalize_torch(mel_spec)

    return mel_spec


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device, weights_only=True)
    print("Complete.")
    return checkpoint_dict


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)
