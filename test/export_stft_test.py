###############################################################################
# Test Imports
###############################################################################
import torch
import unittest

from AudioMlSpecTools.flexible_features import ExportableSTFT
from AudioMlSpecTools.wav import load_wav


###############################################################################
# Helpers
###############################################################################
def _stft(wav: torch.Tensor, n_fft: int, hop_len: int, window: torch.Tensor) -> torch.Tensor:
    # Pack batch
    shape = wav.size()
    wav = wav.reshape(-1, shape[-1])

    # Default values are consistent with librosa.core.spectrum._spectrogram
    spec_f = torch.stft(
        input=wav,
        n_fft=n_fft,
        hop_length=hop_len,
        win_length=n_fft,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    # Unpack batch
    spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-2:])
    return spec_f.abs().pow(2.0)


###############################################################################
# Config
###############################################################################
N_FFT = 1024
HOP_LEN = N_FFT // 4
HANN_WINDOW = torch.hann_window(N_FFT)

AUDIO = load_wav("test/res/380ACP-7-7WYYO4zK0hPS-9.wav")
REF_SPEC = _stft(AUDIO, N_FFT, HOP_LEN, HANN_WINDOW)


###############################################################################
# Tests
###############################################################################
class TestStft(unittest.TestCase):
    def test_results(self):
        my_stft = ExportableSTFT(N_FFT, HOP_LEN)
        my_spec = my_stft.forward(AUDIO)
        is_close = torch.allclose(REF_SPEC, my_spec, atol=0.002)
        self.assertTrue(is_close)
