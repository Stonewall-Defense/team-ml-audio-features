###############################################################################
# Global Imports
###############################################################################
from enum import Enum
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
import torch


###############################################################################
# Enumerated Types
###############################################################################
class SpecType(Enum):
    '''
    Possible spectrum types for feature extraction.

    The literature is conflicted on which, if any, provides the best results.
    '''
    STFT = "STFT"
    LOG_STFT = "LOG_STFT"
    LFCC = "LFCC"
    MEL = "MEL"
    LOG_MEL = "LOG_MEL"
    MFCC = "MFCC"


class WindowFunction(Enum):
    HANN = "Hann"
    HAMMING = "Hamming"
    BLACKMAN = "Blackman"


###############################################################################
# New Functions
###############################################################################
def ideal_hop_length(window_type: WindowFunction, n_fft: int):
    if window_type in [WindowFunction.HANN, WindowFunction.HAMMING]:
        return n_fft // 4
    elif window_type == WindowFunction.BLACKMAN:
        return n_fft // 6
    else:   # Should never happen; probably better than using a default
        raise ValueError(window_type)


def make_window(window_type: WindowFunction, n_fft: int):
    if window_type == WindowFunction.HANN:
        return torch.hann_window(n_fft)
    elif window_type == WindowFunction.HAMMING:
        return torch.hamming_window(n_fft)
    elif window_type == WindowFunction.BLACKMAN:
        return torch.blackman_window(n_fft)
    else:   # Should never happen; probably better than using a default
        raise ValueError(window_type)


###############################################################################
# New Classes
###############################################################################
class ExportableSTFT(torch.nn.Module):
    '''
        Exportable to Executorch. Created by Claude with supervision.

        Results are very slightly different than Torch but show no noticeable difference in model accuracy or training.
    '''
    def __init__(self,
                 n_fft: int,
                 *,
                 window_type: Optional[WindowFunction] = None,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 ):
        super().__init__()

        _window_type = window_type or WindowFunction.HANN
        _window = make_window(_window_type, n_fft)

        _window_len = win_length or n_fft
        if _window_len < n_fft:
            pad_left = (n_fft - _window_len) // 2
            pad_right = n_fft - _window_len - pad_left
            _window = torch.nn.functional.pad(_window, (pad_left, pad_right))

        # Only compute onesided bins: n_fft//2+1
        k = torch.arange(n_fft // 2 + 1).unsqueeze(1)   # (freq, 1)
        n = torch.arange(n_fft).unsqueeze(0)            # (1, n_fft)
        angles = -2 * torch.pi * k * n / n_fft          # (freq, n_fft)

        self.hop_length = hop_length or ideal_hop_length(_window_type, n_fft)
        self.n_fft = n_fft
        self.pad = n_fft // 2

        # Shape: (n_fft//2+1, n_fft)
        self.register_buffer("dft_real", torch.cos(angles) * _window)
        self.register_buffer("dft_imag", torch.sin(angles) * _window)

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        elif x.dim() == 2:
            x = x.unsqueeze(1)               # (B, 1, T)

        x = torch.nn.functional.pad(x, (self.pad, self.pad), mode="reflect")
        x = x.squeeze(1)                                          # (B, T_padded)

        frames = x.unfold(-1, self.n_fft, self.hop_length)        # (B, frames, n_fft)

        real = torch.matmul(frames, self.dft_real.T)              # (B, frames, freq)
        imag = torch.matmul(frames, self.dft_imag.T)

        power = real ** 2 + imag ** 2                             # (B, frames, freq)

        # Match torch.stft output layout: (B, freq, frames)
        return power.permute(0, 2, 1)


###############################################################################
# New Functions
###############################################################################
def determine_spec_type(calc_mels: bool, calc_logs: bool, calc_cepstrum: bool):
    if calc_mels:
        if calc_cepstrum:
            return SpecType.MFCC
        elif calc_logs:
            return SpecType.LOG_MEL
        else:
            return SpecType.MEL
    else:
        if calc_cepstrum:
            return SpecType.LFCC
        elif calc_logs:
            return SpecType.LOG_STFT
        else:
            return SpecType.STFT
