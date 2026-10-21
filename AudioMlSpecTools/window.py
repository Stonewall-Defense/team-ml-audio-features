###############################################################################
# Global Imports
###############################################################################
from enum import Enum

###############################################################################
# 3PP Imports
###############################################################################
import torch


###############################################################################
# Enumerated Types
###############################################################################
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
