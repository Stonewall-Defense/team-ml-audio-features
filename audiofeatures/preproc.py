###############################################################################
# Global Imports
###############################################################################
from abc import ABC, abstractmethod

###############################################################################
# 3PP Imports
###############################################################################
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
import torch


###############################################################################
# Interfaces
###############################################################################
class AudioPreprocessor(torch.nn.Module, ABC):
    @abstractmethod
    def __call__(self, wav: torch.Tensor | np.ndarray) -> torch.Tensor:
        ...


###############################################################################
# Classes
###############################################################################
class HighPassFilter(AudioPreprocessor):
    def __init__(self,
                 *,
                 sample_rate: int,
                 cutoff_freq: int,
                 rolloff_db: int,
                 ):
        nyquist = sample_rate // 2
        if cutoff_freq < 0 or cutoff_freq > nyquist:
            raise ValueError(f"Cutoff freq must be >0 Hz and less than the Nyquist rate ({nyquist})")
        elif rolloff_db < 6 or rolloff_db % 6 != 0:
            raise ValueError("Rolloff dB must be >0 and a multiple of 6")

        self.cutoff_freq = cutoff_freq
        self.rolloff_db = rolloff_db

        self.sos = butter(rolloff_db // 6,
                          cutoff_freq,
                          btype="highpass",
                          analog=False,
                          fs=sample_rate,
                          output="sos",
                          )

        self.zi = sosfilt_zi(self.sos)

    def __call__(self, wav: torch.Tensor | np.ndarray) -> torch.Tensor:
        _wav = wav.numpy() if isinstance(wav, torch.Tensor) else wav
        if _wav.ndim != 1:
            raise ValueError(f"Improper input dim; must be 1 but is {_wav.ndim}")

        processed_samples, _ = sosfilt(self.sos, _wav, zi=self.zi * _wav[0])
        processed_samples = processed_samples.astype(np.float32)
        return torch.from_numpy(processed_samples)
