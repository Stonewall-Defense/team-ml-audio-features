###############################################################################
# Global Imports
###############################################################################
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
import torch
import torchaudio


###############################################################################
# Helpers
###############################################################################
def _is_stereo(wave: torch.Tensor) -> bool:
    return wave.size(0) == 2


###############################################################################
# Functions
###############################################################################
def load_input(path: str,
               *,
               target_sr: Optional[int] = None,
               duration_secs: Optional[int] = None,
               ) -> torch.Tensor:
    '''
    Load a WAV file into memory for processing.

    Other file types may work but have not been tested.

    Positional arguments:
        path -- Location of the audio file

    Keyword arguments:
        target_sr -- Sample rate to which audio should be resampled
        duration_secs -- Consistent duration of output audio, either by truncation or zero-padding as needed
    '''

    wave, sr = torchaudio.load(path, normalize=True)

    # Resample
    final_sr = target_sr or sr

    if target_sr is not None and target_sr != sr:
        wave = torchaudio.functional.resample(wave, orig_freq=sr, new_freq=target_sr)

    # Stereo -> Mono
    if _is_stereo(wave):
        wave = torch.mean(wave, dim=0, keepdim=True)

    if duration_secs is not None:
        # Truncate as needed
        start_secs = 0
        end_secs = duration_secs
        st_idx, end_idx = int(start_secs * final_sr), int(end_secs * final_sr)
        wave = wave[:, st_idx:end_idx]

        # Zero Padding
        num_samples = int(final_sr * duration_secs)
        padding_size = max(num_samples - wave.size(1), 0)
        if padding_size > 0:
            wave = torch.cat([wave, torch.zeros(1, padding_size)], dim=1)

        # Trim
        wave = wave[:, :num_samples]

    return wave
