###############################################################################
# Global Imports
###############################################################################
import os
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
import torch
import torchcodec

###############################################################################
# Local Imports
###############################################################################
from ._common import resample


###############################################################################
# Helpers
###############################################################################
def _is_multichannel(wave: torch.Tensor) -> bool:
    return wave.size(0) > 1


###############################################################################
# Functions
###############################################################################
def set_audio_length(wave: torch.Tensor, sr: int, duration_secs: Optional[int]):
    if duration_secs is None:
        return wave
    else:
        # Truncate as needed
        start_secs = 0
        end_secs = duration_secs
        st_idx, end_idx = int(start_secs * sr), int(end_secs * sr)
        wave = wave[:, st_idx:end_idx]

        # Zero Padding
        num_samples = int(sr * duration_secs)
        padding_size = max(num_samples - wave.size(1), 0)
        if padding_size > 0:
            wave = torch.cat([wave, torch.zeros(1, padding_size)], dim=1)

        # Trim
        wave = wave[:, :num_samples]

        return wave


def load_wav(path: str,
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

    audio = torchcodec.decoders.AudioDecoder(path).get_all_samples()
    wave = audio.data

    # Resample
    final_sr = target_sr or audio.sample_rate

    if target_sr is not None:
        wave = resample(wave, orig_freq=audio.sample_rate, new_freq=target_sr)

    # Stereo -> Mono
    if _is_multichannel(wave):
        wave = torch.mean(wave, dim=0, keepdim=True)

    wave = set_audio_length(wave, final_sr, duration_secs)

    return wave


def list_audio_files(dir: str) -> list[str]:
    return sorted([f for f in os.listdir(dir) if f.endswith(".wav")])


###############################################################################
# Classes
###############################################################################
class WavReader:
    def __init__(self, target_sr: int, duration_secs: Optional[int] = None):
        self.target_sr = target_sr
        self.duration_secs = duration_secs

    def load(self, path: str) -> torch.Tensor:
        return load_wav(path, target_sr=self.target_sr, duration_secs=self.duration_secs)

    def __call__(self, path: str) -> torch.Tensor:
        return self.load(path)

    def clip(self, wav: torch.Tensor, start_sec: int, end_sec: int) -> torch.Tensor:
        start_frame = start_sec * self.target_sr
        end_frame = end_sec * self.target_sr
        return wav[:, start_frame:end_frame]
