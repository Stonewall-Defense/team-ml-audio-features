###############################################################################
# Global Imports
###############################################################################
from enum import Enum
import os
from pathlib import Path
from typing import Optional
import warnings

###############################################################################
# 3PP Imports
###############################################################################
import soundfile
import torch
import torchcodec

###############################################################################
# Local Imports
###############################################################################
from AudioMlSpecTools.resample import resample, Resample


###############################################################################
# Enums, Dataclasses, and Types
###############################################################################
class AudioEncoding(Enum):
    PCM_S = "PCM_S"
    PCM_U = "PCM_U"
    PCM_F = "PCM_F"
    ULAW = "ULAW"
    ALAW = "ALAW"


class BitsPerSample(Enum):
    _8 = 8
    _16 = 16
    _24 = 24
    _32 = 32


###############################################################################
# Functions
###############################################################################
def set_audio_length(wave: torch.Tensor, sr: int, duration_secs: Optional[float], pad=False):
    if duration_secs is None:
        return wave
    else:
        # Truncate as needed
        start_secs = 0
        end_secs = duration_secs
        st_idx, end_idx = int(start_secs * sr), int(end_secs * sr)
        wave = wave[:, st_idx:end_idx]

        target_num_samples = int(sr * duration_secs)

        # Zero Padding
        padding_size = max(target_num_samples - wave.size(1), 0) if pad else 0
        if padding_size > 0:
            wave = torch.cat([wave, torch.zeros(1, padding_size)], dim=1)

        # Trim
        wave = wave[:, :target_num_samples]

        return wave


def load_wav(path: Path | str,
             *,
             target_sr: Optional[int] = None,
             duration_secs: Optional[int] = None,
             mono=True,
             pad=False,
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
    wave, final_sr = _prepare_audio(audio, target_sr=target_sr, mono=mono)
    wave = set_audio_length(wave, final_sr, duration_secs, pad)

    return wave


def is_multichannel(wave: torch.Tensor) -> bool:
    return wave.size(0) > 1


def to_mono(wave: torch.Tensor) -> torch.Tensor:
    return torch.mean(wave, dim=0, keepdim=True) if is_multichannel(wave) else wave


def list_audio_files(dir: str) -> list[str]:
    return sorted([f for f in os.listdir(dir) if f.endswith(".wav")])


###############################################################################
# Classes
###############################################################################
class WavReader:
    def __init__(self,
                 *,
                 target_sr: Optional[int] = None,
                 pad=False,
                 mono=True,

                 # For efficient resampling
                 expected_file_sr: Optional[int] = None,
                 ):
        self.target_sr = target_sr
        self.pad = pad
        self.mono = mono

        self.expected_file_sr = expected_file_sr

        if self.target_sr and self.expected_file_sr:
            self.resample = Resample(self.expected_file_sr, self.target_sr)
        else:
            self.resample = None

    def __call__(self, filename: Path | str, *, start_sec: Optional[float] = None, end_sec: Optional[float] = None):
        return self.read(filename, start_sec=start_sec, end_sec=end_sec)

    def read(self, filename: Path | str, *, start_sec: Optional[float] = None, end_sec: Optional[float] = None):
        if start_sec and start_sec < 0:
            raise ValueError("start_sec must be at least zero")
        elif end_sec and end_sec < 0:
            raise ValueError("end_sec must be at least zero")
        elif start_sec and end_sec and end_sec <= start_sec:
            raise ValueError("end_sec must be strictly higher than start_sec if both are provided")

        f = torchcodec.decoders.AudioDecoder(filename)

        if start_sec is None and end_sec is None:
            audio = f.get_all_samples()
        else:
            audio = f.get_samples_played_in_range(start_sec or 0, end_sec)

        file_sr = audio.sample_rate

        if self.expected_file_sr and self.expected_file_sr != file_sr:
            msg = f"File {filename} has sample rate {file_sr} but expected {self.expected_file_sr}"
            if self.target_sr:
                raise ValueError(msg)
            else:
                warnings.warn(msg)

        wave, final_sr = _prepare_audio(audio, target_sr=self.target_sr, mono=self.mono, resampler=self.resample)

        duration = end_sec - (start_sec or 0) if end_sec else None
        wave = set_audio_length(wave, final_sr, duration, self.pad)

        return wave


###############################################################################
# Helpers
###############################################################################
def _prepare_audio(audio: torchcodec.AudioSamples,
                   *,
                   resampler: Optional[Resample] = None,
                   target_sr: Optional[int] = None,
                   mono=True,
                   ):
    wave = audio.data

    # Resample
    final_sr = target_sr or audio.sample_rate
    wave = resampler(wave) if resampler else resample(wave, orig_freq=audio.sample_rate, new_freq=target_sr)

    # Stereo -> Mono
    if mono:
        wave = to_mono(wave)

    return wave, final_sr


###############################################################################
# Imported Legacy Code
# This code was extracted from v2.8.0 of [torchaudio](https://github.com/pytorch/audio)
# torchaudio is licensed under the BSD 2-Clause License, reprinted below
###############################################################################
#
# Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# From `torchaudio._backend.soundfile`
# Updated by Ryan Quinn 12 June 2026
###############################################################################
def _get_subtype(dtype: torch.dtype,
                 encoding: Optional[AudioEncoding],
                 bits_per_sample: Optional[BitsPerSample],
                 ):
    enc_ = encoding.value if encoding else None
    bps_ = bits_per_sample.value if bits_per_sample else None

    if not enc_:
        if not bps_:
            subtype = {
                torch.uint8: "PCM_U8",
                torch.int16: "PCM_16",
                torch.int32: "PCM_32",
                torch.float32: "FLOAT",
                torch.float64: "DOUBLE",
            }.get(dtype)
            if not subtype:
                raise ValueError(f"Unsupported dtype for wav: {dtype}")
            return subtype
        if bps_ == 8:
            return "PCM_U8"
        return f"PCM_{bps_}"
    if enc_ == "PCM_S":
        if not bps_:
            return "PCM_32"
        if bps_ == 8:
            raise ValueError("wav does not support 8-bit signed PCM encoding.")
        return f"PCM_{bps_}"
    if enc_ == "PCM_U":
        if bps_ in (None, 8):
            return "PCM_U8"
        raise ValueError("wav only supports 8-bit unsigned PCM encoding.")
    if enc_ == "PCM_F":
        if bps_ in (None, 32):
            return "FLOAT"
        if bps_ == 64:
            return "DOUBLE"
        raise ValueError("wav only supports 32/64-bit float PCM encoding.")
    if enc_ == "ULAW":
        if bps_ in (None, 8):
            return "ULAW"
        raise ValueError("wav only supports 8-bit mu-law encoding.")
    if enc_ == "ALAW":
        if bps_ in (None, 8):
            return "ALAW"
        raise ValueError("wav only supports 8-bit a-law encoding.")
    raise ValueError(f"wav does not support {enc_}.")


def save_wav(
    filepath: str,
    src: torch.Tensor,
    sample_rate: int,
    *,
    channels_first: bool = True,
    encoding: Optional[AudioEncoding | str] = None,
    bits_per_sample: Optional[BitsPerSample | int] = None,
):
    if isinstance(encoding, str):
        encoding = AudioEncoding(encoding)

    if isinstance(bits_per_sample, int):
        bits_per_sample = BitsPerSample(bits_per_sample)

    if src.ndim == 1:
        src = src.unsqueeze(0)
    elif src.ndim != 2:
        raise ValueError(f"Expected an at most 2D Tensor, got {src.ndim}D.")
    elif bits_per_sample is not None and bits_per_sample.value == 24:
        warnings.warn(
            "Saving audio with 24 bits per sample might warp samples near -1. "
            "Using 16 bits per sample might be able to avoid this."
        )

    subtype = _get_subtype(src.dtype, encoding, bits_per_sample)

    if channels_first:
        src = src.t()

    soundfile.write(file=filepath, data=src, samplerate=sample_rate, subtype=subtype)
