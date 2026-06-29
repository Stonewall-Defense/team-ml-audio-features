###############################################################################
# Global Imports
###############################################################################
from enum import Enum
import math
import os
from typing import Optional
import warnings

###############################################################################
# 3PP Imports
###############################################################################
import soundfile
import torch
import torchcodec


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
    wave, final_sr = _prepare_audio(audio, target_sr)
    wave = set_audio_length(wave, final_sr, duration_secs)

    return wave


def load_wav_as_is(path: str) -> tuple[torch.Tensor, int]:
    '''
    Load a WAV file into memory for processing.

    Other file types may work but have not been tested.

    Positional arguments:
        path -- Location of the audio file

    Returns:
        Tuple of: entire wav file as a PyTorch Tensor, sample rate
    '''
    audio = torchcodec.decoders.AudioDecoder(path).get_all_samples()
    return _prepare_audio(audio)


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
    def __init__(self, filename: str, target_sr: int):
        self.filename = filename
        self.target_sr = target_sr

    def load(self, *, start_sec: Optional[int], end_sec: Optional[int], pad=False) -> torch.Tensor:
        if start_sec is None and end_sec is None:
            return load_wav(self.filename, target_sr=self.target_sr)
        elif start_sec and start_sec < 0:
            raise ValueError("start_sec must be at least zero")
        elif end_sec and end_sec < 0:
            raise ValueError("end_sec must be at least zero")
        elif start_sec and end_sec and end_sec <= start_sec:
            raise ValueError("end_sec must be strictly higher than start_sec if both are provided")

        audio = torchcodec.decoders.AudioDecoder(self.filename).get_samples_played_in_range(start_sec or 0, end_sec)
        wave, final_sr = _prepare_audio(audio, self.target_sr)

        if start_sec and end_sec and pad:
            wave = set_audio_length(wave, final_sr, end_sec - start_sec)

        return wave

    def __call__(self, *, start_sec: Optional[int], end_sec: Optional[int], pad=False) -> torch.Tensor:
        return self.load(start_sec=start_sec, end_sec=end_sec, pad=pad)


###############################################################################
# Helpers
###############################################################################
def _prepare_audio(audio: torchcodec.AudioSamples, target_sr: Optional[int] = None):
    wave = audio.data

    # Resample
    final_sr = target_sr or audio.sample_rate
    wave = resample(wave, orig_freq=audio.sample_rate, new_freq=target_sr)

    # Stereo -> Mono
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
###############################################################################

###############################################################################
# From `torchaudio.functional`
###############################################################################

def _get_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    device: torch.device = torch.device("cpu"),
    dtype: Optional[torch.dtype] = None,
):
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    if lowpass_filter_width <= 0:
        raise ValueError("Low pass filter width should be positive.")
    base_freq = min(orig_freq, new_freq)
    # This will perform antialiasing filtering by removing the highest frequencies.
    # At first I thought I only needed this when downsampling, but when upsampling
    # you will get edge artifacts without this, as the edge is equivalent to zero padding,
    # which will add high freq artifacts.
    base_freq *= rolloff

    # The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor)
    # using the sinc interpolation formula:
    #   x(t) = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - t))
    # We can then sample the function x(t) with a different sample rate:
    #    y[j] = x(j / new_freq)
    # or,
    #    y[j] = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))

    # We see here that y[j] is the convolution of x[i] with a specific filter, for which
    # we take an FIR approximation, stopping when we see at least `lowpass_filter_width` zeros crossing.
    # But y[j+1] is going to have a different set of weights and so on, until y[j + new_freq].
    # Indeed:
    # y[j + new_freq] = sum_i x[i] sinc(pi * orig_freq * ((i / orig_freq - (j + new_freq) / new_freq))
    #                 = sum_i x[i] sinc(pi * orig_freq * ((i - orig_freq) / orig_freq - j / new_freq))
    #                 = sum_i x[i + orig_freq] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))
    # so y[j+new_freq] uses the same filter as y[j], but on a shifted version of x by `orig_freq`.
    # This will explain the F.conv1d after, with a stride of orig_freq.
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    # If orig_freq is still big after GCD reduction, most filters will be very unbalanced, i.e.,
    # they will have a lot of almost zero values to the left or to the right...
    # There is probably a way to evaluate those filters more efficiently, but this is kept for
    # future work.
    idx_dtype = dtype if dtype is not None else torch.float64

    idx = torch.arange(-width, width + orig_freq, dtype=idx_dtype, device=device)[None, None] / orig_freq

    t = torch.arange(0, -new_freq, -1, dtype=dtype, device=device)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)

    # we do not use built in torch windows here as we need to evaluate the window
    # at specific positions, not over a regular grid.
    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2

    t *= math.pi

    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale

    if dtype is None:
        kernels = kernels.to(dtype=torch.float32)

    return kernels, width


def _apply_sinc_resample_kernel(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
    gcd: int,
    kernel: torch.Tensor,
    width: int,
):
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    # pack batch
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

    num_wavs, length = waveform.shape
    waveform = torch.nn.functional.pad(waveform, (width, width + orig_freq))
    resampled = torch.nn.functional.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()
    resampled = resampled[..., :target_length]

    # unpack batch
    resampled = resampled.view(shape[:-1] + resampled.shape[-1:])
    return resampled


def resample(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: Optional[int],
    *,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
) -> torch.Tensor:
    if orig_freq == new_freq or new_freq is None:
        return waveform

    gcd = math.gcd(int(orig_freq), int(new_freq))

    kernel, width = _get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd,
        lowpass_filter_width,
        rolloff,
        waveform.device,
        waveform.dtype,
    )
    resampled = _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width)
    return resampled


###############################################################################
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
