###############################################################################
# Global Imports
###############################################################################
from enum import Enum
import math
from typing import Optional
import warnings

###############################################################################
# 3PP Imports
###############################################################################
import torch


###############################################################################
# Enumerated Types
###############################################################################
class MelType(Enum):
    OSHAUGHNESSY = "O'Shaughnessy"
    FANT = "Fant"
    LINDSAY_NORMAN = "Lindsay & Norman"
    SLANEY = "Slaney"


###############################################################################
# New Functions
###############################################################################
def generate_filters(n_fft: int, n_filters: int, sample_rate: int, mel_type: Optional[MelType]):
    n_freqs = n_fft // 2 + 1
    f_min = 20.0                        # PyTorch uses zero, but that causes issues with the filter banks
    f_max = float(sample_rate // 2)

    if mel_type:
        return melscale_fbanks(
            mel_type,
            f_min,
            f_max,
            n_freqs,
            n_filters,
            sample_rate,
        )
    else:
        return linear_fbanks(
            n_freqs,
            f_min,
            f_max,
            n_filters,
            sample_rate,
        )


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
def hz_to_mel(freq: float, mel_type: MelType) -> float:
    if mel_type == MelType.OSHAUGHNESSY:
        return 2595.0 * math.log10(1.0 + (freq / 700.0))
    elif mel_type == MelType.FANT:
        return (1000 / math.log10(2)) * math.log10(1.0 + (freq / 1000.0))
    elif mel_type == MelType.LINDSAY_NORMAN:
        return 2410.0 * math.log10(1.0 + (freq / 625.0))
    else:   # MelType.SLANEY
        min_log_hz = 1000.0
        f_sp = 200.0 / 3

        if freq < min_log_hz:
            return freq / f_sp
        else:
            min_log_mel = min_log_hz / f_sp
            logstep = math.log(6.4) / 27.0
            return min_log_mel + math.log(freq / min_log_hz) / logstep


def mel_to_hz(mels: torch.Tensor, mel_type: MelType) -> torch.Tensor:
    if mel_type == MelType.OSHAUGHNESSY:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    elif mel_type == MelType.FANT:
        mul = math.log10(2) / 1000
        return 1000.0 * (10 ** (mels * mul) - 1.0)
    elif mel_type == MelType.LINDSAY_NORMAN:
        return 625.0 * (10.0 ** (mels / 2410.0) - 1.0)
    else:
        min_log_hz = 1000.0
        f_sp = 200.0 / 3

        freqs = f_sp * mels
        min_log_mel = min_log_hz / f_sp

        logstep = math.log(6.4) / 27.0

        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

        return freqs


def create_triangular_filterbank(
    all_freqs: torch.Tensor,
    f_pts: torch.Tensor,
) -> torch.Tensor:
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    return fb


def melscale_fbanks(
        mel_type: MelType,
        f_min: float,
        f_max: float,
        n_freqs: int,
        n_mels: int,
        sample_rate: int,
):
    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    m_min = hz_to_mel(f_min, mel_type=mel_type)
    m_max = hz_to_mel(f_max, mel_type=mel_type)

    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts, mel_type=mel_type)

    # create filterbank
    fb = create_triangular_filterbank(all_freqs, f_pts)

    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(
            "At least one mel filterbank has all zero values. "
            f"The value for `n_mels` ({n_mels}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb


def linear_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_filter: int,
    sample_rate: int,
) -> torch.Tensor:
    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # filter mid-points
    f_pts = torch.linspace(f_min, f_max, n_filter + 2)

    # create filterbank
    fb = create_triangular_filterbank(all_freqs, f_pts)

    return fb
