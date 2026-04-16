###############################################################################
# Global Imports
###############################################################################
from enum import Enum
import json
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


class MelType(Enum):
    OSHAUGHNESSY = "O'Shaughnessy"
    FANT = "Fant"
    LINDSAY_NORMAN = "Lindsay & Norman"
    SLANEY = "Slaney"


class ScalingType(Enum):
    POWER = "power"
    MAGNITUDE = "magnitude"
    LOG = "log"


###############################################################################
# New Classes
###############################################################################
class ExportableSTFT(torch.nn.Module):
    """
        Exportable to Executorch. Created by Claude with supervision.

        Results are very slightly different than Torch but show no noticeable difference in model accuracy or training.
    """
    def __init__(self, n_fft: int, hop_length: int, win_length: int, window: torch.Tensor):
        super().__init__()

        w = window if window is not None else torch.ones(n_fft)
        if win_length < n_fft:
            pad_left = (n_fft - win_length) // 2
            pad_right = n_fft - win_length - pad_left
            w = torch.nn.functional.pad(w, (pad_left, pad_right))

        # Only compute onesided bins: n_fft//2+1
        k = torch.arange(n_fft // 2 + 1).unsqueeze(1)  # (freq, 1)
        n = torch.arange(n_fft).unsqueeze(0)            # (1, n_fft)
        angles = -2 * torch.pi * k * n / n_fft          # (freq, n_fft)

        self.hop_length = hop_length
        self.n_fft = n_fft
        self.pad = n_fft // 2

        # Shape: (n_fft//2+1, n_fft)
        self.register_buffer("dft_real", torch.cos(angles) * w)
        self.register_buffer("dft_imag", torch.sin(angles) * w)

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
def load_params(params: str | list | dict):
    if not isinstance(params, str):
        return params
    else:
        with open(params, "r") as infile:
            return json.loads(infile.read())


def write_params(filename: str, params: list | dict):
    with open(filename, "r") as outfile:
        return outfile.write(json.dumps(params, indent=2))


def power_of_two(n: int):
    return (n & (n - 1) == 0) and n != 0


def generate_filters(n_fft: int, n_filters: int, sample_rate: int, mel_type: Optional[MelType]):
    n_freqs = n_fft // 2 + 1
    f_min = 20.0
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


def create_scaler(scaling_type: ScalingType):
    if scaling_type == ScalingType.LOG:
        return log_scale
    else:
        return AmplitudeToDB(stype=scaling_type.value, top_db=80.0)


def scale_spec(spec: torch.Tensor) -> torch.Tensor:
    min_in_val = torch.min(spec)
    max_in_val = torch.max(spec)
    in_span = max_in_val - min_in_val

    min_out_val = torch.zeros(1)
    max_out_val = torch.ones(1)
    out_span = max_out_val - min_out_val

    scale_factor = out_span / in_span
    return (spec - min_in_val) * scale_factor


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
# Generate features
###############################################################################

class AmplitudeToDB(torch.nn.Module):
    def __init__(self, stype: str = "power", top_db: Optional[float] = None) -> None:
        super(AmplitudeToDB, self).__init__()
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError("top_db must be positive value")
        self.top_db = top_db
        self.multiplier = 10.0 if stype == "power" else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))
        x_db -= self.multiplier * self.db_multiplier

        if self.top_db:
            max_ref = (x_db.max() - self.top_db)
            x_db = torch.max(x_db, max_ref)

        return x_db


def log_scale(waveform: torch.Tensor) -> torch.Tensor:
    log_offset = 1e-6
    return torch.log(waveform + log_offset)


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


def create_dct(n_cepstrum: int, n_filters: int) -> torch.Tensor:
    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    n = torch.arange(float(n_filters))
    k = torch.arange(float(n_cepstrum)).unsqueeze(1)
    dct = torch.cos(math.pi / float(n_filters) * (n + 0.5) * k)  # size (n_mfcc, n_mels)

    dct[0] *= 1.0 / math.sqrt(2.0)
    dct *= math.sqrt(2.0 / float(n_filters))
    return dct.t()


###############################################################################
# Load audio
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
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
) -> torch.Tensor:
    if orig_freq == new_freq:
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
