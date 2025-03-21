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
import torchaudio
from torchaudio import functional as F

###############################################################################
# Local Imports
###############################################################################
from .common import power_of_two


###############################################################################
# Enumerated Types
###############################################################################
class AdvSpecType(Enum):
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
# Helpers
###############################################################################
def _log_scale(waveform: torch.Tensor) -> torch.Tensor:
    log_offset = 1e-6
    return torch.log(waveform + log_offset)


def _hz_to_mel(freq: float, mel_type: MelType) -> float:
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


def _mel_to_hz(mels: torch.Tensor, mel_type: MelType) -> torch.Tensor:
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


def _create_triangular_filterbank(
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


def _melscale_fbanks(
        mel_type: MelType,
        n_freqs: int,
        n_mels: int,
        sample_rate: int,
):
    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    f_min = 0.0
    f_max = float(sample_rate // 2)
    m_min = _hz_to_mel(f_min, mel_type=mel_type)
    m_max = _hz_to_mel(f_max, mel_type=mel_type)

    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_type=mel_type)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(
            "At least one mel filterbank has all zero values. "
            f"The value for `n_mels` ({n_mels}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb


def _create_scaler(scaling_type: ScalingType):
    if scaling_type == ScalingType.LOG:
        return _log_scale
    else:
        return torchaudio.transforms.AmplitudeToDB(stype=scaling_type.value, top_db=80.0)


def _generate_filters(mel_type: MelType, n_fft: int, n_filters: int, sample_rate: int):
    n_freqs = n_fft // 2 + 1
    f_min = 0.0
    f_max = float(sample_rate // 2)

    if mel_type:
        return _melscale_fbanks(
            mel_type,
            n_freqs,
            n_filters,
            sample_rate,
        )
    else:
        return F.linear_fbanks(
            n_freqs,
            f_min,
            f_max,
            n_filters,
            sample_rate,
        )


def _determine_spec_type(calc_mels, calc_logs, calc_cepstrum):
    if calc_mels:
        if calc_cepstrum:
            return AdvSpecType.MFCC
        elif calc_logs:
            return AdvSpecType.LOG_MEL
        else:
            return AdvSpecType.MEL
    else:
        if calc_cepstrum:
            return AdvSpecType.LFCC
        elif calc_logs:
            return AdvSpecType.LOG_STFT
        else:
            return AdvSpecType.STFT


###############################################################################
# Classes
###############################################################################
class FeatureExtractor(torch.nn.Module):
    def __init__(self,
                 sample_rate: int,
                 *,
                 # For all spectra
                 n_fft: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 normalized: Optional[bool] = None,

                 # Shared
                 n_filters: Optional[int] = None,

                 # For all mel spectra
                 is_mel: Optional[bool] = None,
                 mel_type: Optional[MelType] = None,

                 # For all log spectra
                 is_logarithmic: Optional[bool] = None,
                 scaling_type: ScalingType = ScalingType.POWER,

                 # For all cepstra
                 is_cepstrum: Optional[bool] = None,
                 cepstral_coefficients: Optional[int] = None,
                 ):
        super(FeatureExtractor, self).__init__()

        # Required configs
        self.sample_rate = sample_rate

        # Universal configs
        if n_fft is not None and not power_of_two(n_fft):
            raise ValueError('n_fft must be a power of 2')
        self.n_fft = n_fft or 1024

        if hop_length is not None and hop_length > (self.n_fft // 2):
            warnings.warn(f'hop_length should be set to no more than 1/2 the FFT window size, or {self.n_fft // 2} mels for n_fft = {self.n_fft} (currently {hop_length})')
        self.hop_length = hop_length or self.n_fft // 4

        self.normalized = normalized or False

        # Shared configs (mels, MFCC, LFCC)
        if n_filters is not None and n_filters > (self.n_fft // 8):
            warnings.warn(f'n_filters should be set to no more than 1/8 the FFT window size, or {self.n_fft // 8} filters for n_fft = {self.n_fft} (currently {n_filters})')
        self.n_filters = n_filters or self.n_fft // 8

        # Mel configs
        calc_mels = is_mel or (mel_type is not None)
        self.mel_type = mel_type or (MelType.OSHAUGHNESSY if calc_mels else None)

        # Cepstral configs
        calc_cepstrum = is_cepstrum or (cepstral_coefficients is not None)

        if cepstral_coefficients is not None and cepstral_coefficients > self.n_filters:
            raise ValueError(f'cepstral_coefficients must be no greater than n_mels (currently {cepstral_coefficients}/{self.n_filters})')
        self.cepstral_coefficients = cepstral_coefficients or self.n_filters

        # Log configs
        calc_logs = is_logarithmic or calc_cepstrum

        ###################
        # Spec gen code
        ###################

        # Basic spectrogram
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=self.normalized,
        )
        self.fb = _generate_filters(self.mel_type, self.n_fft, self.n_filters, self.sample_rate)

        # DB scaling, if necessary
        self.amplitude_to_DB = _create_scaler(scaling_type) if calc_logs else None

        # Cepstrum, if necessary
        if calc_cepstrum:
            self.dct_mat = F.create_dct(self.cepstral_coefficients, self.n_filters, "ortho")
        else:
            self.dct_mat = None

        self.spec_type = _determine_spec_type(calc_mels, calc_logs, calc_cepstrum)

    def get_spec_type(self):
        return self.spec_type

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self.stft(wav)

        if self.fb is not None:
            spec = torch.matmul(spec.transpose(-1, -2), self.fb).transpose(-1, -2)

        if self.amplitude_to_DB is not None:
            spec = self.amplitude_to_DB(spec)

        if self.dct_mat is not None:
            spec = torch.matmul(spec.transpose(-1, -2), self.dct_mat).transpose(-1, -2).squeeze()

        return spec


class FeatureSource(torch.nn.Module):
    def __init__(self,
                 feature_channels: list[FeatureExtractor],
                 *,
                 stack_spectra: Optional[bool] = True,
                 ):
        super(FeatureSource, self).__init__()

        self.feature_channels = feature_channels
        self.stack_spectra = stack_spectra

        if len(feature_channels) == 0:
            raise ValueError('Must include at least one spec type')

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spectra = [chan(wav) for chan in self.feature_channels]
        if self.stack_spectra:
            return torch.stack(spectra, dim=0).unsqueeze(0)
        else:
            return spectra
