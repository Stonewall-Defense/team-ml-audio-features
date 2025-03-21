###############################################################################
# Global Imports
###############################################################################
from enum import Enum
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
class SpecType(Enum):
    '''
    Possible spectrum types for feature extraction.

    The literature is conflicted on which, if any, provides the best results.
    '''
    MEL = 1
    LOG_MEL = 2
    MFCC = 3


###############################################################################
# Classes
###############################################################################
class AudioFeatureExtractor(torch.nn.Module):
    '''
    Convenience class to efficiently convert raw audio into spectrogram features.
    '''

    def __init__(self,
                 spec_types: list[SpecType],
                 sample_rate: int,
                 *,
                 n_fft: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 n_mels: Optional[int] = None,
                 n_mfcc: Optional[int] = None,
                 top_db: Optional[float] = None,
                 normalized: Optional[bool] = None,
                 stack_spectra: bool = True,
                 ):
        '''
        Positional arguments:
            spec_types -- One or more spectra to generate as features
            sample_rate -- Sample rate of the input audio (every input must match)

        Keyword arguments:
            n_fft -- Size of the FFT window, should represent at most ~5 ms
            hop_length -- Shift distance for each FFT window, should be between 25-50% of `n_fft`
            n_mels -- Number of mel filters to use, should be no more than 1/8 `n_fft`
            n_mfcc -- Number of MFCC bands, should be equal to `n_mels` in most cases
            top_db -- dB cutoff, optional (80 is a good starting point)
            normalized -- Whether to normalize by magnitude after STFT
            stack_spectra -- Stack all generated spectra together into a single tensor
        '''

        super(AudioFeatureExtractor, self).__init__()

        # Set up spec types - Always need mel spec, need log mels for MFCC, MFCC is not a dep
        if not len(spec_types):
            raise ValueError('Must include at least one spec type')
        self.spec_types = spec_types
        self.calc_log_mels = len(self.spec_types) > 1 or self.spec_types[0] != SpecType.MEL
        self.calc_mfcc = SpecType.MFCC in self.spec_types

        # Begin other configs
        self.sample_rate = sample_rate

        if n_fft is not None and not power_of_two(n_fft):
            raise ValueError('n_fft must be a power of 2')
        self.n_fft = n_fft or 1024

        if hop_length is not None and hop_length > (self.n_fft // 2):
            warnings.warn(f'hop_length should be set to no more than 1/2 the FFT window size, or {self.n_fft // 2} mels for n_fft = {self.n_fft} (currently {hop_length})')
        self.hop_length = hop_length or self.n_fft // 4

        if n_mels is not None and n_mels > (self.n_fft // 8):
            warnings.warn(f'n_mels should be set to no more than 1/8 the FFT window size, or {self.n_fft // 8} mels for n_fft = {self.n_fft} (currently {n_mels})')
        self.n_mels = n_mels or self.n_fft // 8

        if n_mfcc is not None and n_mfcc > self.n_mels:
            raise ValueError(f'n_mfcc must be no greater than n_mels (currently {n_mfcc}/{self.n_mels})')
        self.n_mfcc = n_mfcc or self.n_mels

        if top_db is not None and top_db <= 0.0:
            raise ValueError(f'top_db must be greater than 0 (currently {top_db})')
        self.top_db = top_db or 80.0

        self.normalized = normalized or False

        self.stack_spectra = stack_spectra or True

        # Set up all spec gen code for convenience
        self.mel_spec_gen = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            normalized=self.normalized,
        )

        self.log_mel_spec_gen = torchaudio.transforms.AmplitudeToDB(top_db=self.top_db)

        self.norm = 'ortho'
        dct_mat = F.create_dct(self.n_mfcc, self.mel_spec_gen.n_mels, self.norm)
        self.register_buffer('dct_mat', dct_mat)

    def _calc_mfcc(self, log_mel_spec: torch.Tensor) -> torch.Tensor:
        # (..., time, n_mels) dot (n_mels, n_mfcc) -> (..., n_nfcc, time)
        mfccs = torch.matmul(log_mel_spec.transpose(-1, -2), self.dct_mat).transpose(-1, -2)
        return mfccs.squeeze()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        '''
        Convert raw audio into spectrogram features.

        Arguments:
            wav -- Raw audio input

        Returns: One tensor of concatenated spectra if stack_spectra is set, otherwise a list of spectra
        '''
        tmp: dict[SpecType, torch.Tensor] = {}

        # They have to be done in this order regardless of return order
        tmp[SpecType.MEL] = self.mel_spec_gen(wav).squeeze()
        tmp[SpecType.LOG_MEL] = self.log_mel_spec_gen(tmp[SpecType.MEL]) if self.calc_log_mels else None
        tmp[SpecType.MFCC] = self._calc_mfcc(tmp[SpecType.LOG_MEL]) if self.calc_mfcc else None

        # Sort data in requested order, NOT math order
        sorted_spectra = [tmp[spec_type] for spec_type in self.spec_types]

        if self.stack_spectra:
            return torch.stack(sorted_spectra, dim=0).unsqueeze(0)
        else:
            return sorted_spectra
