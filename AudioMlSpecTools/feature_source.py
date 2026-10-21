###############################################################################
# Global Imports
###############################################################################
from typing import Optional, Sequence
import warnings

###############################################################################
# 3PP Imports
###############################################################################
import torch

###############################################################################
# Local Imports
###############################################################################
from .amp_scale import ScalingType, create_scaler
from .cepstrum import create_dct
from .freq_scale import MelType, generate_filters
from .stft import TypePreservingSTFT
from .types import AudioPreprocessor, AudioPostprocessor, BaseFeatureSource
from .util import power_of_two, scale_spec
from .window import WindowFunction


###############################################################################
# Flexible Features
###############################################################################
class FlexibleFeatureChannel(torch.nn.Module):
    def __init__(self,
                 sample_rate: int,
                 *,
                 # For all spectra
                 n_fft: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 window_type: Optional[WindowFunction] = None,
                 scale_spec: bool = True,
                 nan_to_zero: bool = False,

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
        super(FlexibleFeatureChannel, self).__init__()

        # Required configs
        self.sample_rate = sample_rate

        # Universal configs
        if n_fft is not None and not power_of_two(n_fft):
            raise ValueError("n_fft must be a power of 2")
        self.n_fft = n_fft or 1024

        if hop_length is not None and hop_length > (self.n_fft // 2):
            warnings.warn(f"hop_length should be set to no more than 1/2 the FFT window size, or {self.n_fft // 2} for n_fft = {self.n_fft} (currently {hop_length})")

        self.scale_spec = scale_spec
        self.nan_to_zero = nan_to_zero

        # Shared configs (mels, MFCC, LFCC)
        if n_filters is not None and n_filters > (self.n_fft // 8):
            warnings.warn(f"n_filters should be set to no more than 1/8 the FFT window size, or {self.n_fft // 8} filters for n_fft = {self.n_fft} (currently {n_filters})")
        self.n_filters = n_filters or self.n_fft // 8

        # Mel configs
        calc_mels = is_mel or (mel_type is not None)
        self.mel_type = mel_type or (MelType.OSHAUGHNESSY if calc_mels else None)

        # Cepstral configs
        calc_cepstrum = is_cepstrum or (cepstral_coefficients is not None)

        if cepstral_coefficients is not None and cepstral_coefficients > self.n_filters:
            raise ValueError(f"cepstral_coefficients must be no greater than n_mels (currently {cepstral_coefficients}/{self.n_filters})")
        self.cepstral_coefficients = cepstral_coefficients or self.n_filters

        # Log configs
        calc_logs = is_logarithmic or calc_cepstrum

        ###################
        # Spec gen code
        ###################

        # Basic spectrogram
        self._stft = TypePreservingSTFT(self.n_fft, hop_length=hop_length, window_type=window_type)

        fb = generate_filters(self.n_fft, self.n_filters, self.sample_rate, self.mel_type)
        self.register_buffer("fb", fb)

        # DB scaling, if necessary
        self.scaling_type = scaling_type
        self.amplitude_to_DB = create_scaler(scaling_type) if calc_logs else None

        # Cepstrum, if necessary
        if calc_cepstrum:
            dct_mat = create_dct(self.cepstral_coefficients, self.n_filters)
            self.register_buffer("dct_mat", dct_mat)
        else:
            self.dct_mat = None

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self._stft(wav)

        if self.fb is not None:
            spec = torch.matmul(spec.transpose(-1, -2), self.fb).transpose(-1, -2)

        if self.amplitude_to_DB is not None:
            spec = self.amplitude_to_DB(spec)

        if self.dct_mat is not None:
            spec = torch.matmul(spec.transpose(-1, -2), self.dct_mat).transpose(-1, -2)

        if self.scale_spec:
            spec = scale_spec(spec)

        if self.nan_to_zero:
            spec = torch.nan_to_num(spec)

        return spec


class FlexibleFeatureSource(BaseFeatureSource):
    def __init__(self,
                 feature_channels: list[FlexibleFeatureChannel],
                 *,
                 # Parent Params
                 preprocessors: Sequence[AudioPreprocessor] = [],
                 postprocessors: Sequence[AudioPostprocessor] = [],
                 ):
        super(FlexibleFeatureSource, self).__init__(preprocessors=preprocessors,
                                                    postprocessors=postprocessors,
                                                    )

        if len(feature_channels) == 0:
            raise ValueError("Must include at least one spec type")

        self.fc = feature_channels
        self.feature_channels = torch.nn.ModuleList(feature_channels)
        self.preprocessors = torch.nn.ModuleList(preprocessors)

    def num_channels(self):
        return len(self.fc)

    def _make_specs(self, wav: torch.Tensor) -> torch.Tensor:
        spectra = [chan(wav) for chan in self.feature_channels]
        return torch.stack(spectra, dim=1)


###############################################################################
# Efficient Features
###############################################################################
class EfficientFeatureChannel:
    def __init__(self,
                 *,
                 is_mel: bool,
                 is_log: bool,
                 is_cepstrum: bool,
                 ):
        self.is_mel = is_mel
        self.is_log = is_log
        self.is_cepstrum = is_cepstrum

        self.key = self._make_channel_key()

    def get_key(self):
        return self.key

    def _make_channel_key(self):
        if self.is_cepstrum:
            return "mfcc" if self.is_mel else "lfcc"
        elif self.is_log:
            return "mel_log_spec" if self.is_mel else "lin_log_spec"
        else:
            return "mel_freq_spec" if self.is_mel else "lin_freq_spec"


class EfficientFeatureSource(BaseFeatureSource):
    '''
        Calculates one or more feature channels with a restricted set of inputs.
        Uses the fewest-possible operations, within reason.
    '''
    def __init__(self,
                 sample_rate: int,
                 channels: list[EfficientFeatureChannel],
                 *,
                 # Parent Params
                 preprocessors: Sequence[AudioPreprocessor] = [],
                 postprocessors: Sequence[AudioPostprocessor] = [],

                 # For all spectra
                 n_fft: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 window_type: Optional[WindowFunction] = None,

                 # Shared
                 n_filters: Optional[int] = None,

                 # For all cepstra
                 cepstral_coefficients: Optional[int] = None,
                 ):
        super(EfficientFeatureSource, self).__init__(preprocessors=preprocessors,
                                                     postprocessors=postprocessors,
                                                     )

        self.channels = channels

        self.has_lin_freq = any([not c.is_mel for c in channels])
        self.has_mel_freq = any([c.is_mel for c in channels])

        self.has_lfcc = self.has_lin_freq and any([c.is_cepstrum for c in channels])
        self.has_mfcc = self.has_mel_freq and any([c.is_cepstrum for c in channels])
        self.has_cepstrum = self.has_lfcc or self.has_mfcc

        self.has_lin_log = any([not c.is_mel and c.is_log for c in channels])
        self.has_mel_log = any([c.is_mel and c.is_log for c in channels])
        self.has_log_scale = any([c.is_log for c in channels]) or self.has_cepstrum

        # Required configs
        self.sample_rate = sample_rate

        # Universal configs
        if n_fft is not None and not power_of_two(n_fft):
            raise ValueError("n_fft must be a power of 2")
        self.n_fft = n_fft or 1024

        if hop_length is not None and hop_length > (self.n_fft // 2):
            warnings.warn(f"hop_length should be set to no more than 1/2 the FFT window size, or {self.n_fft // 2} for n_fft = {self.n_fft} (currently {hop_length})")

        # Shared configs (mels, MFCC, LFCC)
        if n_filters is not None and n_filters > (self.n_fft // 8):
            warnings.warn(f"n_filters should be set to no more than 1/8 the FFT window size, or {self.n_fft // 8} filters for n_fft = {self.n_fft} (currently {n_filters})")
        self.n_filters = n_filters or self.n_fft // 8

        # Cepstral configs
        if cepstral_coefficients is not None and cepstral_coefficients > self.n_filters:
            raise ValueError(f"cepstral_coefficients must be no greater than n_mels (currently {cepstral_coefficients}/{self.n_filters})")
        self.cepstral_coefficients = cepstral_coefficients or self.n_filters

        ###################
        # Spec gen code
        ###################

        # Basic spectrogram
        self._stft = TypePreservingSTFT(self.n_fft, hop_length=hop_length, window_type=window_type)

        if self.has_lin_freq:
            self.register_buffer("lin_filt", generate_filters(self.n_fft, self.n_filters, self.sample_rate, None))
        else:
            self.lin_filt = None

        if self.has_mel_freq:
            self.register_buffer("mel_filt", generate_filters(self.n_fft, self.n_filters, self.sample_rate, MelType.OSHAUGHNESSY))
        else:
            self.mel_filt = None

        # DB scaling, if necessary
        self.amplitude_to_DB = create_scaler(ScalingType.POWER) if self.has_log_scale else None

        # Cepstrum, if necessary
        if self.has_cepstrum:
            self.register_buffer("dct", create_dct(self.cepstral_coefficients, self.n_filters))
        else:
            self.dct = None

    def _make_specs(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self._stft(wav)

        if self.lin_filt is not None:
            lin_freq_spec = torch.matmul(spec.transpose(-1, -2), self.lin_filt).transpose(-1, -2)
        else:
            lin_freq_spec = None

        if self.mel_filt is not None:
            mel_freq_spec = torch.matmul(spec.transpose(-1, -2), self.mel_filt).transpose(-1, -2)
        else:
            mel_freq_spec = None

        if self.has_lin_log and (lin_freq_spec is not None) and self.amplitude_to_DB:
            lin_log_spec = self.amplitude_to_DB(lin_freq_spec)
        else:
            lin_log_spec = None

        if self.has_mel_log and (mel_freq_spec is not None) and self.amplitude_to_DB:
            mel_log_spec = self.amplitude_to_DB(mel_freq_spec)
        else:
            mel_log_spec = None

        if self.has_lfcc and (lin_log_spec is not None) and (self.dct is not None):
            lfcc = torch.matmul(lin_log_spec.transpose(-1, -2), self.dct).transpose(-1, -2)
        else:
            lfcc = None

        if self.has_mfcc and (mel_log_spec is not None) and (self.dct is not None):
            mfcc = torch.matmul(mel_log_spec.transpose(-1, -2), self.dct).transpose(-1, -2)
        else:
            mfcc = None

        return self._collate(**{
            "lin_freq_spec": lin_freq_spec,
            "mel_freq_spec": lin_freq_spec,
            "lin_log_spec": lin_log_spec,
            "mel_log_spec": mel_log_spec,
            "lfcc": lfcc,
            "mfcc": mfcc,
        })

    def _collate(self, **kwargs):
        specs = [scale_spec(kwargs[c.key]) for c in self.channels]
        return torch.stack(specs, dim=1)


###############################################################################
# Full-Range Features
###############################################################################
class FullRangeStftFeatureSource(BaseFeatureSource):
    '''
        Unlike other feature sources, this one does not apply linear filters to the STFT output.
        This gives higher resolution but makes the size (slightly) harder to control.
    '''
    def __init__(self,
                 sample_rate: int,
                 *,
                 # Parent Params
                 preprocessors: Sequence[AudioPreprocessor] = [],
                 postprocessors: Sequence[AudioPostprocessor] = [],

                 # For all spectra
                 n_fft: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 window_type: Optional[WindowFunction] = None,

                 # For all log spectra
                 is_logarithmic: bool = True,
                 ):
        super(FullRangeStftFeatureSource, self).__init__(preprocessors=preprocessors,
                                                         postprocessors=postprocessors,
                                                         )

        # Internal configs
        self.sample_rate = sample_rate

        if n_fft is not None and not power_of_two(n_fft):
            raise ValueError("n_fft must be a power of 2")
        self.n_fft = n_fft or 1024

        if hop_length is not None and hop_length > (self.n_fft // 2):
            warnings.warn(f"hop_length should be set to no more than 1/2 the FFT window size, or {self.n_fft // 2} for n_fft = {self.n_fft} (currently {hop_length})")

        # Basic spectrogram
        self._stft = TypePreservingSTFT(self.n_fft, hop_length=hop_length, window_type=window_type)

        # DB scaling, if necessary
        self.amplitude_to_DB = create_scaler(ScalingType.POWER) if is_logarithmic else None

    def _make_specs(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self._stft(wav)

        if self.amplitude_to_DB is not None:
            spec = self.amplitude_to_DB(spec)

        return scale_spec(spec).unsqueeze(0)
