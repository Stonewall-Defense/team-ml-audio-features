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
from ._filters import MelType, generate_filters
from ._math import power_of_two, create_dct, scale_spec
from ._scale import ScalingType, create_scaler
from ._spec import ExportableSTFT, determine_spec_type, WindowFunction
from ._util import AudioPreprocessor, AudioPostprocessor, BaseFeatureSource, load_params, write_params


###############################################################################
# Export Classes
###############################################################################
class FeatureChannel(torch.nn.Module):
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
        super(FeatureChannel, self).__init__()

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
        self._stft = ExportableSTFT(self.n_fft, hop_length=hop_length, window_type=window_type)

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

        self.spec_type = determine_spec_type(calc_mels, calc_logs, calc_cepstrum)

    def get_spec_type(self):
        return self.spec_type

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

    @staticmethod
    def from_json(params: str | dict):
        loaded_params = load_params(params)
        if not isinstance(loaded_params, dict):
            raise ValueError(f"Invalid {FeatureChannel.__name__} parameters")

        mel_type_raw = loaded_params.get("mel_type", None)
        scaling_type_raw = loaded_params.get("scaling_type", None)

        return FeatureChannel(sample_rate=loaded_params["sample_rate"],
                              # For all spectra
                              n_fft=loaded_params.get("n_fft", None),
                              hop_length=loaded_params.get("hop_length", None),
                              scale_spec=loaded_params.get("scale_spec", None),

                              # Shared
                              n_filters=loaded_params.get("n_filters", None),

                              # For all mel spectra
                              is_mel=loaded_params.get("is_mel", None),
                              mel_type=MelType(mel_type_raw) if mel_type_raw else None,

                              # For all log spectra
                              is_logarithmic=loaded_params.get("is_logarithmic", None),
                              scaling_type=ScalingType(scaling_type_raw) if scaling_type_raw else ScalingType.POWER,

                              # For all cepstra
                              is_cepstrum=loaded_params.get("is_cepstrum", None),
                              cepstral_coefficients=loaded_params.get("cepstral_coefficients", None),
                              )

    def get_params(self):
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "scale_spec": self.scale_spec,

            # Shared
            "n_filters": self.n_filters,

            # For all mel spectra
            "is_mel": self.mel_type is not None,
            "mel_type": self.mel_type,

            # For all log spectra
            "is_logarithmic": self.amplitude_to_DB is not None,
            "scaling_type": self.scaling_type,

            # For all cepstra
            "is_cepstrum": self.dct_mat is not None,
            "cepstral_coefficients": self.cepstral_coefficients if self.dct_mat is not None else None,
        }

    def to_json(self, filename: str):
        params = self.get_params()
        write_params(filename, params)


class FeatureSource(BaseFeatureSource):
    def __init__(self,
                 feature_channels: list[FeatureChannel],
                 *,
                 # Parent Params
                 preprocessors: Sequence[AudioPreprocessor] = [],
                 postprocessors: Sequence[AudioPostprocessor] = [],
                 ):
        super(FeatureSource, self).__init__(preprocessors=preprocessors,
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

    @staticmethod
    def from_json(params: str | list):
        loaded_params = load_params(params)
        if not isinstance(loaded_params, list):
            raise ValueError(f"Invalid {FeatureSource.__name__} parameters")

        feature_channels = [FeatureChannel.from_json(p) for p in loaded_params]
        return FeatureSource(feature_channels)

    def to_json(self, filename: str):
        params = [fc.get_params() for fc in self.feature_channels.children() if isinstance(fc, FeatureChannel)]
        write_params(filename, params)
