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
from ._math import power_of_two, scale_spec
from ._scale import ScalingType, create_scaler
from ._spec import ExportableSTFT, WindowFunction
from ._util import AudioPreprocessor, AudioPostprocessor, BaseFeatureSource


###############################################################################
# Export Classes
###############################################################################
class FullRangeStftFeatureSource(BaseFeatureSource):
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
        self._stft = ExportableSTFT(self.n_fft, hop_length=hop_length, window_type=window_type)

        # DB scaling, if necessary
        self.amplitude_to_DB = create_scaler(ScalingType.POWER) if is_logarithmic else None

    def _make_specs(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self._stft(wav)

        if self.amplitude_to_DB is not None:
            spec = self.amplitude_to_DB(spec)

        return scale_spec(spec).unsqueeze(0)
