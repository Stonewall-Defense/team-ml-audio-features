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
from ._common import ScalingType, ExportableSTFT, AudioPreprocessor, BaseFeatureSource
from ._common import power_of_two, create_scaler, scale_spec


###############################################################################
# Export Classes
###############################################################################
class FullRangeStftFeatureSource(BaseFeatureSource):
    def __init__(self,
                 sample_rate: int,
                 preprocessors: Sequence[AudioPreprocessor] = [],
                 *,
                 # For all spectra
                 n_fft: Optional[int] = None,
                 hop_length: Optional[int] = None,

                 # For all log spectra
                 is_logarithmic: bool = True,
                 ):
        super(FullRangeStftFeatureSource, self).__init__(preprocessors)

        # Internal configs
        self.sample_rate = sample_rate

        if n_fft is not None and not power_of_two(n_fft):
            raise ValueError("n_fft must be a power of 2")
        self.n_fft = n_fft or 1024

        if hop_length is not None and hop_length > (self.n_fft // 2):
            warnings.warn(f"hop_length should be set to no more than 1/2 the FFT window size, or {self.n_fft // 2} mels for n_fft = {self.n_fft} (currently {hop_length})")
        self.hop_length = hop_length or self.n_fft // 4

        # Basic spectrogram
        self._stft = ExportableSTFT(self.n_fft, self.hop_length)

        # DB scaling, if necessary
        self.amplitude_to_DB = create_scaler(ScalingType.POWER) if is_logarithmic else None

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        for preproc in self.preprocessors:
            wav = preproc(wav)

        spec = self._stft(wav)

        if self.amplitude_to_DB is not None:
            spec = self.amplitude_to_DB(spec)

        return scale_spec(spec).unsqueeze(0)
