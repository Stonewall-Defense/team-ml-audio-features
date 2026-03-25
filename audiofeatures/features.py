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
# Helper Classes
###############################################################################
class AmpScaler(torch.nn.Module):
    def __init__(self, stype: str = "power", top_db: Optional[float] = None) -> None:
        super(AmpScaler, self).__init__()
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


class ManualSTFT(torch.nn.Module):
    """
        Exportable to Executorch. Created by Claude with supervision. Results are very slightly different than Torch.
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
# Helper Functions
###############################################################################
def _load_params(params: str | list | dict):
    if not isinstance(params, str):
        return params
    else:
        with open(params, "r") as infile:
            return json.loads(infile.read())


def _write_params(filename: str, params: list | dict):
    with open(filename, "r") as outfile:
        return outfile.write(json.dumps(params, indent=2))


def _power_of_two(n: int):
    return (n & (n - 1) == 0) and n != 0


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


def _linear_fbanks(
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
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    return fb


def _generate_filters(n_fft: int, n_filters: int, sample_rate: int, mel_type: Optional[MelType]):
    n_freqs = n_fft // 2 + 1

    if mel_type:
        return _melscale_fbanks(
            mel_type,
            n_freqs,
            n_filters,
            sample_rate,
        )
    else:
        f_min = 0.0
        f_max = float(sample_rate // 2)

        return _linear_fbanks(
            n_freqs,
            f_min,
            f_max,
            n_filters,
            sample_rate,
        )


def _create_scaler(scaling_type: ScalingType):
    if scaling_type == ScalingType.LOG:
        return _log_scale
    else:
        return AmpScaler(stype=scaling_type.value, top_db=80.0)


def _create_dct(n_mfcc: int, n_mels: int) -> torch.Tensor:
    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    n = torch.arange(float(n_mels))
    k = torch.arange(float(n_mfcc)).unsqueeze(1)
    dct = torch.cos(math.pi / float(n_mels) * (n + 0.5) * k)  # size (n_mfcc, n_mels)

    dct[0] *= 1.0 / math.sqrt(2.0)
    dct *= math.sqrt(2.0 / float(n_mels))
    return dct.t()


def _scale_spec(spec: torch.Tensor) -> torch.Tensor:
    min_in_val = torch.min(spec)
    max_in_val = torch.max(spec)
    in_span = max_in_val - min_in_val

    min_out_val = torch.zeros(1)
    max_out_val = torch.ones(1)
    out_span = max_out_val - min_out_val

    scale_factor = out_span / in_span
    return (spec - min_in_val) * scale_factor


def _determine_spec_type(calc_mels: bool, calc_logs: bool, calc_cepstrum: bool):
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
# Export Classes
###############################################################################
class FeatureChannel(torch.nn.Module):
    def __init__(self,
                 sample_rate: int,
                 *,
                 # For all spectra
                 n_fft: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 scale_spec: Optional[bool] = None,

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
        if n_fft is not None and not _power_of_two(n_fft):
            raise ValueError("n_fft must be a power of 2")
        self.n_fft = n_fft or 1024

        if hop_length is not None and hop_length > (self.n_fft // 2):
            warnings.warn(f"hop_length should be set to no more than 1/2 the FFT window size, or {self.n_fft // 2} mels for n_fft = {self.n_fft} (currently {hop_length})")
        self.hop_length = hop_length or self.n_fft // 4

        self.scale_spec = scale_spec if scale_spec is not None else True

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
        window = torch.hann_window(self.n_fft)
        self.register_buffer("window", window)
        self._stft = ManualSTFT(self.n_fft, self.hop_length, self.n_fft, self.window)

        fb = _generate_filters(self.n_fft, self.n_filters, self.sample_rate, self.mel_type)
        self.register_buffer("fb", fb)

        # DB scaling, if necessary
        self.scaling_type = scaling_type
        self.amplitude_to_DB = _create_scaler(scaling_type) if calc_logs else None

        # Cepstrum, if necessary
        if calc_cepstrum:
            dct_mat = _create_dct(self.cepstral_coefficients, self.n_filters)
            self.register_buffer("dct_mat", dct_mat)
        else:
            self.dct_mat = None

        self.spec_type = _determine_spec_type(calc_mels, calc_logs, calc_cepstrum)

    def get_spec_type(self):
        return self.spec_type

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self._stft(wav)

        if self.fb is not None:
            spec = torch.matmul(spec.transpose(-1, -2), self.fb).transpose(-1, -2)

        if self.amplitude_to_DB is not None:
            spec = self.amplitude_to_DB(spec)

        if self.dct_mat is not None:
            spec = torch.matmul(spec.transpose(-1, -2), self.dct_mat).transpose(-1, -2).squeeze()

        if self.scale_spec:
            spec = _scale_spec(spec)

        return spec

    @staticmethod
    def from_json(params: str | dict):
        loaded_params = _load_params(params)
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
        _write_params(filename, params)


class FeatureSource(torch.nn.Module):
    def __init__(self, feature_channels: list[FeatureChannel]):
        super(FeatureSource, self).__init__()

        if len(feature_channels) == 0:
            raise ValueError("Must include at least one spec type")

        self.fc = feature_channels
        self.feature_channels = torch.nn.ModuleList(feature_channels)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spectra = [chan(wav) for chan in self.feature_channels]
        return torch.stack(spectra, dim=1)

    @staticmethod
    def from_json(params: str | list):
        loaded_params = _load_params(params)
        if not isinstance(loaded_params, list):
            raise ValueError(f"Invalid {FeatureSource.__name__} parameters")

        feature_channels = [FeatureChannel.from_json(p) for p in loaded_params]
        return FeatureSource(feature_channels)

    def to_json(self, filename: str):
        params = [fc.get_params() for fc in self.feature_channels.children() if isinstance(fc, FeatureChannel)]
        _write_params(filename, params)
