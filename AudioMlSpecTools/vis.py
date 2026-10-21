###############################################################################
# Global Imports
###############################################################################
import math
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

###############################################################################
# Local Imports
###############################################################################
from .util import num_specs, has_channels


###############################################################################
# Helpers
###############################################################################
def _calc_plot_shape(num_spectra: int) -> tuple[int, int]:
    if num_spectra >= 12:
        return math.ceil(num_spectra / 4), 4
    if num_spectra >= 9:
        return math.ceil(num_spectra / 3), 3
    elif num_spectra >= 4:
        return math.ceil(num_spectra / 2), 2
    else:
        return num_spectra, 1


###############################################################################
# Plotters
###############################################################################
def plot_spectra(specs: Tensor,
                 subtitles: list[str] = [],
                 sup_title: Optional[str] = None,
                 ) -> None:
    _specs = specs.squeeze()
    _num_specs = num_specs(_specs)
    _has_channels = has_channels(_specs)

    if subtitles and len(subtitles) != _num_specs:
        raise ValueError("If provided, there must be one subtitle per spectrum")

    nrows, ncols = _calc_plot_shape(_num_specs)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    fig.suptitle(sup_title or f"Display of {_num_specs} Spectrograms")

    for idx in range(_num_specs):
        row = idx % nrows
        col = idx // nrows

        im = (_specs[idx, :, :] if _has_channels else _specs).squeeze()

        sub_p = axs[row][col] if ncols > 1 else axs[idx] if nrows > 1 else axs
        sub_p.imshow(im, origin='lower', aspect='auto')

        if subtitles:
            subtitle = subtitles[idx]
            sub_p.set_title(subtitle)
        else:
            sub_p.set_title(f"Spectrum {idx}")

    fig.tight_layout()
    fig.set_figwidth(ncols * 5)
    fig.set_figheight(nrows * 3)
    plt.show()


def plot_wav(wav: Tensor, sample_rate: int, title: str = "") -> None:
    td = wav.squeeze(0)
    time = np.linspace(0, len(td) / sample_rate, num=len(td))
    plt.plot(time, td)
    plt.title(title or "Audio Amplitude vs. Time")
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude")
    plt.show()


def plot_with_time_domain(specs: Tensor,
                          wav: Optional[Tensor] = None,
                          sample_rate: Optional[int] = None,
                          title: Optional[str] = None,
                          ) -> None:
    if wav is not None and sample_rate is None:
        raise ValueError("If `wav` is provided, `sample_rate` must also be provided")

    # Config
    _specs = specs.squeeze()
    _num_specs = num_specs(_specs)
    _has_channels = has_channels(_specs)

    has_audio = (wav is not None and sample_rate is not None)
    total_plots = _num_specs + int(has_audio)

    fig, ax = plt.subplots(nrows=total_plots)

    fig.suptitle(title or f"Plot of {_num_specs} With{'' if has_audio else 'out'} Audio")
    fig.tight_layout()
    fig.set_figheight(8)

    # Spectrogram(s)
    for idx in range(_num_specs):
        _chan_spec = (_specs[idx, :, :] if _has_channels else _specs).squeeze()
        ax[idx].imshow(_chan_spec, origin='lower', aspect='auto')
        ax[idx].set_xlabel("STFT Window")
        ax[idx].set_ylabel("Freq Band")

    # Waveform
    if wav is not None and sample_rate is not None:
        audio = wav.squeeze(0)
        time = np.linspace(0, len(audio) / sample_rate, num=len(audio))
        ax[_num_specs].set_xlim(left=0, right=time[-1])
        ax[_num_specs].plot(time, audio)
        ax[_num_specs].set_xlabel("Time (sec)")
        ax[_num_specs].set_ylabel("Amplitude")

    # Display the plots
    plt.show()
