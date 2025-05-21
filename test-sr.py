###############################################################################
# Global Imports
###############################################################################
import math

###############################################################################
# 3PP Imports
###############################################################################
import matplotlib.pyplot as plt
from torch import Tensor

###############################################################################
# Certus Imports
###############################################################################
from audiofeatures import load_input

###############################################################################
# Local Imports
###############################################################################
from audiofeatures import FeatureSource, FeatureExtractor


###############################################################################
# Constants
###############################################################################
SAMPLE_RATES = [8000, 16000, 22050, 44100]
FILENAME = "./dev/1-31482-A-42.wav"


###############################################################################
# Helpers
###############################################################################
def calc_plot_shape(num_spectra: int) -> tuple[int, int]:
    if num_spectra >= 12:
        return math.ceil(num_spectra / 4), 4
    if num_spectra >= 9:
        return math.ceil(num_spectra / 3), 3
    elif num_spectra >= 4:
        return math.ceil(num_spectra / 2), 2
    else:
        return num_spectra, 1


def plot(spectra: Tensor, sources: list[FeatureExtractor]) -> None:
    num_spectra = len(spectra)
    nrows, ncols = calc_plot_shape(num_spectra)
    is_tiled = ncols > 1

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

    for idx in range(num_spectra):
        row = idx % nrows
        col = idx // nrows

        sub_p = axs[row][col] if is_tiled else axs[idx]
        sub_p.imshow(spectra[idx].squeeze(0), origin='lower', aspect='auto')
        sub_p.set_title(f"{sources[idx].get_spec_type().value}/{idx}")

    fig.tight_layout()
    plt.show()


###############################################################################
# ! MAIN
###############################################################################
spectra = []
fc = []

for sample_rate in SAMPLE_RATES:
    # feature_channels = [
    #     FeatureExtractor(sample_rate, n_fft=2048, is_logarithmic=True, is_mel=True),
    #     FeatureExtractor(sample_rate, n_fft=1024, is_logarithmic=True, is_mel=True),
    #     FeatureExtractor(sample_rate, n_fft=512, is_logarithmic=True, is_mel=True),
    # ]
    feature_channels = [
        FeatureExtractor(sample_rate, n_fft=2048, is_logarithmic=True, is_mel=False),
        FeatureExtractor(sample_rate, n_fft=1024, is_logarithmic=True, is_mel=False),
        FeatureExtractor(sample_rate, n_fft=512, is_logarithmic=True, is_mel=False),
    ]
    source = FeatureSource(feature_channels, stack_spectra=False)

    wav = load_input(FILENAME, target_sr=sample_rate, duration_secs=1)

    spectra += source.forward(wav)
    fc += feature_channels

plot(spectra, fc)
