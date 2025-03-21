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
from audiofeatures import FeatureSource, FeatureExtractor, MelType, ScalingType


###############################################################################
# Constants
###############################################################################
SAMPLE_RATE = 22050

###############################################################################
# Config
###############################################################################
# feature_channels = [
#     FeatureExtractor(SAMPLE_RATE),
#     FeatureExtractor(SAMPLE_RATE, is_logarithmic=True),
#     FeatureExtractor(SAMPLE_RATE, is_cepstrum=True),
#     FeatureExtractor(SAMPLE_RATE, is_mel=True),
#     FeatureExtractor(SAMPLE_RATE, is_mel=True, is_logarithmic=True),
#     FeatureExtractor(SAMPLE_RATE, is_mel=True, is_cepstrum=True),
# ]

# feature_channels = [
#     FeatureExtractor(SAMPLE_RATE, mel_type=MelType.OSHAUGHNESSY, is_logarithmic=True),
#     FeatureExtractor(SAMPLE_RATE, mel_type=MelType.FANT, is_logarithmic=True),
#     FeatureExtractor(SAMPLE_RATE, mel_type=MelType.LINDSAY_NORMAN, is_logarithmic=True),
#     FeatureExtractor(SAMPLE_RATE, mel_type=MelType.SLANEY, is_logarithmic=True),
# ]

# feature_channels = [
#     FeatureExtractor(SAMPLE_RATE, is_mel=True, is_logarithmic=True, scaling_type=ScalingType.POWER),
#     FeatureExtractor(SAMPLE_RATE, is_mel=True, is_logarithmic=True, scaling_type=ScalingType.MAGNITUDE),
#     FeatureExtractor(SAMPLE_RATE, is_mel=True, is_logarithmic=True, scaling_type=ScalingType.LOG),
# ]

feature_channels = [
    FeatureExtractor(SAMPLE_RATE, n_fft=512, is_mel=True, is_logarithmic=True),
    FeatureExtractor(SAMPLE_RATE, n_fft=1024, is_mel=True, is_logarithmic=True),
    FeatureExtractor(SAMPLE_RATE, n_fft=2048, is_mel=True, is_logarithmic=True),
]

# feature_channels = [
#     FeatureExtractor(SAMPLE_RATE, is_mel=False, is_cepstrum=True),
#     FeatureExtractor(SAMPLE_RATE, is_mel=True, is_cepstrum=True),
# ]

for chan in feature_channels:
    print(chan.get_spec_type().value, end=" ")
print()


###############################################################################
# Helpers
###############################################################################
def calc_plot_shape(num_spectra: int) -> tuple[int, int]:
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
# Main
###############################################################################
# wav = load_input("./dev/0_ryan_2.wav", target_sr=SAMPLE_RATE)
wav = load_input("./dev/gunshot.wav", target_sr=SAMPLE_RATE)
# wav = load_input("./dev/12-18-4.wav", target_sr=SAMPLE_RATE, duration_secs=3)

source = FeatureSource(feature_channels, stack_spectra=False)

spectra = source.forward(wav)
plot(spectra, feature_channels)
