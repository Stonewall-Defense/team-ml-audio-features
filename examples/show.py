###############################################################################
# 3PP Imports
###############################################################################
import click
import torch

###############################################################################
# Certus Imports
###############################################################################
from audio_tensor_plotter import plot_with_time_domain

###############################################################################
# Local Imports
###############################################################################
from AudioMlSpecTools import load_wav, FeatureChannel, FeatureSource, HighPassFilter

###############################################################################
# Constants
###############################################################################
SAMPLE_RATE = 44_100
AUDIO_DURATION_SEC = 1

N_FFT = 1024
HOP_LEN = 256
N_MELS = 128


###############################################################################
# Config
###############################################################################
FEATURE_EXTRACTOR = FeatureSource(
    [
        FeatureChannel(SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LEN, n_filters=N_MELS, is_logarithmic=True, is_mel=True, nan_to_zero=True),
        FeatureChannel(SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LEN, n_filters=N_MELS, is_logarithmic=True, is_mel=False, nan_to_zero=True),
    ],
    preprocessors=[
        HighPassFilter(sample_rate=SAMPLE_RATE, cutoff_freq=7_000, rolloff_db=6),
    ],
    postprocessors=[]
)


###############################################################################
# ! MAIN
###############################################################################
@click.command()
@click.argument("filename", required=True)
def main(filename: str):
    wav, _ = load_wav(filename, target_sr=SAMPLE_RATE, duration_secs=AUDIO_DURATION_SEC, mono=True, pad=True)
    specs = FEATURE_EXTRACTOR(wav).squeeze(0)
    print(specs.shape, specs)
    plot_with_time_domain(specs, wav, SAMPLE_RATE, "Example")


if __name__ == "__main__":
    main()
