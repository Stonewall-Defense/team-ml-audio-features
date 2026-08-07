###############################################################################
# Global Imports
###############################################################################
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
import click

###############################################################################
# Certus Imports
###############################################################################
from audio_tensor_plotter import plot_with_time_domain

###############################################################################
# Local Imports
###############################################################################
from AudioMlSpecTools import load_wav, FeatureChannel, FeatureSource, HighPassFilter, LowPassFilter


###############################################################################
# Constants
###############################################################################
SAMPLE_RATE = 44_100

N_FFT = 1024
HOP_LEN = 256
N_MELS = 128


###############################################################################
# Config
###############################################################################
FEATURE_EXTRACTOR = FeatureSource(
    [
        FeatureChannel(SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LEN, n_filters=N_MELS, is_logarithmic=True, is_mel=True),
        FeatureChannel(SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LEN, n_filters=N_MELS, is_logarithmic=True, is_mel=False),
    ],
    preprocessors=[
        # HighPassFilter(sample_rate=SAMPLE_RATE, cutoff_freq=7_000, rolloff_db=6),
        LowPassFilter(sample_rate=SAMPLE_RATE, cutoff_freq=3_000, rolloff_db=6),
    ],
    postprocessors=[]
)


###############################################################################
# ! MAIN
###############################################################################
@click.command()
@click.argument("filename", required=True)
@click.option("--duration_secs", "-d", type=int)
def main(filename: str, duration_secs: Optional[int]):
    wav, _ = load_wav(filename, target_sr=SAMPLE_RATE, duration_secs=duration_secs, mono=True, pad=True)
    specs = FEATURE_EXTRACTOR(wav).squeeze(0)
    plot_with_time_domain(specs, wav, SAMPLE_RATE, "Filt 1st Order @ 3 kHz cutoff")
    # plot_with_time_domain(specs, wav, SAMPLE_RATE, "Baseline")


if __name__ == "__main__":
    main()
