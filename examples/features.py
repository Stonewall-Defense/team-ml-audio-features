###############################################################################
# Local Imports
###############################################################################
from AudioMlSpecTools import load_wav, plot_with_time_domain, FlexibleFeatureChannel, FlexibleFeatureSource, FullRangeStftFeatureSource, HighPassFilter, RemoveBgSpectralSubtraction


###############################################################################
# Constants
###############################################################################
SAMPLE_RATE = 44_100
AUDIO_DURATION_SEC = 1

N_FFT = 1024


###############################################################################
# Setup
###############################################################################
PREPROCESSORS = [
    HighPassFilter(sample_rate=SAMPLE_RATE, cutoff_freq=7_000, rolloff_db=6),
]

POSTPROCESSORS = [
    RemoveBgSpectralSubtraction()
]

CHANNELS = [
    FlexibleFeatureChannel(SAMPLE_RATE, n_fft=N_FFT, is_logarithmic=True, is_mel=True),
    FlexibleFeatureChannel(SAMPLE_RATE, n_fft=N_FFT, is_logarithmic=True, is_mel=False)
]
DUAL_CHANNEL_FEATURE_SOURCE = FlexibleFeatureSource(CHANNELS, preprocessors=PREPROCESSORS, postprocessors=POSTPROCESSORS)
DUAL_CHANNEL_FEATURE_SOURCE_BG = FlexibleFeatureSource(CHANNELS, preprocessors=PREPROCESSORS)

FULL_RANGE_FEATURE_SOURCE = FullRangeStftFeatureSource(SAMPLE_RATE, preprocessors=PREPROCESSORS, postprocessors=POSTPROCESSORS)

AUDIO, _ = load_wav("test/res/380ACP-7-7WYYO4zK0hPS-9.wav", target_sr=SAMPLE_RATE, duration_secs=AUDIO_DURATION_SEC)


###############################################################################
# ! MAIN
###############################################################################
def main():
    specs = DUAL_CHANNEL_FEATURE_SOURCE.forward(AUDIO)   # Or just DUAL_CHANNEL_FEATURE_SOURCE(AUDIO)
    print(specs.shape)
    plot_with_time_domain(specs, AUDIO, SAMPLE_RATE, "DUAL")

    specs = DUAL_CHANNEL_FEATURE_SOURCE_BG.forward(AUDIO)
    print(specs.shape)
    plot_with_time_domain(specs, AUDIO, SAMPLE_RATE, "With BG")

    full_range_spec = FULL_RANGE_FEATURE_SOURCE.forward(AUDIO)
    print(full_range_spec.shape)
    plot_with_time_domain(full_range_spec, AUDIO, SAMPLE_RATE, "FULL")


if __name__ == "__main__":
    main()
