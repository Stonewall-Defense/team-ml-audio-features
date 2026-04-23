###############################################################################
# Local Imports
###############################################################################
from audiofeatures import load_wav, FeatureChannel, FeatureSource, FullRangeStftFeatureSource, HighPassFilter


###############################################################################
# Constants
###############################################################################
SAMPLE_RATE = 44_100
AUDIO_DURATION_SEC = 1

N_FFT = 1024
HOP_LEN = N_FFT // 4
N_MELS = N_FFT // 8


###############################################################################
# Setup
###############################################################################
PREPROCESSORS = [
    HighPassFilter(sample_rate=SAMPLE_RATE, cutoff_freq=7_000, rolloff_db=6),
]

CHANNELS = [
    FeatureChannel(SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LEN, n_filters=N_MELS, is_logarithmic=True, is_mel=True),
    FeatureChannel(SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LEN, n_filters=N_MELS, is_logarithmic=True, is_mel=False)
]
DUAL_CHANNELFEATURE_SOURCE = FeatureSource(CHANNELS, preprocessors=PREPROCESSORS)

FULL_RANGE_FEATURE_SOURCE = FullRangeStftFeatureSource(SAMPLE_RATE, preprocessors=PREPROCESSORS)

AUDIO = load_wav("test/res/380ACP-7-7WYYO4zK0hPS-9.wav", target_sr=SAMPLE_RATE, duration_secs=AUDIO_DURATION_SEC)


###############################################################################
# ! MAIN
###############################################################################
def main():
    specs = DUAL_CHANNELFEATURE_SOURCE.forward(AUDIO)   # Or just DUAL_CHANNELFEATURE_SOURCE(AUDIO)
    print(specs.shape)

    full_range_spec = FULL_RANGE_FEATURE_SOURCE.forward(AUDIO)
    print(full_range_spec.shape)


if __name__ == "__main__":
    main()
