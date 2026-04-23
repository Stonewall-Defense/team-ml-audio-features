###############################################################################
# 3PP Imports
###############################################################################
import torch

###############################################################################
# Testing Imports
###############################################################################
import unittest

###############################################################################
# Local Imports
###############################################################################
from audiofeatures.flexible_features import FeatureChannel, FeatureSource
from audiofeatures.stft_features import FullRangeStftFeatureSource
from audiofeatures.wav import load_wav

###############################################################################
# Config
###############################################################################
# Constants
SAMPLE_RATE = 44_100
N_RUNS = 100


# Audio data
wav = load_wav("test/res/380ACP-7-7WYYO4zK0hPS-9.wav")

# Standard to compare
channels = [
    FeatureChannel(SAMPLE_RATE, is_mel=False, is_logarithmic=True, is_cepstrum=False, n_filters=513),
]
ref_source = FeatureSource(channels)

# New code
stft_source = FullRangeStftFeatureSource(SAMPLE_RATE)


###############################################################################
# Tests
###############################################################################
class TestStft(unittest.TestCase):
    def test_results(self):
        new_features = stft_source.forward(wav)
        ref_features = ref_source.forward(wav).squeeze(0)
        is_close = torch.all(torch.cosine_similarity(new_features, ref_features) > 0.93)
        self.assertTrue(is_close)
