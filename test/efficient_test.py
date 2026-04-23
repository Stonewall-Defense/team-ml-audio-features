###############################################################################
# Global Imports
###############################################################################
import time

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
from audiofeatures.efficient_features import ChannelConfig, EfficientFeatureSource
from audiofeatures.flexible_features import FeatureChannel, FeatureSource
from audiofeatures.wav import load_wav


###############################################################################
# Config
###############################################################################
# Constants
SAMPLE_RATE = 44_100
N_RUNS = 100


# Audio data
wav = load_wav("test/res/380ACP-7-7WYYO4zK0hPS-9.wav")


# Efficient Features
e_channels = [
    ChannelConfig(is_mel=False, is_log=False, is_cepstrum=False),
    ChannelConfig(is_mel=False, is_log=True, is_cepstrum=False),
    ChannelConfig(is_mel=True, is_log=True, is_cepstrum=False),
    ChannelConfig(is_mel=True, is_log=True, is_cepstrum=True),
]
e_source = EfficientFeatureSource(SAMPLE_RATE, e_channels)


# Flexible Features
f_channels = [
    FeatureChannel(SAMPLE_RATE, is_mel=False, is_logarithmic=False, is_cepstrum=False),
    FeatureChannel(SAMPLE_RATE, is_mel=False, is_logarithmic=True, is_cepstrum=False),
    FeatureChannel(SAMPLE_RATE, is_mel=True, is_logarithmic=True, is_cepstrum=False),
    FeatureChannel(SAMPLE_RATE, is_mel=True, is_logarithmic=True, is_cepstrum=True),
]
f_source = FeatureSource(f_channels)


###############################################################################
# Tests
###############################################################################
class TestBench(unittest.TestCase):
    def test_results(self):
        e_features = e_source.forward(wav)
        f_features = f_source.forward(wav)
        is_close = torch.allclose(e_features, f_features)
        self.assertTrue(is_close)

    def test_speed(self):
        t_e = 0
        for _ in range(N_RUNS):
            t_1 = time.time()
            e_source.forward(wav)
            t_2 = time.time()
            t_e += t_2 - t_1

        t_f = 0
        for _ in range(N_RUNS):
            t_1 = time.time()
            f_source.forward(wav)
            t_2 = time.time()
            t_f += t_2 - t_1

        self.assertLess(t_e, t_f)
