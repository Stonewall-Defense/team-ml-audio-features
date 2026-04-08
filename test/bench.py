###############################################################################
# GLobal Imports
###############################################################################
import time

###############################################################################
# 3PP Imports
###############################################################################
import torch

###############################################################################
# Local Imports
###############################################################################
from audiofeatures.efficient_features import ChannelConfig, EfficientFeatureSource
from audiofeatures.flexible_features import FeatureChannel, FeatureSource
from audiofeatures.wav import load_wav


###############################################################################
# Config
###############################################################################
SAMPLE_RATE = 44_100
N_RUNS = 1_000

###############################################################################
# ! MAIN
###############################################################################
e_channels = [
    ChannelConfig(is_mel=False, is_log=False, is_cepstrum=False),
    ChannelConfig(is_mel=False, is_log=True, is_cepstrum=False),
    ChannelConfig(is_mel=True, is_log=True, is_cepstrum=False),
    ChannelConfig(is_mel=True, is_log=True, is_cepstrum=True),
]
e_source = EfficientFeatureSource(SAMPLE_RATE, e_channels)

f_channels = [
    FeatureChannel(SAMPLE_RATE, is_mel=False, is_logarithmic=False, is_cepstrum=False),
    FeatureChannel(SAMPLE_RATE, is_mel=False, is_logarithmic=True, is_cepstrum=False),
    FeatureChannel(SAMPLE_RATE, is_mel=True, is_logarithmic=True, is_cepstrum=False),
    FeatureChannel(SAMPLE_RATE, is_mel=True, is_logarithmic=True, is_cepstrum=True),
]
f_source = FeatureSource(f_channels)

wav = load_wav("test/res/380ACP-7-7WYYO4zK0hPS-9.wav")

t_e = 0
for _ in range(N_RUNS):
    t_1 = time.time()
    e_specs = e_source.forward(wav)
    t_2 = time.time()
    t_e += t_2 - t_1

t_f = 0
for _ in range(N_RUNS):
    t_1 = time.time()
    f_specs = f_source.forward(wav)
    t_2 = time.time()
    t_f += t_2 - t_1

print(f"Efficient: {t_e:.03f} s / {(t_e/N_RUNS) * 1000:.03f} ms")
print(f"Flexible: {t_f:.03f} s / {(t_f/N_RUNS) * 1000:.03f} ms")

print(torch.allclose(e_specs, f_specs))
