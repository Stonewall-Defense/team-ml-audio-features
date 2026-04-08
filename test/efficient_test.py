###############################################################################
# Certus Imports
###############################################################################
from plot_spec import plot

###############################################################################
# Local Imports
###############################################################################
from audiofeatures.efficient_features import ChannelConfig, EfficientFeatureSource
from audiofeatures.wav import load_wav


###############################################################################
# ! MAIN
###############################################################################
channels = [
    ChannelConfig(is_mel=False, is_log=False, is_cepstrum=False),
    ChannelConfig(is_mel=False, is_log=True, is_cepstrum=False),
    ChannelConfig(is_mel=True, is_log=True, is_cepstrum=False),
    ChannelConfig(is_mel=True, is_log=True, is_cepstrum=True),
]
source = EfficientFeatureSource(44_100, channels)

wav = load_wav("test/res/380ACP-7-7WYYO4zK0hPS-9.wav")
specs = source.forward(wav).squeeze()

plot([specs[idx].squeeze() for idx in range(4)], ["Lin/Lin", "Lin/Log", "Mel/Log", "MFCC"], "Efficient Specs")
