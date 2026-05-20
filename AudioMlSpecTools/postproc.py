###############################################################################
# 3PP Imports
###############################################################################
from maad.sound import remove_background
from scipy.signal import wiener
import torch


###############################################################################
# Local Imports
###############################################################################
from ._util import AudioPostprocessor


###############################################################################
# Classes
###############################################################################
class WienerFilter(AudioPostprocessor):
    def __init__(self):
        super(WienerFilter, self).__init__()

    def _process(self, spec: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(wiener(spec.numpy()))


class RemoveBgSpectralSubtraction(AudioPostprocessor):
    def __init__(self,
                 *,
                 gauss_win: int = 50,
                 gauss_std: int = 25,
                 ):
        super(RemoveBgSpectralSubtraction, self).__init__()

        self.gauss_win = gauss_win
        self.gauss_std = gauss_std

    def _process(self, spec: torch.Tensor) -> torch.Tensor:
        result, _, _ = remove_background(spec.squeeze().numpy(), self.gauss_win, self.gauss_std)
        return torch.from_numpy(result)
