###############################################################################
# Global Imports
###############################################################################
from abc import ABC, abstractmethod
import json
from typing import Sequence

###############################################################################
# 3PP Imports
###############################################################################
import torch


###############################################################################
# New Classes
###############################################################################
class AudioPreprocessor(torch.nn.Module, ABC):
    def __init__(self):
        super(AudioPreprocessor, self).__init__()

    @abstractmethod
    def _process(self, wav: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self._process(wav)


class AudioPostprocessor(torch.nn.Module, ABC):
    def __init__(self):
        super(AudioPostprocessor, self).__init__()

    @abstractmethod
    def _process(self, spec: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, specs: torch.Tensor) -> torch.Tensor:
        has_batch = len(specs.shape) == 4
        chan_dim = 1 if has_batch else 0
        n_chans = specs.shape[chan_dim]

        for chan_idx in range(n_chans):
            if has_batch:
                specs[:, chan_idx] = self._process(specs[:, chan_idx])
            else:
                specs[chan_idx] = self._process(specs[:, chan_idx])

        return specs


class BaseFeatureSource(torch.nn.Module, ABC):
    def __init__(self,
                 *,
                 preprocessors: Sequence[AudioPreprocessor] = [],
                 postprocessors: Sequence[AudioPostprocessor] = [],
                 ):
        super(BaseFeatureSource, self).__init__()

        self.preprocessors = preprocessors
        self.postprocessors = postprocessors

    @abstractmethod
    def _make_specs(self, wav: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        for preproc in self.preprocessors:
            wav = preproc(wav)

        spec = self._make_specs(wav)

        for postproc in self.postprocessors:
            spec = postproc(spec)

        return spec


###############################################################################
# New Functions
###############################################################################
def load_params(params: str | list | dict):
    if not isinstance(params, str):
        return params
    else:
        with open(params, "r") as infile:
            return json.loads(infile.read())


def write_params(filename: str, params: list | dict):
    with open(filename, "r") as outfile:
        return outfile.write(json.dumps(params, indent=2))
