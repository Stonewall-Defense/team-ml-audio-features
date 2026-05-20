###############################################################################
# Global Imports
###############################################################################
import math

###############################################################################
# 3PP Imports
###############################################################################
import torch


###############################################################################
# New Functions
###############################################################################
def power_of_two(n: int):
    return (n & (n - 1) == 0) and n != 0


def scale_spec(spec: torch.Tensor) -> torch.Tensor:
    min_in_val = torch.min(spec)
    max_in_val = torch.max(spec)
    in_span = max_in_val - min_in_val

    min_out_val = torch.zeros(1)
    max_out_val = torch.ones(1)
    out_span = max_out_val - min_out_val

    scale_factor = out_span / in_span
    return (spec - min_in_val) * scale_factor


###############################################################################
# Imported Legacy Code
# This code was extracted from v2.8.0 of [torchaudio](https://github.com/pytorch/audio)
# torchaudio is licensed under the BSD 2-Clause License, reprinted below
###############################################################################
#
# Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
def create_dct(n_cepstrum: int, n_filters: int) -> torch.Tensor:
    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    n = torch.arange(float(n_filters))
    k = torch.arange(float(n_cepstrum)).unsqueeze(1)
    dct = torch.cos(math.pi / float(n_filters) * (n + 0.5) * k)  # size (n_mfcc, n_mels)

    dct[0] *= 1.0 / math.sqrt(2.0)
    dct *= math.sqrt(2.0 / float(n_filters))
    return dct.t()
