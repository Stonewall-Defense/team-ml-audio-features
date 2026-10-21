###############################################################################
# 3PP Imports
###############################################################################
import torch


###############################################################################
# Functions
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


def has_batch(spec: torch.Tensor):
    return len(spec.shape) == 4


def has_channels(spec: torch.Tensor):
    return len(spec.shape) >= 3


def num_specs(spec: torch.Tensor):
    if has_batch(spec):
        return spec.shape[1]
    elif has_channels(spec):
        return spec.shape[0]
    else:
        return 1
