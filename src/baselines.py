import torch
import torch.nn.functional as F

from src.aux import torch_batch_linspace


__all__ = ['bounded_random_walk']


def bounded_random_walk(start: torch.Tensor, 
                        end: torch.Tensor, 
                        std: float=0.1, 
                        lower_bound: float=-1, 
                        upper_bound: float=1,
                        length: int=512):
    '''
    The dimensions of start and end should match the dimensions of the desired number of unique baselines.
    It is recommended to smooth the baselines before adding to the spectra.
    '''
    size = start.shape
    size[-1] = length
    
    assert (lower_bound <= start and lower_bound <= end)
    assert (start <= upper_bound and end <= upper_bound)

    bounds = upper_bound - lower_bound

    rand = (std * (torch.rand([size]) - 0.5)).cumsum(-1)
    rand_trend = torch_batch_linspace(rand[..., 0].unsqueeze(-1), rand[...,-1].unsqueeze(-1), length)
    rand_deltas = (rand - rand_trend)
    rand_deltas /= torch.clip(torch.max((rand_deltas.max(-1) - rand_deltas.min(-1)) / bounds,
                                        dim=-1, keepdims=True),
                              min=1, max=None).unsqueeze(-1)

    trend_lines = torch_batch_linspace(start, end, length)
    upper_bound_delta = upper_bound - trend_lines 
    lower_bound_delta = lower_bound - trend_lines 

    upper_slips_mask = (rand_deltas - upper_bound_delta) >= 0
    upper_deltas =  rand_deltas - upper_bound_delta
    rand_deltas[upper_slips_mask] = (upper_bound_delta - upper_deltas)[upper_slips_mask]

    lower_slips_mask = (lower_bound_delta-rand_deltas) >= 0
    lower_deltas =  lower_bound_delta - rand_deltas
    rand_deltas[lower_slips_mask] = (lower_bound_delta + lower_deltas)[lower_slips_mask]

    return trend_lines + rand_deltas
