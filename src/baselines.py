import torch
import torch.nn.functional as F

from .aux import batch_linspace

__all__ = ['bounded_random_walk']


def bounded_random_walk(start: torch.Tensor, 
                        end: torch.Tensor, 
                        std: (float,torch.Tensor)=0.1, 
                        lower_bound: float=-1, 
                        upper_bound: float=1,
                        length: int=512):
    '''
    Code modified from: 
    https://stackoverflow.com/questions/46954510/random-walk-series-between-start-end-values-and-within-minimum-maximum-limits
    The dimensions of start and end should match the dimensions of the 
    desired number of unique baselines. It is recommended to smooth the 
    baselines before adding to the spectra.
    '''
    size = list([d for d in start.shape])
    size[-1] = length

    if isinstance(std, float):
        std = torch.as_tensor(std)
    if isinstance(std, torch.Tensor): 
        for _ in range(start.ndim - std.ndim): std = std.unsqueeze(-1)
    
    assert ((lower_bound <= start).all() and (lower_bound <= end).all())
    assert ((start <= upper_bound).all() and (end <= upper_bound).all())

    bounds = upper_bound - lower_bound

    rand = (std * (torch.rand(tuple(size)) - 0.5)).cumsum(-1)
    rand_trend = batch_linspace(rand[..., 0].unsqueeze(-1), 
                                rand[...,-1].unsqueeze(-1), length)
    rand_deltas = (rand - rand_trend)
    rand_deltas /= torch.clamp(torch.max((rand_deltas.amax(-1) - \
                                          rand_deltas.amin(-1)).unsqueeze(-1) / bounds, 
                                         dim=-1, keepdims=True).values,
                               min=1, max=None)
    
    trend_lines = batch_linspace(start, end, length)
    upper_bound_delta = upper_bound - trend_lines 
    lower_bound_delta = lower_bound - trend_lines 

    upper_slips_mask = (rand_deltas - upper_bound_delta) >= 0
    upper_deltas =  rand_deltas - upper_bound_delta
    rand_deltas[upper_slips_mask] = (upper_bound_delta - upper_deltas)[upper_slips_mask]

    lower_slips_mask = (lower_bound_delta-rand_deltas) >= 0
    lower_deltas =  lower_bound_delta - rand_deltas
    rand_deltas[lower_slips_mask] = (lower_bound_delta + lower_deltas)[lower_slips_mask]

    return trend_lines + rand_deltas
