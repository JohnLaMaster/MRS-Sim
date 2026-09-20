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
    # BUGFIX (v2.0, handover section 13): start/end must be at least 3-D
    # (matching every real call site's (batch, 1, 1) convention -- e.g.
    # PhysicsModel.baselines()/residual_water()) for this function's
    # internal arithmetic to stay consistent with batch_linspace()'s own
    # shape-promotion (src/aux/aux.py, unsqueezes a 2-D `stop` to 3-D). A
    # 2-D (batch, 1) start/end used to silently produce a cross-broadcast
    # (batch, batch, length) result instead of (batch, 1, length): `rand`
    # (computed directly here) stayed at the input's original ndim while
    # `rand_trend`/`trend_lines` (via batch_linspace) were promoted to one
    # more dimension, so subtracting them broadcast the batch axis against
    # the wrong dimension. Normalized here (not inside batch_linspace()
    # itself, a shared utility other call sites already rely on as-is).
    while start.ndim < 3: start = start.unsqueeze(-1)
    while end.ndim < 3: end = end.unsqueeze(-1)

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
