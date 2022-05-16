import numpy as np
import torch
import torch.nn.functional as F

from src.aux import np_batch_linspace


__all__ = ['bounded_random_walk']


def bounded_random_walk(start: torch.Tensor, 
                        end: torch.Tensor, 
                        std: float=0.1, 
                        lower_bound: float=-1, 
                        upper_bound: float=1,
                        length: int=512):
    start, end = start.numpy(), end.numpy()
    batchSize = start.shape[0]
    
    assert (lower_bound <= start and lower_bound <= end)
    assert (start <= upper_bound and end <= upper_bound)

    bounds = upper_bound - lower_bound

    rand = (std * (np.random.random([batchSize, 1, length]) - 0.5)).cumsum(-1)
    rand_trend = np.expand_dims(np_batch_linspace(rand[...,0], rand[...,-1], np.asarray(length)), 1)
    rand_deltas = (rand - rand_trend)
    rand_deltas /= np.expand_dims(np.clip(np.max((rand_deltas.max(-1) - rand_deltas.min(-1)) / bounds,
                                                 axis=-1, keepdims=True),
                                          a_min=1, a_max=None), 
                                  axis=-1)

    trend_line = np.expand_dims(np_batch_linspace(start, end, np.asarray(length)), (0,1))
    upper_bound_delta = upper_bound - trend_line #np.repeat(upper_bound - trend_line, batchSize, axis=0)
    lower_bound_delta = lower_bound - trend_line #np.repeat(lower_bound - trend_line, batchSize, axis=0)

    upper_slips_mask = (rand_deltas - upper_bound_delta) >= 0
    upper_deltas =  rand_deltas - upper_bound_delta
    rand_deltas[upper_slips_mask] = (upper_bound_delta - upper_deltas)[upper_slips_mask]

    lower_slips_mask = (lower_bound_delta-rand_deltas) >= 0
    lower_deltas =  lower_bound_delta - rand_deltas
    rand_deltas[lower_slips_mask] = (lower_bound_delta + lower_deltas)[lower_slips_mask]

    return torch.from_numpy(trend_line + rand_deltas, dtype=torch.float32)
