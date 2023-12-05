import copy
import math
import os
from collections import OrderedDict
import json

import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.interpolate import CubicHermiteMAkima as CubicHermiteInterp
# from pm_v3 import PhysicsModel
from torch.fft import fft, fftshift, ifft, ifftshift, irfft, rfft
from types import SimpleNamespace

__all__ = ['batch_linspace', 'batch_smooth', 'complex_exp', 'concat_dict', 
           'convertdict', 'counter', 'dict2tensors', 'Fourier_Transform', 
           'HilbertTransform', 'inv_Fourier_Transform', 'normalize', 
           'OrderOfMagnitude', 'rand_omit', 'sample_baselines', 
           'sample_resWater', 'sim2acquired', 'sort_parameters', 
           'torch2numpy', 'unwrap']


PI = torch.from_numpy(np.asarray(np.pi)).squeeze().float()


# Only works with 3 dimensions. Is it used with transients at any point?
def batch_linspace(start: torch.Tensor, 
                   stop: torch.Tensor, 
                   steps: int) -> torch.Tensor:
    stop = stop.unsqueeze(-1) if stop.ndim==2 else stop
    for _ in range(3-start.ndim): start = start.unsqueeze(-1)
    start = start.expand_as(stop).to(stop.device)
    out = torch.arange(0, steps).expand([int(start.numel()), 1, 
                    int(steps)]).to(stop.device).clone().float()
    delta = stop.clone() - start.clone()
    out *= delta.float()
    out /= (steps - 1)
    out += start
    out[...,-1] = stop[...,-1]
    return out


def batch_smooth(x: torch.Tensor,
                 window_len: torch.Tensor,
                 window: str='flat',
                 mode: str='reflect'
                ) -> torch.Tensor:
    assert(x.ndim==3)
    x = x.permute(1,0,2)
    if isinstance(window_len, float): 
        window_len = torch.as_tensor(window_len)
        for _ in range(x.ndim - window_len.ndim): 
            window_len = window_len.unsqueeze(0)
        window_len = window_len.repeat(x.shape[1], 1, 1)

    w_len = (x.shape[-1] * window_len).int()
    mx, threshold = w_len.max().item(), torch.div(w_len.squeeze(), 2, 
                                                    rounding_mode='floor')
    even = mx if mx % 2 == 0 else mx + 1

    w = torch.zeros((1, mx), dtype=torch.float32).repeat(x.shape[1], 1)
    for i in range(even//2):
        ind = (i<threshold)
        w[ind, i] = 1
    w = w.roll(even//2, -1)
    w = w + w.flip(-1)
    w[(w==2)] = 1
    w /= w.sum(dim=-1, keepdims=True)
    w = w.unsqueeze(1)

    out = F.conv1d(input=F.pad(x, (even, even), mode=mode), 
                   weight=w, groups=w.shape[0]).permute(1,0,2)
    start, stop = even//2, -even//2-1 if even==mx else -even//2-2
    out = out[...,start:stop]

    return out


def complex_exp(signal: torch.Tensor, 
                theta: torch.Tensor, 
                real: bool=False) -> torch.Tensor:
    '''
    complex_exp expects a complex exponent. If dim=1 has only size=1, it is 
    assumed to be the imaginary component. Therefore, the real component is 
    assumed to be zero. If a real component should be used, it must be 
    supplied at index = 0 along dimension 1.
    '''
    if real:
        theta = theta if theta.shape[-2]==2 else torch.cat([
                    theta, torch.zeros_like(theta)], dim=-2)
    else:
        theta = theta if theta.shape[-2]==2 else torch.cat([
                    torch.zeros_like(theta), theta], dim=-2)
    
    real = signal[...,0,:].mul(torch.cos(theta[...,1,:])) - \
            signal[...,1,:].mul(torch.sin(theta[...,1,:]))
    imag = signal[...,0,:].mul(torch.sin(theta[...,1,:])) + \
            signal[...,1,:].mul(torch.cos(theta[...,1,:]))

    return torch.cat([real.unsqueeze(-2), imag.unsqueeze(-2)], dim=-2)


def concat_dict(target: dict,
                new: dict,
               ) -> np.ndarray:
    assert([(key in target.keys().lower()) for key in new.keys().lower()])
    for k, v in new.items():
        target[k] = np.concatenate([target[k], v], axis=0)
    return target


def convertdict(file, simple=False, device='cpu'):
    delete = []
    for k, v in file.items():
        if not k.startswith('__') and not ('None' in k):
            if isinstance(v, dict):
                convertdict(file[k], device)
            elif k=='linenames':
                file[k] = dict({str(a): torch.from_numpy(np.asarray(b, 
                    dtype=np.float32)).to(device) for a, b in zip(file[k][0,:], 
                                                                file[k][1,:])})
            elif k in ['notes', 'seq','vendor','pulse_sequence']:
                delete.append(k)
            else:
                file[k] = torch.from_numpy(np.asarray(file[k], 
                                    dtype=np.float32)).squeeze().to(device)
        else:
            delete.append(k)
    if len(delete)>0:
        for k in delete:
            file.pop(k, None)

    return file if not simple else SimpleNamespace(**file)


class counter():
    def __init__(self, start=0):
        super().__init__()
        self.start = copy.copy(start)
        self._count = start

    def __call__(self, num: int=1):
        self._count += num
        return self._count

    def __repr__(self):
        return str(self._count)

    def __enter__(self):
        # self._count += 1
        return int(self._count)

    def reset(self):
        self._count = copy.copy(self.start)


def dict2tensors(dct: dict) -> dict:
    for key, value in dct.items():
        if isinstance(value, dict):
            dict2tensors(value)

                      
def Fourier_Transform(signal: torch.Tensor) -> torch.Tensor: 
    assert(signal.ndim>=3)
    signal = signal.transpose(-1,-2)
    assert(signal.shape[-1]==2)
    signal = torch.view_as_complex(signal.contiguous())
    signal = torch.view_as_real(fftshift(fft(signal, dim=-1), 
                                dim=-1)).transpose(-1,-2)
    return signal.contiguous()


def HilbertTransform(data: torch.Tensor, 
                     dim: int=-1) -> torch.Tensor:  # checked 10.10.21
    '''
    Confirmed to match scipy.linalg.hilbert and Matlab's hilbert 
    implementations.
    Suggestion from Fern: 
        Remove the mean before the transform, can add it back later
    https://dsp.stackexchange.com/questions/46291/why-is-scipy-implementation-
        of-hilbert-function-different-from-matlab-implemen
    Added nested copies of the (inverse) Fourier transfrom because this code 
    does not use (i)fftshift.
    
    TODO: check in a notebook to make sure lo de la media works well.
    '''
    assert(data.ndim>=3)
    def Fourier(data): return fft(data, dim=-1)
    def invFourier(data): return ifft(data, dim=-1)

    N = data.shape[dim]
    mn = torch.mean(data, dim=-1, keepdim=True)
    Xf = Fourier(data - mn)
    h = torch.zeros_like(Xf)
    if N % 2 == 0:
        h[..., (0,N // 2)] = 1
        h[..., 1:N // 2] = 2
    else:
        h[..., 0] = 1
        h[..., 1:(N + 1) // 2] = 2

    a = Xf.mul(h)
    out = invFourier(a) + mn

    return torch.cat([out.real, out.imag], dim=-2)


def inv_Fourier_Transform(signal: torch.Tensor, 
                          normalize: bool=False) -> torch.Tensor:
    assert(signal.ndim>=3)
    signal = signal.transpose(-1,-2)
    assert(signal.shape[-1]==2)

    signal = torch.view_as_complex(signal.contiguous())
    signal = torch.view_as_real(ifft(ifftshift(signal, dim=-1),
                                     dim=-1)).transpose(-1,-2)
    return signal.contiguous()


def normalize_old(tensor: torch.Tensor, dims=tuple) -> torch.Tensor:
    denom = torch.amax(tensor, dim=dims, keepdims=True).values - \
                torch.amin(tensor, dim=dims, keepdims=True).values
    return (tensor - torch.amin(tensor.abs(), dim=dims, 
                                keepdims=True).values) / denom


# Source: https://stackoverflow.com/questions/52859751/most-efficient-way-to-find-order-of-magnitude-of-float-in-python
def OrderOfMagnitude(data: torch.Tensor,
                     eps: float=1e-8):
    data = data.abs()
    for d in range(data.ndim-1,0,-1):
        data = data.amax(d, keepdims=True)

    return torch.floor(torch.log10(data+eps))
    

def prepareConfig(N: int, cfg: dict) -> dict:#, pt_density: int):
    # try: cfg['pt_density']
    # except: cfg['pt_density'] = pt_density
    # print('N = ',N)
    length = round((cfg['ppm_range'][1]-cfg['ppm_range'][0]) * \
                   cfg['pt_density'])
    if length % 2 != 0: length += 1
    if len(cfg['start'])==1: cfg['start'] = [cfg['start'][0], cfg['start'][0]]
    if len(cfg['end'])==1: cfg['end'] = [cfg['end'][0], cfg['end'][0]]
    if len(cfg['std'])==1: cfg['std'] = [cfg['std'][0], cfg['std'][0]]
    if len(cfg['upper'])==1: cfg['upper'] = [cfg['upper'][0], cfg['upper'][0]]
    if len(cfg['lower'])==1: cfg['lower'] = [cfg['lower'][0], cfg['lower'][0]]
    if len(cfg['window'])==1: cfg['window'] = [cfg['window'][0], cfg['window'][0]]
    if len(cfg['scale'])==1: cfg['scale'] = [cfg['scale'][0], cfg['scale'][0]]
    cfg['ppm_range'] = [torch.as_tensor([val]) for val in cfg['ppm_range']]

    return {
        'start': torch.zeros(N,1,1).uniform_(cfg['start'][0],
                                             cfg['start'][1]),
        'end': torch.zeros(N,1,1).uniform_(cfg['end'][0],
                                           cfg['end'][1]),
        'std': torch.zeros(N,1,1).uniform_(cfg['std'][0],
                                           cfg['std'][1]),
        'upper_bnd': torch.ones(N,1,1).uniform_(cfg['upper'][0],
                                                cfg['upper'][1]),
        'lower_bnd': torch.ones(N,1,1).uniform_(cfg['lower'][0],
                                                cfg['lower'][1]),
        'windows': torch.ones(N,1,1).uniform_(cfg['window'][0],
                                              cfg['window'][1]),
        'length': length,
        'ppm_range': cfg['ppm_range'],
        'scale': torch.ones(N,1,1).uniform_(cfg['scale'][0],
                                            cfg['scale'][1]),
        'rand_omit': cfg['drop_prob'],
    }


def rand_omit(data: torch.Tensor, value: float=0., p: float=0.2):
    assert(data.ndim>=2)
    sign = torch.tensor([True if torch.rand([1]) > (1.0 - p) else \
                            False for _ in range(data.shape[0])])
    data[sign,...] = value
    return data, sign


def load_parameters(path: str, prepare: tuple):
    assert os.path.isfile(path)
    with open(path,'rb') as file:
        params = io.loadmat(file, variable_names='params')['params']

    out = list(prepare)
    out = out.append(params)

    return out


def load_default_values(path: str, pm: nn.Module, quant: str="T2", spins: bool=False):
    assert quant.lower() in ["t2", "conc"]
    key = "spins" if spins else "metab"
    with open(path) as file:
        values = json.load(file)
    
    num_spins = pm._num_spins
    end = len(pm.spins) - 1
    
    minimum = torch.zeros(end)
    maximum = torch.zeros(end)

    metabs, _ = pm.metab

    for i, ind in enumerate(range(0,end,num_spins)):
        met = values[metabs[i]][quant][key]
        for n in range(0,len(met["min"])):
            minimum[:,ind+n] = met["min"][n]
            maximum[:,ind+n] = met["max"][n]
    return minimum, maximum


def smooth(x: torch.Tensor,
           window_len: float=0.1,
           window: str='flat') -> torch.Tensor:
    assert(x.ndim==3)
    w_len = int(x.shape[-1] * window_len)
    w = torch.ones((1, 1, w_len), dtype=torch.float32)
    w /= w.sum()
    out = F.conv1d(input=F.pad(x, (w_len, w_len), mode='reflect'), weight=w)
    out = out[...,w_len//2:-w_len//2-1]
    assert(x.shape==out.shape)
    return out


def sample_baselines(N: int, **cfg): 
    if not isinstance(cfg, type(None)):
        try: cfg['pt_density']
        except: cfg['pt_density'] = 128
        dct = prepareConfig(N=N, cfg=cfg)#, pt_density=cfg['pt_density'])
        return dct
    return None


def sample_resWater(N: int, **cfg):
    if not isinstance(cfg, type(None)):
        try: cfg['pt_density']
        except: cfg['pt_density'] = 1204
        dct = prepareConfig(N=N, cfg=cfg)#, pt_density=cfg['pt_density'])
        # start, _ = rand_omit(torch.zeros(N,1,1).uniform_(0,cfg['prime']), 
        #                      0.0, cfg['drop_prob'])
        # end, _   = rand_omit(torch.zeros(N,1,1).uniform_(0,cfg['prime']), 
        #                      0.0, cfg['drop_prob'])
        start, _ = rand_omit(torch.zeros(N,1,1).uniform_(-1*cfg['prime'],
                                                         cfg['prime']), 
                             0.0, cfg['drop_prob'])
        end, _   = rand_omit(torch.zeros(N,1,1).uniform_(-1*cfg['prime'],
                                                         cfg['prime']), 
                             0.0, cfg['drop_prob'])
        dct.update({#'cropRange_resWater': cfg['cropRange_water'],
                    'start_prime': start,
                    'end_prime': end})

        return dct
    return None


def sim2acquired(line: torch.Tensor, 
                 sim_range: list, 
                 target_ppm: torch.Tensor,
                ) -> torch.Tensor:
    '''
    This approach uses nonuniform sampling density to reduce the memory 
    footprint. This is possible because the tails being padded are always 
    zero. Having 10e1 or 10e10 zeros gives the same result. So, small tails 
    are padded to the input, 
    '''
    raw_ppm = [target_ppm.amin(), target_ppm.amax()]
    if target_ppm.amin(keepdims=True)[0]!=line.shape[0]: 
        raw_ppm = [raw_ppm[0].repeat(line.shape[0]), 
                   raw_ppm[1].repeat(line.shape[0])]
    if sim_range[0].shape[0]!=line.shape[0]: 
        sim_range[0] = sim_range[0].repeat_interleave(line.shape[0], dim=0)
    if sim_range[1].shape[0]!=line.shape[0]: 
        sim_range[1] = sim_range[1].repeat_interleave(line.shape[0], dim=0)
    for _ in range(3 - sim_range[0].ndim): 
        sim_range[0] = sim_range[0].unsqueeze(-1)
    for _ in range(3 - sim_range[1].ndim): 
        sim_range[1] = sim_range[1].unsqueeze(-1)
    for _ in range(3 - raw_ppm[0].ndim): 
        raw_ppm[0] = raw_ppm[0].unsqueeze(-1)
    for _ in range(3 - raw_ppm[1].ndim): 
        raw_ppm[1] = raw_ppm[1].unsqueeze(-1)

    pad = 100 # number of points added to each side
    pad_left, pad_right = 0, 0
    
    # Middle side
    xaxis = batch_linspace(sim_range[0], sim_range[1], int(line.shape[-1]))

    # Left side
    if (raw_ppm[0]<sim_range[0]).all():
        xaxis = torch.cat([batch_linspace(raw_ppm[0], sim_range[0], 
                                          pad+1)[...,:-1], 
                           xaxis], dim=-1) 
        pad_left = pad
   
    # Right side
    if (raw_ppm[1]>sim_range[1]).all():
        xaxis = torch.cat([xaxis, batch_linspace(sim_range[1], raw_ppm[1], 
                                                 pad+1)[...,1:]], dim=-1) 
        pad_right = pad

    padding = tuple([pad_left, pad_right])
    signal = torch.nn.functional.pad(input=line, pad=padding, 
                                     mode="constant", value=0)
    #signal = batch_smooth(x=signal, window_len=float(3.0/signal.shape[-1]))
    
    ch_interp = CubicHermiteInterp(xaxis=xaxis, signal=signal)
    return ch_interp.interp(xs=target_ppm)


def sort_parameters(params: torch.Tensor,
                    index: dict,
                   ) -> dict:
    parameters = OrderedDict()
    for i, k in enumerate(index.keys()):
        if not k in ['metabolites', 'parameters', 'overall']:
            parameters.update({k: params[:,index[k]]})
    return parameters


def torch2numpy(input: dict):
    if isinstance(input, dict):
        for k, v in input.items():
            if isinstance(input[k], dict):
                torch2numpy(input[k])
            elif torch.is_tensor(v):
                input[k] = v.numpy()
            else:
                pass
    return input


def unwrap(phase: torch.Tensor, 
           period: torch.Tensor=2*PI, 
           dim: int=-1,
          ) -> torch.Tensor:
    ind = [slice(None, None)]*phase.ndim
    ind[dim] = slice(1, None)
    dd = torch.diff(phase, dim=dim)
    dtype = torch.result_type(dd, period)
    interval_high = period / 2
    interval_low = -interval_high
    ddmod = torch.remainder(dd - interval_low, period) + interval_low
    ddmod = torch.where((ddmod==interval_low) & (dd>0), interval_high, ddmod)
    ph_correct = torch.where(torch.abs(dd) < (period / 2), 
                             torch.tensor(0.), ddmod - dd)
    out = phase
    out[ind] = phase[ind] + ph_correct.cumsum(dim=dim)
    return out



def normalize(signal: torch.Tensor,
              fid: bool=False,
              denom: torch.Tensor=None,
              noisy: int=-3, # dim for noisy/clean
             ) -> torch.Tensor:
    '''
    Normalize each sample of single or multi-echo spectra. 
       Step 1: Find the max of the real and imaginary components separately
       Step 2: Pick the larger value for each spectrum
    If the signal is separated by metabolite, then an additional max() 
       is necessary
    Reimplemented according to: https://stackoverflow.com/questions/4157653
       6/normalizing-complex-values-in-numpy-python
    '''
    if isinstance(denom, type(None)):
        if signal.shape[-2]==1:
            denom = torch.amax(signal.abs(), dim=-1, keepdim=True)#.values
            # print(type(denom))
        elif signal.shape[-2]==2:
            denom = torch.amax(torch.sqrt(signal[...,0,:]**2 + 
                                          signal[...,1,:]**2), 
                               dim=-1, keepdim=True)#.values
            # print('denom.shape ',denom.shape)
            denom = denom.unsqueeze(-2)
        else:
            denom = torch.amax(signal[...,2,:].unsqueeze(-2),#.abs(), 
                               dim=-1, keepdim=True)#.values

        denom[denom.isnan()] = 1e-6
        denom[denom==0.0] = 1e-6
        denom = torch.amax(denom, dim=noisy, keepdim=True)

    for _ in range(denom.ndim-signal.ndim): signal = signal.unsqueeze(1)

    return signal / denom, denom

