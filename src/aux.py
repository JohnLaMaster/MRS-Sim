import copy
import math
import os
from collections import OrderedDict

import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, fftshift, ifft, ifftshift, irfft, rfft
from types import SimpleNamespace

__all__ = ['complex_exp', 'convertdict', 'Fourier_Transform', 'HilbertTransform', 
           'inv_Fourier_Transform', 'batch_smooth', 'torch_batch_linspace', 'counter',
           'rand_omit','OrderOfMagnitude','sample_resWater','sample_baselines',
           'torch2numpy','sort_parameters','concat_dict']


def complex_exp(signal: torch.Tensor, 
                theta: torch.Tensor, 
                real: bool=False) -> torch.Tensor:
    '''
    complex_exp expects a complex exponent. If dim=1 has only size=1, it is assumed to be the imaginary component.
    Therefore, the real component is assumed to be zero. If a real component should be used, it must be supplied
    at index = 0 along dimension 1.
    '''
    theta = theta if theta.shape[-2]==2 else torch.cat([torch.zeros_like(theta), theta], dim=-2)
    
    if real: 
        real = signal.mul(torch.cos(theta[...,1,:].unsqueeze(1)))
        imag = signal.mul(torch.sin(theta[...,1,:].unsqueeze(1)))
        return torch.cat([real, imag], dim=-2).mul(torch.exp(theta[...,0,:].unsqueeze(1)).expand_as(signal))
    
    real = signal[...,0,:].mul(torch.cos(theta[...,1,:])) - signal[...,1,:].mul(torch.sin(theta[...,1,:]))
    imag = signal[...,0,:].mul(torch.sin(theta[...,1,:])) + signal[...,1,:].mul(torch.cos(theta[...,1,:]))

    return torch.cat([real.unsqueeze(-2), imag.unsqueeze(-2)], dim=-2)


def convertdict(file, simple=False, device='cpu'):
    delete = []
    for k, v in file.items():
        if not k.startswith('__') and not ('None' in k):
            if isinstance(v, dict):
                convertdict(file[k], device)
            elif k=='linenames':
                file[k] = dict({str(a): torch.from_numpy(np.asarray(b, dtype=np.float32)).to(device) for a, b in zip(file[k][0,:], file[k][1,:])})
            elif k in ['notes', 'seq','vendor']:
                delete.append(k)
            else:
                file[k] = torch.from_numpy(np.asarray(file[k], dtype=np.float32)).squeeze().to(device)
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


def rand_omit(data: torch.Tensor, value: float=0., p: float=0.2):
    assert(data.ndim>=2)
    sign = torch.tensor([True if torch.rand([1]) > (1.0 - p) else False for _ in range(data.shape[0])])
    data[sign,...] = value
    return data, sign


def Fourier_Transform(signal: torch.Tensor) -> torch.Tensor: 
    assert(signal.ndim>=3)
    signal = signal.transpose(-1,-2)
    assert(signal.shape[-1]==2)
    signal = torch.view_as_complex(signal.contiguous())
    signal = torch.view_as_real(fftshift(fft(signal, dim=-1), dim=-1)).transpose(-1,-2)
    return signal.contiguous()


def HilbertTransform(data: torch.Tensor, 
                     dim: int=-1) -> torch.Tensor:  # checked 10.10.21
    '''
    Confirmed to match scipy.linalg.hilbert and Matlab's hilbert implementations.
    Suggestion from Fern: remove the mean before the transform, can add it back later
    https://dsp.stackexchange.com/questions/46291/why-is-scipy-implementation-of-hilbert-function-different-from-matlab-implemen
    Added nested copies of the (inverse) Fourier transfrom because this code does not use (i)fftshift.
    
    TODO: check in a notebook to make sure lo de la media works well.
    '''
    assert(data.ndim>=3)
    def Fourier(data):
        return rfft(data, dim=-1)
    def invFourier(data):          
        return irfft(data, dim=-1)
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
    return torch.cat([data, out], dim=-2)


def inv_Fourier_Transform(signal: torch.Tensor, 
                          normalize: bool=False) -> torch.Tensor:#, 
    assert(signal.ndim>=3)
    signal = signal.transpose(-1,-2)
    assert(signal.shape[-1]==2)
    signal = torch.view_as_complex(signal.contiguous())
    signal = torch.view_as_real(ifft(ifftshift(signal, dim=-1),dim=-1)).transpose(-1,-2)
    return signal.contiguous()


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

    mx, threshold = w_len.max().item(), torch.div(w_len.squeeze(), 2, rounding_mode='floor')
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

    out = F.conv1d(input=F.pad(x, (even, even), mode=mode), weight=w, groups=w.shape[0]).permute(1,0,2)
    start, stop = even//2, -even//2-1 if even==mx else -even//2-2
    return out[...,start:stop]
    
    
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

# Only works with 3 dimensions. Is it used with transients at any point?
def torch_batch_linspace(start: torch.Tensor, 
                         stop: torch.Tensor, 
                         steps: int) -> torch.Tensor:
    stop = stop.unsqueeze(-1) if stop.ndim==2 else stop
    for _ in range(3-start.ndim): start = start.unsqueeze(-1)
    start = start.expand_as(stop).to(stop.device)
    out = torch.arange(0, steps).expand([int(start.numel()), 1, int(steps)]).to(stop.device).clone().float()
    delta = stop.clone() - start.clone()
    out *= delta.float()
    out /= (steps - 1)
    out += start
    out[...,-1] = stop[...,-1]
    return out

def _save(path: str, 
          spectra: torch.Tensor, 
          parameters: dict, 
          ppm_range: torch.Tensor, 
          baselines: torch.Tensor=None, 
          residual_water: torch.Tensor=None, 
          spectral_fit: torch.Tensor=None, 
          quantities: dict=None):
    print('>>> Saving Spectra')
    base, _ = os.path.split(path)
    os.makedirs(base, exist_ok=True)
    num_test = 0
    dict = {'spectra': spectra.numpy(),
            'params': parameters,
            'ppm_range': ppm_range,
            'baselines': bl.numpy(),
            'residual_water': blf.numpy(),
            'spectral_fit': None,
            'quantities': quantities,
            }
    io.savemat(path + '_spectra.mat', do_compression=True, mdict=dict)
    print(path + '_spectra.mat')

# Source: https://stackoverflow.com/questions/52859751/most-efficient-way-to-find-order-of-magnitude-of-float-in-python
def OrderOfMagnitude(data: torch.Tensor):
    for d in range(data.ndim-1,0,-1):
        data = data.amax(d, keepdims=True)
    return torch.floor(torch.log10(data))


def sample_resWater(N: int, **cfg):
    if not isinstance(cfg, type(None)):
        try: cfg['pt_density']
        except: cfg['pt_density'] = 1204
        start, _ = rand_omit(torch.zeros(N,1,1).uniform_(0,cfg['prime']), 0.0, cfg['drop_prob'])
        end, _   = rand_omit(torch.zeros(N,1,1).uniform_(0,cfg['prime']), 0.0, cfg['drop_prob'])
        length = round((cfg['cropRange_water'][1] - cfg['cropRange_water'][0])*cfg['pt_density'])
        if length % 2 != 0: length += 1
        return {
            'start': torch.zeros(N,1,1),
            'end': torch.zeros(N,1,1),
            'std': torch.zeros(N,1,1).uniform_(cfg['std'][0],cfg['std'][1]),
            'upper_bnd': torch.ones(N,1,1).uniform_(cfg['upper'][0],cfg['upper'][1]),
            'lower_bnd': torch.ones(N,1,1).uniform_(cfg['lower'][0],cfg['lower'][1]) * -1,
            'windows': torch.zeros(N,1,1).fill_(cfg['window']),
            'length': length,
            'cropRange_resWater': cfg['cropRange_water'],
            'cropRange': cfg['cropRange'],
            'start_prime': start,
            'end_prime': end,
            'scale': cfg['scale'],
            'rand_omit': cfg['drop_prob'],
        }
    return None


def sample_baselines(N: int, **cfg): 
    if not isinstance(cfg, type(None)):
        try: cfg['pt_density']
        except: cfg['pt_density'] = 128
        length = round((cfg['cropRange'][1]-cfg['cropRange'][0])*cfg['pt_density'])
        if length % 2 != 0: length += 1
        return {
            'start': torch.zeros(N,1,1).uniform_(cfg['start'][0],cfg['start'][1]),
            'end': torch.zeros(N,1,1).uniform_(cfg['end'][0],cfg['end'][1]),
            'std': torch.zeros(N,1,1).uniform_(cfg['std'][0],cfg['std'][1]),
            'upper_bnd': cfg['upper'],
            'lower_bnd': cfg['lower'],
            'windows': torch.ones(N,1,1).uniform_(cfg['window'][0],cfg['window'][1]),
            'length': length,
            'cropRange': cfg['cropRange'],
            'scale': cfg['scale'],
            'rand_omit': cfg['drop_prob'],
        }
    return None

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

def sort_parameters(params: torch.Tensor,
                    index: dict):
    parameters = OrderedDict()
    for i, k in enumerate(index.keys()):
        if not k in ['metabolites', 'parameters', 'overall']:
            parameters.update({k: params[:,index[k]]})
    return parameters

def concat_dict(target: dict,
                new: dict):
    assert([(key in target.keys().lower()) for key in new.keys().lower()])
    for k, v in new.items():
        target[k] = np.concatenate([target[k], v], axis=0)
    return target
