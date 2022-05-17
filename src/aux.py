import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import (fft, ifft, fftshift, ifftshift, rfft, irfft)
import numpy as np
    
    
__all__ = ['complex_exp', 'convertdict', 'Fourier_Transform', 'HilbertTransform', 
           'inv_Fourier_Transform', 'smooth', 'torch_batch_linspace', ]


def complex_exp(signal: torch.Tensor, 
                theta: torch.Tensor, 
                real: bool=False) -> torch.Tensor:
    '''
    complex_exp expects a complex exponent. If dim=1 has only size=1, it is assumed to be the imaginary component.
    Therefore, the real component is assumed to be zero. If a real component should be used, it must be supplied
    at index = 0 along dimension 1.
    '''
    theta = theta if theta.shape[1]==2 else torch.cat([torch.zeros_like(theta), theta], dim=1)
    
    if real: 
        real = signal.mul(torch.cos(theta[:,1,:].unsqueeze(1)))
        imag = signal.mul(torch.sin(theta[:,1,:].unsqueeze(1)))
        return torch.cat([real, imag], dim=1).mul(torch.exp(theta[:,0,:].unsqueeze(1)).expand_as(signal))
    
    real = signal[::,0,:].mul(torch.cos(theta[::,1,:])) - signal[::,1,:].mul(torch.sin(theta[::,1,:]))
    imag = signal[::,0,:].mul(torch.sin(theta[::,1,:])) + signal[::,1,:].mul(torch.cos(theta[::,1,:]))

    return torch.cat([real.unsqueeze(1), imag.unsqueeze(1)], dim=-2)


def convertdict(file, simple=False, device='cpu'):
    from types import SimpleNamespace
    import numpy as np
    if simple:
        p = SimpleNamespace(**file) # self.basisSpectra
        keys = [y for y in dir(p) if not y.startswith('__')]
        for i in range(len(keys)):
            file[keys[i]] = torch.FloatTensor(np.asarray(file[keys[i]], dtype=np.float32)).squeeze().to(device)
        return SimpleNamespace(**file)
    else:
        delete = []
        for k, v in file.items():
            if not k.startswith('__') and not ('None' in k):
                if isinstance(v, dict):
                    for kk, vv in file[k].items():
                        file[k][kk] = torch.FloatTensor(np.asarray(file[k][kk], dtype=np.float32)).squeeze().to(device)
                elif k=='linenames':
                    file[k] = dict({str(a): torch.FloatTensor(np.asarray(b, dtype=np.float32)).to(device) for a, b in zip(file[k][0,:], file[k][1,:])})
                elif k in ['notes', 'seq']: #isinstance(v, str):
                    delete.append(k)
                else:
                    try:
                        file[k] = torch.FloatTensor(np.asarray(file[k], dtype=np.float32)).squeeze().to(device)
                    except TypeError:
                        pass
            else:
                delete.append(k)
        if len(delete)>0:
            for k in delete:
                file.pop(k, None)
        return file


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
    return torch.cat([data, out], dim=1)


def inv_Fourier_Transform(signal: torch.Tensor, 
                          normalize: bool=False) -> torch.Tensor:#, 
    assert(signal.ndim>=3)
    signal = signal.transpose(-1,-2)
    assert(signal.shape[-1]==2)
    signal = torch.view_as_complex(signal.contiguous())
    signal = torch.view_as_real(ifft(ifftshift(signal, dim=-1),dim=-1)).transpose(-1,-2)
    return signal.contiguous()


def smooth(x: torch.Tensor,
           window_len: float: 0.1,
           window: str='flat') -> torch.Tensor:
    w_len = int(x.shape[-1] * window_len)
    w = torch.ones((1, 1, w_len), dtype=torch.float32)
    w /= w.sum()
    out = F.conv1d(input=F.pad(x, (w_len, w_len), mode='reflect'), weight=w)
    out = out[...,w_len//2:-w_len//2]
    assert(x.shape==out.shape)
    return out


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

