'''
https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch
By: chausies
Nov 13, 2020 - Converted to 3D tensors for PyTorch NN compatability by Ing. John T LaMaster
Mar 16, 2021 - replaced remaining NumPy with PyTorch implementation
Jan 21, 2022 - Added the Cubic Hermite Modified Akima Interpolation which better handles undulations and over-/undershoots

Modified Akima is prefered for general MRS applications as it can reliably handle spectra at various stages of processing
'''
import torch
import torch.nn as nn

__all__ = ['CubicHermiteSplines', 'CubicHermiteMAkima']


class CubicHermite(nn.Module):
    def __init__(self, xaxis, signal):
        super().__init__()
        self.device = signal.device
        for _ in range(signal.ndim - xaxis.ndim): xaxis = xaxis.unsqueeze(0)
        self.x = xaxis.expand_as(signal).to(self.device).contiguous()
        self.y = signal.contiguous()
        self.m = (signal[...,1:] - signal[...,:-1]) / (self.x[...,1:] - self.x[...,:-1])

    @staticmethod
    def h_poly_helper(tt):
        out = torch.empty_like(tt)
        A = torch.tensor([
                          [1, 0, -3,  2],
                          [0, 1, -2,  1],
                          [0, 0,  3, -2],
                          [0, 0, -1,  1]
                         ], dtype=tt[-1].dtype, device=tt.device)

        for r in range(4):
            out[...,r,:] = A[r,0] * tt[...,0,:] + A[r,1] * tt[...,1,:] + \
                           A[r,2] * tt[...,2,:] + A[r,3] * tt[...,3,:]
        return out

    def h_poly(self, t):
        tt = torch.empty_like(t).unsqueeze(-2).repeat_interleave(4, dim=-2).to(self.device)
        tt[...,0,:] = torch.ones_like(tt[...,0,:])
        for i in range(1, 4):
            tt[...,i,:] = tt[...,i-1,:].clone() * t
        return self.h_poly_helper(tt)

    def interp(self, xs):
        size, size0 = list([i for i in self.x.shape]), list([1 for _ in range(self.x.ndim)])
        for _ in range(self.x.ndim - xs.ndim): xs = xs.unsqueeze(0)
        for i in range(2, xs.ndim-1): size0[i] = xs.shape[i]
        size[-1] = -1
           
        xs = xs.expand(size).contiguous()
        I  = torch.searchsorted(self.x[...,0,1:-1].unsqueeze(-2).contiguous(), 
                                xs[...,0,:].unsqueeze(-2).contiguous()).repeat(size0)

        x  = torch.gather(self.x, -1, I)
        dx = torch.gather(self.x, -1, I+1) - x
        
        hh = self.h_poly((xs - x)/dx)
            
        return hh[...,0,:]*torch.gather(self.y,-1,I)   + hh[...,1,:]*torch.gather(self.m,-1,I)*dx + \
               hh[...,2,:]*torch.gather(self.y,-1,I+1) + hh[...,3,:]*torch.gather(self.m,-1,I+1)*dx

    
    
class CubicHermiteSplines(CubicHermite):
    def __init__(self, xaxis: torch.Tensor, signal: torch.Tensor):
        super().__init__(xaxis, signal)
        self.m = torch.cat([self.m[...,0].unsqueeze(-1), (self.m[...,1:] + self.m[...,:-1]) / 2, self.m[...,-1].unsqueeze(-1)], dim=-1)
    
    
class CubicHermiteMAkima(CubicHermite):
    '''
    Cubic Hermite Modified Akima Interpolation
    https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/
    
    Written by: Ing. John T LaMaster, 2022
    
    References:
    - https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    - https://de-m-wikipedia-org.translate.goog/wiki/Akima-Interpolation?_x_tr_sl=de&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=sc
    - https://en.wikipedia.org/wiki/Akima_spline ***
    '''
    def __init__(self, xaxis: torch.Tensor, signal: torch.Tensor):
        super().__init__(xaxis, signal)
        d0 = 2 * self.m[...,0] - self.m[...,1]
        m0 = 2 * d0 - self.m[...,0]
        dn = 2 * self.m[...,-1] - self.m[...,-2]
        mn = 2 * dn - self.m[...,-1]
        self.m = torch.cat([m0.unsqueeze(-1), d0.unsqueeze(-1), self.m, dn.unsqueeze(-1), mn.unsqueeze(-1)], dim=-1)
        
        weights = torch.abs(self.m[...,:-1] - self.m[...,1:]) + torch.abs((self.m[...,:-1] + self.m[...,1:])/2)
        weights1 = weights[...,:-2].clone()
        weights2 = weights[...,2:].clone()
        delta1 = self.m[...,1:-2].clone()
        delta2 = self.m[...,2:-1].clone()
        weights12 = weights1 + weights2 + 1e-6
        
        self.m = (weights2 / weights12) * delta1 + (weights1 / weights12) * delta2
        self.m[(weights12==0)] = 0.
