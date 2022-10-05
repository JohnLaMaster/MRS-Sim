import copy
import os
from collections import OrderedDict
from functools import reduce
from math import ceil, floor

import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
from aux import *
from baselines import bounded_random_walk
from interpolate import CubicHermiteMAkima as CubicHermiteInterp  # , batch_linspace
from numpy import pi
# from torch.utils import _pair
from types import SimpleNamespace

__all__ = ['PhysicsModel'] #


const = torch.FloatTensor([1.0e6]).squeeze()
zero = torch.FloatTensor([0.0]).squeeze().float()
def check(x, label):
    x = x.clone()
    x = x.float()
    if torch.isnan(x).any(): print(label,': NaN found in ',label)
    if not torch.isfinite(x).all(): print(label,': Inf found in ',label)
    if (x>1e6).any(): 
        print(x.dtype, const.dtype, zero.dtype)
        a = torch.where(x>const.to(x.device), x, zero.to(x.device))
        a = a.float()
        ind = a.nonzero()
#         ind = torch.where(x>1.0e6, x, 0.0).float().nonzero()
        print(label,': Value greater than 1e6: ',x[ind])

PI = torch.from_numpy(np.asarray(np.pi)).squeeze().float()


def dict2tensors(dct: dict) -> dict:
    for key, value in dct.items():
        if isinstance(value, dict):
            dict2tensors(value)

@torch.no_grad()
class PhysicsModel(nn.Module):
    def __init__(self, 
                 PM_basis_set: str,
                ):
        super().__init__()
        '''
        Args:
            cropRange:    specified fitting range
            length:       number of data points in the fitted spectra
            apodization:  amount of apodization in Hz. Should only be included if 
                          noise=True in the forward pass
        '''
        # Load basis spectra, concentration ranges, and units
        path = './src/basis_sets/' + PM_basis_set # 'fitting_basis_ge_PRESS144.mat'

        with open(path, 'rb') as file:
            dct = convertdict(io.loadmat(file,simplify_cells=True))
            self.basisFcns = {}
            for key, value in dct.items():
                if str(key)=='metabolites': self.basisFcns['metabolites'] = value
                elif str(key)=='artifacts': self.basisFcns['artifacts'] = value
                elif str(key)=='header':
                    for k, v in dct[key].items():
                        if str(k)=='ppm': 
                            k, v = '_ppm', torch.flip(v.unsqueeze(0), dims=[-1,0]).squeeze(0)
                        if not isinstance(v, str): 
                            self.register_buffer(str(k), v.float())


        
    def __repr__(self):
        lines = sum([len(listElem) for listElem in self.totals])   
        out = 'MRS_PhysicsModel(basis={}, lines={}'.format('Osprey, GE_PRESS144', lines)
        return out + ', range={}ppm, resample={}pts)'.format(self.cropRange, self.length)
    
    @property
    def basis_metab(self):
        temp = [x for x, _ in enumerate(self.totals)]
        metab_list = []
        for i, k in enumerate(self._basis_metab):
            if not ('mm' in k.lower() or 'lip' in k.lower()):
                metab_list.append(k)
        return metab_list, temp[0:len(metab_list)]
    
    @property
    def index(self):
        return self._index
    
    @property
    def metab(self):
        temp = [x for x, _ in enumerate(self.totals)]
        return self._metab, temp
    
    @property
    def ppm(self, 
            cropped: bool=False
           ) -> torch.Tensor:
        return self._ppm if not cropped else self.ppm_cropped

       
    def initialize(self, 
                   metab: list=['Cho','Cre','Naa','Glx','Ins','Mac','Lip'],
                   cropRange: list=[0,5],
                   length: float=512,
                   basisFcn_len: float=1024,
                   ppm_ref: float=4.65,
                   transients: int=8,
                   coil_sens: bool=False,
                   spectral_resolution: list=[10.0, 10.0, 10.0],
                   image_resolution: list=[0.5, 0.5, 0.5],
                   lineshape: str='voigt',
                   fshift_i: bool=False,
                   difference_editing: list=False, # should be a list of basis function names that gets subtracted
                  ):
        '''
        Steps: 
        - based on the desired fitting range, calculate the necessary number of splines
        - initialize the difference matrix
        - prepare the parameter index
        - prepare the parameter dictionary template
        - prepare the fids
        
        '''
        
        self._metab, l, self.MM = self.order_metab(metab)
        if difference_editing: 
            self._difference_editing, l_diff, mm_diff  = self.order_metab(difference_editing)
            assert(l==l_diff)
            assert(self.MM==mm_diff)
            self.difference_editing_fids = torch.stack([torch.as_tensor(self.basisFcns['metabolites'][m.lower()]['fid'], dtype=torch.float32) for m in self._difference_editing], dim=0).unsqueeze(0)
            # Resample the basis functions, ppm, and t to the desired resolution
            self.difference_editing_fids = self.resample_(signal=self.difference_editing_fids,
                                                          ppm=self._ppm,
                                                          length=basisFcn_len,
                                                          target_range=[self._ppm.min(), self._ppm.max()])
        self.MM = self.MM + 1 if  self.MM>-1 else False
        self.lineshape_type = lineshape

#         if isinstance(cropRange, type(None)): cropRange = [self._ppm.min(), self._ppm.max()]

        '''
        bandwidth = 1/dwelltime
        carrier_freq = 127.8 # MHz
        freq_ref = 4.68 * carrier_freq / 10e6
        degree = 2*pi*bandwidth*(1/(bandwidth + freq_ref))
        
        freq_ref = ppm_ref * self.carrier_freq / 10e6
        phi1_ref = 2*self.PI*self.spectralwidth * (torch.linspace(-0.5*self.spectralwidth, 0.5*self.spectralwidth, self.l) + freq_ref)
        '''
#         self.ppm.float()
        self.cropRange = cropRange if cropRange else [self._ppm.min(), self._ppm.max()]
        self.t = self.t.unsqueeze(-1).float()
        self._basis_metab = []

        dct = OrderedDict()
        for m in self._metab: dct.update({str(m): torch.empty(1)})  # Add metabolite names to dictionary
        # \todo Add a check to make sure the specified metabolites are in the basis set
        self.syn_basis_fids = torch.stack([torch.as_tensor(self.basisFcns['metabolites'][m.lower()]['fid'], dtype=torch.float32) for m in self._metab], dim=0).unsqueeze(0)
        # Resample the basis functions, ppm, and t to the desired resolution
        self.syn_basis_fids = self.resample_(signal=self.syn_basis_fids,
                                             ppm=self._ppm,
                                             length=basisFcn_len,
                                             target_range=[self._ppm.min(), self._ppm.max()])
        self._ppm = torch.linspace(self._ppm.min(), self._ppm.max(), basisFcn_len).unsqueeze(0)
        self.t = torch.linspace(self.t.min(), self.t.max(), basisFcn_len).unsqueeze(-1)

        # Define variables for later
        self.register_buffer('l', torch.FloatTensor([self.syn_basis_fids.shape[-1]]).squeeze())
        self.register_buffer('length', torch.FloatTensor([length]).squeeze())
        self.register_buffer('ppm_ref', torch.FloatTensor([ppm_ref]).squeeze())
        self.register_buffer('ppm_cropped', torch.fliplr(torch.linspace(cropRange[0], cropRange[1], length).unsqueeze(0)))
        self.spectral_resolution = spectral_resolution
        self.image_resolution = image_resolution
    
        # Define the first-order phase reference in the time-domain
        freq_ref = ppm_ref * self.carrier_frequency / 10e6
        spectralwidth = torch.linspace(-0.5*self.spectralwidth, 0.5*self.spectralwidth, int(self.l))
        phi1_ref = 2 * PI * spectralwidth / (spectralwidth + freq_ref)
        self.register_buffer('phi1_ref', torch.as_tensor(phi1_ref, dtype=torch.float32).squeeze())

        # Define the index used to specify the variables in the forward pass and in the sampling code
        num_bF = l+self.MM if self.MM else l
        header, cnt = self._metab, counter(start=int(3*num_bF)-1)
        g = 1 if not self.MM else 2
        names = ['d',   'dmm', 'g',   'gmm', 'fshift', 'snr', 'phi0', 'phi1', 'b0', 'bO_dir']
        mult  = [  l, self.MM,   l, self.MM,        1,     1,      1,      1,    1,        3] 
        if fshift_i: # Should be a global fshift then individual metabolites and MM/Lip fsfhitfs
            names.insert(-6,'fshiftmet')
            names.insert(-6,'fshiftmm')
            mult.insert(-6,l), mult.insert(-6,self.MM)
        if transients>1: # Minimum 2 transients for the variables to be included in the model
            names.append('transients')
            mult.append(transients)
            if coil_sens:
                names.append('coil_sens')
                mult.append(transients)
        for n, m in zip(names, mult): 
            for _ in range(m): header.append(n)
            
        # Define the min/max ranges for quantifying the variables
        self.min_ranges = torch.zeros([1,len(header)], dtype=torch.float32)
        self.max_ranges = torch.zeros_like(self.min_ranges)
        for i, m in enumerate(header):
            met, temp, strt = False, None, None
            if m in self.basisFcns['metabolites'].keys(): 
                temp = self.basisFcns['metabolites'][m]
                met = True
            elif m in self.basisFcns['artifacts'].keys(): 
                temp = self.basisFcns['artifacts'][m]
            elif m in ['fshiftmet','fshiftmm']:
                if not strt: strt = i 
                try: 
                    temp = self.basisFcns['metabolites'][i]['fshift']
                    # Workaround to allow for zero-shifts
                    if temp['min']==0: temp['min'] = -1e-10
                    if temp['max']==0: temp['max'] =  1e-10
                except KeyError:
                    # If not included in the basis set, then set a default
                    default = 5 # Hz
                    temp = {'min': -default, 'max': default}
            if temp:
                self.min_ranges[0,i] = temp['min']
                self.max_ranges[0,i] = temp['max']

        self.totals = []
        for i in range(num_bF):
            self.totals.append(1)

        # Begin defining the indices of each variable
        # # Metabolites
        ind = list(int(x) for x in torch.arange(0,num_bF))
        
        # # Line shape corrections 
        ind.append(tuple(int(x) for x in torch.arange(0,num_bF) + num_bF)) # Lorentzian corrections
        ind.append(tuple(int(x) for x in torch.arange(0,num_bF) + 2*num_bF)) # Single Gaussian correction factor

        # # Frequency Shift / Scaling Factor / Noise depending on supervised T/F
        ind.append(cnt(1)) # Global fshift
        if fshift_i: ind.append(tuple(cnt(1) for _ in range(num_bF))) # Individual fshifts
        # ind.append(cnt(1)) # Scaling
        ind.append(cnt(1)) # SNR

        # # Phase
        ind.append(cnt(1)) # Phi0
        ind.append(cnt(1)) # Phi1

        # # B0 inhomogeneities
        ind.append(cnt(1)) # B0 - mean
        ind.append(tuple(int(cnt(1)) for _ in torch.arange(0,3))) # directional deltas

        # # Coil sensitivities
        if transients>1:
            ind.append(tuple(int(cnt(1)) for _ in torch.arange(0,transients)))
            if coil_sens:
                ind.append(tuple(int(cnt(1)) for _ in torch.arange(0,transients)))

        # # Cummulative
        total = cnt(1)
        ind.append(tuple(int(x) for x in torch.arange(0,num_bF)))        # Metabolites
        ind.append(tuple(int(x) for x in torch.arange(num_bF,total)))    # Parameters
        ind.append(tuple(int(x) for x in torch.arange(0,total)))         # Overall

        # Define the remaining dictionary keys
        dct.update({'D': torch.empty(1), 'G': torch.empty(1), 'F_Shift': torch.empty(1)})
        if fshift_i: dct.update({'F_Shifts': torch.empty(1)})
        dct.update({'SNR': torch.empty(1), 'Phi0': torch.empty(1), 'Phi1': torch.empty(1), 'B0': torch.empty(1), 
                    'B0_dir': torch.empty(1)})
        if transients: 
            dct.update({'Transients': torch.empty(1)})
            if coil_sens: dct.update({'Coil_Sens': torch.empty(1)})
        dct.update({'Metabolites': torch.empty(1), 'Parameters': torch.empty(1), 'Overall': torch.empty(1)})
        
        # Combine and define the index for internal use in the model
        self._index = OrderedDict({d.lower(): i for d,i in zip(dct.keys(),ind)})

        return dct, ind

    
    def add_offsets(self,
                    fid: torch.Tensor,
                    offsets: tuple=None,
                    drop_prob: float=0.2,
                   ) -> dict:
        '''
        Used for adding residual water and baselines. config dictionaries are needed for each one.
        '''
        out, baselines, res_water = offsets
        scale = 10**(OrderOfMagnitude(fid) - OrderOfMagnitude(out))
        offset = out.clone() * scale

        out, ind = rand_omit(out, 0.0, drop_prob)
        if not isinstance(baselines, type(None)): 
            baselines *= scale
            if drop_prob: baselines[ind,...] = 0.0
        if not isinstance(res_water, type(None)): 
            res_water *= scale
            if drop_prob: res_water[ind,...] = 0.0
        if not isinstance(out, int):# == 0:
            fid += inv_Fourier_Transform(out*scale)

        return fid, {'baselines': baselines, 
                     'residual_water': res_water, 
                     'offset': offset}

    
    def apodization(self,
                    fid: torch.Tensor, 
                    hz: int=4,
                   ) -> torch.Tensor:
        exp = torch.exp(-self.t * hz).t()
        for _ in range(fid.ndim - exp.ndim): exp = exp.unsqueeze(0)
        return fid * exp.expand_as(fid)


    def baselines(self,
                  config: dict,
                 ) -> tuple():
        cfg = SimpleNamespace(**config)

        baselines = HilbertTransform(
                            batch_smooth(bounded_random_walk(cfg.start, cfg.end, cfg.std, cfg.lower_bnd, 
                                                             cfg.upper_bnd, cfg.length), 
                                         cfg.windows)
                            )

        if cfg.rand_omit>0: 
            baselines, _ = rand_omit(baselines, 0.0, cfg.rand_omit)

        ch_interp = CubicHermiteInterp(xaxis=torch.linspace(self.cropRange[0], 
                                                            self.cropRange[1], 
                                                            cfg.length),
                                       signal=baselines)

        out = ch_interp.interp(xs=self.ppm_cropped.fliplr()) * config['scale']
        ppm = self._ppm.clone().unsqueeze(-1).repeat(1,baselines.shape[0],1)

        raw_baseline = self.sim2acquired(out, [ppm.amin(keepdims=True), ppm.amax(keepdims=True)], self.ppm)

        return out.fliplr(), raw_baseline.fliplr()
    
    
    def B0_inhomogeneities(self, 
                           b0: torch.Tensor,
                           param: torch.Tensor, # hz
                          ) -> torch.Tensor:
        '''
        I need spatial resolution of images and spectra as well as another parameter(s) defining the range of
        the B0 variation across the MRS voxel. 
        complex_exp(torch.ones_like(fid), param*t).sum(across this extra spatial dimension)
        return fid * sum(complex_exp)
        '''
        t = self.t.clone().mT # [1, 8192] 
        # FID should be 4 dims = [bS, basis fcns, channels, length]
        for _ in range(4 - t.ndim): t = t.unsqueeze(0)
        for _ in range(4 - param.ndim): param = param.unsqueeze(1)
        # if t.shape[-1]==1: t = t.mT # [1, 1, 8192].mT => [1, 8192, 1]
        # param = param.unsqueeze(-2) # [bS, basisfcns, channels=1, extra=1, params]

        # spectral_resolution = [10.0, 10.0, 10.0]
        # image_resolution    = [ 0.5,  0.5,  0.5]
        num_pts = [int(m/n) for m, n in zip(self.spectral_resolution, self.image_resolution)]
        mean = b0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        dx = param[...,0]#.unsqueeze(-1)
        dy = param[...,1]#.unsqueeze(-1)
        dz = param[...,2]#.unsqueeze(-1)

        '''
        Matlab code confirming it works!
        Gradient of x goes top to bottom. Gradient of y runs through the center.
        n = 10; x = 1; y = 4; z = 2; xvec = -x:2*x/(n-1):x; yvec = -y:2*y/(n-1):y; zvec=-z:2*z/(n-1):z;
        dB0 = xvec .* xvec' ./ x; dB1 = dB0 + yvec;
        dB2 = reshape(dB1,[10,10,1]) + reshape((zvec+1),[1,1,10]);
        '''

        # output.shape = [bS, 1, length, 1, 1]
        x = torch_batch_linspace(1 - dx, 1 + dx, num_pts[0]).permute(0,2,1).unsqueeze(-1)
        # output.shape = [bS, 1, 1, length, 1]
        y = torch_batch_linspace(1 - dy, 1 + dy, num_pts[1]).unsqueeze(-1)
        # output.shape = [bS, 1, 1, 1, length]
        z = torch_batch_linspace(1 - dz, 1 + dz, num_pts[2]).unsqueeze(-1).permute(0,1,3,2)

        # Define the changes in B0
        dB0  = x * x.transpose(-3,-2) / dx.unsqueeze(-1)
        dB0 += y
        dB0 = dB0.repeat(1,1,1,z.shape[-1]) + z
        dB0 += mean

        # output.shape = [bS, length, length, length]
        dB0 = dB0.unsqueeze(1).flatten(start_dim=2, end_dim=-1).unsqueeze(-1)
        # output.shape = [bS, 1, length^3, 1] * [1, 1, 1, 8192] => [bS, 1, length^3, 8192]
        dB0 = dB0 * t

        identity = torch.ones_like(t).repeat(1,1,2,1)

        return complex_exp(identity, (-1*dB0.unsqueeze(-2)).deg2rad()).sum(dim=-3)


    def coil_sensitivity(self,
                         fid: torch.Tensor,
                         coil_sens: torch.Tensor,
                        ) -> torch.Tensor:
        '''
        Used to apply scaling factors to simulate the effect of coil combination weighting
        '''
        coil_sens = coil_sens.unsqueeze(1) # Adds the transient dim
        for _ in range(fid.ndim - coil_sens.ndim): 
            coil_sens = coil_sens.unsqueeze(-1)
        return fid * coil_sens


    def eddyCurrents(self,
                     fid: torch.Tensor,
                     phase: torch.Tensor,
                    ) -> torch.Tensor:
        '''
        Not fully implemented. Is there a way to manipulate a simulated water fid and then
        use atan2 to get the phase?
        '''
        return complex_exp(fid, -1*phase.unsqueeze(-1).deg2rad()*self.t)


    def firstOrderPhase(self, 
                        fid: torch.Tensor, 
                        phi1: torch.Tensor,
                       ) -> torch.Tensor:
        '''
        Current implementation does this in the time domain!
        '''
        for _ in range(fid.ndim - self.phi1_ref.ndim): 
            self.phi1_ref = self.phi1_ref.unsqueeze(0)
        for _ in range(fid.ndim - phi1.ndim): 
            phi1 = phi1.unsqueeze(-1)

        return complex_exp(fid, -1*(phi1 + self.phi1_ref).deg2rad())

        
    def frequency_shift(self, 
                        fid: torch.Tensor, 
                        param: torch.Tensor,
                        t: torch.Tensor=None,
                       ) -> torch.Tensor:
        '''
        Do NOT forget to specify the dimensions for the (i)fftshift!!! 
        Will reorder the batch samples!!!
        Also, keep fshift in Hz
        '''
        t = self.t if t==None else t
        t = t.t() if t.shape[-1]==1 else t
            
        for _ in range(fid.ndim - param.ndim): param = param.unsqueeze(-1)
        for _ in range(fid.ndim - t.ndim): t = t.unsqueeze(0)
        f_shift = param.mul(t)
        
        return complex_exp(fid, f_shift)
        
        
    def generate_noise(self, 
                       fid: torch.Tensor, 
                       param: torch.Tensor,
                       max_val: torch.Tensor,
                       transients: torch.Tensor=None,
                      ) -> torch.Tensor:
        '''
        RMS coefficient is used because this is done in the time domain with sinusoids
        SNR formula (MRI!):
            snr_db = 10*log10(snr_lin * 0.66) # 0.66 Rayleigh distribution correction factor to calculate the true SNR
            snr_lin = max(real(spectra)) / std_noise # Not sure whether to use real or magnitude spectra
        '''
        for _ in range(fid.ndim-max_val.ndim): max_val = max_val.unsqueeze(-1)
        for _ in range(fid.ndim-param.ndim): param = param.unsqueeze(-1)

        lin_snr = 10**(param / 10) # convert from decibels to linear scale
        if not isinstance(transients, type(None)): 
            # Scale the mean SNR accourding to the number of transients
            s = int(transients.shape[-1])
            lin_snr = lin_snr / s**0.5
            for _ in range(fid.ndim-transients.ndim): 
                transients = transients.unsqueeze(-1)

            # Allows the transients' SNR to come from a distribution
            lin_snr = lin_snr * transients

        k = 1 / lin_snr         # scaling coefficient
        a_signal = max_val      # RMS coefficient for sine wave
        scale = k * a_signal    # signal apmlitude scaled for desired noise amplitude

        scale[torch.isnan(scale)] = 1e-6
        scale[scale==0] += 1e-6
        if scale.ndim==3:
            if scale.shape[-2]==1 and scale.shape[-1]==1: 
                scale = scale.squeeze(-1)

        e = torch.distributions.normal.Normal(0,scale).sample([fid.shape[-1]])
        if e.ndim==2: e = e.unsqueeze(1)
        if e.ndim==3: e = e.permute(1,2,0)
        elif e.ndim==4: e = e.permute(1,2,3,0).repeat_interleave(fid.shape[1], dim=1)
        elif e.ndim==5: e = e.permute(1,2,3,4,0).squeeze(-2)

        return HilbertTransform(e)


    def line_summing(self,
                     fid: torch.Tensor,
                     params: torch.Tensor,
                     mm: int,
                     l: int,
                    ) -> tuple:
        if not mm:
            fidSum = fid.sum(dim=-3) 
            spectral_fit = fidSum.clone()
            mx_values = torch.amax(fid[...,0,:].unsqueeze(-2).sum(dim=-3).unsqueeze(1), dim=-1) 
        else:
            mm = fid[...,l:,:,:].sum(dim=-3)
            fidSum = fid[...,0:l,:,:].sum(dim=-3)
            spectral_fit = fidSum.clone()
            mx_values = torch.amax(fidSum[...,0,:].unsqueeze(-2), dim=-1, keepdims=True) 
            fidSum += mm
        return fidSum, spectral_fit, mx_values
    

    def lineshape_correction(self, 
                             fid: torch.Tensor, 
                             d: torch.Tensor=None, 
                             g: torch.Tensor=None,
                            ) -> torch.Tensor:
        '''
        In a Voigt lineshape model, each basis line has its own Lorentzian value. Fat- and Water-based 
        peaks use one Gaussian value per group.
        '''
        if 'gaussian' in  self.lineshape_type:
            return self.lineshape_gaussian(fid, g)
        
        if 'lorentzian' in  self.lineshape_type:
            return self.lineshape_lorentzian(fid, d)

        if 'voigt' in self.lineshape_type:
            return self.lineshape_voigt(fid, d, g)


    def lineshape_gaussian(self, 
                           fid: torch.Tensor, 
                           g: torch.Tensor
                          ) -> torch.Tensor:
        t = self.t.clone().t().unsqueeze(0)
        g = g.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2,1)
        
        return fid * torch.exp(-g * t.unsqueeze(0).pow(2)) 


    def lineshape_lorentzian(self, 
                             fid: torch.Tensor, 
                             d: torch.Tensor, 
                            ) -> torch.Tensor:
        t = self.t.clone().t().unsqueeze(0)
        d = d.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2,1)
        if d.dtype==torch.float64: d = d.float()
        
        return fid * torch.exp(-d * t.unsqueeze(0))


    def lineshape_voigt(self, 
                        fid: torch.Tensor, 
                        d: torch.Tensor, 
                        g: torch.Tensor
                       ) -> torch.Tensor:
        '''
        In a Voigt lineshape model, each basis line has its own Lorentzian value. Fat- and Water-based 
        peaks use one Gaussian value per group.
        '''
        t = self.t.clone().t().unsqueeze(0)
        d = d.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2,1)
        g = g.unsqueeze(-1).unsqueeze(-1).expand_as(d)
        if d.dtype==torch.float64: d = d.float()
        
        return fid * torch.exp((-d - g * t.unsqueeze(0)) * t.unsqueeze(0))


    def magnitude(self, 
                  signal: torch.Tensor, 
                  normalize: bool=False,
                  individual: bool=False,
                  eps: float=1e-6,
                 ) -> torch.Tensor:
        '''
        Calculate the magnitude spectrum of the input spectrum.
        '''
        magnitude = torch.sqrt(signal[...,0,:].pow(2) + signal[...,1,:].pow(2) + eps).unsqueeze(-2)
        out = torch.cat([signal, magnitude], dim=-2)
            
        if normalize:
            if individual: 
                data = torch.sum(signal, dim=-3, keepdim=True)
                magnitude = torch.sqrt(data[...,0,:].pow(2) + data[...,1,:].pow(2) + eps).unsqueeze(-2)
            return out.div(torch.max(magnitude, dim=-1, keepdim=True).values + eps)

        return out
    

    def modulate(self, 
                 fids: torch.Tensor,
                 params: torch.Tensor,
                ) -> torch.Tensor:
        params = params.unsqueeze(-1).unsqueeze(-1).repeat_interleave(2, dim=-2)
        return params.mul(fids)


    def normalize(self, 
                  signal: torch.Tensor,
                  fid: bool=False,
                  denom: torch.Tensor=None,
                 ) -> torch.Tensor:
        '''
        Normalize each sample of single or multi-echo spectra. 
            Step 1: Find the max of the real and imaginary components separately
            Step 2: Pick the larger value for each spectrum
        If the signal is separated by metabolite, then an additional max() is necessary
        Reimplemented according to: https://stackoverflow.com/questions/41576536/normalizing-complex-values-in-numpy-python
        '''
        if isinstance(denom, type(None)):
            if not signal.shape[-2]==3:
                denom = torch.max(torch.sqrt(signal[...,0,:]**2 + signal[...,1,:]**2), dim=-1, keepdim=True).values.unsqueeze(-2)
            else:
                denom = torch.max(signal[...,2,:].unsqueeze(-2).abs(), dim=-1, keepdim=True).values

            denom[denom.isnan()] = 1e-6
            denom[denom==0.0] = 1e-6

        for _ in range(denom.ndim-signal.ndim): signal = signal.unsqueeze(1)

        return signal / denom, denom


    def order_metab(self, 
                    metab: list,
                   ) -> tuple:
        mm_lip, temp, num_mm = [], [], -1
        for k in sorted(metab):
            if 'mm' in k: 
                mm_lip.append(k)
                temp.append(k)
                num_mm += 1
        for k in sorted(metab):
            if 'lip' in k:
                mm_lip.append(k)
                temp.append(k)
                num_mm += 1
        for term in temp: metab.pop(term)
        metab = sorted(metab)
        if num_mm>-1:
            return metab + mm_lip, len(metab), num_mm
        return metab, len(metab), num_mm
                    

    def quantify_params(self, 
                        params: torch.Tensor,
                        label=[],
                       ) -> torch.Tensor:
        delta = self.max_ranges - self.min_ranges
        minimum = self.min_ranges.clone()
        params = params.mul(delta) + minimum

        return params

    
    def quantify_metab(self, 
                       params: torch.Tensor, 
                       norm: torch.Tensor=None,
                       wrt_metab: str='cr',
                       b0: bool=False,
                      ) -> dict:
        '''
        Both peak height and peak area are returned for each basis line. The basis fids are first 
        modulated and then broadened. The recovered spectra are normalized and then multiplied by
        the normalizing values from the original spectra.
        '''
        if params.ndim==3: params = params.squeeze(-1)
        assert(params.ndim==2)
        wrt_metab = wrt_metab.lower()
        
        # Quantify parameters
        params = self.quantify_params(params, label='quantify_metab')

        if b0:
            B0 = self.B0_inhomogeneities(b0=params[:,self.index['b0']],
                                         param=params[:,self.index['b0_dir']])

        # Define basis spectra coefficients
        fid = params[:,self.index['metabolites']].unsqueeze(2).unsqueeze(-1) #* self.syn_basis_fids
        fid = torch.cat([fid.mul(self.syn_basis_fids[:,:,0,:].unsqueeze(2)),
                         fid.mul(self.syn_basis_fids[:,:,1,:].unsqueeze(2))], dim=2).contiguous()
        check(fid, 'quantify_metab basis line modulation')

#         Line broadening **if included in model**
        if params.shape[1]>len(self.index['metabolites']):
            fid = self.lineshape_correction(fid, params[:,self.index['d']],
                                            params[:,self.index['g']])

        if b0: fid *= B0
        
        # Recover and normalize the basis lines
        specs = self.magnitude(Fourier_Transform(fid), normalize=True)
        
        # Scale the lines to the magnitude of the input spectra
        if norm: spec = spec.mul(norm)

        # # Combine related metabolite lines
        # specs = spec[:,tuple(self.totals[0]),0,:].sum(dim=1, keepdim=True).unsqueeze(-2)
        # for ind in range(1, len(self.totals)):
        #     specs = torch.cat([specs, spec[:,tuple(self.totals[ind]),0,:].sum(dim=1, keepdim=True).unsqueeze(-2)], dim=1)

        # Quantities
        area = torch.stack([torch.trapz(specs[:,i,:], self.ppm, dim=-1).unsqueeze(-1) for i in range(specs.shape[1])], dim=1)
        height = torch.max(specs, dim=-1, keepdims=True).values
        
        # Normalize to creatine (CRE)
        rel_area = area.div(area[:,self.index[wrt_metab],::].unsqueeze(1))
        rel_height = height.div(height[:,self.index[wrt_metab],::].unsqueeze(1))

        return {'area': area.squeeze(-1), 'height': height.squeeze(-1), 'rel_area': rel_area.squeeze(-1), 'rel_height': rel_height.squeeze(-1), 'params': params}


    def resample_(self, 
                  signal: torch.Tensor, 
                  ppm: torch.Tensor=None,
                  length: int=512,
                  target_range: list=None,
                 ) -> torch.Tensor:
        '''
        Basic Cubic Hermite Spline Interpolation of :param signal: with no additional scaling.
        :param signal: input torch tensor
        :param new: new target x-axis
        return interpolated signal
        
        I am flipping the ppm vector instead of the basis lines - 02.01.2022 JTL
        '''
        dims, dims[-1] = list(signal.shape), -1
        ppm = ppm.unsqueeze(0).squeeze() if not isinstance(ppm, type(None)) else self.ppm.unsqueeze(0).squeeze()
        if not (ppm[...,0]<ppm[...,-1]): ppm = torch.flip(ppm.unsqueeze(0), dims=[-1]).squeeze(0)
        
        if isinstance(target_range, type(None)): target_range = self.cropRange
        new = torch.linspace(start=target_range[0], end=target_range[1], steps=int(length)).to(signal.device)
        for i in range(signal.ndim - new.ndim): new = new.unsqueeze(0)

        chs_interp = CubicHermiteInterp(ppm, torch.flip(signal, dims=[-1]))

        return torch.flip(chs_interp.interp(new), dims=[-1])
    
    
    def residual_water(self,
                       config: dict,
                      ) -> tuple():
        cfg = SimpleNamespace(**config)

        start_prime = cfg.cropRange_resWater[0] + cfg.start_prime
        end_prime   = cfg.cropRange_resWater[1] -   cfg.end_prime
        ppm = torch_batch_linspace(start=start_prime, stop=end_prime, steps=int(cfg.length))

        res_water = HilbertTransform(
                         batch_smooth(bounded_random_walk(cfg.start, cfg.end, cfg.std, cfg.lower_bnd, 
                                                          cfg.upper_bnd, cfg.length),
                                      cfg.windows, 'constant') * config['scale']
                         )

        if cfg.rand_omit>0: 
            res_water = rand_omit(res_water, 0.0, cfg.rand_omit)

        raw_res_water = self.sim2acquired(res_water, [start_prime, end_prime], self.ppm)
        res_water = self.sim2acquired(res_water, [start_prime, end_prime], self.ppm_cropped)

        return res_water, raw_res_water


    def set_parameter_constraints(self, cfg: dict):
        cfg_keys = cfg.keys()

        for k, ind in self._index.items():
            if k in cfg_keys:
                if isinstance(ind, tuple):
                    for i in ind:
                        self.min_ranges[:,i] = cfg[k][0]
                        self.max_ranges[:,i] = cfg[k][1]
                else:
                    self.min_ranges[:,ind] = cfg[k][0]
                    self.max_ranges[:,ind] = cfg[k][1]
    
        
    def simulate_offsets(self,
                         baselines: dict=None,
                         residual_water: dict=None,
                         drop_prob: float=0.2,
                        ) -> torch.Tensor:
        out = 0
        if baselines: 
            print('>>>>> Baselines')
            baselines, raw_baselines = self.baselines(baselines)
            out += raw_baselines.clone()
        else: baselines = None

        if residual_water: 
            print('>>>>> Residual Water')
            res_water, raw_res_water = self.residual_water(residual_water)
            out += raw_res_water.clone()
        else: res_water = None

        return out, baselines, res_water


    def sim2acquired(self, 
                     line: torch.Tensor, 
                     sim_range: list, 
                     target_ppm: torch.Tensor,
                    ) -> torch.Tensor:
        '''
        This approach uses nonuniform sampling density to reduce the memory footprint. This 
        is possible because the tails being padded are always zero. Having 10e1 or 10e10 zeros 
        gives the same result. So, small tails are padded to the input, 
        '''
        raw_ppm = [target_ppm.amin(), target_ppm.amax()]
        if target_ppm.amin(keepdims=True)[0]!=line.shape[0]: 
            raw_ppm = [raw_ppm[0].repeat(line.shape[0]), raw_ppm[1].repeat(line.shape[0])]
        if sim_range[0].shape[0]!=line.shape[0]: sim_range[0] = sim_range[0].repeat_interleave(line.shape[0], dim=0)
        if sim_range[1].shape[0]!=line.shape[0]: sim_range[1] = sim_range[1].repeat_interleave(line.shape[0], dim=0)
        for _ in range(3 - sim_range[0].ndim): sim_range[0] = sim_range[0].unsqueeze(-1)
        for _ in range(3 - sim_range[1].ndim): sim_range[1] = sim_range[1].unsqueeze(-1)
        for _ in range(3 - raw_ppm[0].ndim): raw_ppm[0] = raw_ppm[0].unsqueeze(-1)
        for _ in range(3 - raw_ppm[1].ndim): raw_ppm[1] = raw_ppm[1].unsqueeze(-1)

        pad = 100 # number of points added to each side
        pad_left, pad_right = 0, 0
        xaxis = torch_batch_linspace(sim_range[0], sim_range[1], int(line.shape[-1])) # middle side

        # Left side
        if (raw_ppm[0]<sim_range[0]).all():
            xaxis = torch.cat([torch_batch_linspace(raw_ppm[0], sim_range[0], pad+1)[...,:-1], # left side
                               xaxis], dim=-1) 
            pad_left = pad
        # Right side
        if (raw_ppm[1]>sim_range[1]).all():
            xaxis = torch.cat([xaxis, torch_batch_linspace(sim_range[1], raw_ppm[1], pad+1)[...,1:]], dim=-1) 
            pad_right = pad

        padding = tuple([pad_left, pad_right])
        ch_interp = CubicHermiteInterp(xaxis=xaxis, signal=torch.nn.functional.pad(input=line, pad=padding))
        return ch_interp.interp(xs=target_ppm)

            
    def transients(self, 
                   fid: torch.Tensor, 
                   coil_sens: torch.Tensor,
                  ) -> torch.Tensor:
        '''
        This function simply creates the number of transients. Noise and scaling
        are done separately.
        The SNR dB value provided is the SNR of the final, coil combined spectrum. Therefore, each of the 
        transients will have a much higher linear SNR that is dependent upon the expected final SNR and 
        the number of transients being simulated.
        '''
        # assert(fid.ndim==3) # Using difference editing would make it [bS, ON/OFF, channels, length] 
        # output.shape = [bS, ON/OFF, transients, channels, length] 
        return fid.unsqueeze(-3).repeat_interleave(repeats=coil_sens.shape[-1], dim=-3)

    
    def zero_fill(self,
                  fidSum: torch.Tensor,
                  fill: int,
                 ) -> torch.Tensor:
        '''
        This function will append the FID with zeros to a final length of fill.
        '''
        dim = []
        for i in fidSum.shape: 
            dim.appned(i)
        dim.append(fill - fidSum.shape[-1])
        return fidSum.cat(torch.zeros(dim, dim=-1))

    
    def zeroOrderPhase(self, 
                       fid: torch.Tensor, 
                       phi0: torch.Tensor,
                      ) -> torch.Tensor:
        for _ in range(fid.ndim - phi0.ndim): phi0 = phi0.unsqueeze(-1)
        return complex_exp(fid, -1*phi0.deg2rad())


    def forward(self, 
                params: torch.Tensor, 
                diff_edit: torch.Tensor=None,
                b0: bool=True,
                gen: bool=True,
                eddy: bool=False,
                fids: bool=False,
                phi0: bool=True,
                phi1: bool=True,
                noise: bool=True,
                apodize: bool=False,
                offsets: bool=True,
                fshift_g: bool=True,
                fshift_i: bool=True,
                resample: bool=True,
                snr_both: bool=False,
                baselines: dict=None, 
                coil_sens: bool=False,
                magnitude: bool=True,
                snr_combo: str=False,
                zero_fill: int=False,
                broadening: bool=True,
                transients: bool=False,
                residual_water: dict=None,
                drop_prob: float=None,
               ) -> torch.Tensor:
        
        if params.ndim>=3: params = params.squeeze()  # convert 3d parameter matrix to 2d [batchSize, parameters]
        if params.ndim==1: params = params.unsqueeze(0) # Allows for batchSize = 1

#         params = self.quantify_params(params, label='forward')

        # B0 inhomogeneities
        if b0:
            # Memory limitations require this to be calculated either before or after the spectra
            B0 = self.B0_inhomogeneities(b0=params[:,self.index['b0']],
                                         param=params[:,self.index['b0_dir']])

        # Simulate the Residual Water and Baselines
        if offsets:
            offset = self.simulate_offsets(baselines=baselines, 
                                           residual_water=residual_water, 
                                           drop_prob=drop_prob)

        # Define basis spectra coefficients
        if gen: print('>>>>> Preparing metabolite coefficients')
        fid = self.modulate(fids=self.syn_basis_fids, params=params[:,self.index['metabolites']])
        if not isinstance(diff_edit, type(None)): 
            fid = torch.stack((fid, self.modulate(fids=self.difference_editing_fids, 
                                                  params=diff_edit[:,self.index['metabolites']])),
                              dim=1)

        # Apply B0 inhomogeneities
        if b0: fid *= B0
        
        # Line broadening
        if broadening:
            if gen: print('>>>>> Applying line shape distortions')
            fid = self.lineshape_correction(fid, params[:,self.index['d']], 
                                                 params[:,self.index['g']])

        # Basis Function-wise Frequency Shift
        if fshift_i:
            if gen: print('>>>>> Shifting individual frequencies')
            fidSum = self.frequency_shift(fid=fid, 
                                          param=params[:,self.index['f_shifts']])
            
        # Summing the basis lines
        # # Saves the unadulterated original as the spectral_fit and the max values
        # # for the SNR calculations which should not consider artifacts.
        l = len(self._metab) - self.MM if self.MM else len(self._metab)
        fidSum, spectral_fit, mx_values = self.line_summing(fid=fid, 
                                                            params=params, 
                                                            mm=self.MM, 
                                                            l=l)

        # Add the Residual Water and Baselines
        if offsets:
            if gen: print('>>> Offsets')
            fidSum, offset = self.add_offsets(fid=fidSum, 
                                              offsets=offset, 
                                              drop_prob=drop_prob)

        # Create the transient copies
        if transients:
            if gen: print('>>> Transients')
            fidSum = self.transients(fid=fidSum, 
                                     coil_sens=params[:,self.index['coil_sens']])
            spectral_fit = self.transients(fid=spectral_fit, 
                                           coil_sens=params[:,self.index['coil_sens']])

        # Add Noise
        if noise:
            if gen: print('>>>>> Adding noise')
            transient = params[:,self.index['transients']] if transients else None
            noise = self.generate_noise(fid=fidSum, 
                                        max_val=mx_values, 
                                        param=params[:,self.index['snr']], 
                                        transients=transient)
            if transients:
                if snr_combo=='both':
                    # output.shape: [bS, ON\OFF, [noisy, noiseless], transients, channels, length]
                    fidSum = torch.stack((fidSum.clone() + noise, fidSum), dim=-4)
                elif snr_combo=='avg':
                    # Produces more realistic noise profile
                    # output.shape: [bS, ON\OFF, channels, length]  
                    fidSum = fidSum.mean(dim=-3)
                    spectral_fit = spectral_fit.mean(dim=-3)
            else:
                fidSum += noise

        # Scale with coil senstivities
        if coil_sens:
            if gen: print('>>> Coil sensitivity')
            fidSum = self.coil_sensitivity(fid=fidSum, 
                                           coil_sens=params[:,self.index['coil_sens']])
            spectral_fit = self.coil_sensitivity(fid=spectral_fit, 
                                                 coil_sens=params[:,self.index['coil_sens']])
            
        # Rephasing Spectrum
        if phi0:
            if gen: print('>>>>> Rephasing spectra - zero-order')
            fidSum = self.zeroOrderPhase(fid=fidSum, 
                                         phi0=params[:,self.index['phi0']])

        # Rephasing Spectrum
        if phi1:
            if gen: print('>>>>> Rephasing spectra - first-order')
            fidSum = self.firstOrderPhase(fid=fidSum, 
                                          phi1=params[:,self.index['phi1']])
        
        # Frequency Shift
        if fshift_g:
            if gen: print('>>>>> Shifting global frequencies')
            fidSum = self.frequency_shift(fid=fidSum, 
                                          param=params[:,self.index['f_shift']])

        # Eddy Currents
        if eddy:
            fidSum = self.eddyCurrents(fid=fidSum, phase=params[:,self.index['ecc']])

        # Apodize
        if apodize:
            fidSum = self.apodization(fid=fidSum, hz=apodize)

        # Zero-filling
        if zero_fill:
            fidSum = self.zero_fill(fid=fidSum, fill=zero_fill)

        # Recover Spectrum
        if gen: print('>>>>> Recovering spectra')
        specSummed = Fourier_Transform(fidSum)
        spectral_fit = Fourier_Transform(spectral_fit)

        # Crop and resample spectra
        if resample and not fids:
            specSummed = self.resample_(specSummed, length=self.length)
            spectral_fit = self.resample_(spectral_fit, length=self.length)
                     
        # Calculate magnitude spectra
        if magnitude:
            if gen: print('>>>>> Generating magnitude spectra')
            specSummed = self.magnitude(specSummed)
            spectral_fit = self.magnitude(spectral_fit)
        
        # Normalize
        specSummed, denom = self.normalize(specSummed)
        spectral_fit, _ = self.normalize(spectral_fit, denom)

        # Convert normalized spectra back to time-domain
        if fids:
            specSummed = self.magnitude(inv_Fourier_Transform(specSummed[...,0:2,:]))
            spectral_fit = self.magnitude(inv_Fourier_Transform(spectral_fit[...,0:2,:]))
            if resample:
                num_pts = zero_fill if zero_fill else self.length
                t = torch.linspace(self.t.amin(), self.t.amax(), max(self.t.squeeze().shape))
                specSummed = self.resample_(specSummed.flip(dims=[-1]), length=self.length, ppm=t, 
                                            target_range=[self.t.amin(), self.t.amax()]).flip(dims=[-1])
                spectral_fit = self.resample_(spectral_fit.flip(dims=[-1]), length=self.length, ppm=t, 
                                              target_range=[self.t.amin(), self.t.amax()]).flip(dims=[-1])


        if not isinstance(diff_edit, type(None)): 
            print('>>> Creating the difference spectra')
            # Consensus paper recommends dividing difference spectra by 2. Not sure about any other consequenctial effects
            specSummed = torch.cat(specSummed, (specSummed[:,0,...] - specSummed[:,1,...]).unsqueeze(1) / 2, dim=1)
            spectral_fit = torch.cat(spectral_fit, (spectral_fit[:,0,...] - spectral_fit[:,1,...]).unsqueeze(1) / 2, dim=1)
            
            
        print('>>>>> Compiling spectra')
        return self.compile_outputs(specSummed, spectral_fit, offset, params, denom, b0)

    def compile_outputs(self, 
                        specSummed: torch.Tensor, 
                        spectral_fit: torch.Tensor,
                        offsets: dict,
                        params: torch.Tensor, 
                        denom: torch.Tensor,
                        b0: bool,
                       ) -> torch.Tensor:
        if offsets:
            if not isinstance(offsets['baselines'], type(None)):
                offsets['baselines'], _ = self.normalize(offsets['baselines'], 
                                                         denom=denom)
            if not isinstance(offsets['residual_water'], type(None)):
                offsets['residual_water'], _ = self.normalize(offsets['residual_water'], 
                                                              denom=denom)

        offsets = torch2numpy(offsets)

        try: baselines = offsets['baselines']
        except Exception: baselines = None
        try: residual_water = offsets['residual_water']
        except Exception: residual_water = None
        quantities = torch2numpy(self.quantify_metab(params, b0=b0))

        return specSummed.numpy(), spectral_fit.numpy(), baselines, \
               residual_water, params.numpy(), quantities
