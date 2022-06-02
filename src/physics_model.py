import copy
import os
from collections import OrderedDict
from functools import reduce
from math import ceil, floor

import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
from numpy import pi
from types import SimpleNamespace

from .aux import *
from .baselines import bounded_random_walk
from .interpolate import CubicHermiteMAkima as CubicHermiteInterp 


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

class PhysicsModel(nn.Module):
    def __init__(self, 
                 PM_basis_set: str,
                 # opt,
                 # cropRange: tuple=(0,5), 
                 # length: int=512,
                 # ppm_ref: float=4.65
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
        paths = ['./src/basis_sets/' + PM_basis_set] # 'fitting_basis_ge_PRESS144.mat'

        for path in paths:
            with open(path, 'rb') as file:
                dct = convertdict(io.loadmat(file))
                for key, value in dct.items():
                    # print('key: ',key)
                    if str(key)=='ppm': key, value = '_ppm', torch.flip(value.unsqueeze(0), dims=[-1,0]).squeeze(0)
                    if not isinstance(value, dict):
                        self.register_buffer(str(key), value.float())
                    if key=='linenames':
                        self.linenames = value

        
    def __repr__(self):
        lines = sum([len(listElem) for listElem in self.totals])   
        out = 'MRS_PhysicsModel(basis={}, lines={}'.format('Osprey, GE_PRESS144', lines)
        return out + ', range={}ppm, resample={}pts)'.format(self.cropRange, self.length)
    
    @property
    def metab(self):
        temp = [x for x, _ in enumerate(self.totals)]
        return self._metab, temp#.append(temp)
    
    @property
    def basis_metab(self):
        temp = [x for x, _ in enumerate(self.totals)]
        metab_list = []
        for i, k in enumerate(self._basis_metab):
            if not ('mm' in k.lower() or 'lip' in k.lower()):
                metab_list.append(k)
        return metab_list, temp[0:len(metab_list)]#.append(temp)    

       
    def initialize(self, 
                   metab: list=['Cho','Cre','Naa','Glx','Ins','Mac','Lip'],
                   cropRange: list=[0,5],
                   length: float=512,
                   ppm_ref: float=4.65,
                   spectral_resolution: list=[10.0, 10.0, 10.0],
                   image_resolution: list=[0.5, 0.5, 0.5],
                  ):
        '''
        Steps: 
        - based on the desired fitting range, calculate the necessary number of splines
        - initialize the difference matrix
        - prepare the parameter index
        - prepare the parameter dictionary template
        - prepare the fids
        
        Modes:
        - supervised: Supervised models use different variables and are intended for supervised fitting
        - not supervised: Unsupervised fitting using p-splines and scaling, no noise
        - metabonly: Only includes the metabolite coefficients
        - metabfit: Intended for the MLPs, includes metabolite coefficients and Voigt lineshape profiles
        '''
        self._metab = metab

        self.register_buffer('l', torch.FloatTensor([self.fid.shape[-1]]).squeeze())
        self.register_buffer('length', torch.FloatTensor([length]).squeeze())
        self.register_buffer('ppm_ref', torch.FloatTensor([ppm_ref]).squeeze())
        self.register_buffer('ppm_cropped', torch.fliplr(torch.linspace(cropRange[0], cropRange[1], length).unsqueeze(0)))

        freq_ref = ppm_ref * self.carrier_frequency / 10e6
        spectralwidth = torch.linspace(-0.5*self.spectralwidth, 0.5*self.spectralwidth, int(self.l))
        phi1_ref = 2 * PI * spectralwidth / (spectralwidth + freq_ref)

        self.register_buffer('phi1_ref', torch.as_tensor(phi1_ref, dtype=torch.float32).squeeze())
        self.spectral_resolution = spectral_resolution
        self.image_resolution = image_resolution


        '''
        bandwidth = 1/dwelltime
        carrier_freq = 127.8 # MHz
        freq_ref = 4.68 * carrier_freq / 10e6
        degree = 2*pi*bandwidth*(1/(bandwidth + freq_ref))
        
        freq_ref = ppm_ref * self.carrier_freq / 10e6
        phi1_ref = 2*self.PI*self.spectralwidth * (torch.linspace(-0.5*self.spectralwidth, 0.5*self.spectralwidth, self.l) + freq_ref)
        '''
        self.ppm.float()
        self.cropRange = cropRange
        self.t = self.t.unsqueeze(-1).float()
        self._basis_metab = []


        ind, indA, dict, self.totals = [], [], OrderedDict([]), []
        self.MM = -1
        for m in metab: 
            if 'Cho' in m: m = 'Ch'
            if 'Cre' in m: m = 'Cr'
            if 'Mac' in m: m = 'MM'
            temp = []
            for i, (name, value) in enumerate(self.linenames.items()):
                # print(name, value)
                case1 = (m.lower() in name.lower()[2:-2])
                case2 = (name.lower()[2:-2] in m.lower())
                if case1 and case2:
                    if (m=='Ch' and 'Cr' not in name) or \
                        (m!='Ch') or ('mm' in name.lower()) or \
                        ('lip' in name.lower()):
                        ind.append(value), indA.append(int(i))
                        temp.append(int(len(ind)-1))
                        dict.update({name[2:-2]: torch.empty(1)})
                        self._basis_metab.append(name[2:-2])
                    if 'mm' in name.lower() or 'lip' in name.lower():
                        self.MM += 1
            self.totals.append(temp)
        self.MM = self.MM + 1 if  self.MM>-1 else False
        
        self.syn_basis_fids = torch.as_tensor(self.fids[tuple(torch.cat([x for x in ind],  dim=0).long()),::], dtype=torch.float32).unsqueeze(0)
        l = len(ind)
        assert(self.syn_basis_fids.ndim==4)
        header, cnt = [], counter(start=int(3*l)-1)
        # print(cnt)
        g = 1 if not self.MM else 2
        names = ['_d','_g','fshift','scale','snr','phi0','phi1','b0','bO_dir']
        mult  = [   l,   l,       g,      1,    1,     1,     1,   1,       3]
        for n, m in zip(names, mult):
            for _ in range(m): header.append(n)

        for m in header:
            for i, name in enumerate(self.linenames.keys()):
                if name.lower()[2:-2] in m.lower(): 
                    indA.append(int(i))

        self.max_ranges = self.max_ranges.unsqueeze(0)[:,tuple(indA)].float()
        self.min_ranges = self.min_ranges.unsqueeze(0)[:,tuple(indA)].float()

        # Metabolites
        ind = list(int(x) for x in torch.arange(0,l))
        
        # Line shape corrections 
        ind.append(tuple(int(x) for x in torch.arange(0,l) + l)) # Lorentzian corrections
        ind.append(tuple(int(x) for x in torch.arange(0,l) + 2*l)) # Single Gaussian correction factor

        # Frequency Shift / Scaling Factor / Noise depending on supervised T/F
        ind.append(tuple(cnt(1) for _ in range(2 if self.MM else 1))) # Fshift
        ind.append(cnt(1)) # Scaling
        ind.append(cnt(1)) # SNR

        # # Phase
        ind.append(cnt(1)) # Phi0
        ind.append(cnt(1)) # Phi1

        # # B0 inhomogeneities
        ind.append(tuple(int(x) for x in torch.arange(0,4) + cnt(0))), cnt(4)

        # Cummulative
        total = cnt(1)

        ind.append(tuple(int(x) for x in torch.arange(0,l)))        # Metabolites
        ind.append(tuple(int(x) for x in torch.arange(l,total)))    # Parameters
        ind.append(tuple(int(x) for x in torch.arange(0,total)))    # Overall

        dict.update({'D': torch.empty(1), 'G': torch.empty(1), 'F_Shift': torch.empty(1), 'Scale': torch.empty(1), 
                     'SNR': torch.empty(1), 'Phi0': torch.empty(1), 'Phi1': torch.empty(1), 'B0': torch.empty(1),
                     # 'B0_dir': torch.empty(1), 
                     'Metabolites': torch.empty(1), 'Parameters': torch.empty(1), 
                     'Overall': torch.empty(1)})
        self._index = OrderedDict({d.lower(): i for d,i in zip(dict.keys(),ind)})

        return dict, ind
    
    @property
    def index(self):
        return self._index

    def apodization(self,
                    fid: torch.Tensor, 
                    hz: int=4,
                   ) -> torch.Tensor:
        exp = torch.exp(-self.t * hz).t()
        for _ in range(fid.ndim - exp.ndim): exp = exp.unsqueeze(0)
        return fid * exp.expand_as(fid)
        

    def add_artifacts(self,
                      baselines: torch.Tensor,
                      params: torch.Tensor=None,
                     ) -> torch.Tensor:
        '''Spectra are normalized to [-1,1], therefore, the splines need to be able to cover that distance'''
        return self.frequency_shift(self.dephase(HilbertTransform(baselines, dim=-1), # shape: [batchSize, num_splines, spectrum_length]
                                                 phi=params[:,(self.index['phi0'],self.index['phi1'])],
                                                 ppm=self.ppm_cropped,
                                                 quantify=quantify,
                                                 baseline=True),
                                    param=params[:,self.index['f_shift']],
                                    quantify=quantify,
                                    t=self.cropped_t)


    def add_offsets(self,
                    fid: torch.Tensor,
                    baselines: dict=None,
                    residual_water: dict=None,
                    drop_prob: float=0.2) -> torch.Tensor:
        '''
        Used for adding residual water and baselines. config dictionaries are needed for each one.
        '''
        if baselines: 
            print('>>>>> Baselines')
            baselines, raw_baselines = self.baselines(baselines)
            # print('baselines.amin() {}, baselines.amax() {}'.format(baselines.amin(), baselines.amax()))
            out = raw_baselines.clone()
        else: baselines = None

        if residual_water: 
            print('>>>>> Residual Water')
            res_water, raw_res_water = self.residual_water(residual_water)
            if torch.is_tensor(out): out += raw_res_water.clone()
            else: out = raw_res_water.clone()
            # print('res_water.amin() {}, res_water.amax() {}'.format(res_water.amin(), res_water.amax()))
        else: res_water = None

        if drop_prob: 
            out, ind = rand_omit(out, 0.0, drop_prob)
            if baselines: baselines[ind] = 0.0
            if res_water: res_water[ind] = 0.0

        offset = out.clone()
        scale = 10**(OrderOfMagnitude(fid) - OrderOfMagnitude(out)) * 5

        return {'baselines': baselines * scale, 'residual_water': res_water * scale, 
                'offset': offset * scale, 'offset_fid': inv_Fourier_Transform(out) * scale}


    def sim2acquired_inefficient(self, 
                                 line: torch.Tensor, 
                                 sim_range: list, 
                                 target_ppm: torch.Tensor,
                                ) -> torch.Tensor:
        ppm_raw = [target_ppm.min(), target_ppm.max()]
        pt_per_ppm = line.shape[-1] / (sim_range[1] - sim_range[0]) # points per ppm #target_ppm.shape[-1]
        pt_density = target_ppm.shape[-1] / (target_ppm.amin() - target_ppm.amax())

        pad_left  = ((sim_range[0] - ppm_raw[0]) * pt_per_ppm).ceil()
        pad_right = ((ppm_raw[1] - sim_range[1]) * pt_per_ppm).ceil()

        for i, pad in enumerate([pad_left, pad_right]):
            cropRange = []
            targetRange = []
            if i==0:
                padding = tuple(int(pad), 0)  
                cropRange.append(sim_range[0] - pad / pt_per_ppm)
                cropRange.append(sim_range[1])
                targetRange = [target_ppm[0], sim_range[1]]
            else:
                padding = tuple(0, int(pad))
                cropRange.append(sim_range[0])
                cropRange.append(sim_range[1] + pad / pt_per_ppm)
                targetRange = [target_ppm[0], target_ppm[1]]

            line = torch.nn.functional.pad(input=line, pad=padding)
            xaxis = torch_batch_linspace(cropRange[0], cropRange[1], int(line.shape[-1]))
            length = (targetRange[1] - targetRange[0]) * pt_density
            new_ppm = torch_batch_linspace(targetRange[0], targetRange[1], int(length))
            ch_interp = CubicHermiteInterp(xaxis=xaxis, signal=line)
            line = ch_interp.interp(xs=new_ppm)

        return line

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
        if sim_range[0].shape[0]!=line.shape[0]: sim_range[0] = sim_range[0].repeat(line.shape[0])
        if sim_range[1].shape[0]!=line.shape[0]: sim_range[1] = sim_range[1].repeat(line.shape[0])
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
            baselines = rand_omit(baselines, 0.0, cfg.rand_omit)

        ch_interp = CubicHermiteInterp(xaxis=torch.linspace(self.cropRange[0], self.cropRange[1], cfg.length),
                                       signal=baselines)

        out = ch_interp.interp(xs=self.ppm_cropped.fliplr()) * config['scale']
        ppm = self._ppm.clone().unsqueeze(-1).repeat(baselines.shape[0],1)

        raw_baseline = self.sim2acquired(out, [ppm.amin(keepdims=True), ppm.amax(keepdims=True)], self.ppm)

        return out.fliplr(), raw_baseline.fliplr()
    
    
    def B0_inhomogeneities(self, 
                           # fid: torch.Tensor, 
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

        spectral_resolution = [10.0, 10.0, 10.0]
        image_resolution = [1,1,1]#[0.5, 0.5, 0.5]

        # num_pts = [m/n for m, n in zip(self.spectral_resolution, self.image_resolution)]
        num_pts = [int(m/n) for m, n in zip(spectral_resolution, image_resolution)]
        mean = param[...,0]#.unsqueeze(-1)
        dx = param[...,1]#.unsqueeze(-1)
        dy = param[...,2]#.unsqueeze(-1)
        dz = param[...,3]#.unsqueeze(-1)
        # print(mean.shape, dx.shape)
        x = torch_batch_linspace(1 - dx, 1 + dx, num_pts[0]).permute(0,2,1).unsqueeze(-1) + (mean - 1)
        y = torch_batch_linspace(1 - dy, 1 + dy, num_pts[1]).unsqueeze(-1) + (mean - 1)
        z = torch_batch_linspace(1 - dz, 1 + dz, num_pts[2]).unsqueeze(-1).permute(0,1,3,2) + (mean - 1)

        # print('x.shape: ',x.shape, y.shape, z.shape)

        dB0 = torch.ones([1, num_pts[0], num_pts[1], num_pts[2]]).to(param.device) * x * y * z
        # dB0 = base * x * y * z
        # print('dB0.shaped: ',dB0.shape)
        dB0 = dB0.unsqueeze(1).flatten(start_dim=2, end_dim=-1).unsqueeze(-1) * t
        # print('dB0.shaped: ',dB0.shape)
        # print('t.shaped: ',t.shape)
        # dB0 = dB0 * t#.mT
        # print('dB0.shaped: ',dB0.shape)
        # dB0 = dB0.unsqueeze(-2)
        # print('dB0.shaped: ',dB0.shape)

        # dB0 = ((x.mT * y).unsqueeze(-1) * z.unsqueeze(-2)).flatten(start_dim=2, end_dim=-1).mT
        # shape [bS, basisfcns, 8000, channel=1] * [1, 1, 1, 8192] = [bS, 1, 8000, 8192]
        # [bS, 1, 8000, 8192].unsqueeze(-2) = [bS, 1, 8000, 1, 8192]
        identity = torch.ones_like(t).repeat(1,1,2,1)

        return complex_exp(identity, -1*dB0.unsqueeze(-2)).sum(dim=-3)


    def dephase(self,
                spectra: torch.Tensor,
                phi: torch.Tensor,
                ppm: torch.Tensor=None,
                baseline: bool=False,
                quantify: bool=True,
                correction: bool=True,
               ) -> torch.Tensor:
        phi0, phi1 = phi[:,0].unsqueeze(-1), phi[:,1].unsqueeze(-1)
        mult = -1 if correction else 1
        if quantify:
            # Values are in the range [0,1] and the quantification ranges are zero-centered
            # Therefore, 1 - phi and -1 * phi0 are equivalent
            # phi0, phi1 = 1 - phi0, 1 - phi1
            phi0 = mult * ((self.phi0_max - self.phi0_min) * phi0 + self.phi0_min)
            phi1 = mult * ((self.phi1_max - self.phi1_min) * phi1 + self.phi1_min)
            
        ppm = self.ppm_cropped if isinstance(ppm, type(None)) else ppm

        if baseline: 
            spectra = inv_Fourier_Transform(spectra)
            spectra = self.zeroOrderPhase(fid=spectra, phi0=phi0)
            spectra = Fourier_Transform(spectra)
            return self.firstOrderPhase(spectra=spectra, phi1=phi1, ppm=ppm)
        
        if ppm[::,0]>ppm[::,-1]: ppm = ppm.flip(dims=[-1,-2]).to(spectra.device)
        spectra = self.firstOrderPhase(spectra=spectra, phi1=phi1, ppm=ppm)#.flip(dims=[-1,-2]))
        fid = inv_Fourier_Transform(spectra)
        fid = self.zeroOrderPhase(fid=fid, phi0=phi0)
        return Fourier_Transform(fid)


    def firstOrderPhase(self, 
                        spectra: torch.Tensor, 
                        phi1: torch.Tensor,
                        ppm=None,
                        dephase: bool=False,
                       ) -> torch.Tensor:
        # FID should be in the frequency domain for this step
        # mult = self.ppm_ref - self._ppm if ppm==None else self.ppm_ref - ppm
        # New implementation does this in the time domain!
        # mult = self.phi1_ref
        # for _ in range(spectra.ndim - mult.ndim): mult = mult.unsqueeze(0)
        for _ in range(spectra.ndim - self.phi1_ref.ndim): self.phi1_ref = self.phi1_ref.unsqueeze(0)
        for _ in range(spectra.ndim - phi1.ndim): phi1 = phi1.unsqueeze(-1)
        return complex_exp(spectra, -1*(phi1 + self.phi1_ref).deg2rad())
        '''
        carrier_freq = 127.8 # MHz
        freq_ref = 4.68 * carrier_freq / 10e6
        degree = 2*pi*bandwidth*(1/(bandwidth + freq_ref))
        '''
        
    def frequency_shift(self, 
                        fid: torch.Tensor, 
                        param: torch.Tensor,
                        t: torch.Tensor=None,
                       ) -> torch.Tensor:
        '''
        Do NOT forget to specify the dimensions for the (i)fftshift!!! Will reorder the batch samples!!!
        '''
        t = self.t if t==None else t
        t = t.t() if t.shape[-1]==1 else t
            
        for _ in range(fid.ndim - param.ndim): param = param.unsqueeze(-1)
        for _ in range(fid.ndim - t.ndim): t = t.unsqueeze(0)
        f_shift = param.mul(t)
        
        return complex_exp(fid, -1*f_shift)
        # # Convert back to time-domain
        # fid = inv_Fourier_Transform(fid)
        # # Apply TD complex exponential
        # fid = complex_exp(fid, f_shift)# -1*f_shift)
        # # Return to the frequency domain
        # return Fourier_Transform(fid)
        
        
    def generate_noise(self, 
                       fid: torch.Tensor, 
                       param: torch.Tensor,
                       max_val: torch.Tensor,
                       transients: torch.Tensor=None,
                      ) -> torch.Tensor:
        '''
        RMS coefficient is used because this is done in the time domain with sinusoids
        SNR formula:
            snr_db = 10*log10(snr_lin * 0.66) # 0.66 Rayleigh distribution correction factor to calculate the true SNR
            snr_lin = max(real(spectra)) / std_noise # Not sure whether to use real or magnitude spectra
        '''
        for _ in range(fid.ndim-max_val.ndim): max_val = max_val.unsqueeze(-1)
        for _ in range(fid.ndim-param.ndim): param = param.unsqueeze(-1)


        lin_snr = 10**(param / 10) # convert from decibels to linear scale
        if not isinstance(transients, type(None)): 
            lin_snr = lin_snr * torch.sqrt(transients)
        k = 1 / lin_snr # scaling coefficient
        a_signal = torch.FloatTensor([2]).sqrt().pow(-1).to(fid.device) * max_val # RMS coefficient for sine wave
        scale = k * a_signal # signal apmlitude scaled for desired noise amplitude
        scale[torch.isnan(scale)] = 1e-6
        if scale.ndim==3:
            if scale.shape[-2]==1 and scale.shape[-1]==1: scale = scale.squeeze(-1)

        e = torch.distributions.normal.Normal(0,scale).rsample([fid.shape[-1]])
        if e.ndim==2: e = e.unsqueeze(1).permute(1,2,0)
        elif e.ndim==3: e = e.permute(1,2,0)
        elif e.ndim==4: e = e.permute(1,2,3,0).repeat_interleave(fid.shape[1], dim=1)

        return HilbertTransform(e)


    def snr_lin2db(self, snr):
        return 10*torch.log10(snr)
    

#     @property
    def snr_var(self, 
                spectra: torch.Tensor, 
                SNR: torch.Tensor,
               ) -> torch.Tensor:
        std_noise = torch.FloatTensor([2]).sqrt().pow(-1).to(spectra.device) * spectra[:,0,:].max(dim=-1,keepdim=True).values.unsqueeze(-1).div(10**SNR) / 10
        return std_noise.pow(2)
    

#     @property
    def noise_var(self, 
                  fid: torch.Tensor, 
                  param: torch.Tensor,
                 ) -> torch.Tensor:
        if param.ndim==3: param = param.squeeze(-1)
        lin_snr = 10**(param[:,self.index['snr']].unsqueeze(-1).unsqueeze(-1) / 10) # convert from decibels to linear scale
        k = 1 / lin_snr # scaling coefficient
        a_signal = torch.FloatTensor([2]).sqrt().pow(-1).to(fid.device) * torch.max(fid[:,0,:].unsqueeze(1), dim=-1, keepdim=True).values#.unsqueeze(-1) # RMS coefficient for sine wave
        scale = k * a_signal # signal apmlitude scaled for desired noise amplitude
        scale[torch.isnan(scale)] = 1e-6
        return scale.pow(2)


    def lineshape_correction(self, 
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
        
        return fid * torch.exp((-d - g * t.unsqueeze(0)) * t.unsqueeze(0))#d + g * t.unsqueeze(0)) * t.unsqueeze(0))# 


    def lineshape_lorentzian(self, 
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
        if d.dtype==torch.float64: d = d.float()
        
        return fid * torch.exp(-d * t.unsqueeze(0))


    def lineshape_gaussian(self, 
                           fid: torch.Tensor, 
                           d: torch.Tensor, 
                           g: torch.Tensor
                          ) -> torch.Tensor:
        '''
        In a Voigt lineshape model, each basis line has its own Lorentzian value. Fat- and Water-based 
        peaks use one Gaussian value per group.
        '''
        t = self.t.clone().t().unsqueeze(0)
        g = g.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2,1)
        
        return fid * torch.exp(-g * t.unsqueeze(0).pow(2)) 


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

        return signal / denom, denom


    @property
    def ppm(self, 
            cropped: bool=False
           ) -> torch.Tensor:
        return self._ppm if not cropped else self.ppm_cropped

    def get_ppm(self, cropped: bool=False):
        return self.ppm_cropped if cropped else self._ppm


    def set_parameter_constraints(self, cfg):
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

    def quantify_params(self, 
                        params: torch.Tensor,
                        label=[]) -> torch.Tensor:
        delta = self.max_ranges - self.min_ranges
        minimum = self.min_ranges.clone()
        '''Allows partial, but structured, parameter quantification: can remove variables from right to left only'''
        if params.shape[1]<delta.shape[1]: 
            if params.ndim==3 and params.shape[-1]==1: params = params.squeeze(-1)
            delta = delta[:,0:params.shape[1]]
            minimum = minimum[:,0:params.shape[1]].expand_as(params)
        params = params.mul(delta.expand_as(params)) + minimum

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
            B0 = self.B0_inhomogeneities(fid, params[:,self.index['b0']])

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
        spec = self.magnitude(Fourier_Transform(fid), normalize=True)
        
        # Scale the lines to the magnitude of the input spectra
        if norm: spec = spec.mul(norm)

        # Combine related metabolite lines
        specs = spec[:,tuple(self.totals[0]),0,:].sum(dim=1, keepdim=True).unsqueeze(-2)
        for ind in range(1, len(self.totals)):
            specs = torch.cat([specs, spec[:,tuple(self.totals[ind]),0,:].sum(dim=1, keepdim=True).unsqueeze(-2)], dim=1)

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

        chs_interp = CubicHermiteInterp(ppm, torch.flip(signal, dims=[-1]))#self.ppm, torch.fliplr(signal))
        
        return torch.flip(chs_interp.interp(new.expand(dims)), dims=[-1])

    def _resample_(self, 
                   signal: torch.Tensor, 
                   scale: torch.Tensor, 
                   ppm: torch.Tensor=None,
                   quantify: bool=False,
                   length: int=512,
                  ) -> torch.Tensor:
        '''
        The unit of scale is ppm/point. Multiplying the scale by the desired number of points, length,  
        gives the number of ppm needed for the new range. Adding the starting point gives the new end point.
        '''
        dims, dims[-1] = list(signal.shape), -1
        if quantify: 
            scale = scale * 0.6 - 0.3
            scale = scale.float()
        ppm = ppm.unsqueeze(0).squeeze() if not isinstance(ppm, type(None)) else self.ppm.unsqueeze(0).squeeze()
        if not (ppm[0]<ppm[-1]): ppm = torch.flip(ppm.unsqueeze(0), dims=[-1]).squeeze(0)
        stop = self.cropRange[1] + scale.unsqueeze(-1)
        if stop.ndim==4: stop = stop.squeeze(-1)
        start = torch.as_tensor([self.cropRange[0]], device=scale.device)
        new_ppm_range = torch_batch_linspace(start=start, stop=stop, steps=length).to(signal.device)
        
        chs_interp = CubicHermiteInterp(ppm, signal)#torch.flip(signal, dims=[-1]))
        return torch.flip(chs_interp.interp(new_ppm_range.expand(dims)), dims=[-1])
    
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

        return res_water.fliplr(), raw_res_water.fliplr()

            
    def transients(self, 
                   fid: torch.Tensor, 
                   params: torch.Tensor,
                   snr: torch.Tensor,
                  ) -> torch.Tensor:
        '''
        The SNR dB value provided is the SNR of the final, coil combined spectrum. Therefore, each of the 
        transients will have a much higher linear SNR that is dependent upon the expected final SNR and 
        the number of transients being simulated.
        '''
        assert(fid.ndim==3)
        num_transients = params.shape[-1]
        fid = fid.unsqueeze(1).repeat_interleave(repeats=params.shape[-1], dim=1)
        max_val = fid.max(-1, keepdims=True).values
        for _ in range(fid.ndim - params.ndim): params = params.unsqueeze(-1)
        out = fid * params
        return out + self.generate_noise(fid, param, max_val, transients=num_transients)
            

    def zeroOrderPhase(self, 
                       fid: torch.Tensor, 
                       phi0: torch.Tensor,
                      ) -> torch.Tensor:
        for _ in range(fid.ndim - phi0.ndim): phi0 = phi0.unsqueeze(-1)
        return complex_exp(fid, -1*phi0.deg2rad())
    

    def modulate(self, 
                 params: torch.Tensor,
                ) -> torch.Tensor:
        return params.unsqueeze(-1).unsqueeze(-1).mul(self.syn_basis_fids)


    def forward(self, 
                params: torch.Tensor, 
                b0: bool=True,
                gen: bool=True, 
                phi0: bool=True,
                phi1: bool=True,
                noise: bool=True,
                scale: bool=True,
                fshift: bool=True,
                apodize: bool=False,
                offsets: bool=True,
                baselines: dict=None, 
                magnitude: bool=True,
                broadening: bool=True,
                transients: bool=False,
                residual_water: dict=None,
                drop_prob: float=None,
               ) -> torch.Tensor:
        
        if params.ndim>=3: params = params.squeeze()  # convert 3d parameter matrix to 2d [batchSize, parameters]
        if params.ndim==1: params = params.unsqueeze(0) # Allows for batchSize = 1

        params = self.quantify_params(params, label='forward')

        if b0:
            # Memory limitations require this to be calculated either before or after the spectra
            B0 = self.B0_inhomogeneities(fid, params[:,self.index['b0']])

        # Define basis spectra coefficients
        if gen: print('>>>>> Preparing metabolite coefficients')
        fid = self.modulate(params[:,self.index['metabolites']])

        if b0: fid *= B0
        
        # Line broadening
        if broadening:
            if gen: print('>>>>> Applying line shape distortions')
            fid = self.lineshape_correction(fid, params[:,self.index['d']], 
                                            params[:,self.index['g']])
                
        # Summing the basis lines
        l = len(self._metab) - self.MM if self.MM else len(self._metab)
        if not self.MM:
            fidSum = fid.sum(dim=-3) 
            spectral_fit = fidSum.clone()
            mx_values = torch.amax(fid[...,0,:].unsqueeze(-2).sum(dim=-3).unsqueeze(1), dim=-1) 
        else:
            mm = fid[:,l:,:,:].sum(dim=-3)
            fidSum = fid[:,0:l,:,:].sum(dim=-3)
            spectral_fit = fidSum.clone()
            mx_values = torch.amax(fidSum[...,0,:].unsqueeze(-2), dim=-1, keepdims=True) 
            print('type(mx_values): ',type(mx_values))
            # print(self.index['f_shift'])#, params[:,self.index['f_shift']][1])
            fidSum = fidSum + self.frequency_shift(mm, params[:,self.index['f_shift'][1]])
        # Save these values for the SNR calculation. Should only consider metabolites, not artifacts!


        # Scaling and Adding the Residual Water and Baselines
        if offsets:
            offsets = self.add_offsets(fid=fidSum, baselines=baselines, 
                                       residual_water=residual_water, drop_prob=drop_prob)
            if torch.is_tensor(offsets['offset_fid']):
                # print('fidSum.shape {}, offsets["offset_fid"].shape {}'.format(fidSum.shape,offsets['offset_fid'].shape))
                assert(fidSum.shape==offsets['offset_fid'].shape)
                fidSum += offsets['offset_fid'].clone()
                offsets.pop('offset_fid')
                # This last part calculates the difference in order of magnitude between the offsets and
                # the FIDs they are being added to. Then the offsets are scaled up to match the magnitude
        
            check(fidSum, 'offsets')
        # # Make sure that the noise generator considers transients properly -> specifically the mx_values!!!
        if transients:
            fidSum = self.transients(fidSum, params[:,self.index['transients']])

        # Add Noise
        if noise:
            if gen: print('>>>>> Adding noise')
            fidSum += self.generate_noise(fidSum, params[:,self.index['snr']], mx_values)
            check(fidSum, 'noise')

        # Rephasing Spectrum
        if phi0:
            if gen: print('>>>>> Rephasing spectra')
            fidSum = self.zeroOrderPhase(fidSum, params[:,self.index['phi0']])
            check(fidSum, 'phi0')

        # Rephasing Spectrum
        if phi1:
            if gen: print('>>>>> Rephasing spectra')
            fidSum = self.firstOrderPhase(fidSum, params[:,self.index['phi1']])
            check(fidSum, 'phi1')
        
        # # Apodize
        if apodize:
            fidSum = self.apodization(fidSum, hz=apodize)

        # Frequency Shift
        if fshift:
            if gen: print('>>>>> Shifting frequencies')
            fidSum = self.frequency_shift(fidSum, params[:,self.index['f_shift'][0]])
            check(fidSum, 'fshift')

        # Recover Spectrum
        if gen: print('>>>>> Recovering spectra')
        specSummed = Fourier_Transform(fidSum)
        spectral_fit = Fourier_Transform(spectral_fit)#.flip(-1,-2)
        check(specSummed, 'fft')

        # # Rephasing Spectrum
        # if phi1:
        #     if gen: print('>>>>> Rephasing spectra')
        #     specSummed = self.firstOrderPhase(specSummed, params[:,self.index['phi1']])

        # # Frequency Shift
        # if fshift:
        #     if gen: print('>>>>> Shifting frequencies')
        #     specSummed = self.frequency_shift(specSummed, params[:,self.index['f_shift']])

        # Resampling the spectra
        # if scale:
        #     specSummed = self._resample_(specSummed.flip(-1,-2), length=self.length, scale=params[:,self.index['scale']])
        #     spectral_fit = self._resample_(spectral_fit.flip(-1,-2), length=self.length, scale=params[:,self.index['scale']])
        #     check(specSummed, '_resample_')
        # else:
        specSummed = self.resample_(specSummed.flip(-1,-2), length=self.length)
        spectral_fit = self.resample_(spectral_fit.flip(-1,-2), length=self.length)
        check(specSummed, 'resample_')
                     
        if magnitude:
            if gen: print('>>>>> Generating magnitude spectra')
            specSummed = self.magnitude(specSummed)#, normalize=True)
            spectral_fit = self.magnitude(spectral_fit)
            check(specSummed, 'magnitude')
        # else:
        
        specSummed, denom = self.normalize(specSummed)
        spectral_fit, _ = self.normalize(spectral_fit, denom)
        check(specSummed, 'normalize')

        return self.compile_outputs(specSummed.flip(-1,-2), spectral_fit.flip(-1,-2), offsets, params, denom, b0)


    def compile_outputs(self, 
                        specSummed: torch.Tensor, 
                        spectral_fit: torch.Tensor,
                        offsets: dict,#(torch.Tensor,...), 
                        params: torch.Tensor, 
                        denom: torch.Tensor,
                        b0: bool,
                       ) -> torch.Tensor:
        if offsets:
            offsets['baselines'], _ = self.normalize(offsets['baselines'], denom=denom)
            offsets['residual_water'], _ = self.normalize(offsets['residual_water'], denom=denom)

        offsets = torch2numpy(offsets)
        # if offsets:
        try: baselines = offsets['baselines']
        except Exception: baselines = None
        try: residual_water = offsets['residual_water']
        except Exception: residual_water = None
        quantities = torch2numpy(self.quantify_metab(params, b0=b0))

        return specSummed.numpy(), spectral_fit.numpy(), baselines, residual_water, params.numpy(), quantities
