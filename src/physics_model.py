import copy
import os
from collections import OrderedDict
from functools import reduce

import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
from src.aux import *
from src.interpolate import CubicHermiteMAkima as CubicHermiteInterp  # , batch_linspace

__all__ = ['PhysicsModel'] #


const = torch.FloatTensor([1.0e6]).squeeze()
zero = torch.FloatTensor([0.0]).squeeze().float()
def check(x, label):
    x = x.clone()
    x = x.float()
    if torch.isnan(x).any(): print(label,': NaN found')
    if not torch.isfinite(x).all(): print(label,': Inf found')
    if (x>1e6).any(): 
        print(x.dtype, const.dtype, zero.dtype)
        a = torch.where(x>const.to(x.device), x, zero.to(x.device))
        a = a.float()
        ind = a.nonzero()
#         ind = torch.where(x>1.0e6, x, 0.0).float().nonzero()
        print(label,': Value greater than 1e6: ',x[ind])



class PhysicsModel(nn.Module):
    def __init__(self, 
                 opt,
                 cropRange: tuple=(0,5), 
                 length: int=512,
                 apodization: int=False):
        super(PhysicsModel, self).__init__()
        '''
        Args:
            cropRange:    specified fitting range
            length:       number of data points in the fitted spectra
            apodization:  amount of apodization in Hz. Should only be included if 
                          noise=True in the forward pass
        '''
        # Load basis spectra, concentration ranges, and units
        paths = ['./dataset/basis_spectra/' + opt.PM_basis_set] # 'fitting_basis_ge_PRESS144.mat'

        for path in paths:
            with open(path, 'rb') as file:
                dct = convertdict(io.loadmat(file))
                for key, value in dct.items():
#                     print('key: ',key)
                    if str(key)=='ppm': key, value = '_ppm', torch.flip(value.unsqueeze(0), dims=[-1,0]).squeeze(0)
                    if not isinstance(value, dict):
                        self.register_buffer(str(key), value.float())
                    if key=='linenames':
                        self.linenames = value

        self.register_buffer('l', torch.FloatTensor([8192]).squeeze())
        self.register_buffer('length', torch.FloatTensor([length]).squeeze())
        self.register_buffer('ppm_ref', torch.FloatTensor([4.68]).squeeze())
        self.register_buffer('ppm_cropped', torch.fliplr(torch.linspace(cropRange[0], cropRange[1], length).unsqueeze(0)))
        self.ppm.float()

        self.apodize = apodization        
        self.cropRange = cropRange

        self.t = self.t.unsqueeze(-1).float()
        self.baseline_t = self.baseline_t.unsqueeze(-1).float()
        self.cropped_t = self.cropped_t.unsqueeze(-1).float()
        self._basis_metab = []

        
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
        return self._basis_metab, temp#.append(temp)    

       
    def initialize(self, 
                   metab: list=['Cho','Cre','Naa','Glx','Ins','Mac','Lip'],
                   cropRange: list=[0,5],
                   splines: int=15,
                   length: float=512,
                   supervised: bool=False,
                   metabonly: bool=False,
                   metabfit: bool=False,
                   baselines: bool=True,
                   metab_as_is: bool=False,
                   nophase: bool=False):
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
        self.num_splines = round((cropRange[1] - cropRange[0]) * splines) + 3
        basis_splines = create_splines_linspace(self.length.clone().cpu().numpy(), self.num_splines-3, remove_last_spline=False)
        self.register_buffer('basis_splines', torch.from_numpy(basis_splines).float().unsqueeze(0))#[::,1:-1])
        self.difference_matrix()
        self.supervised = supervised

        ind, indA, dict, self.totals = [], [], OrderedDict([]), []
        for m in metab: 
            if 'Cho' in m: m = 'Ch'
            if 'Cre' in m: m = 'Cr'
            if 'Mac' in m: m = 'MM'
            temp = []
            for i, (name, value) in enumerate(self.linenames.items()):
                case1 = (m.lower() in name.lower()[2:-2])
                case2 = (name.lower()[2:-2] in m.lower() and metab_as_is)
                if (case1 and not metab_as_is) or (case1 and case2):
                    if (m=='Ch' and 'Cr' not in name) or m!='Ch':
                        ind.append(value), indA.append(int(i))
                        temp.append(int(len(ind)-1))
                        dict.update({name[2:-2]: torch.empty(1)})
                        self._basis_metab.append(name[2:-2])
            self.totals.append(temp)
        
        self.fitting_basis_fids, l = torch.as_tensor(self.fids[tuple(torch.cat([x for x in ind], dim=0).long()),::], dtype=torch.float32).unsqueeze(0), len(ind)
        self.syn_basis_fids        = torch.as_tensor(self.fid[tuple(torch.cat([x for x in ind],  dim=0).long()),::], dtype=torch.float32).unsqueeze(0)
        assert(self.fitting_basis_fids.ndim==4)
        assert(self.syn_basis_fids.ndim==4)
        
        if not supervised:
            self.cropRange = tuple([torch.as_tensor(cR) for cR in self.cropRange])
            header = []
            names = ['_d','_g','fshift','scale','phi0','phi1','baseline','lambda']
            mult = [l, 1, 1, 1, 1, 1, self.num_splines, 1]
            for n, m in zip(names, mult):
                for _ in range(m): header.append(n)
            for m in header: 
                for i, name in enumerate(self.linenames.keys()):
                    if m.lower() in name.lower():
                        indA.append(int(i))
            mx_ind = 2
        elif supervised: 
            self.syn_basis_fids, l = self.fid[tuple(torch.cat([x for x in ind], dim=0).long()),::].float(), len(ind)
            indB = tuple([i for i, name in enumerate(self.linenames.keys()) if 'baseline' in name.lower()] * self.baseline_basis_set.shape[0])
            self.max_baseline = self.max_ranges[1,indB]#5
            self.min_baseline = self.min_ranges[0,indB]#*5
            header = []
            names = ['_d','_g','fshift','snr','phi0','phi1','baseline']
            mult = [l, 1, 1, 1, 1, 1, 5]
            for n, m in zip(names, mult):
                for _ in range(m): header.append(n)
            for m in header:
                for i, name in enumerate(self.linenames.keys()):
                    if name.lower()[2:-2] in m.lower(): 
                        indA.append(int(i))
            mx_ind = 1
        elif metabfit:
            names = ['d']*l
            names.append('g')
            for m in names:
                for i, (name, value) in enumerate(self.linenames.items()):
                    if name.lower()[2:-2] in m.lower(): indA.append(int(i))
            mx_ind = 0

        # Min range uses the first row because it is mainly zeroed out allowing parameters to be turned off
        self.max_ranges = self.max_ranges[mx_ind,tuple(indA)].unsqueeze(0).float()
        self.min_ranges = self.min_ranges[mx_ind,tuple(indA)].unsqueeze(0).float()
        
        # Metabolites
        ind = list(int(x) for x in torch.arange(0,l))

        # Line shape corrections 
        ind.append(tuple(int(x) for x in torch.arange(0,l) + l)) # Lorentzian corrections
        ind.append(int(2 * l)) # Single Gaussian correction factor

        # Frequency Shift / Scaling Factor or Noise depending on supervised T/F
        ind.append(int(2 * l + 1))
        ind.append(int(2 * l + 2))

        # Phase
        if not nophase:
            ind.append(2 * l + 3)
            ind.append(2 * l + 4)
            n = 0
        else: 
            n = 2

        # Baseline splines and labmda
        if not supervised:
            ind.append(tuple(torch.arange(0,self.num_splines,1) + (2 * l + 5 - n)))
            total = int(ind[-1][-1] + 1)
            ind.append(total)

            # Cummulative
            ind.append(tuple(int(x) for x in torch.arange(0,l)))        # Metabolites
            ind.append(tuple(int(x) for x in torch.arange(l,total)))    # Parameters
            ind.append(tuple(int(x) for x in torch.arange(0,total)))    # Overall

            dict.update({'D': torch.empty(1), 'G': torch.empty(1), 'F_Shift': torch.empty(1), 'Scale': torch.empty(1), 'Phi0': torch.empty(1), 'Phi1': torch.empty(1), #'Noise': [],
                         'Baseline': torch.empty(1), 'Lambda': torch.empty(1), 'Metabolites': torch.empty(1), 'Parameters': torch.empty(1), 'Overall': torch.empty(1)})
            self._index = {d.lower(): i for d,i in zip(dict.keys(),ind)}
            self._index['snr'] = copy.deepcopy(self._index['scale'])
            
            return dict, ind
        
        if baselines: ind.append(tuple(int(x) for x in torch.arange(0,5,1) + (2 * l + 5 - n)))
        try: total = int(ind[-1][-1]+1)
        except TypeError: total = int(ind[-1]+1)

        # Cummulative
        ind.append(tuple(int(x) for x in torch.arange(0,l)))        # Metabolites
        ind.append(tuple(int(x) for x in torch.arange(l,total)))    # Parameters
        ind.append(tuple(int(x) for x in torch.arange(0,total)))    # Overall

        dict.update({'D': torch.empty(1), 'G': torch.empty(1), 'F_Shift': torch.empty(1), 'SNR': torch.empty(1), 'Phi0': torch.empty(1), 'Phi1': torch.empty(1)})
        if baselines: dict.update({'Baseline': torch.empty(1)})
        dict.update({'Metabolites': torch.empty(1), 'Parameters': torch.empty(1), 'Overall': torch.empty(1)})
        self._index = {d.lower(): i for d,i in zip(dict.keys(),ind)}
        
        
        if metabonly or metabfit: 
            ind, total, delete = [], [], ['D','G','F_Shift','SNR','Phi0','Phi1','Baseline','Parameters']
            if metabfit: delete = delete[2:]
            for val in delete: dict.pop(val)
            for k, v in self._index.items(): 
                if k in [key.lower() for key in dict.keys()] and k not in 'overall': 
                    ind.append(v)
                    '''This generates Overall indices regardless of which variables are included'''
                    if not isinstance(v, tuple): 
                        total.append(v)
                    else: 
                        for i in v: 
                            total.append(i)
            ind.append(tuple(set(total)))
            
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
    
    
    def B0_inhomogeneities(self, 
                           fid: torch.Tensor, 
                           param: torch.Tensor, # hz
                          ) -> torch.Tensor:
        t = self.t.clone()
        for _ in range(fid.ndim - t.ndim): t = t.unsqueeze(0)
        for _ in range(fid.ndim - param.ndim): param = param.unsqueeze(-1)
        if t.shape[-1]==1: t = t.t()
        return complex_exp(fid, param * t)


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
        mult = self.ppm_ref - self._ppm if ppm==None else self.ppm_ref - ppm
        for _ in range(spectra.ndim - mult.ndim): mult = mult.unsqueeze(0)
        for _ in range(spectra.ndim - phi1.ndim): phi1 = phi1.unsqueeze(-1)
        return complex_exp(spectra, phi1.deg2rad() * mult)
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
            
        for _ in range(fid.ndim - param.ndim): param = param.unsqueeze(-1)
        for _ in range(fid.ndim - t.ndim): t = t.unsqueeze(0)
        f_shift = param.mul(t.mT)#.mul(-1.0).mul(t.mT)
        
        # # Convert back to time-domain
        fid = inv_Fourier_Transform(fid)
        # # Apply TD complex exponential
        fid = complex_exp(fid, f_shift)# -1*f_shift)
        # # Return to the frequency domain
        return Fourier_Transform(fid)
        
        
    def generate_noise(self, 
                       fid: torch.Tensor, 
                       param: torch.Tensor,
                       max_val: torch.Tensor,
                      ) -> torch.Tensor:
        '''
        RMS coefficient is used because this is done in the time domain with sinusoids
        SNR formula:
            snr_db = 10*log10(snr_lin * 0.66) # 0.66 Rayleigh distribution correction factor to calculate the true SNR
            snr_lin = max(real(spectra)) / std_noise # Not sure whether to use real or magnitude spectra
        '''
        max_val = max_val.values
        for _ in range(fid.ndim-max_val.ndim): max_val = max_val.unsqueeze(-1)
        for _ in range(fid.ndim-param.ndim): param = param.unsqueeze(-1)
        lin_snr = 10**(param / 10) # convert from decibels to linear scale
        k = 1 / lin_snr # scaling coefficient
        a_signal = torch.FloatTensor([2]).sqrt().pow(-1).to(fid.device) * max_val # RMS coefficient for sine wave
        scale = k * a_signal # signal apmlitude scaled for desired noise amplitude
        scale[torch.isnan(scale)] = 1e-6

        e = torch.distributions.normal.Normal(0,scale).rsample([fid.shape[-1]])
        if e.ndim==2: e = e.unsqueeze(1).permute(1,2,0)
        elif e.ndim==3: e = e.permute(1,2,0)
        elif e.ndim==4: e = e.permute(1,2,3,0).repeat_interleave(fid.shape[1], dim=1)
        
        return HilbertTransform(e)
    
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
        assert(fid.ndim==3)
        t = self.t.clone().t().unsqueeze(0)
        d = d.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2,1)
        g = g.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(d)
        if d.dtype==torch.float64: d = d.float()
        
        return fid * torch.exp((-d - g * t.unsqueeze(0)) * t.unsqueeze(0))#d + g * t.unsqueeze(0)) * t.unsqueeze(0))# 

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
                  fid: bool=False
                 ) -> torch.Tensor:
        '''
        Normalize each sample of single or multi-echo spectra. 
            Step 1: Find the max of the real and imaginary components separately
            Step 2: Pick the larger value for each spectrum
        If the signal is separated by metabolite, then an additional max() is necessary
        Reimplemented according to: https://stackoverflow.com/questions/41576536/normalizing-complex-values-in-numpy-python
        '''
        if fid: 
            return signal / torch.max(torch.sqrt(out[...,0,:]**2 + out[...,1,:]**2), dim=-1, keepdim=True).values.unsqueeze(-2)

        mx = torch.max(signal[...,2,:].unsqueeze(-2).abs(), dim=-1, keepdim=True).values
        mx[mx.isnan()] = 1e-6
        mx[mx==0.0] = 1e-6
        return signal / (mx+1e-6)

    @property
    def ppm(self, 
            cropped: bool=False
           ) -> torch.Tensor:
        return self._ppm if not cropped else self.ppm_cropped

    def quantify_params(self, 
                        params: torch.Tensor,
                        label=[]) -> torch.Tensor:
        delta = self.max_ranges - self.min_ranges
        minimum = self.min_ranges.clone()
        '''Allows partial, but structured, parameter quantification: can remove variables from right to left only'''
        if params.shape[1]<<delta.shape[1]: 
            if params.ndim==3 and params.shape[-1]==1: params = params.squeeze(-1)
            delta = delta[:,0:params.shape[1]]
            minimum = minimum[:,0:params.shape[1]].expand_as(params)
        params = params.mul(delta.expand_as(params)) + minimum
        
        if not self.supervised:
            try: params[:, self.index['lambda']] = 1 / torch.sqrt(params[:, self.index['lambda']])
            except IndexError: pass
            try:
                params[:, self.index['scale']] -= 1
                params[:, self.index['scale']] *= 0.2
            except Exception as e: pass

        return params

    def quantify_metab(self, 
                       params: torch.Tensor, 
                       norm: torch.Tensor=None,
                       wrt_metab: str='cr',
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

        # Define basis spectra coefficients
        fid = params[:,self.index['metabolites']].unsqueeze(2).unsqueeze(-1) #* self.syn_basis_fids
        fid = torch.cat([fid.mul(self.syn_basis_fids[:,:,0,:].unsqueeze(2)),
                         fid.mul(self.syn_basis_fids[:,:,1,:].unsqueeze(2))], dim=2).contiguous()
        check(fid, 'quantify_metab basis line modulation')

#         Line broadening **if included in model**
        if params.shape[1]>len(self.index['metabolites']):
            fid = self.lineshape_correction(fid, params[:,self.index['d']],
                                            params[:,self.index['g']], 'quant_metab')
        
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
                 ) -> torch.Tensor:
        '''
        Basic Cubic Hermite Spline Interpolation of :param signal: with no additional scaling.

        :param signal: input torch tensor
        :param new: new target x-axis
        return interpolated signal
        
        I am flipping the ppm vector instead of the basis lines - 02.01.2022 JTL
        '''
        ppm = ppm.unsqueeze(0).squeeze() if not isinstance(ppm, type(None)) else self.ppm.unsqueeze(0).squeeze()
        if not (ppm[0]<ppm[-1]): ppm = torch.flip(ppm.unsqueeze(0), dims=[-1]).squeeze(0)
#         chs_interp = CubicHermiteInterp(torch.flip(ppm,dims=[0]), torch.flip(signal, dims=[0]))#self.ppm, torch.fliplr(signal))
        chs_interp = CubicHermiteInterp(ppm, torch.flip(signal, dims=[-1]))#self.ppm, torch.fliplr(signal))
#         chs_interp = CubicHermiteInterp(ppm, signal)#self.ppm, torch.fliplr(signal))
        new = torch.linspace(start=self.cropRange[0], end=self.cropRange[1], steps=int(length)).to(signal.device)
        for i in range(signal.ndim - new.ndim): new = new.unsqueeze(0)
        dims, dims[-1] = list(signal.shape), -1
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
        new_ppm_range = batch_linspace(start=start, stop=stop, steps=length).to(signal.device)
        
        chs_interp = CubicHermiteInterp(ppm, signal)#torch.flip(signal, dims=[-1]))
        return torch.flip(chs_interp.interp(new_ppm_range.expand(dims)), dims=[-1])
    
            
    def transients(self, 
                   fid: torch.Tensor, 
                   params: torch.Tensor,
                  ) -> torch.Tensor:
        assert(fid.ndim==3)
        fid = fid.unsqueeze(1).repeat_interleave(repeats=params.shape[-1], dim=1)
        for _ in range(fid.ndim - params.ndim): params = params.unsqueeze(-1)
        return fid * params
            

    def zeroOrderPhase(self, 
                       fid: torch.Tensor, 
                       phi0: torch.Tensor,
                      ) -> torch.Tensor:
        for _ in range(fid.ndim - phi0.ndim): phi0 = phi0.unsqueeze(-1)
        return complex_exp(fid, phi0.deg2rad())
    

    def forward(self, 
                params: torch.Tensor, 
                gen: bool=True, 
                phi0: bool=True,
                phi1: bool=True,
                noise: bool=True,
                scale: bool=True,
                fshift: bool=True,
                apodize: bool=False,
                baselines: bool=True, 
                magnitude: bool=True,
                broadening: bool=True,
                transients: bool=False,
               ) -> torch.Tensor:
        
        if params.ndim>=3: params = params.squeeze()  # convert 3d parameter matrix to 2d [batchSize, parameters]
        if params.ndim==1: params = params.unsqueeze(0) # Allows for batchSize = 1

        params = self.quantify_params(params, label='forward')

        # Define basis spectra coefficients
        if gen: print('>>>>> Preparing metabolite coefficients')
        fid = params[:,self.index['metabolites']].unsqueeze(2).unsqueeze(-1)
        fid = torch.cat([fid.mul(self.syn_basis_fids[:,0,:].unsqueeze(1)),
                         fid.mul(self.syn_basis_fids[:,1,:].unsqueeze(1))], dim=2)
        
        # Line broadening
        if broadening:
            if gen: print('>>>>> Applying line shape distortions')
            fidSum = self.lineshape_correction(fid, params[:,self.index['d']], 
                                                params[:,self.index['g']], 'foward')
        
        fidSum = fidSum.sum(dim=-3)
        mx_values = torch.max(fidSum[:,0,:].unsqueeze(1), dim=-1) 
        # Save these values for the SNR calculation
        # Should not be using max values that include the baseline!
        
        # # Make sure that the noise generator considers transients properly -> specifically the mx_values!!!
        
        if transients:
            fidSum = self.transients(fidSum, params[:,self.index['transients']])

        # Add Noise
        if noise:
            if gen: print('>>>>> Adding noise')
            fidSum += self.generate_noise(fidSum, params[:,self.index['snr']], mx_values)

        # Rephasing Spectrum
        if phi0:
            if gen: print('>>>>> Rephasing spectra')
            specSummed = self.zeroOrderPhase(fidSum, params[:,self.index['phi0']])
        
        # # Apodize
        if apodize:
            specSummed = self.apodization(specSummed, hz=apodize)

        # Recover Spectrum
        if gen: print('>>>>> Recovering spectra')
        specSummed = Fourier_Transform(specSummed)

        # Rephasing Spectrum
        if phi1:
            if gen: print('>>>>> Rephasing spectra')
            specSummed = self.firstOrderPhase(specSummed, params[:,self.index['phi1']])

        # Frequency Shift
        if fshift:
            if gen: print('>>>>> Shifting frequencies')
            specSummed = self.frequency_shift(specSummed, params[:,self.index['f_shift']])

        # Resampling the spectra
        specSummed = torch.cat([self.resample_(specSummed[...,0,:].unsqueeze(-2), length=self.length),
                                self.resample_(specSummed[...,1,:].unsqueeze(-2), length=self.length)], dim=-2)
        
        # Add Baselines
        if isinstance(baseline, torch.Tensor):
            if gen: print('>>>>> Adding baselines')
            baseline = self.add_artifacts(baselines=baseline,
                                          params=params)
            ppm = self.cropped_ppm #if not splines else self.ppm_cropped
            specSummed += self._resample_(baseline, ppm=ppm, length=self.length, scale=params[:,self.index['scale']])
            # specSummed += torch.cat([self.resample_(baseline[...,0,:].unsqueeze(-2), ppm=ppm, length=self.length, scale=params[:,self.index['scale']]),
            #                          self.resample_(baseline[...,1,:].unsqueeze(-2), ppm=ppm, length=self.length)], dim=-2)
            
        if magnitude:
            if gen: print('>>>>> Generating magnitude spectra')
            return self.magnitude(specSummed, normalize=True)
        
        return self.normalize(specSummed)

    '''
        # Recover Spectrum
        specSummed = Fourier_Transform(fid)
        
#         specSummed = self.resample_(specSummed, length=self.length, ppm=self.ppm)
        if not individual:
            specSummed = torch.cat([self.resample_(specSummed[::,0,:].unsqueeze(-2), length=self.length, ppm=self.ppm),
                                    self.resample_(specSummed[::,1,:].unsqueeze(-2), length=self.length, ppm=self.ppm)], dim=-2)
#             return self.magnitude(specSummed, normalize=True)
        else:
            temp = torch.stack([self.resample_(specSummed[:,0,0,:].unsqueeze(-3), length=self.length, ppm=self.ppm),
                                self.resample_(specSummed[:,0,1,:].unsqueeze(-3), length=self.length, ppm=self.ppm)], dim=-2).transpose(1,0)
            for i in range(1,specSummed.shape[-3]):
                temp = torch.cat([temp,
                                  torch.stack([self.resample_(specSummed[:,i,0,:].unsqueeze(-3), length=self.length, ppm=self.ppm),
                                               self.resample_(specSummed[:,i,1,:].unsqueeze(-3), length=self.length, ppm=self.ppm)], dim=-2).transpose(1,0)],
                                  dim=-3)
            specSummed = temp
#         if not individual:
#             specSummed = torch.cat([self._resample_(specSummed[::,0,:].unsqueeze(-2), length=self.length, scale=scale),
#                                     self._resample_(specSummed[::,1,:].unsqueeze(-2), length=self.length, scale=scale)], dim=-2)
# #             return self.magnitude(specSummed, normalize=True)
#         else:
#             temp = torch.stack([self._resample_(specSummed[:,0,0,:].unsqueeze(-3), length=self.length, scale=scale),
#                                 self._resample_(specSummed[:,0,1,:].unsqueeze(-3), length=self.length, scale=scale)], dim=-2).transpose(1,0)
#             for i in range(1,specSummed.shape[-3]):
#                 temp = torch.cat([temp,
#                                   torch.stack([self._resample_(specSummed[:,i,0,:].unsqueeze(-3), length=self.length, scale=scale),
#                                                self._resample_(specSummed[:,i,1,:].unsqueeze(-3), length=self.length, scale=scale)], dim=-2).transpose(1,0)],
#                                   dim=-3)
#             specSummed = temp
#             specSummed = self.resample_(specSummed, length=self.length, ppm=self.ppm)
#             return self.normalize(temp)
#             print('after specSummed.shape {}, temp.shape {}'.format(specSummed.shape, temp.shape))
# # #         scale = torch.as_tensor([0.4892]).unsqueeze(0).repeat(params.shape[0],1).pow(-1) *0 + 2
# #         scale = torch.ones_like(params[:,self.index['snr']])
# #         specSummed = torch.cat([self._resample_(specSummed[:,0,:].unsqueeze(1), scale, length=self.length), \
# #                                 self._resample_(specSummed[:,1,:].unsqueeze(1), scale, length=self.length)], dim=1)#, \
# #         specSummed = torch.cat([self._resample_(specSummed[:,0,:], scale, length=self.length), \
# #                                 self._resample_(specSummed[:,1,:], scale, length=self.length)], dim=1)#, \
#         specSummed = torch.cat([self._resample_(specSummed[:,0,:].unsqueeze(1), params[:,self.index['scale']], length=self.length), \
#                                 self._resample_(specSummed[:,1,:].unsqueeze(1), params[:,self.index['scale']], length=self.length)], dim=1)#, \


        return self.magnitude(specSummed, normalize=True, individual=individual)
        
#         ## Normalize Spectra #by dividing by the norm of the area under the 3 major peaks
#         return self.normalize(specSummed)

'''    
