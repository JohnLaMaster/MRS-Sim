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
from interpolate import CubicHermiteMAkima as CubicHermiteInterp
from numpy import pi
from types import SimpleNamespace



__all__ = ['PhysicsModel']



PI = torch.from_numpy(np.asarray(np.pi)).squeeze().float()

@torch.no_grad()
class PhysicsModel(nn.Module):
    def __init__(self, 
                 PM_basis_set: str,
                ):
        super().__init__()
        # Load basis spectra, concentration ranges, and units
        path = './src/basis_sets/' + PM_basis_set # 'fitting_basis_ge_PRESS144.mat'

        with open(path, 'rb') as file:
            dct = convertdict(io.loadmat(file,simplify_cells=True))
            self.basisFcns = {}
            for key, value in dct.items():
                if str(key)=='metabolites': 
                    self.basisFcns['metabolites'] = value
                elif str(key)=='artifacts': 
                    self.basisFcns['artifacts'] = value
                elif str(key)=='header':
                    for k, v in dct[key].items():
                        if str(k)=='ppm': 
                            k, v = '_ppm', torch.flip(v.unsqueeze(0), 
                                                      dims=[-1,0]).squeeze(0)
                        if not isinstance(v, str): 
                            self.register_buffer(str(k), v.float())

        
    def __repr__(self):
        lines = sum([len(listElem) for listElem in self.totals])   
        out = 'MRS-Sim(basis={}, lines={}'.format('Osprey, GE_PRESS144', lines)
        return out + ', range={}ppm, resample={}pts)'.format(self.cropRange, 
                                                             self.length)
    
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
                   basisFcn_len: float=1024,
                   b0: bool=False,
                   coil_fshift: bool=False,
                   coil_phi0: bool=False,
                   coil_sens: bool=False,
                   cropRange: list=[0,5],
                   difference_editing: list=False, 
                   # should be a list of basis function names 
                   # that gets subtracted
                   eddycurrents: bool=False,
                   fshift_i: bool=False,
                   image_resolution: list=[0.5, 0.5, 0.5],
                   length: float=512,
                   lineshape: str='voigt',
                   num_coils: int=1,
                   ppm_ref: float=4.65,
                   spectral_resolution: list=[10.0, 10.0, 10.0],
                   spectralwidth=None,
                  ) -> tuple:
        # # Sort metabs and group met vs mm/lip
        self._metab, l, self.MM = self.order_metab(metab)
        self.MM = self.MM + 1 if  self.MM>-1 else False
        # Initialize basic variables
        self.lineshape_type = lineshape
        self.cropRange = cropRange if cropRange else [self._ppm.min(), 
                                                      self._ppm.max()]
        self.t = self.t.unsqueeze(-1).float()
        self._basis_metab = []

        # Add metabolite names to dictionary
        dct = OrderedDict()
        for m in self._metab: dct.update({str(m): torch.empty(1)})  

        # Check that the specified metabolites are in the basis set
        assert(m.lower() in list(self.basisFcns['metabolites'].keys()) for \
                m in self._metab)

        # # Compile the selected basis functions
        self.syn_basis_fids = torch.stack([torch.as_tensor(
            self.basisFcns['metabolites'][m.lower()]['fid'], 
            dtype=torch.float32) for m in self._metab], dim=0).unsqueeze(0)

        # # Resample basis functions to match simulation parameters
        if not isinstance(spectralwidth, type(None)):
            # If the spectral bandwidth is different from the basis set, f
            self.spectralwidth = torch.as_tensor(spectralwidth)
            target_range = [-0.5*spectralwidth / self.carrier_frequency, 
                            0.5*spectralwidth / self.carrier_frequency]
            t = 1 / spectralwidth * basisFcn_len
        else:
            target_range = [self._ppm.min(), self._ppm.max()]
            t = 1 / self.spectralwidth * basisFcn_len #  self.t.max()

        # Resample the basis functions, ppm, and t to the desired resolution
        self.syn_basis_fids = inv_Fourier_Transform(
                                self.resample_(signal=Fourier_Transform(
                                                        self.syn_basis_fids),
                                               ppm=self._ppm,
                                               length=basisFcn_len,
                                               target_range=target_range)
                                                    )
        self._ppm = torch.linspace(target_range[0], target_range[1], 
                                   basisFcn_len).unsqueeze(0) + ppm_ref
        self.t = torch.linspace(0, t, basisFcn_len).unsqueeze(-1)

        if difference_editing:
            self.difference_editing_fids = self.syn_basis_fids.clone()
            ind = [idx for idx, string in enumerate(difference_editing) if 
                    string in self._metab]
            for m in zip(ind, difference_editing):
                self.difference_editing_fids[0,ind,...] = torch.as_tensor(
                    self.basisFcns['metabolites'][m.lower()]['fid_OFF'], 
                    dtype=torch.float32)

        # Define variables for later
        self.register_buffer('l', torch.FloatTensor([
                             self.syn_basis_fids.shape[-1]]).squeeze())
        self.register_buffer('length', torch.FloatTensor([length]).squeeze())
        self.register_buffer('ppm_ref', torch.FloatTensor([ppm_ref]).squeeze())
        self.register_buffer('ppm_cropped', torch.fliplr(torch.linspace(
                            float(self.cropRange[0]), float(self.cropRange[1]), 
                            length).unsqueeze(0)))
        self.spectral_resolution = spectral_resolution
        self.image_resolution = image_resolution
    
        # Define the first-order phase reference in the time-domain
        '''
        bandwidth = 1/dwelltime
        carrier_freq = 127.8 # MHz
        freq_ref = 4.68 * carrier_freq / 10e6
        degree = 2*pi*bandwidth*(1/(bandwidth + freq_ref))
        
        freq_ref = ppm_ref * self.carrier_freq / 10e6
        phi1_ref = 2*self.PI*self.spectralwidth * (torch.linspace(-0.5*
            self.spectralwidth, 0.5*self.spectralwidth, self.l) + freq_ref)
        '''        
        freq_ref = ppm_ref * self.carrier_frequency / 10e6
        spectralwidth = torch.linspace(-0.5*float(self.spectralwidth), 
                                       0.5*float(self.spectralwidth), 
                                       int(self.l))
        phi1_ref = 2 * PI * spectralwidth / (spectralwidth + freq_ref)
        self.register_buffer('phi1_ref', torch.as_tensor(phi1_ref, 
                             dtype=torch.float32).squeeze())

        ### Define the index used to specify the variables in the forward pass 
        # # and in the sampling code
        num_bF = l+self.MM if self.MM else l
        header, cnt = self._metab, counter(start=int(3*num_bF)-1)
        g = 1 if not self.MM else 2
        names = ['d',   'dmm', 'g',   'gmm', 'fshift', 'snr', 'phi0', 'phi1']
        mult  = [  l, self.MM,   l, self.MM,        1,     1,      1,      1] 

        # Should be a global fshift then individual metabolites 
        # and MM/Lip fsfhitfs
        names.insert(-4,'fshiftmet')
        names.insert(-4,'fshiftmm')
        mult.insert(-4,l), mult.insert(-6,self.MM)
        # if b0:
        names.insert(-1,'b0')
        names.insert(-1,'bO_dir')
        mult.insert(-1,1), mult.insert(-1,3)
        # if eddycurrents:
        names.insert(-1,'eddyCurrents_A')
        mult.insert(-1,1)
        names.insert(-1,'eddyCurrents_tc')
        mult.insert(-1,1)
        if num_coils>1: 
            # Minimum 2 num_coils for the variables to be included in the model
            names.append('coil_snr')
            mult.append(num_coils)
            if coil_sens:
                names.append('coil_sens')
                mult.append(num_coils)
            if coil_fshift:
                names.append('coil_fshift')
                mult.append(num_coils)
            if coil_phi0:
                names.append('coil_phi0')
                mult.append(num_coils)
        for n, m in zip(names, mult): 
            for _ in range(m): header.append(n)
            
        # Define the min/max ranges for quantifying the variables
        self.min_ranges = torch.zeros([1,len(header)], dtype=torch.float32)
        self.max_ranges = torch.zeros_like(self.min_ranges)
        for i, m in enumerate(header):
            met, temp, strt = False, None, None
            if m.lower() in self.basisFcns['metabolites'].keys(): 
                temp = self.basisFcns['metabolites'][m.lower()]
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
        
        # # Line shape corrections - Lorenztian and Gaussian
        ind.append(tuple(int(x) for x in torch.arange(0,num_bF) + num_bF))   
        ind.append(tuple(int(x) for x in torch.arange(0,num_bF) + 2*num_bF)) 

        # # Frequency Shift
        ind.append(cnt(1))                               # Global fshift
        ind.append(tuple(cnt(1) for _ in range(num_bF))) # Individual fshifts
        
        # # Noise
        ind.append(cnt(1)) # SNR

        # # Phase
        ind.append(cnt(1)) # Phi0
        ind.append(cnt(1)) # Phi1

        # # B0 inhomogeneities
        ind.append(cnt(1))                                        # B0 - mean
        ind.append(tuple(int(cnt(1)) for _ in torch.arange(0,3))) # directional Δs

        # # Eddy currents
        ind.append(tuple(int(cnt(1)) for _ in torch.arange(0,2)))

        # # Coil transients
        if num_coils>1:      # Number of coils
            ind.append(tuple(int(cnt(1)) for _ in torch.arange(0,num_coils)))
            if coil_sens:    # Coil sensitivities
                ind.append(tuple(int(cnt(1)) for _ in torch.arange(0,num_coils)))
            if coil_fshift:  # Frequency unalignment
                ind.append(tuple(int(cnt(1)) for _ in torch.arange(0,num_coils)))
            if coil_phi0:    # Zero-Order phase unalignment
                ind.append(tuple(int(cnt(1)) for _ in torch.arange(0,num_coils)))

        # # Cummulative
        total = cnt(1)
        ind.append(tuple(int(x) for x in torch.arange(0,num_bF)))     # Metabolites
        ind.append(tuple(int(x) for x in torch.arange(num_bF,total))) # Parameters
        ind.append(tuple(int(x) for x in torch.arange(0,total)))      # Overall


        # Define the remaining dictionary keys
        dct.update({'D': torch.empty(1), 
                    'G': torch.empty(1), 
                    'F_Shift': torch.empty(1)})
        dct.update({'F_Shifts': torch.empty(1)})
        dct.update({'SNR': torch.empty(1), 
                    'Phi0': torch.empty(1), 
                    'Phi1': torch.empty(1)})
        dct.update({'B0': torch.empty(1), 
                    'B0_dir': torch.empty(1)})
        dct.update({'ECC': torch.empty(1)})
        if num_coils>1: 
            dct.update({'Coil_SNR': torch.empty(1)})
            if coil_sens:   dct.update({'Coil_Sens': torch.empty(1)})
            if coil_fshift: dct.update({'Coil_fShift': torch.empty(1)})
            if coil_phi0:   dct.update({'Coil_Phi0': torch.empty(1)})
        dct.update({'Metabolites': torch.empty(1), 
                    'Parameters': torch.empty(1), 
                    'Overall': torch.empty(1)})
        
        # Combine and define the index for internal use in the model
        self._index = OrderedDict({d.lower(): i for d,i in zip(dct.keys(),ind)})

        return dct, ind

    
    def add_offsets(self,
                    fid: torch.Tensor,
                    offsets: tuple=None,
                    drop_prob: float=0.2,
                   ) -> dict:        
        '''
        Used for adding residual water and baselines. config dictionaries are 
        needed for each one.
        '''
        out, baselines, res_water = offsets
        scale = 10**(OrderOfMagnitude(fid) - OrderOfMagnitude(out))
        out, ind = rand_omit(out, 0.0, drop_prob)
        offset = out.clone() * scale

        if not isinstance(baselines, type(None)): 
            baselines *= scale
            if drop_prob: baselines[ind,...] = 0.0
        if not isinstance(res_water, type(None)): 
            res_water *= scale
            if drop_prob: res_water[ind,...] = 0.0
        if not isinstance(out, int):# == 0: meaning offsets were not included
            fid += inv_Fourier_Transform(out*scale)

        return fid, {'baselines':      baselines, 
                     'residual_water': res_water, 
                     'offset':         offset}

    
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
        '''
        Simulate baseline offsets
        '''
        cfg = SimpleNamespace(**config)

        baselines = batch_smooth(bounded_random_walk(cfg.start, cfg.end, 
                                                     cfg.std, cfg.lower_bnd, 
                                                     cfg.upper_bnd, 
                                                     cfg.length), 
                                 cfg.windows)

        if cfg.rand_omit>0: 
            baselines, _ = rand_omit(baselines, 0.0, cfg.rand_omit)

        # Convert simulated residual water from local to clinical range before 
        # Hilbert transform makes the imaginary component. Then resample 
        # acquired range to cropped range.
        ppm = self._ppm.clone()#.unsqueeze(-1)#.repeat(1,baselines.shape[0],1)
        raw_baseline = HilbertTransform(
                        self.sim2acquired(baselines * config['scale'], 
                                          [ppm.amin(keepdims=True), 
                                           ppm.amax(keepdims=True)], self.ppm)
                       )
        ch_interp = CubicHermiteInterp(xaxis=self.ppm, signal=raw_baseline)
        baselines = ch_interp.interp(xs=self.ppm_cropped)

        return baselines.fliplr(), raw_baseline.fliplr()
    
    
    def B0_inhomogeneities(self, 
                           b0: torch.Tensor,
                           param: torch.Tensor, # hz
                          ) -> torch.Tensor:
        '''
        I need spatial resolution of images and spectra as well as another 
        parameter(s) defining the range of the B0 variation across the MRS 
        voxel. complex_exp(torch.ones_like(fid), param*t).sum(across this extra 
        spatial dimension) return fid * sum(complex_exp)
        '''
        t = self.t.clone().mT # [1, 8192] 
        # FID should be 4 dims = [bS, basis fcns, channels, length]
        for _ in range(4 - t.ndim): t = t.unsqueeze(0)
        for _ in range(4 - param.ndim): param = param.unsqueeze(1)
        # if t.shape[-1]==1: t = t.mT # [1, 1, 8192].mT => [1, 8192, 1]
        # param = param.unsqueeze(-2) # [bS, basisfcns, channels=1, 
        #                                   extra=1, params]

        # spectral_resolution = [10.0, 10.0, 10.0]
        # image_resolution    = [ 0.5,  0.5,  0.5]
        num_pts = [int(m/n) for m, n in zip(self.spectral_resolution, 
                                            self.image_resolution)]
        mean = b0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        dx = param[...,0]
        dy = param[...,1]
        dz = param[...,2]

        '''
        Matlab code confirming it works!
        Gradient of x goes top to bottom. Gradient of y runs through the center.
        n = 10; x = 1; y = 4; z = 2; xvec = -x:2*x/(n-1):x; 
        yvec = -y:2*y/(n-1):y; zvec=-z:2*z/(n-1):z;
        dB0 = xvec .* xvec' ./ x; dB1 = dB0 + yvec;
        dB2 = reshape(dB1,[10,10,1]) + reshape((zvec+1),[1,1,10]);
        '''

        # output.shape = [bS, 1, length, 1, 1]
        x = batch_linspace(1-dx,1+dx,num_pts[0]).permute(0,2,1).unsqueeze(-1)
        # output.shape = [bS, 1, 1, length, 1]
        y = batch_linspace(1-dy,1+dy,num_pts[1]).unsqueeze(-1)
        # output.shape = [bS, 1, 1, 1, length]
        z = batch_linspace(1-dz,1+dz,num_pts[2]).unsqueeze(-1).permute(0,1,3,2)

        # Define the changes in B0
        dB0  = x * x.transpose(-3,-2) / dx.unsqueeze(-1)
        dB0 += y
        dB0  = dB0.repeat(1,1,1,z.shape[-1]) + z
        dB0 += mean

        # output.shape = [bS, length, length, length]
        dB0 = dB0.unsqueeze(1).flatten(start_dim=2, end_dim=-1).unsqueeze(-1)
        # output.shape = [bS, 1, length^3, 1] * [1, 1, 1, 8192] 
        #                => [bS, 1, length^3, 8192]
        dB0 = dB0 * t

        identity = torch.ones_like(t).repeat(1,1,2,1)

        return complex_exp(identity, 
                           (-1*dB0.unsqueeze(-2)).deg2rad()).sum(dim=-3)


    def coil_freq_drift(self,
                        fid: torch.Tensor,
                        f_shift: torch.Tensor,
                        t: torch.Tensor=None,
                       ) -> torch.Tensor:
        if fid.ndim>4:
            for _ in range(fid.ndim - f_shift.ndim - 2): 
                f_shift = f_shift.unsqueeze(1)
                # => [bS, [[ON/OFF], [noisy/noiseless]], transients_weights]

        for _ in range(fid.ndim - f_shift.ndim): 
            f_shift = f_shift.unsqueeze(-1)
            # => [bS, [[ON/OFF], [noisy/noiseless]], transients_weights, 
            #     channels, spectra]

        t = self.t if t==None else t
        t = t.t() if t.shape[-1]==1 else t
        for _ in range(fid.ndim - t.ndim): t = t.unsqueeze(0)
        
        return complex_exp(fid, f_shift.mul(t))


    def coil_phi0_drift(self,
                        fid: torch.Tensor,
                        phi0: torch.Tensor,
                       ) -> torch.Tensor:
        if fid.ndim>4:
            for _ in range(fid.ndim - phi0.ndim - 2): phi0 = phi0.unsqueeze(1)
                # => [bS, [[ON/OFF], [noisy/noiseless]], transients_phi]
                
        for _ in range(fid.ndim - phi0.ndim): phi0 = phi0.unsqueeze(-1)
            # => [bS, [[ON/OFF], [noisy/noiseless]], transients_phi, 
            #     channels, spectra]
            
        return complex_exp(fid, -1*phi0.deg2rad())


    def coil_sensitivity(self,
                         fid: torch.Tensor,
                         coil_sens: torch.Tensor,
                        ) -> torch.Tensor:
        '''
        Used to apply scaling factors to simulate the effect of coil 
        combination weighting
        '''
        # Add the transient dim => [bS, transients, weights]
        coil_sens = coil_sens.unsqueeze(1) 
        if fid.ndim>4:
            for _ in range(fid.ndim - coil_sens.ndim - 4): 
                # => [bS, [[ON/OFF], [noisy/noiseless]], transients, weights]
                coil_sens = coil_sens.unsqueeze(1)
        for _ in range(fid.ndim - coil_sens.ndim): 
            # => [bS, [[ON/OFF], [noisy/noiseless]], transients, 
            #     channels, weights]
            coil_sens = coil_sens.unsqueeze(-1)
        return fid * coil_sens


    def eddyCurrentKloseRemoval(self,
                                fid: torch.Tensor,
                                water: torch.Tensor,
                               ) -> torch.Tensor:
        phase = torch.angle(torch.view_as_complex(water.transpose(-1,-2)))
        return complex_exp(fid, -phase)


    def firstOrderEddyCurrents(self,
                               fid: torch.Tensor,
                               params: torch.Tensor,
                              ) -> torch.Tensor:
        '''
        Came from Jamie Near's code: 
            https://github.com/CIC-methods/FID-A/blob/master/processingTools/op
            _makeECArtifact.m
        A:  amplitude of EC artifact in time domain [Hz]
        tc: time constant [s] of the exponentially decaying phase artifact in 
            time domain.
        Default parameter values were provided by Jamie directly.
        '''
        t = self.t.t() if self.t.shape[0]!=1 else self.t # [1x8192]
        A, tc = params[:,0].unsqueeze(-1), params[:,1].unsqueeze(-1)
        for _ in range(fid.ndim - 2):
            t  = t.unsqueeze(0)
            A  = A.unsqueeze(-1)
            tc = tc.unsqueeze(-1)

        f_mod = A * complex_exp(torch.ones_like(fid), 
                                (-t / tc).expand_as(fid), real=True)

        return complex_exp(fid, -1*f_mod*t*2*PI)


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
        f_shift = param.mul(t).expand_as(fid)
        
        return complex_exp(fid, f_shift)
        
        
    def generate_noise(self, 
                       fid: torch.Tensor, 
                       param: torch.Tensor,
                       max_val: torch.Tensor,
                       zeros: torch.Tensor, # number of coils with no signal
                       transients: torch.Tensor=None,
                       uncorrelated: bool=False,
                      ) -> torch.Tensor:
        '''
        SNR formula (MRS): snr_lin = max(real(spectra)) / std_noise 
        Signal averaging improves SNR by a factor of sqrt(num_coil)
        '''
        for _ in range(fid.ndim-max_val.ndim): max_val = max_val.unsqueeze(-1)
        for _ in range(fid.ndim-param.ndim): param = param.unsqueeze(-1)

        lin_snr = 10**(param / 10) # convert from decibels to linear scale
        if not isinstance(transients, type(None)): 
            # Scale the mean SNR accourding to the number of transients
            s = torch.zeros_like(param) + int(transients.shape[-1]) - \
                    zeros.unsqueeze(-1).unsqueeze(-1)
            lin_snr /= s**0.5
            for _ in range(fid.ndim-transients.ndim): 
                transients = transients.unsqueeze(-1)

            # Allows the transients' SNR to come from a distribution
            lin_snr = lin_snr * transients

        std_dev = max_val / lin_snr
        std_dev[torch.isnan(std_dev)] = 1e-6
        std_dev[std_dev==0] += 1e-6
        if std_dev.ndim==2:
            std_dev = std_dev.squeeze(0)
        elif std_dev.ndim>=4:
            if std_dev.shape[-2]==1: std_dev = std_dev.squeeze(-2)
            elif std_dev.shape[-1]==1: std_dev = std_dev.squeeze(-1)
        
        if uncorrelated: 
            # Quadrature coils where real/phase are recorded separately
            # Separate coils ==> uncorrelated noise for real/phase channels
            std_dev = std_dev.unsqueeze(-2)

        '''
        Notes:
        # dim=2: no change                      [channels, length] => [1, channels, length]
        # dim=3: no change                                           [bS, channels, length]
        # dim=4: spectral editing or transients :: [bS, (edit/transient), channels, length]
        # dim=5: editing & transients ::           [bS, edit, transients, channels, length]
        # output.shape: [bS, ON/OFF, [noisy, noiseless], transients, channels, length]
        '''
        e = torch.distributions.normal.Normal(0,std_dev)
        e = torch.movedim(e.sample([fid.shape[-1]]),0,-1)

        if uncorrelated:
            return inv_Fourier_Transform(e)

        return inv_Fourier_Transform(HilbertTransform(e))


    def line_summing(self,
                     fid: torch.Tensor,
                     params: torch.Tensor,
                     mm: int,
                     l: int,
                    ) -> tuple:
        if not mm:
            fidSum = fid.sum(dim=-3) 
            spectral_fit = fidSum.clone()
            mx_values = torch.amax(
                Fourier_Transform(fid)[...,0,:].unsqueeze(-2).sum(dim=-3, 
                    keepdims=True), dim=-1, keepdims=True) 
        else:
            mm = fid[...,l:,:,:].sum(dim=-3)
            fidSum = fid[...,0:l,:,:].sum(dim=-3)
            spectral_fit = fidSum.clone()
            mx_values = torch.amax(
                Fourier_Transform(fidSum)[...,0,:].unsqueeze(-2), dim=-1, 
                keepdims=True) 
            fidSum += mm
        return fidSum, spectral_fit, mx_values
    

    def lineshape_correction(self, 
                             fid: torch.Tensor, 
                             d: torch.Tensor=None, 
                             g: torch.Tensor=None,
                            ) -> torch.Tensor:
        '''
        In a Voigt lineshape model, each basis line has its own Lorentzian 
        value. Fat- and Water-based peaks use one Gaussian value per group.
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
        In a Voigt lineshape model, each basis line has its own Lorentzian 
        value. Fat- and Water-based peaks use one Gaussian value per group.
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
        magnitude = torch.sqrt(signal[...,0,:].pow(2) + 
                               signal[...,1,:].pow(2) + eps).unsqueeze(-2)
        out = torch.cat([signal, magnitude], dim=-2)
            
        if normalize:
            if individual: 
                data = torch.sum(signal, dim=-3, keepdim=True)
                magnitude = torch.sqrt(data[...,0,:].pow(2) + 
                                    data[...,1,:].pow(2) + eps).unsqueeze(-2)
            return out.div(torch.max(magnitude, dim=-1, 
                                     keepdim=True).values + eps)

        return out
    

    def modulate(self, 
                 fids: torch.Tensor,
                 params: torch.Tensor,
                ) -> torch.Tensor:
        for i in range(2): params = params.unsqueeze(-1)
        return params.repeat_interleave(2, dim=-2).mul(fids)

            
    def multicoil(self, 
                  fid: torch.Tensor, 
                  multi_coil: torch.Tensor,
                 ) -> torch.Tensor:
        '''
        This function creates transients according to the number of specified 
        coils. Noise and scaling are done separately.
        The SNR dB value provided is the SNR of the final, coil combined 
        spectrum. Therefore, each of the transients will have a much higher 
        linear SNR that is dependent upon the expected final SNR and the number 
        of transients being simulated.
        '''
        # assert(fid.ndim==3) # Using difference editing would make it 
        #   [bS, ON/OFF, channels, length] 
        # output.shape = [bS, ON/OFF, transients, channels, length] 
        return fid.unsqueeze(-3).repeat_interleave(repeats=multi_coil.shape[-1], 
                                                   dim=-3)


    def normalize(self, 
                  signal: torch.Tensor,
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
            if not signal.shape[-2]==3:
                denom = torch.max(torch.sqrt(signal[...,0,:]**2 + 
                                             signal[...,1,:]**2), 
                                  dim=-1, keepdim=True).values.unsqueeze(-2)
            else:
                denom = torch.max(signal[...,2,:].unsqueeze(-2).abs(), 
                                  dim=-1, keepdim=True).values

            denom[denom.isnan()] = 1e-6
            denom[denom==0.0] = 1e-6
            denom = torch.amax(denom, dim=noisy, keepdim=True)

        for _ in range(denom.ndim-signal.ndim): signal = signal.unsqueeze(1)

        return signal / denom, denom


    def order_metab(self, 
                    metab: list,
                   ) -> tuple:
        mm_lip, mm, lip, temp, num_mm = [], [], [], [], -1
        for i, k in enumerate(metab):
            if 'mm' in k.lower(): 
                mm.append(k)
                temp.append(i)
                num_mm += 1
            mm = sorted(mm)
        mm_lip += mm
        for i, k in enumerate(metab):
            if 'lip' in k.lower():
                lip.append(k)
                temp.append(i)
                num_mm += 1
            lip = sorted(lip)
        mm_lip += lip
        temp.reverse()
        for term in temp: 
            metab.pop(term)
        metab = sorted(metab, key=str.casefold)

        if num_mm>-1:
            return metab + mm_lip, len(metab), num_mm
        return metab, len(metab), num_mm

    
    def quantify_metab(self, 
                       fid: torch.Tensor, 
                       wrt_metab: str='cr',
                      ) -> dict:

        '''
        Both peak height and peak area are returned for each basis line. The 
        basis fids are first modulated and then broadened. The recovered 
        spectra are then quantified.
        '''
        ind = tuple([self.index[m.lower()] for m in wrt_metab.split(',')])

        # Quantities
        specs = Fourier_Transform(fid)[...,0,:].unsqueeze(-2)

        area = torch.stack([torch.trapz(specs[...,i,:,:], self.ppm, 
                dim=-1).unsqueeze(-1) for i in range(specs.shape[-3])], dim=-3)
        height = torch.max(specs, dim=-1, keepdims=True).values

        area_denom = area[...,ind,:,:].sum(dim=-3, keepdims=True)
        height_denom = height[...,ind,:,:].sum(dim=-3, keepdims=True)
        
        # Normalize to metabolite(s) wrt_metab
        rel_area = area.div(area_denom)
        rel_height = height.div(height_denom)

        return {'area': area.squeeze(-2), 
                'height': height.squeeze(-2), 
                'rel_area': rel_area.squeeze(-2), 
                'rel_height': rel_height.squeeze(-2)}
                    

    def quantify_params(self, 
                        params: torch.Tensor,
                        label=[],
                       ) -> torch.Tensor:
        delta = self.max_ranges - self.min_ranges
        minimum = self.min_ranges.clone()
        params = params.mul(delta) 
        params += minimum
        return params


    def resample_(self, 
                  signal: torch.Tensor, 
                  ppm: torch.Tensor=None,
                  length: int=512,
                  target_range: list=None,
                  flip: bool=True,
                 ) -> torch.Tensor:
        '''
        Basic Cubic Hermite Spline Interpolation of :param signal: with no 
            additional scaling.
        :param signal: input torch tensor
        :param new: new target x-axis
        return interpolated signal
        
        I am flipping the ppm vector instead of the basis lines 
        - 02.01.2022 JTL
        '''
        dims, dims[-1] = list(signal.shape), -1
        ppm = ppm if not isinstance(ppm, type(None)) else self.ppm
        ppm = ppm.unsqueeze(0).squeeze()
        if not (ppm[...,0]<ppm[...,-1]): 
            ppm = torch.flip(ppm.unsqueeze(0), dims=[-1]).squeeze(0)
        
        if isinstance(target_range, type(None)): target_range = self.cropRange
        new = torch.linspace(start=target_range[0], end=target_range[1], 
                             steps=int(length)).to(signal.device)
        for i in range(signal.ndim - new.ndim): new = new.unsqueeze(0)

        if flip: signal = torch.flip(signal, dims=[-1])

        chs_interp = CubicHermiteInterp(ppm, signal)
        signal = chs_interp.interp(new)

        if flip: return torch.flip(signal, dims=[-1])

        return signal
    
    
    def residual_water(self,
                       config: dict,
                      ) -> tuple():
        cfg = SimpleNamespace(**config)

        start_prime = cfg.cropRange_resWater[0] + cfg.start_prime
        end_prime   = cfg.cropRange_resWater[1] -   cfg.end_prime
        ppm = batch_linspace(start=start_prime, stop=end_prime, 
                             steps=int(cfg.length))

        res_water = batch_smooth(bounded_random_walk(cfg.start, cfg.end, 
                                                     cfg.std, cfg.lower_bnd, 
                                                     cfg.upper_bnd, 
                                                     cfg.length),
                                 cfg.windows, 'constant') * config['scale']

        if cfg.rand_omit>0: 
            res_water, ind = rand_omit(res_water, 0.0, cfg.rand_omit)

        # Convert simulated residual water from local to clinical range before 
        # Hilbert transform makes the imaginary component. Then resample 
        # acquired range to cropped range.
        raw_res_water = HilbertTransform(self.sim2acquired(res_water, 
                            [start_prime, end_prime], self.ppm))
        ch_interp = CubicHermiteInterp(xaxis=self.ppm, signal=raw_res_water)
        res_water = ch_interp.interp(xs=self.ppm_cropped)

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
            print('line 990 raw_baselines.shape ',raw_baselines.shape)
        else: 
            baselines, raw_baselines = None, None

        if residual_water: 
            print('>>>>> Residual Water')
            res_water, raw_res_water = self.residual_water(residual_water)
            out += raw_res_water.clone()
        else: 
            res_water, raw_res_water = None, None

        return out, baselines, res_water
        # return (raw_baselines, raw_res_water), baselines, res_water


    def sim2acquired(self, 
                     line: torch.Tensor, 
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
        ch_interp = CubicHermiteInterp(xaxis=xaxis, 
                       signal=torch.nn.functional.pad(input=line, pad=padding))
        return ch_interp.interp(xs=target_ppm)

    
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
                baselines: dict=None, 
                coil_phi0: bool=False,
                coil_sens: bool=False,
                magnitude: bool=True,
                multicoil: bool=False,
                snr_combo: str=False,
                wrt_metab: str='cr',
                zero_fill: int=False,
                broadening: bool=True,
                coil_fshift: bool=False,
                residual_water: dict=None,
                drop_prob: float=None,
               ) -> torch.Tensor:
        if params.ndim==1: params = params.unsqueeze(0) # Allows batchSize = 1

        # B0 inhomogeneities
        if b0:
            # Memory limitations require this to be calculated either before 
            # or after the spectra
            B0 = self.B0_inhomogeneities(b0=params[:,self.index['b0']],
                                         param=params[:,self.index['b0_dir']])

        # Simulate the Residual Water and Baselines
        if offsets:
            offset = self.simulate_offsets(baselines=baselines, 
                                           residual_water=residual_water, 
                                           drop_prob=drop_prob)

        # Define basis spectra coefficients
        if gen: print('>>>>> Preparing metabolite coefficients')
        fid = self.modulate(fids=self.syn_basis_fids, 
                            params=params[:,self.index['metabolites']])

        '''
        Not implemented yet
        if not isinstance(diff_edit, type(None)): 
            fid = torch.stack((fid, 
                               self.modulate(fids=self.difference_editing_fids, 
                               params=diff_edit[:,self.index['metabolites']])),
                              dim=1)
        '''

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
        # # Saves the unadulterated original as the spectral_fit and the max 
        # # values for the SNR calculations which should not consider artifacts
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
        if multicoil>1:
            if gen: print('>>> Transients')
            fidSum = self.multicoil(fid=fidSum, 
                                multi_coil=params[:,self.index['coil_snr']])
            spectral_fit = self.multicoil(fid=spectral_fit, 
                                multi_coil=params[:,self.index['coil_snr']])

        # Add Noise
        if noise:
            if gen: print('>>>>> Adding noise')
            transients = None
            if multicoil>1: transients = params[:,self.index['coil_snr']]
            zeros = torch.where(params[:,self.index['coil_sens']]<=0.0,
                                1,0).sum(dim=-1,keepdims=True)
            noise = self.generate_noise(fid=fidSum, 
                                        max_val=mx_values, 
                                        param=params[:,self.index['snr']], 
                                        zeros=zeros, # num of zeroed out coils
                                        transients=transients,
                                        uncorrelated=False)
            d = -4 if multicoil>1 else -3
            fidSum = torch.stack((fidSum.clone() + noise, fidSum), dim=d)
            spectral_fit = torch.stack((spectral_fit.clone() + noise, 
                                        spectral_fit), dim=d)
            # Keep both noisey transients and clean transients
            # output.shape: [bS, ON\OFF, [noisy/clean], transients, channels, length]
            #                    transients, channels, length]

        # Scale with coil senstivities
        if coil_sens:
            # input.shape: [bS, [ON\OFF], [noisy, clean], 
            #                   transients, channels, length]
            assert(multicoil>1)
            if gen: print('>>> Coil sensitivity')
            fidSum = self.coil_sensitivity(fid=fidSum, 
                           coil_sens=params[:,self.index['coil_sens']])
            spectral_fit = self.coil_sensitivity(
                           fid=spectral_fit.unsqueeze(1), 
                           coil_sens=params[:,self.index['coil_sens']])

        if coil_fshift:
            assert(multicoil>1)
            if gen: print('>>> Coil Frequency Drift')
            fidSum = self.coil_freq_drift(fid=fidSum, 
                           f_shift=params[:,self.index['coil_fshift']])

        if coil_phi0:
            assert(multicoil)
            if gen: print('>>> Coil Phase Drift')
            fidSum = self.coil_phi0_drift(fid=fidSum, 
                           phi0=params[:,self.index['coil_phi0']])

        if snr_combo=='avg' and multicoil>1:
            # output.shape: [bS, ON\OFF, [noisy/clean], ~transients~, channels, length]  
            fidSum = fidSum.mean(dim=-3)
            
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
            fidSum = self.firstOrderEddyCurrents(fid=fidSum, 
                                   params=params[:,self.index['ecc']])

        # Apodize
        if apodize:
            fidSum = self.apodization(fid=fidSum, hz=apodize)

        # Zero-filling
        if zero_fill:
            fidSum = self.zero_fill(fid=fidSum, fill=zero_fill)

        # Recover Spectrum
        if not fids:
            if gen: print('>>>>> Recovering spectra')
            specSummed = Fourier_Transform(fidSum)
            spectral_fit = Fourier_Transform(spectral_fit)

            # Crop and resample spectra
            if resample:
                specSummed = self.resample_(specSummed, length=self.length)
                spectral_fit = self.resample_(spectral_fit, length=self.length)
        else:
            specSummed = fidSum
            # spectral_fit = spectral_fit # redundant
                     
        # Calculate magnitude spectra
        if magnitude:
            if gen: print('>>>>> Generating magnitude spectra')
            specSummed = self.magnitude(specSummed)
            spectral_fit = self.magnitude(spectral_fit)
        
        # Normalize
        specSummed, denom = self.normalize(specSummed, noisy=d)
        spectral_fit, _ = self.normalize(spectral_fit, denom)

        # Convert normalized spectra back to time-domain
        if fids and resample:
            num_pts = zero_fill if zero_fill else self.length
            t = torch.linspace(self.t.amin(), self.t.amax(), 
                               max(self.t.squeeze().shape))
            specSummed = self.resample_(specSummed, length=self.length, 
                                        ppm=t, flip=False, 
                                        target_range=[self.t.amin(), 
                                                      self.t.amax()])
            spectral_fit = self.resample_(spectral_fit, length=self.length, 
                                          ppm=t, flip=False, target_range=[
                                          self.t.amin(), self.t.amax()])

        if not isinstance(diff_edit, type(None)): 
            print('>>> Creating the difference spectra')
            # Consensus paper recommends dividing difference spectra by 2. Not 
            # sure about any other consequenctial effects
            specSummed = torch.cat(specSummed, (specSummed[:,0,...] - 
                            specSummed[:,1,...]).unsqueeze(1) / 2, dim=1)
            spectral_fit = torch.cat(spectral_fit, (spectral_fit[:,0,...] - 
                            spectral_fit[:,1,...]).unsqueeze(1) / 2, dim=1)
            
        print('>>>>> Compiling spectra')
        return self.compile_outputs(specSummed, spectral_fit, offset, params, 
                                    denom, self.quantify_metab(fid, wrt_metab))

    def compile_outputs(self, 
                        specSummed: torch.Tensor, 
                        spectral_fit: torch.Tensor,
                        offsets: dict,
                        params: torch.Tensor, 
                        denom: torch.Tensor,
                        quantities: dict,
                       ) -> torch.Tensor:
        if offsets:
            if not isinstance(offsets['baselines'], type(None)):
                offsets['baselines'], _ = \
                        self.normalize(offsets['baselines'], denom=denom)
            if not isinstance(offsets['residual_water'], type(None)):
                offsets['residual_water'], _ = \
                        self.normalize(offsets['residual_water'], denom=denom)

        offsets = torch2numpy(offsets)

        try: baselines = offsets['baselines']
        except Exception: baselines = None
        try: residual_water = offsets['residual_water']
        except Exception: residual_water = None
        quantities = torch2numpy(quantities)

        return specSummed.numpy(), spectral_fit.numpy(), baselines, \
               residual_water, params.numpy(), quantities
