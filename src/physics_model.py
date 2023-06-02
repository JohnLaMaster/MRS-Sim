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
# Gyromagnetic ratio of protons
# Citation: The National Institute of Standards and Technology, USA
# https://physics.nist.gov/cgi-bin/cuu/Value?gammapbar
gamma_p = torch.FloatTensor(42.577478518)


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
                    self.header = value
                    for k, v in dct[key].items():
                        if str(k)=='ppm': 
                            k, v = '_ppm', v#torch.flip(v.unsqueeze(0), 
                                             #         dims=[-1,0]).squeeze(0)
                        if not isinstance(v, str): 
                            self.register_buffer(str(k), v.float())

        
    def __repr__(self):
        lines = sum([len(listElem) for listElem in self.totals])   
        out = 'MRS-Sim(basis={}, lines={}'.format(PM_basis_set, lines)
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
                   image_resolution: list=[0.5, 0.5, 0.5], # mm
                   length: float=512,
                   lineshape: str='voigt',
                   num_coils: int=1,
                   ppm_ref: float=4.65,
                   spectral_resolution: list=[10.0, 10.0, 10.0], # mm
                   spectralwidth=None,
                   snr_metab: str='PCr',
                  ) -> tuple:
        # # Sort metabs and group met vs mm/lip
        self._metab, l, self.MM = self.order_metab(metab)
        self.MM = self.MM + 1 if  self.MM>-1 else False
        # Initialize basic variables
        self.lineshape_type = lineshape
        self.cropRange = cropRange if cropRange else [self._ppm.min(), 
                                                      self._ppm.max()]
        self.t = self.t.unsqueeze(-1).float().squeeze().unsqueeze(0)
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

#         # # Resample basis functions to match simulation parameters
#         if not isinstance(spectralwidth, type(None)):
#             # If the spectral bandwidth is different from the basis set, f
#             self.spectralwidth = torch.as_tensor(spectralwidth)
#             target_range = [-0.5*spectralwidth / self.carrier_frequency, 
#                             0.5*spectralwidth / self.carrier_frequency] + self.ppm_ref
#             t = 1 / spectralwidth * basisFcn_len # max t value
#         else:
#             target_range = [self._ppm.min(), self._ppm.max()]
#             t = self.t.max() # 1 / self.spectralwidth * basisFcn_len #  

        # Resample the basis functions, ppm, and t to the desired resolution
        target_range = [[self.t.min(), self.t.max()],
                        [self._ppm.min(), self._ppm.max()]]
        # print('target_range: ',target_range)
        # print('basisFcn_len: ',basisFcn_len)
        # print('self.syn_basis_fids.shape {}'.format(self.syn_basis_fids.shape))
        # chint = CubicHermiteInterp(xaxis=self.t, signal=self.syn_basis_fids)
        # self.syn_basis_fids = chint.interp(torch.linspace(self.t.min(), self.t.max(),basisFcn_len))
        # self.syn_basis_fids = self.resample_(signal=self.syn_basis_fids, 
        #                                      ppm=self.t,
        #                                      length=2048,#basisFcn_len,
        #                                      target_range=target_range[0],
        #                                      flip=False)
        """
        Things work when self.resample_(syn_basis_fids, length=8192), self.t(len=8192) and self._ppm(len=8192)
        but not when length=basisFcn_len
        """
        # print('self.syn_basis_fids.shape {}'.format(self.syn_basis_fids.shape))
        # self.syn_basis_fids = inv_Fourier_Transform(self.resample_(signal=Fourier_Transform(
        #                                                             self.syn_basis_fids),
        #                                        ppm=self._ppm,
        #                                        length=basisFcn_len,
        #                                        target_range=target_range[1]
        #                                        )
        #                                             )
        # self._ppm = torch.linspace(target_range[1][0], target_range[1][1], basisFcn_len).unsqueeze(0)#
                                   # basisFcn_len).unsqueeze(0)# + ppm_ref
        # # self._ppm /= (self.B0 * self.carrier_frequency)
        # self.t = torch.linspace(self.t.min(), self.t.max(), basisFcn_len).unsqueeze(0)#basisFcn_len).unsqueeze(0)
        # print('min(self._ppm): {}, max(self._ppm): {}'.format(self._ppm.min(),self._ppm.max()))
        # print('min(self.t): {}, max(self.t): {}'.format(self.t.min(),self.t.max()))

        #self.syn_basis_fids = self.syn_basis_fids * torch.exp(-1*t).unsqueeze(0).unsqueeze(0)
        # print(self.syn_basis_fids.shape, torch.exp(-1*self.t).shape)
        # test = torch.stack([torch.exp(-1*self.t),torch.ones_like(self.t)],dim=1).unsqueeze(0)
        # print('test.shape: ',test.shape)
        lw = 1 - self.linewidth if self.linewidth==0 else 0
        self.syn_basis_fids *= torch.exp(-lw*self.t).unsqueeze(-2).unsqueeze(0).expand_as(self.syn_basis_fids)
        # self.syn_basis_fids = self.syn_basis_fids * torch.stack([torch.exp(-lw*self.t),
        #                                                          torch.exp(-lw*self.t)],
        #                                                         dim=-2).unsqueeze(0).expand_as(self.syn_basis_fids)
        # X =  torch.stack([torch.exp(-lw*self.t), torch.exp(-lw*self.t)], dim=-2).unsqueeze(0)
        # print('X.shape ', X.shape )
        # X = X.expand_as(self.syn_basis_fids)
        # print('self.syn_basis_fids.shape {}, X.shape {}'.format(self.syn_basis_fids.shape, X.shape))
        # self.syn_basis_fids *= X
        # # print(self.syn_basis_fids.shape)

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
        # self.register_buffer('ppm_cropped', torch.linspace(
        #                     float(self.cropRange[0]), float(self.cropRange[1]), 
        #                     length).unsqueeze(0))
        self.spectral_resolution = spectral_resolution
        self.image_resolution = image_resolution
        # print('ppm_cropped: ',self.ppm_cropped.min(), self.ppm_cropped.max())
    
        # Define the first-order phase reference in the time-domain
        '''
        bandwidth = 1/dwelltime # Hz
        carrier_freq = 127.8 # MHz
        freq_ref = 4.68 * carrier_freq * 10e6 # Hz  ## / 10e6
        degree = 2*pi*bandwidth*(1/(bandwidth + freq_ref))
        
        freq_ref = ppm_ref * self.carrier_freq / 10e6
        phi1_ref = 2*self.PI*self.spectralwidth * (torch.linspace(-0.5*
            self.spectralwidth, 0.5*self.spectralwidth, self.l) + freq_ref)

        0.5*PI*Hz = 1 rad/s
        deg * 2*PI = rad

        freq_ref = (self._ppm - ppm_ref) * gamma_p * B0
        time_ref = 1 / freq_ref
        phase[radians] = f[Hz] * timeshift[s] * 2 * PI
                       = (f[Hz] * 2 * PI) * timeshift[deg] 
                       deg * 2 * PI = radians
                       (Hz = 1 / s) * (s)


        ph1 = ph1 * pi/180 # convert from deg/ppm to rad/ppm
        correction: specs .* exp(1j(ph1*(_ppm-ppm_ref)))
        artifact: specs .* exp(-1j(ph1*(_ppm-ppm_ref)))
        '''        
        # freq_ref = (self._ppm - ppm_ref) * self.carrier_frequency * 10e6
        phi1_ref = self._ppm - ppm_ref
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
        names.insert(-4,'fshiftmet'),       mult.insert(-4,l)
        names.insert(-4,'fshiftmm'),        mult.insert(-6,self.MM)
        names.insert(-1,'b0'),              mult.append(1)
        names.insert(-1,'bO_dir'),          mult.append(3)
        names.insert(-1,'eddyCurrents_A'),  mult.append(1)
        names.insert(-1,'eddyCurrents_tc'), mult.append(1)
        names.append('coil_snr'),           mult.append(num_coils)
        names.append('coil_sens'),          mult.append(num_coils)
        names.append('coil_fshift'),        mult.append(num_coils)
        names.append('coil_phi0'),          mult.append(num_coils)
        for n, m in zip(names, mult): 
            for _ in range(m): header.append(n)
            
        # Define the min/max ranges for quantifying the variables
        self.define_parameter_ranges(header=header)



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
        ind.append(tuple(int(cnt(1)) for _ in torch.arange(0,num_coils)))
        # Coil sensitivities
        ind.append(tuple(int(cnt(1)) for _ in torch.arange(0,num_coils)))
        # Frequency unalignment
        ind.append(tuple(int(cnt(1)) for _ in torch.arange(0,num_coils)))
        # Zero-Order phase unalignment
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
        dct.update({'Coil_SNR': torch.empty(1)})
        dct.update({'Coil_Sens': torch.empty(1)})
        dct.update({'Coil_fShift': torch.empty(1)})
        dct.update({'Coil_Phi0': torch.empty(1)})
        dct.update({'Metabolites': torch.empty(1), 
                    'Parameters': torch.empty(1), 
                    'Overall': torch.empty(1)})
        
        # Combine and define the index for internal use in the model
        self._index = OrderedDict({d.lower(): i for d,i in zip(dct.keys(),ind)})

        return dct, ind

    def define_parameter_ranges(header: list):
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

    
    def add_inhomogeneities(self,
                            fid: torch.Tensor,
                            B0: torch.Tensor
                           ) -> torch.Tensor:
        '''
        This adds the B0 field effects and allows the FIDs to be individual
        moieties- or metabolite-level.
        '''
        for _ in range(fid.ndim - B0.ndim): B0 = B0.unsqueeze(1)
        # # B0.shape = torch.Size([5, 1, 2, 8192])
        # # data.shape = torch.Size([5, 26, 2, 8192])
        # # output.shape = torch.Size([5, 26, 2, 8192])
        return fid * B0


    def add_offsets(self,
                    fid: torch.Tensor,
                    offsets: tuple=None,
                    max_val: torch.Tensor=None,
                    drop_prob: float=0.2,
                   ) -> dict:        
        '''
        Used for adding residual water and baselines. config dictionaries are 
        needed for each one.
        '''
        out, baselines, res_water = offsets
        # print('max_val: ',max_val.squeeze())
        # print('max_base: ',baselines.shape,torch.max(baselines).squeeze(), torch.min(baselines).squeeze())
        scale = max_val#10**(OrderOfMagnitude(fid) - OrderOfMagnitude(out))
        out, ind = rand_omit(out, 0.0, drop_prob)
        offset = out.clone() * scale

        if not isinstance(baselines, type(None)): 
            baselines *= scale
            if drop_prob: baselines[ind,...] = 0.0
        if not isinstance(res_water, type(None)): 
            res_water *= scale
            if drop_prob: res_water[ind,...] = 0.0
        if not isinstance(out, int):# == 0: meaning offsets were not included
            # fid += inv_Fourier_Transform(out*scale)
            fid = inv_Fourier_Transform(Fourier_Transform(fid) + out*scale)


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
        # print('start: ',torch.squeeze(cfg.start))
        # print('end: ',torch.squeeze(cfg.end))
        baselines = batch_smooth(bounded_random_walk(cfg.start, cfg.end, 
                                                     cfg.std, cfg.lower_bnd, 
                                                     cfg.upper_bnd, 
                                                     cfg.length), 
                                 cfg.windows, 'constant')

        # baselines, _ = self.normalize(signal=baselines, fid=False, denom=None, noisy=-3)

        trend = batch_linspace(baselines[...,0].unsqueeze(-1),
                               baselines[...,-1].unsqueeze(-1), 
                               cfg.length)
        # print('trend.shape: {}, start: {}, stop: {}'.format(trend.shape,baselines[...,0].squeeze(),
        #                        baselines[...,-1].squeeze()))
        baselines = baselines - trend
        # # print('def baselines:: baselines.shape ',baselines.shape)
        baselines, _ = self.normalize(signal=baselines, fid=False, denom=None, noisy=-3)

        if cfg.rand_omit>0: 
            baselines, _ = rand_omit(baselines, 0.0, cfg.rand_omit)

        # print(type(baselines), type(config['scale']))

        """
        TODO for Tuesday:
        figure out why the baselines form a giant U shape from end to end
        - It's the Hilbert Transform! Not sure why it does that though
        """

        # Convert simulated residual water from local to clinical range before 
        # Hilbert transform makes the imaginary component. Then resample 
        # acquired range to cropped range.
        ppm = self._ppm.clone()#.unsqueeze(-1)#.repeat(1,baselines.shape[0],1)
        # print(ppm.amin(),ppm.amax(), self.ppm.amin(),self.ppm.amax())
        # original = batch_linspace(start=cfg.start, stop=cfg.end, steps=baselines.shape[-1])
        # # print(baselines.shape,original.shape)
        # # print('isnan test: ',torch.isnan(baselines).any())
        # a = self.sim2acquired(baselines * config['scale'], 
        #                                   [cfg.start, cfg.end],
        #                                   self.ppm)
        # # print('isnan test: ',torch.isnan(a).any())
        # a = HilbertTransform(a)
        # # print('isnan test: ',torch.isnan(a).any())
        # raw_baseline = a
        # raw_baseline = HilbertTransform(
        #                 self.sim2acquired(baselines * config['scale'], 
        #                                   [cfg.start, cfg.end],
        #                                   self.ppm)
        #                )
        # print('baselines.shape: {}'.format(baselines.shape))
        cropRange = [torch.as_tensor(val).unsqueeze(-1) for val in cfg.cropRange]
        raw_baseline = self.sim2acquired(baselines * config['scale'], 
                                         [cropRange[0], cropRange[1]],
                                         self.ppm)
        # print('min: {}, max: {}'.format(raw_baseline.min().squeeze(), raw_baseline.max().squeeze()))
        # raw_baseline = Fourier_Transform(rinv_Fourier_Transform(raw_baseline))
        # print('raw_baseline.shape: {}'.format(raw_baseline.shape))
        # raw_baseline = torch.cat([raw_baseline, torch.zeros_like(raw_baseline)],
        #                          dim=-2)
                       
        # print('isnan test: ',torch.isnan(raw_baseline).any())
        ch_interp = CubicHermiteInterp(xaxis=self.ppm, signal=raw_baseline)
        baselines = ch_interp.interp(xs=self.ppm_cropped)
        # print('min: {}, max: {}'.format(baselines.min().squeeze(), baselines.max().squeeze()))

        # raw_baseline = torch.cat([raw_baseline, torch.zeros_like(raw_baseline)],dim=-2)
        raw_baseline = raw_baseline.repeat_interleave(2,dim=-2)

        # # return baselines.fliplr(), raw_baseline.fliplr()
        # return torch.fliplr(baselines), raw_baseline
        # return baselines, raw_baseline
        # return raw_baseline.flip(dims=[-1]), raw_baseline
        # return raw_baseline.fliplr(), inv_Fourier_Transform(raw_baseline)
        return raw_baseline.fliplr(), raw_baseline
    
    
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
        t = self.t.clone()#.mT # [1, 8192] 
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
        # dB0.shape = [bS, length, length, length]
        # # output.shape = [bS, 1, length^3, 1] * [1, 1, 1, 8192] 
        # #                => [bS, 1, length^3, 8192]

        # Let's make a for-loop to iterate over the voxel z- and y-dimensions
        identity, out = torch.ones_like(t).repeat(1,1,2,1), 0
        for i in range(dB0.shape[-1]): # z-dimension
            for j in range(dB0.shape[-2]): # y-dimension
                temp = dB0[...,j,i].unsqueeze(-1).unsqueeze(-1)
                temp = temp.unsqueeze(1).flatten(start_dim=2, end_dim=-1)
                temp = temp.unsqueeze(-1) * t
                out += complex_exp(identity,
                               -1*temp.unsqueeze(-2).deg2rad()).sum(dim=-3)
        del temp
        # Eqn 4 in Ningzhi Li 2015, "Spectral fitting using basis set..."
        return out

        # # output.shape = [bS, length, length, length]
        # # print('dB0.shape: {}'.format(dB0.shape))
        # dB0 = dB0.unsqueeze(1).flatten(start_dim=2, end_dim=-1).unsqueeze(-1)
        # # output.shape = [bS, 1, length^3, 1] * [1, 1, 1, 8192] 
        # #                => [bS, 1, length^3, 8192]
        # # print('dB0.shape: {}; t.shape: {}'.format(dB0.shape,t.shape))
        # dB0 = dB0 * t

        # identity = torch.ones_like(t).repeat(1,1,2,1)

        # # Eqn 4 in Ningzhi Li 2015, "Spectral fitting using basis set..."
        # out = complex_exp(identity,
        #                    -1*dB0.unsqueeze(-2).deg2rad()).sum(dim=-3)
        # # print('out.shape: {}'.format(out.shape))
        # return out


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
        # print('fid.shape {}, coil_sens.shape {}'.format(fid.shape,coil_sens.shape))
        return fid * coil_sens.expand_as(fid)


    def eddyCurrentKloseRemoval(self,
                                fid: torch.Tensor,
                                water: torch.Tensor,
                               ) -> torch.Tensor:
        phase = torch.angle(torch.view_as_complex(water.transpose(-1,-2)))
        return complex_exp(fid, -phase)


    def first_order_eddy_currents(self,
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
        a = -1*f_mod*t*2*PI

        return complex_exp(fid, -1*f_mod*t*2*PI)
        # print('f_mod.shape: {}, (-1*f_mod*t*2*PI).shape: {}'.format(f_mod.shape,a.shape))


    def first_order_phase(self, 
                        fid: torch.Tensor, 
                        phi1: torch.Tensor,
                       ) -> torch.Tensor:
        '''
        Current implementation does this in the frequency domain. Time domain
        would require convolving the two instead of multiplication.
        '''
        for _ in range(fid.ndim - self.phi1_ref.ndim): 
            self.phi1_ref = self.phi1_ref.unsqueeze(0)
        for _ in range(fid.ndim - phi1.ndim): 
            phi1 = phi1.unsqueeze(-1)

        # exp(i*phase) is the correction, therefore, exp(-i*phase) adds it

        return inv_Fourier_Transform(
                    complex_exp(Fourier_Transform(fid), 
                                (-1*self.phi1_ref*phi.deg2rad()))
            )
# ================================================================================
        # # return inv_Fourier_Transform(complex_exp(torch.ones_like(fid),
        # #         self._ppm*phi1)) * fid

        # return inv_Fourier_Transform(
        #     Fourier_Transform(fid)*complex_exp(torch.ones_like(fid),
        #         self._ppm*phi1)
        #     )

        # # return complex_exp(fid, -1*(phi1*self.phi1_ref.pow(-1))/360)

        # # return complex_exp(fid, -1*(phi1 + self.phi1_ref).deg2rad())

        
    def frequency_shift(self, 
                        fid: torch.Tensor, 
                        param: torch.Tensor,
                        t: torch.Tensor=None,
                       ) -> torch.Tensor:
        '''
        Do NOT forget to specify the dimensions for the (i)fftshift!!! 
        Will reorder the batch samples!!!
        Also, keep fshift in Hz
        Correction: exp(-1j*fshift*t)
        Adding: exp(+1j*fshift*t)
        '''
        t = self.t if t==None else t
        t = t.t() if t.shape[-1]==1 else t
            
        for _ in range(fid.ndim - param.ndim): param = param.unsqueeze(-1)
        for _ in range(fid.ndim - t.ndim): t = t.unsqueeze(0)
        # print('frequency_shift:: fid.shape {}, param.shape {}, t.shape {}'.format(fid.shape, param.shape, t.shape))
        f_shift = param.mul(t).expand_as(fid)
        # f_shift = param.mul(t)
        # print('f_shift.shape: ',f_shift.shape)
        # f_shift = f_shift.expand_as(fid)
        
        return complex_exp(fid, f_shift)
        
        
    def generate_noise(self, 
                       fid: torch.Tensor, 
                       param: torch.Tensor,
                       max_val: torch.Tensor,
                       zeros: torch.Tensor, # number of coils with no signal
                       transients: torch.Tensor=None,
                       uncorrelated: bool=True,
                      ) -> torch.Tensor:
        '''
        SNR formula (MRS): snr_lin = max(real(spectra)) / std_noise 
        Signal averaging improves SNR by a factor of sqrt(num_coil)
        '''
        '''
        The std and lin_snr of the noise vectors generated in this section match the simulation parameters
        near perfectly. The noise seems much more undercontrol after changing the Hilbert Transform fcn
        (switching from rfft/irfft to fft/ifft).

        I still don't know why the overall SNRs vary so much. It has to be something outside of this code.
        Even with zero-order phase turned off, it still varies greatly.

        Without MM/Lip, the mean PCr SNR = 8.5278dB. Parameters set to 8.6dB. STD=0.8199
          min/max = 7.2079/9.7023
        This is MUCH more reasonable, but I still don't know where the problem comes from so I can 
        adjust the simulations to get more accurate nosie profiles.




        '''
        zeros, d = torch.where(zeros<=0.0,1,0).sum(dim=-1,keepdims=True), -4
        if transients.ndim==1: zeros, transients, d = None, None, -3

        # print('generate_noise')
        # print('fid.shape {}, param.shape {}, max_val.shape {}'.format(fid.shape, param.shape, max_val.shape))
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
        if isinstance(transients, type(None)): std_dev = std_dev.squeeze(-2)
        # print('std_dev.shape {}'.format(std_dev.shape))
        
        if uncorrelated: 
            # Quadrature coils where real/phase are recorded separately
            # Separate coils ==> uncorrelated noise for real/phase channels
            std_dev = std_dev.repeat_interleave(repeats=2,dim=-1)
            # print('std_dev.shape: ',std_dev.shape)

        '''
        Notes:
        # dim=2: no change                      [channels, length] => [1, channels, length]
        # dim=3: no change                                           [bS, channels, length]
        # dim=4: spectral editing or transients :: [bS, (edit/transient), channels, length]
        # dim=5: editing & transients ::           [bS, edit, transients, channels, length]
        # output.shape: [bS, ON/OFF, [noisy, noiseless], transients, channels, length]
        '''
        # e = torch.distributions.normal.Normal(0,std_dev)
        # e = e.sample([fid.shape[-1]])
        # # print('e.shape {}'.format(e.shape))
        # e = torch.movedim(e,0,-1)
        # # print('e.shape {}'.format(e.shape))

        # e = torch.movedim(torch.distributions.normal.Normal(0,std_dev).sample([fid.shape[-1]]),0,-1)
        # print('std_dev.shape {}'.format(std_dev.shape))
        # e = torch.movedim(torch.distributions.normal.Normal(0,torch.ones_like(std_dev)).sample([fid.shape[-1]]),0,-1)
        e = torch.distributions.normal.Normal(0,torch.ones_like(std_dev)).sample([fid.shape[-1]])
        # print('e.shape: ',e.shape)
        e = torch.movedim(e,0,-1)
        # print('e.shape: ',e.shape)
        e = rFourier_Transform(e) if not uncorrelated else Fourier_Transform(e)
        # print('e.shape: ',e.shape)
        mn = e.mean(dim=-1, keepdims=True)
        std = e.std(dim=-1, keepdims=True)
        std[std==0] += 1e-6
        e = (e - mn).div(std).mul(std_dev.unsqueeze(-1))
        # print('isnan test: ',torch.isnan(e).any())
        # # print('e.shape: ',e.shape)
        # # e = e.mul(std_dev.unsqueeze(-1))
        # # print('e.shape: ',e.shape)
        # test = True
        # cnt = -1
        # # threshold = 0.05
        # # while test:
        # #     cnt += 1
        # #     print('refinement round {}'.format(cnt))
        # #     check0 = (max_val/e.std(dim=-1, keepdims=True)>=(1-threshold)*lin_snr)
        # #     check1 = (max_val/e.std(dim=-1, keepdims=True)<=(1+threshold)*lin_snr)
        # #     ind = torch.where(check0 | check1, 0, 1)
        # #     # ind = torch.where((e.std(dim=-1, keepdims=True)>=(1-threshold)*std_dev) | (e.std(dim=-1, keepdims=True)<=(1+threshold)*std_dev),0,1)
        # #     if not (ind==1).any(): test = False
        # #     e = self.refine_noise(fid_shape=fid.shape[-1], ind=ind, lin_snr=lin_snr, mx_val=max_val, std_dev=std_dev, e=e)
        # threshold = 0.1
        # e0 = HilbertTransform(e)
        # while test:
        #     cnt += 1
        #     # print('refinement round {}'.format(cnt))
        #     new_snr = max_val/e0.std(dim=-1, keepdims=True)
        #     ind = torch.where(new_snr>=(lin_snr-threshold),   0, 1)
        #     ind = torch.where(new_snr<=(lin_snr+threshold), ind, 1)
        #     # print('lin_snr: {}, new_snr: {}'.format(lin_snr,new_snr))
        #     # ind = torch.where(check0 and check1, 0, 1)
        #     # print(ind)
        #     # ind = torch.where((e.std(dim=-1, keepdims=True)>=(1-threshold)*std_dev) | (e.std(dim=-1, keepdims=True)<=(1+threshold)*std_dev),0,1)
        #     if not (ind==1).any(): test = False
        #     # print('refinement round {}'.format(cnt))
        #     e0 = HilbertTransform(self.refine_noise(fid_shape=fid.shape[-1], ind=ind, lin_snr=lin_snr, mx_val=max_val, std_dev=std_dev, e=e))

        # if uncorrelated:
        #     return inv_Fourier_Transform(e)

        # # return inv_Fourier_Transform(HilbertTransform(e))
        # e = inv_Fourier_Transform(HilbertTransform(e))
        # # print('e.shape: ',e.shape)
        # return e
        # return inv_Fourier_Transform(e)
        e = inv_Fourier_Transform(e)
        # print('isnan test: ',torch.isnan(e).any())
        return e, d

    def refine_noise(self,
                     fid_shape, # fid.shape[-1]
                     ind: torch.Tensor,
                     lin_snr: torch.Tensor,
                     mx_val: torch.Tensor,
                     std_dev: torch.Tensor,
                     e: torch.Tensor) -> torch.Tensor:
        # ind = torch.where(((e.std(dim=-1, keepdims=True)>=0.95*std_dev) and (e.std(dim=-1, keepdims=True)<=1.05*std_dev)),0,1)
        # if (ind==1).any():
        e0  = torch.movedim(torch.distributions.normal.Normal(0,torch.ones_like(std_dev)).sample([fid_shape]),0,-1)
        mn  = e0.mean(dim=-1, keepdims=True)
        std = e0.std(dim=-1, keepdims=True)
        e0  = (e0 - mn).div(std).mul(std_dev.unsqueeze(-1))
        e[ind] = e0[ind]
        return e


    def line_summing(self,
                     fid: torch.Tensor,
                     params: torch.Tensor,
                     mm: int,
                     wrt_metab: str,
                    ) -> tuple:
        """
        This top part uses creatine for the SNR calculations. The bottoms parts that have
        been commented out use the max of the real components.
        """
        l = len(self._metab) - self.MM if self.MM else len(self._metab)
        index = [self.index[m.lower()] for m in wrt_metab.split(',')]
        #    [self.index['pcr']]
        # try: index.append(self.index['cr'])
        # except KeyError as E: pass

        mx_values = torch.amax(
            Fourier_Transform(fid)[...,tuple(index),0,:].unsqueeze(-2).sum(dim=-3, 
                keepdims=True), dim=-1, keepdims=True)
        if not mm:
            fidSum = fid.sum(dim=-3) 
            spectral_fit = fidSum.clone()
            # mx_values = torch.amax(
            #     Fourier_Transform(fid)[...,0,:].unsqueeze(-2).sum(dim=-3, 
            #         keepdims=True), dim=-1, keepdims=True) 
        else:
            mm = fid[...,l:,:,:].sum(dim=-3)
            fidSum = fid[...,0:l,:,:].sum(dim=-3)
            spectral_fit = fidSum.clone()
            # mx_values = torch.amax(
            #     Fourier_Transform(fidSum)[...,0,:].unsqueeze(-2), dim=-1, 
            #     keepdims=True) 
            fidSum += mm
        for _ in range(mx_values.ndim - fidSum.ndim): mx_values = mx_values.squeeze(-1)
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
        t = self.t.clone().unsqueeze(0)
        g = g.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2,1)
        
        return fid * torch.exp(-g * t.unsqueeze(0).pow(2)) 


    def lineshape_lorentzian(self, 
                             fid: torch.Tensor, 
                             d: torch.Tensor, 
                            ) -> torch.Tensor:
        t = self.t.clone().unsqueeze(0)
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
        # t = self.t.clone().t().unsqueeze(0)
        t = self.t.clone().unsqueeze(0)#.unsqueeze(0)
        d = d.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2,1)
        g = g.unsqueeze(-1).unsqueeze(-1).expand_as(d)
        if d.dtype==torch.float64: d = d.float()
        # print('lineshape_voigt:: t.shape {}, d.shape {}, g.shape {}'.format(t.shape, d.shape, g.shape))
        # a = g*t.unsqueeze(0)
        # b = (-d - a) * t.unsqueeze(0)
        # c = torch.exp(b)
        # print(a.shape, b.shape, c.shape)
        
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
                       params: torch.Tensor,
                       wrt_metab: str='cr',
                      ) -> dict:

        '''
        Both peak height and peak area are returned for each basis line. The 
        basis fids are first modulated and then broadened. The recovered 
        spectra are then quantified.
        '''
        # print(type(self.index))
        ind = tuple([self.index[m.lower()] for m in wrt_metab.split(',')])
        coeffs = params[...,self.index['metabolites']].clone()#.unsqueeze(-1)

        # Quantities
        specs = Fourier_Transform(fid)[...,0,:].unsqueeze(-2) # real component

        # print('specs.shape {}, self.ppm.shape {}'.format(specs.shape, self.ppm.shape))

        area = torch.stack([torch.trapz(specs[...,i,:,:], self.ppm, 
                dim=-1).unsqueeze(-1) for i in range(specs.shape[-3])], dim=-3)
        height = torch.max(specs, dim=-1, keepdims=True).values

        area_denom = area[...,ind,:,:].sum(dim=-3, keepdims=True)
        height_denom = height[...,ind,:,:].sum(dim=-3, keepdims=True)
        
        # Normalize to metabolite(s) wrt_metab
        rel_area = area.div(area_denom)
        rel_height = height.div(height_denom)

        return {'coefficient': coeffs,
                'area': area.squeeze(-2), 
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


        # self.syn_basis_fids = self.resample_(signal=self.syn_basis_fids, 
        #                                      ppm=self.t,
        #                                      length=2048,#basisFcn_len,
        #                                      target_range=target_range[0],
        #                                      flip=False)
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
        Nothing should ever be flipped EXCEPT the actual axis when plotting!!!
        - 10.05.2023 JTL
        '''
        dims, dims[-1] = list(signal.shape), -1
        ppm = ppm if not isinstance(ppm, type(None)) else self.ppm
        ppm = ppm.unsqueeze(0).squeeze()
        if not (ppm[...,0]<ppm[...,-1]): 
            # print('flipping ppm')
            ppm = torch.flip(ppm.unsqueeze(0), dims=[-1]).squeeze(0)
        
        if isinstance(target_range, type(None)): target_range = self.cropRange
        # print('def resample_ target_range: ',target_range)
        # print('ppm.amin() {}, ppm.amax() {}'.format(ppm.amin(),ppm.amax()))
        new = torch.linspace(start=target_range[0], end=target_range[1], 
                             steps=int(length)).to(signal.device)
        # print('new[0] {}, new[-1] {}, len {}'.format(new[0],new[-1], int(length)))
        for i in range(signal.ndim - new.ndim): new = new.unsqueeze(0)

        if flip: 
            # print('flipping')
            signal = torch.flip(signal, dims=[-1])

        chs_interp = CubicHermiteInterp(ppm, signal)
        signal = chs_interp.interp(new)

        if flip: 
            # print('flipping')
            return torch.flip(signal, dims=[-1])

        return signal
    
    
    def residual_water(self,
                       config: dict,
                      ) -> tuple():
        cfg = SimpleNamespace(**config)

        start_prime = cfg.cropRange_resWater[0] + cfg.start_prime
        end_prime   = cfg.cropRange_resWater[1] -   cfg.end_prime
        ppm = batch_linspace(start=start_prime, stop=end_prime, 
                             steps=int(cfg.length))
        # print(cfg.lower_bnd)
        res_water = batch_smooth(bounded_random_walk(cfg.start, cfg.end, 
                                                     cfg.std, -1*cfg.lower_bnd, 
                                                     cfg.upper_bnd, 
                                                     cfg.length),
                                 cfg.windows, 'constant') 
        trend = batch_linspace(res_water[...,0].unsqueeze(-1),
                               res_water[...,-1].unsqueeze(-1), 
                               cfg.length)
        res_water -= trend
        res_water, _ = self.normalize(signal=res_water, fid=False, denom=None, noisy=-3)

        if cfg.rand_omit>0: 
            res_water, ind = rand_omit(res_water, 0.0, cfg.rand_omit)

        # Convert simulated residual water from local to clinical range before 
        # Hilbert transform makes the imaginary component. Then resample 
        # acquired range to cropped range.
        # print(type(res_water), type(config['scale']))
        raw_res_water = HilbertTransform(
                        self.sim2acquired(res_water * config['scale'], 
                                          [start_prime, 
                                           end_prime], self.ppm)
                        )
        ch_interp = CubicHermiteInterp(xaxis=self.ppm, signal=raw_res_water)
        res_water = ch_interp.interp(xs=self.ppm_cropped)

        # return res_water, raw_res_water
        return raw_res_water, raw_res_water   # raw_res_water is flat on the tails


    def set_parameter_constraints(self, cfg: dict):
        cfg_keys = [k.lower() for k in cfg.keys()]
        # print(cfg_keys)
        # print(self._index)

        for k, ind in self._index.items():
            if k in cfg_keys:
                # print(k)
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

            # out += baselines.clone()
            # print('line 990 raw_baselines.shape ',raw_baselines.shape)
        else: 
            baselines, raw_baselines = None, None

        if residual_water: 
            print('>>>>> Residual Water')
            res_water, raw_res_water = self.residual_water(residual_water)
            out += raw_res_water.clone()
        else: 
            res_water, raw_res_water = None, None

        return out, baselines, res_water
        # return out, raw_baselines, raw_res_water
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

        pad = 1000 # number of points added to each side
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
        # print('line.shape: {}, padding: {}'.format(line.shape, padding))
        signal = torch.nn.functional.pad(input=line, pad=padding, 
                                         mode="constant", value=0)
        # print('signal.shape: ',signal.shape)
        # print('sim2acquired')
        signal = batch_smooth(x=signal, window_len=float(3.0/signal.shape[-1]))
        # print('isnan(signal) test: ',torch.isnan(signal).any())
        ch_interp = CubicHermiteInterp(xaxis=xaxis, signal=signal)
        return ch_interp.interp(xs=target_ppm)
        # out = ch_interp.interp(xs=target_ppm)
        # # print('isnan test: ',torch.isnan(out).any())
        # # print('out.shape: ',out.shape)
        # return out

    
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

    
    def zero_order_phase(self, 
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
            if gen: print('>>>>> Simulating B0 field heterogeneities')
            # Memory limitations require this to be calculated either before 
            # or after the spectra
            B0 = self.B0_inhomogeneities(b0=params[:,self.index['b0']],
                                         param=params[:,self.index['b0_dir']])

        # Simulate the Residual Water and Baselines
        if offsets:
            if gen: print('>>>>> Generating Baseline/Residual Water offsets')
            offset = self.simulate_offsets(baselines=baselines, 
                                           residual_water=residual_water, 
                                           drop_prob=drop_prob)


        # Define basis spectra coefficients
        if gen: print('>>>>> Preparing metabolite coefficients')
        fid = self.modulate(fids=self.syn_basis_fids, 
                            params=params[:,self.index['metabolites']])
        # fid.shape = torch.Size([bS, num_basisfcns, (num_moieties), 2, spec_length])


        '''
        Not implemented yet
        if not isinstance(diff_edit, type(None)): 
            fid = torch.stack((fid, 
                               self.modulate(fids=self.difference_editing_fids, 
                               params=diff_edit[:,self.index['metabolites']])),
                              dim=1)
        '''

        # Apply B0 inhomogeneities
        if b0: 
            if gen: print('>>>>> Applying B0 field distortions')
            fid = self.add_inhomogeneities(fid, B0)
        
        # Line broadening
        if broadening:
            if gen: print('>>>>> Applying line shape distortions')
            fid = self.lineshape_correction(fid=fid, d=params[:,self.index['d']], 
                                                     g=params[:,self.index['g']])

        # Basis Function-wise Frequency Shift
        if fshift_i:
            if gen: print('>>>>> Shifting individual frequencies')
            fid = self.frequency_shift(fid=fid, 
                                       param=params[:,self.index['f_shifts']])
            
        # Summing the basis lines
        # # If moiety-level basis functions are used, combine them first
        if fid.ndim==5: fid = fid.sum(dim=-3)
        # # Saves the unadulterated original as the spectral_fit and the max 
        # # values for the SNR calculations which should not consider artifacts
        fidSum, spectral_fit, mx_values = self.line_summing(fid=fid, 
                                                            params=params, 
                                                            mm=self.MM)

        # Add the Residual Water and Baselines
        if offsets:
            if gen: print('>>> Offsets')
            fidSum, offsets = self.add_offsets(fid=fidSum, 
                                               offsets=offset, 
                                               max_val=mx_values,
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
            noise, d = self.generate_noise(fid=fidSum, 
                                           max_val=mx_values, 
                                           param=params[:,self.index['snr']], 
                                           zeros=params[:,self.index['coil_sens']], # num of zeroed out coils
                                           transients=params[:,self.index['coil_snr']],
                                           uncorrelated=True,
                                           wrt_metab=wrt_metab)
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
                           fid=spectral_fit,#.unsqueeze(1), 
                           coil_sens=params[:,self.index['coil_sens']])

        
        if multicoil>>1:
            cp0, cp1 = None, None
            if coil_fshift:
                # assert()
                if gen: print('>>> Coil Frequency Drift')
                cp0 = fidSum.clone()
                fidSum = self.coil_freq_drift(fid=fidSum, 
                               f_shift=params[:,self.index['coil_fshift']])
                fidSum = torch.cat([fidSum,cp0,fidSum],dim=d)
                # spectral_fit = self.coil_freq_drift(fid=spectral_fit, 
                #                f_shift=params[:,self.index['coil_fshift']])

            if coil_phi0:
                assert(multicoil)
                if gen: print('>>> Coil Phase Drift')
                cp1 = fidSum.clone()
                fidSum = self.coil_phi0_drift(fid=fidSum[:,0:4,...], 
                               phi0=params[:,self.index['coil_phi0']])
                if not isinstance(cp0, type(None)): fidSum = torch.cat([fidSum,cp0], dim=d)
                fidSum = torch.cat([fidSum,cp1[:,4:,...]], dim=d)

        if snr_combo=='avg' and multicoil>1:
            # output.shape: [bS, ON\OFF, [noisy/clean], ~transients~, channels, length]  
            fidSum = fidSum.mean(dim=-3)
            # spectral_fit = spectral_fit.mean(dim=-3)

            
        # Rephasing Spectrum
        if phi0:
            if gen: print('>>>>> Rephasing spectra - zero-order')
            fidSum = self.zero_order_phase(fid=fidSum, 
                                   phi0=params[:,self.index['phi0']])

        # Rephasing Spectrum
        if phi1:
            if gen: print('>>>>> Rephasing spectra - first-order')
            fidSum = self.first_order_phase(fid=fidSum, 
                                   phi1=params[:,self.index['phi1']])
        
        # Frequency Shift
        if fshift_g:
            if gen: print('>>>>> Shifting global frequencies')
            fidSum = self.frequency_shift(fid=fidSum, 
                                   param=params[:,self.index['f_shift']])

        # Eddy Currents
        if eddy:
            fidSum = self.first_order_eddy_currents(fid=fidSum, 
                                   params=params[:,self.index['ecc']])
            # fid = self.first_order_eddy_currents(fid=fid, 
            #                        params=params[:,self.index['ecc']])

        # Apodize
        if apodize:
            fidSum = self.apodization(fid=fidSum, hz=apodize)
            # fid = self.apodization(fid=fid, hz=apodize)

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
                print('resampling spectra')
                specSummed = self.resample_(signal=specSummed, length=self.length, flip=False)
                spectral_fit = self.resample_(signal=spectral_fit, length=self.length, flip=False)
        else:
            specSummed = fidSum
            # spectral_fit = spectral_fit # redundant
                     
        # Calculate magnitude spectra
        if magnitude:
            if gen: print('>>>>> Generating magnitude spectra')
            specSummed = self.magnitude(specSummed)
            spectral_fit = self.magnitude(spectral_fit)
        
        # Normalize
        if not isinstance(noise, bool): 
            # print(noise)
            # if noise==False:
            specSummed, denom = self.normalize(signal=specSummed, noisy=d)
        else:
            specSummed, denom = self.normalize(signal=specSummed)
        spectral_fit, _ = self.normalize(signal=spectral_fit, denom=denom)

        # Convert normalized spectra back to time-domain
        if fids and resample:
            num_pts = zero_fill if zero_fill else self.length
            t = torch.linspace(self.t.amin(), self.t.amax(), 
                               max(self.t.squeeze().shape))
            specSummed = self.resample_(signal=specSummed, length=self.length, 
                                        ppm=t, flip=False, 
                                        target_range=[self.t.amin(), 
                                                      self.t.amax()])
            spectral_fit = self.resample_(signal=spectral_fit, length=self.length, 
                                          ppm=t, flip=False, target_range=[
                                          self.t.amin(), self.t.amax()])



        # spectral_fit = specSummed[:,1,...];
        if not isinstance(diff_edit, type(None)): 
            print('>>> Creating the difference spectra')
            # Consensus paper recommends dividing difference spectra by 2. Not 
            # sure about any other consequenctial effects
            specSummed = torch.cat(specSummed, (specSummed[:,0,...] - 
                            specSummed[:,1,...]).unsqueeze(1) / 2, dim=1)
            spectral_fit = torch.cat(spectral_fit, (spectral_fit[:,0,...] - 
                            spectral_fit[:,1,...]).unsqueeze(1) / 2, dim=1)
            # spectral_fit = specSummed[:,:,1,...]; 

            
        print('>>>>> Compiling spectra')
        return self.compile_outputs(specSummed, spectral_fit, offsets, params, 
                                    denom, self.quantify_metab(fid, params, 
                                                               wrt_metab)
                                    )

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
