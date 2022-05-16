import os
import scipy.io as io
import torch
import torch.nn as nn
import numpy as np
import copy
from functools import reduce
from collections import OrderedDict
# from modules.aux.auxiliary import matrix_inverse
from modules.physics_model.aux.splines import create_splines_linspace
# from modules.aux.auxiliary import counter
from src.interpolate import CubicHermiteMAkima as CubicHermiteInterp#, batch_linspace 
from modules.physics_model.aux.aux import *


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
#     version = 1
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
        self.first_time = True
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
        # PPM ranges from Sara Nelson's code
        self.regions = [[2.852, 3.460], # Choline and Creatine
                        [1.850, 2.252]] # NAA and NAAG
        self.fshift_regions = [[1.300, 3.970]]
        self.t = self.t.unsqueeze(-1).float()
        self.baseline_t = self.baseline_t.unsqueeze(-1).float()
        self.cropped_t = self.cropped_t.unsqueeze(-1).float()
        self._basis_metab = []
        
        
        self.tracking = []
        
    def __repr__(self):
        lines = sum([len(listElem) for listElem in self.totals])   
        out = 'MRS_Fitting_PhysicsModel(basis={}, lines={}'.format('Osprey, GE_PRESS144', lines)
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
#         for i, m in enumerate(metab): 
        for m in metab: 
            if 'Cho' in m: m = 'Ch'
            if 'Cre' in m: m = 'Cr'
            if 'Mac' in m: m = 'MM'
            temp = []
            for i, (name, value) in enumerate(self.linenames.items()):
                case1 = (m.lower() in name.lower()[2:-2])
                case2 = (name.lower()[2:-2] in m.lower() and metab_as_is)
#                 print('case1 {}, case2 {}'.format(case1,case2))
                if (case1 and not metab_as_is) or (case1 and case2):
                    if (m=='Ch' and 'Cr' not in name) or m!='Ch':
                        ind.append(value), indA.append(int(i))
                        temp.append(int(len(ind)-1))
                        dict.update({name[2:-2]: torch.empty(1)})#len(ind) - 1})
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
#                 print('name: {}, mult: {}'.format(n,m))
                for _ in range(m): header.append(n)
            for m in header: 
                for i, name in enumerate(self.linenames.keys()):
                    if m.lower() in name.lower():# in m.lower(): 
#                         print(m)
                        indA.append(int(i))
            mx_ind = 2
#             print('len(indA): ',len(indA))
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
#             names, b = ['d']*l, []
#             for n in ['g','fshift','snr','phi0','phi1']: names.append(n)
#             if baselines: 
#                 for _ in range(5): names.append('baseline')
            for m in header:
                for i, name in enumerate(self.linenames.keys()):
#                     if m.lower() in name.lower():#[2:-2] in m.lower(): 
                    if name.lower()[2:-2] in m.lower(): 
#                         if 'baseline' not in m: b.append(int(i))
                        indA.append(int(i))
#             print('header: ',header)
#             print('indA: ',indA)
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
#         print('self.max_ranges: ',self.max_ranges)
#         print('self.min_ranges: ',self.min_ranges)
#         print(self.max_ranges.shape)
        
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
            
#         print('self._index: ',self._index)
        return dict, ind
    
    @property
    def index(self):
        return self._index

    def apodization(self,
                fid: torch.Tensor, 
                hz: int=4) -> torch.Tensor:
        if hz:
            return fid * torch.unsqueeze(torch.exp(-self.t * hz).t(), dim=0).expand_as(fid)
        return fid

    def augment(self,
                data: torch.Tensor,
                t: torch.Tensor,
                ppm: torch.Tensor,
                p: float=0.8,
                phase: bool=False,
                fshift: bool=False) -> torch.Tensor:
        '''
        Augment the input data for the neural network. The includes a frequency shift coupled with
        zero- and first-order phase offsets. 
        
        Augmentations are applied randomly with p=0.8
        '''
        data = data[:,0:2,:]
        if fshift:
            sign = torch.tensor([True if torch.rand([1]) > 0.8 else False for _ in range(data.shape[0])], device=data.device)
            param = torch.rand(data.shape[0], device=data.device)#.unsqueeze
            param[sign].fill_(0.5)
            data = self.frequency_shift(fid=data, param=param, t=t, correction=True, quantify=True)

        if phase:
            sign0 = torch.tensor([True if torch.rand([1]) > 0.8 else False for _ in range(data.shape[0])], device=data.device)
            sign1 = torch.tensor([True if torch.rand([1]) > 0.8 else False for _ in range(data.shape[0])], device=data.device)
            param = torch.rand(data.shape[0], 2, device=data.device)
            param[sign0, 0].fill_(0.5)
            param[sign1, 1].fill_(0.5)
            data = self.dephase(spectra=data, phi=param, ppm=ppm, correction=False)
        return self.magnitude(signal=data, normalize=True)

    def baseline_basis(self,
                       amplitudes: torch.Tensor,
                       phi: torch.Tensor,
                       fshift: torch.Tensor,
                       forward: bool=False) -> torch.Tensor:
        amp = amplitudes.unsqueeze(-1).unsqueeze(-1)
        baseline = torch.cat([amp.mul(self.baseline_basis_set[:,0,:].unsqueeze(1)), 
                              amp.mul(self.baseline_basis_set[:,1,:].unsqueeze(1))], dim=-2).sum(dim=-3)        
        if forward:
            baseline = self.dephase(baseline, 1 - phi, baseline=True, ppm=self.baseline_ppm)
            return self.frequency_shift(baseline, fshift / self.l * self.baseline_basis_set.shape[-1], t=self.baseline_t)
        return baseline

    def baseline_splines(self,
                         amplitudes: torch.Tensor,
                         no_artifacts: bool=True,
                         params: torch.Tensor=None,
                         quantify: bool=True) -> torch.Tensor:
        '''Spectra are normalized to [-1,1], therefore, the splines need to be able to cover that distance'''
        B = self.basis_splines.repeat(amplitudes.shape[0],1,1)
        if amplitudes.ndim==2: amplitudes = amplitudes.unsqueeze(-1)

        if no_artifacts: 
            return HilbertTransform(torch.sum(B * amplitudes, dim=1, keepdims=True), dim=-1)

        return self.frequency_shift(self.dephase(HilbertTransform(torch.sum(B * amplitudes, dim=1, keepdims=True), dim=-1), # shape: [batchSize, num_splines, spectrum_length]
                                                phi=params[:,(self.index['phi0'],self.index['phi1'])],
                                                ppm=self.ppm_cropped,
                                                quantify=quantify,
                                                baseline=True),
                                    param=params[:,self.index['f_shift']],
                                    quantify=quantify,
                                    t=self.cropped_t)
    
    def B0_inhomogeneities(self, 
                           fid: torch.Tensor, 
                           param: torch.Tensor) -> torch.Tensor:
        for _ in range(3-param.ndim): param = param.unsqueeze(-1)
        t = self.t.clone().unsqueeze(0)
        if t.shape[2]==1: t = t.t()
        return complex_exp(fid, param * t)

#         out = B * amplitudes
# #         check(out, 'baseline_splines 0')
#         out = out.sum(dim=1, keepdims=True)
# #         check(out, 'baseline_splines 1')
#         out = HilbertTransform(out, dim=-1)   
#         check(out, 'baseline_splines 2')
#         out = self.dephase(out, # shape: [batchSize, num_splines, spectrum_length]
#                            phi=params[:,(self.index['phi0'],self.index['phi1'])],
#                            ppm=self.ppm_cropped,
#                            quantify=quantify,
#                            baseline=True)
# #         check(out, 'baseline_splines 3')
#         out = self.frequency_shift(out,
#                                    param=params[:,self.index['f_shift']],
#                                    quantify=quantify,
#                                    t=self.cropped_t)
# #         check(out, 'baseline_splines 4')
#         return out
    
    @torch.jit.script
    def pinv(fcns: torch.Tensor, splines: torch.Tensor, index: torch.Tensor=None) -> torch.Tensor:
        AtA = fcns.mT.matmul(fcns)
        lu = torch.linalg.lu_factor_ex(AtA)
#         return torch.diagonal(torch.lu_solve(torch.eye(AtA.shape[-1], device=AtA.device).unsqueeze(0).repeat(AtA.shape[0],1,1), lu[0], lu[1]).matmul(fcns.transpose(-2,-1))[:,:,0:splines.shape[1]].matmul(splines), -2, -1).sum(-1).unsqueeze(-1)
        return torch.diagonal(torch.lu_solve(torch.eye(AtA.shape[-1], device=AtA.device).unsqueeze(0).repeat(AtA.shape[0],1,1), \
                                             lu[0], lu[1]).matmul(fcns.mT).matmul(splines), -2, -1).sum(-1).unsqueeze(-1)
        
    def calc_ED(self, 
                lambdas: torch.Tensor) -> torch.Tensor:
        basis_fcns_ = torch.cat([self.basis_splines.repeat(lambdas.shape[0],1,1),
                                 self.diff_mat.repeat(lambdas.shape[0],1,1).mul(lambdas.pow(0.5))], dim=1)
        basis_splines_ = torch.cat([self.basis_splines.clone().repeat(lambdas.shape[0],1,1),
                                    torch.zeros_like(self.diff_mat.repeat(lambdas.shape[0],1,1))], dim=1)
        self._ED = self.pinv(basis_fcns_, basis_splines_)
        return self._ED

    def dephase(self,
                spectra: torch.Tensor,
                phi: torch.Tensor,
                ppm: torch.Tensor=None,
                baseline: bool=False,
                quantify: bool=True,
                correction: bool=True) -> torch.Tensor:
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
#         print('dephase: ppm.shape ',ppm.shape, spectra.shape)
        spectra = self.firstOrderPhase(spectra=spectra, phi1=phi1, ppm=ppm)#.flip(dims=[-1,-2]))
        fid = inv_Fourier_Transform(spectra)
        fid = self.zeroOrderPhase(fid=fid, phi0=phi0)
        return Fourier_Transform(fid)

    def difference_matrix(self): 
        length = int(self.length)
        self.register_buffer('diff_mat', torch.eye(length).float())#.num_splines))
        start, end, value = (1, 2), (length-1, length-2), (-2, 1)#(self.num_splines-1, self.num_splines-2), (-2, 1)
        for i in range(2):
            for r, c in zip(range(end[i]), range(start[i],length)):#num_splines)):
                self.diff_mat[r,c].fill_(value[i])
        return self.diff_mat

    def firstOrderPhase(self, 
                        spectra: torch.Tensor, 
                        phi1: torch.Tensor,
                        ppm=None,
                        dephase: bool=False) -> torch.Tensor:
        # FID should be in the frequency domain for this step
#         print(self.ppm_ref.device, self._ppm.device)
#         if not ppm==None: print(ppm.device)
        mult = self.ppm_ref - self._ppm if ppm==None else self.ppm_ref - ppm
        while phi1.ndim<3: phi1 = phi1.unsqueeze(-1)
        return complex_exp(spectra, phi1.deg2rad() * mult)
        
    def frequency_shift(self, 
                        fid: torch.Tensor, 
                        param: torch.Tensor,
                        t: torch.Tensor=None,
                        quantify: bool=False,
                        correction: bool=False) -> torch.Tensor:
        '''
        Do NOT forget to specify the dimensions for the (i)fftshift!!! Will reorder the batch samples!!!
        '''
        t = self.t if t==None else t
        if quantify: param = (self.fshift_max - self.fshift_min) * param + self.fshift_min
        if correction: param = -1 * param
            
        f_shift = param.unsqueeze(-1).unsqueeze(-1).mul(-1.0).mul(t.mT)
        if f_shift.ndim>3: f_shift = f_shift.squeeze(1).squeeze(-1)
#         print('f_shift0.shape {}, f_shift1.shape {}, f_shift.shape {}, t.mT.shape {}, fid.shape {}'.format(f_shift0.shape, f_shift1.shape, f_shift.shape, t.mT.shape, fid.shape))
#         f_shift = torch.cat([torch.zeros_like(f_shift), f_shift], dim=1)
        
#         # # Convert back to time-domain
#         fid = inv_Fourier_Transform(fid)
#         # # Apply TD complex exponential
#         fid = complex_exp(fid, f_shift)# -1*f_shift)
#         # # Return to the frequency domain
        return Fourier_Transform(complex_exp(inv_Fourier_Transform(fid), f_shift))
        
        
    def generate_noise(self, 
                       fid: torch.Tensor, 
                       param: torch.Tensor,
                       max_val: torch.Tensor) -> torch.Tensor:
        '''
        SNR formula:
            snr_db = 10*log10(snr_lin * 0.66) # 0.66 Rayleigh distribution correction factor to calculate the true SNR
            snr_lin = max(real(spectra)) / std_noise # Not sure whether to use real or magnitude spectra
        '''
        lin_snr = 10**(param.unsqueeze(-1).unsqueeze(-1) / 10) # convert from decibels to linear scale
        k = 1 / lin_snr # scaling coefficient
        a_signal = torch.FloatTensor([2]).sqrt().pow(-1).to(fid.device) * max_val.values.unsqueeze(-1) # RMS coefficient for sine wave
        scale = k * a_signal # signal apmlitude scaled for desired noise amplitude
        scale[torch.isnan(scale)] = 1e-6

        e = torch.distributions.normal.Normal(0,scale).rsample([fid.shape[2]])
        if e.ndim==2: e = e.unsqueeze(1).permute(1,2,0)
        elif e.ndim==3: e = e.permute(1,2,0)
        elif e.ndim==4: e = e.squeeze(-1).permute(1,2,0)
        
        return HilbertTransform(e)
    
#     @property
    def snr_var(self, 
                spectra: torch.Tensor, 
                SNR: torch.Tensor) -> torch.Tensor:
#         a = torch.FloatTensor([2]).sqrt().pow(-1).to(spectra.device)
#         b = spectra[:,0,:].max(dim=-1,keepdim=True).values.unsqueeze(-1)
#         c = b.div(10**SNR / 10)
#         d = c.pow(2)
#         print('PhysicsModel.snr_var shape: a.shape {}, b.shape {}, c.shape {}, d.shape {}'.format(a.shape, b.shape, c.shape, d.shape))
        std_noise = torch.FloatTensor([2]).sqrt().pow(-1).to(spectra.device) * spectra[:,0,:].max(dim=-1,keepdim=True).values.unsqueeze(-1).div(10**SNR) / 10
        return std_noise.pow(2)
    
#     @property
    def noise_var(self, fid: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
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
                             g: torch.Tensor,
                             label: str,
                            ) -> torch.Tensor:
        '''
        Applies Voigt lineshape corrections using a single Gaussian value and separate Lorentzian values
        for the fat and water peaks 
        '''
#         print('lineshape_correction: ', label, ' ', torch.cat([d.clone().squeeze()[0,::], g.clone().squeeze().unsqueeze(-1)[0]], dim=-1))
        t = self.t.clone().t().unsqueeze(0)
        d = d.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2,1)
        g = g.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(d)
        if d.dtype==torch.float64: d = d.float()
#         print('PM.lineshape_correction ',label, 'fid {}, d {}, g {}'.format(fid.shape,d.shape,g.shape))
        
        return fid * torch.exp((-d - g * t.unsqueeze(0)) * t.unsqueeze(0))
#         a = g * t.unsqueeze(0)
# #         check(a,'lineshape_correction 0')
#         b = -d - a
# #         check(b,'lineshape_correction 1')
#         c = b * t.unsqueeze(0)
# #         check(c,'lineshape_correction 2')
#         e = torch.exp(c)
# #         check(e,'lineshape_correction 3')
#         f = fid * e
# #         check(f,'lineshape_correction 4')
# #         print('a {}, b {}, c {}, e {}, f {}'.format(a.shape, b.shape, c.shape, e.shape, f.shape))
# # #         print('concatenated: ',torch.cat([d[0:5,0,0,:], g[0:5,0,0,:]], dim=-1).squeeze())
# # #         print('exp(x): ',e.squeeze()[0:5,0:5])
#         return f

    def magnitude(self, 
                   signal: torch.Tensor, 
                   normalize: bool=False,
                   individual: bool=False,
                   eps: float=1e-6) -> torch.Tensor:
        '''
        Calculate the magnitude spectrum of the input spectrum.
        '''
#         signal = signal / (torch.max(torch.max(signal, dim=-1, keepdim=True).values, dim=-2, keepdim=True).values + eps)
#         data = torch.sum(signal, dim=-3, keepdim=True) if individual else signal
        magnitude = torch.sqrt(signal[...,0,:].pow(2) + signal[...,1,:].pow(2) + eps).unsqueeze(-2)
#         print('signal.shape {}, data.shape {}, magnitude.shape {}'.format(signal.shape, data.shape, magnitude.shape))
#         print('signal.shape {}, magnitude.shape {}'.format(signal.shape, magnitude.shape))
        out = torch.cat([signal, magnitude], dim=-2)
        if torch.isnan(magnitude).any(): print('Found NAN in magnitude() of physics model')
        if not torch.isfinite(magnitude).all(): print('Found Inf in magnitude() of physics model')
        
        if torch.isnan(signal).any(): print('Found NAN in magnitude().signal of physics model')
        if not torch.isfinite(signal).all(): print('Found Inf in magnitude().signal of physics model')
            
        if normalize:
            if individual: 
                data = torch.sum(signal, dim=-3, keepdim=True)
                magnitude = torch.sqrt(data[...,0,:].pow(2) + data[...,1,:].pow(2) + eps).unsqueeze(-2)
            return out.div(torch.max(magnitude, dim=-1, keepdim=True).values + eps)

        return out

    def normalize(self, 
                  signal: torch.Tensor,
                  fid: bool=False) -> torch.Tensor:
        '''
        Normalize each sample of single or multi-echo spectra. 
            Step 1: Find the max of the real and imaginary components separately
            Step 2: Pick the larger value for each spectrum
        If the signal is separated by metabolite, then an additional max() is necessary
        Reimplemented according to: https://stackoverflow.com/questions/41576536/normalizing-complex-values-in-numpy-python
        '''
        if fid: 
            return signal / torch.max(torch.sqrt(out[:,:,0,:]**2 + out[:,:,1,:]**2), dim=-1, keepdim=True).values.unsqueeze(-2)

        mx = torch.max(signal[...,2,:].unsqueeze(-2).abs(), dim=-1, keepdim=True).values
        mx[mx.isnan()] = 1e-6
        mx[mx==0.0] = 1e-6
        return signal / (mx+1e-6)

    @property
    def ppm(self, 
            cropped: bool=False) -> torch.Tensor:
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
                params[:, self.index['scale']] *= 0.3
            except Exception as e: pass

        return params

    def quantify_metab(self, 
                       params: torch.Tensor, 
                       norm: torch.Tensor=None) -> dict:
        '''
        Both peak height and peak area are returned for each basis line. The basis fids are first 
        modulated and then broadened. The recovered spectra are normalized and then multiplied by
        the normalizing values from the original spectra.
        '''
        if params.ndim==3: params = params.squeeze(-1)
        assert(params.ndim==2)
        
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
#         print('quantify_metab: area.shape {}, self.index["cr"] {}'.format(area.shape, self.index['cr']))
        rel_area = area.div(area[:,self.index['cr'],::].unsqueeze(1))
        rel_height = height.div(height[:,self.index['cr'],::].unsqueeze(1))

        return {'area': area.squeeze(-1), 'height': height.squeeze(-1), 'rel_area': rel_area.squeeze(-1), 'rel_height': rel_height.squeeze(-1), 'params': params}

    def region_masking(self, fshift=False):
        regions = self.regions if not fshift else self.fshift_regions
        mask = torch.zeros_like(self.ppm_cropped)
        for region in regions:
            ind = reduce(lambda x, y: [x0 & y0 for x0, y0 in zip(x, y)], [[self.ppm_cropped >= region[0]], 
                                                                          [self.ppm_cropped <= region[1]]])
            mask += ind[0]
        return mask

    def resample_(self, 
                  signal: torch.Tensor, 
                  ppm: torch.Tensor=None,
                  length: int=512) -> torch.Tensor:
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
        return torch.flip(chs_interp.interp(new.expand(dims)), dims=[-1])#signal.shape[0], signal.shape[1], -1)), dims=[-1])
#         return chs_interp.interp(new.expand(signal.shape[0], signal.shape[1], -1))

    def _resample_(self, 
                   signal: torch.Tensor, 
                   scale: torch.Tensor, 
                   ppm: torch.Tensor=None,
                   quantify: bool=False,
                   length: int=512) -> torch.Tensor:
        '''
        The unit of scale is ppm/point. Multiplying the scale by the desired number of points, length,  
        gives the number of ppm needed for the new range. Adding the starting point gives the new end point.
        '''
        if quantify: 
            scale = scale * 0.6 - 0.3
            scale = scale.float()
        ppm = ppm.unsqueeze(0).squeeze() if not isinstance(ppm, type(None)) else self.ppm.unsqueeze(0).squeeze()
        if not (ppm[0]<ppm[-1]): ppm = torch.flip(ppm.unsqueeze(0), dims=[-1]).squeeze(0)
        stop = self.cropRange[1] + scale.unsqueeze(-1)# * self.cropRange[1]# (self.cropRange[1] - self.cropRange[0]) + self.cropRange[0]
#         print('_resample_ stop.shape: ',stop.shape)
        if stop.ndim==4: stop = stop.squeeze(-1)
        start = torch.as_tensor([self.cropRange[0]], device=scale.device)
#         print(type(start), start.dtype, type(stop), stop.dtype)
        new_ppm_range = batch_linspace(start=start, stop=stop, steps=length).to(signal.device)
#         print('_resample_: ',new_ppm_range.shape, signal.shape)
#         print('new_ppm_range.shape: ',start, new_ppm_range.shape,new_ppm_range.min(),new_ppm_range.max(),new_ppm_range[::,0],new_ppm_range[::,-1])
#         new_ppm_range = torch.linspace(start=self.cropRange[0], end=self.cropRange[1], steps=int(length)).to(signal.device)
#         for i in range(signal.ndim - new_ppm_range.ndim): new_ppm_range = new_ppm_range.unsqueeze(0)
        chs_interp = CubicHermiteInterp(ppm, torch.flip(signal, dims=[-1]))#torch.flip(signal, dims=[-1]))# signal)#
#         chs_interp = CubicHermiteInterp(self.ppm.float(), signal)#torch.flip(signal, dims=[0]))
#         return torch.flip(chs_interp.interp(new_ppm_range.expand(signal.shape[0], signal.shape[1], -1)), dims=[-1])
        dims, dims[-1] = list(signal.shape), -1
        return torch.flip(chs_interp.interp(new_ppm_range.expand(dims)), dims=[-1])#signal.shape[0], signal.shape[1], -1)), dims=[-1])
#       return chs_interp.interp(new_ppm_range.expand(signal.shape[0], signal.shape[1], -1))

    def zeroOrderPhase(self, 
                       fid: torch.Tensor, 
                       phi0: torch.Tensor) -> torch.Tensor:
        for _ in range(3-phi0.ndim): phi0 = phi0.unsqueeze(-1)
        return complex_exp(fid, phi0.deg2rad())# * a)

    def forward(self, 
                params: torch.Tensor, 
                gen: bool=False, 
                noise: bool=False,
                phi1: bool=False,
                scale: bool=False,
                splines: bool=False,
                baselines: bool=True, 
                broadening: bool=True,
                reconstruct: bool=False) -> torch.Tensor:
        if params.ndim>=3: params = params.squeeze()  # convert 3d parameter matrix to 2d [batchSize, parameters]
        if params.ndim==1: params = params.unsqueeze(0) # Allows for batchSize = 1

        params = self.quantify_params(params, label='forward')

        # Define basis spectra coefficients
        if gen: print('>>>>> Preparing metabolite coefficients')
        fid = params[:,self.index['metabolites']].unsqueeze(2).unsqueeze(-1)
        if not splines:
            fid = torch.cat([fid.mul(self.syn_basis_fids[:,0,:].unsqueeze(1)),
                             fid.mul(self.syn_basis_fids[:,1,:].unsqueeze(1))], dim=2)
        else:
            fid = torch.cat([fid.mul(self.fitting_basis_fids[:,:,0,:].unsqueeze(2)),
                             fid.mul(self.fitting_basis_fids[:,:,1,:].unsqueeze(2))], dim=2)

        # Line broadening
        if gen: print('>>>>> Applying line shape distortions')
        fidSum = self.lineshape_correction(fid, params[:,self.index['d']], 
                                                params[:,self.index['g']],'foward')
        
        fidSum = fidSum.sum(dim=-3)
        mx_values = torch.max(fidSum[:,0,:].unsqueeze(1), dim=-1) 
        # Save these values for the SNR calculation
        # Should not be using max values that include the baseline!
        
        # Add spline baselines
        if baselines and splines:
            baseline = self.baseline_splines(params[:,self.index['baseline']] - 1., no_artifacts=True, params=params, quantify=False)
            fidSum += inv_Fourier_Transform(baseline)

        # Add Noise
        if noise:
            if gen: print('>>>>> Adding noise')
            fidSum += self.generate_noise(fidSum, params[:,self.index['snr']], mx_values)

        # Rephasing Spectrum
        if gen: print('>>>>> Rephasing spectra')
        specSummed = self.zeroOrderPhase(fidSum, params[:,self.index['phi0']])
        
        # # Apodize
        if self.apodize:
            specSummed = self.apodization(specSummed, hz=self.apodize)

        # Recover Spectrum
        if gen: print('>>>>> Recovering spectra')
        specSummed = Fourier_Transform(specSummed)

        # Rephasing Spectrum
        if phi1:
            if gen: print('>>>>> Rephasing spectra')
            specSummed = self.firstOrderPhase(specSummed, params[:,self.index['phi1']])

        # Frequency Shift
        if gen: print('>>>>> Shifting frequencies')
        specSummed = self.frequency_shift(specSummed, params[:,self.index['f_shift']])

        # Resampling the spectra
        specSummed = torch.cat([self.resample_(specSummed[:,0,:].unsqueeze(1), length=self.length),
                                self.resample_(specSummed[:,1,:].unsqueeze(1), length=self.length)], dim=1)
        
        # Add Baseline 
        if baselines and not splines:
            if gen: print('>>>>> Adding baselines')
            baseline = self.baseline_basis(params[:,self.index['baseline']], 
                                           phi=params[:,tuple([self.index['phi0'],self.index['phi1']])],
                                           fshift=params[:,self.index['f_shift']],
                                           forward=True)
            ppm = self.baseline_ppm #if not splines else self.ppm_cropped
            specSummed += torch.cat([self.resample_(baseline[:,0,:].unsqueeze(1), ppm=ppm, length=self.length),
                                     self.resample_(baseline[:,1,:].unsqueeze(1), ppm=ppm, length=self.length)], dim=1)
            
        if gen: print('>>>>> Generating magnitude spectra')
        return self.magnitude(specSummed, normalize=True)


    def reconstruct(self, 
                    params: torch.Tensor, 
                    gen: bool=False, 
                    noise: bool=False,
                    phi1: bool=False,
                    splines: bool=False,
                    baselines: bool=True, 
                    broadening: bool=True,
                    reconstruct: bool=False) -> torch.Tensor:
        if params.ndim>=3: params = params.squeeze()  # convert 3d parameter matrix to 2d [batchSize, parameters]
        if params.ndim==1: params = params.unsqueeze(0) # Allows for batchSize = 1

        params = self.quantify_params(params, label='reconstruct')
        # print("params[:,self.index['lambda']].max(): ",params[:,self.index['lambda']].max())
        
        # Define basis spectra coefficients
        fid = params[:,self.index['metabolites']].unsqueeze(2).unsqueeze(-1)
        basis_fids = self.syn_basis_fids if not splines else self.fitting_basis_fids
        fid = torch.cat([fid.mul(basis_fids[:,:,0,:].unsqueeze(2)),
                         fid.mul(basis_fids[:,:,1,:].unsqueeze(2))], dim=-2)
        

        # Line broadening
        fid = self.lineshape_correction(fid, params[:,self.index['d']], 
                                             params[:,self.index['g']],'reconstruct')
        
        fidSum = fid.sum(dim=-3)
        mx_values = torch.max(fidSum[:,0,:].unsqueeze(1), dim=-1) 
        # Save these values for the SNR calculation
        # Should not be using max values that include the baseline!
        
        # Add spline baselines
        if baselines and splines:
            baseline = self.baseline_splines(params[:,self.index['baseline']], no_artifacts=False, params=params, quantify=False)
#             print('baseline reconstruction values: min {}, mean {}, max {}'.format(params[:,self.index['baseline']].min(), params[:,self.index['baseline']].mean(), params[:,self.index['baseline']].max()))
            
        # Rephasing Spectrum
        fidSum = self.zeroOrderPhase(fidSum, params[:,self.index['phi0']])
        
#         # # Apodize
#         if self.apodize:
#             specSummed = self.apodization(specSummed, hz=self.apodize)

        # Recover Spectrum
        specSummed = Fourier_Transform(fidSum)

        # Rephasing Spectrum
        if phi1:
            specSummed = self.firstOrderPhase(specSummed, params[:,self.index['phi1']])

        # Frequency Shift
        specSummed = self.frequency_shift(specSummed, params[:,self.index['f_shift']])

        # Resampling the spectra
        specSummed = torch.cat([self.resample_(specSummed[:,0,:].unsqueeze(1), length=self.length),
                                self.resample_(specSummed[:,1,:].unsqueeze(1), length=self.length)], dim=1)
        
        # Add Baseline 
        if baselines:
            if not splines:
                baseline = self.baseline_basis(params[:,self.index['baseline']], 
                                               phi=params[:,tuple([self.index['phi0'],self.index['phi1']])],
                                               fshift=params[:,self.index['f_shift']],
                                               forward=True)
                ppm = self.baseline_ppm# if not splines else self.ppm_cropped
                baseline = torch.cat([self.resample_(baseline[:,0,:].unsqueeze(1), ppm=ppm, length=self.length),
                                      self.resample_(baseline[:,1,:].unsqueeze(1), ppm=ppm, length=self.length)], dim=1)
            specSummed += baseline
            
        return self.magnitude(specSummed, normalize=True)

    
    
    def fit(self, 
            params: torch.Tensor,  
#             scale: torch.Tensor, 
            magnitude: bool=True,
            individual: bool=False) -> torch.Tensor:
        if params.ndim>=3: params = params.squeeze()  # convert 3d parameter matrix to 2d [batchSize, parameters]
        if params.ndim==1: params = params.unsqueeze(0) # Allows for batchSize = 1
        
        params = self.quantify_params(params, label='fit')
#         scale = scale * 0.6 - 0.3 # +/- 0.3 ppm correction available

        # Define basis spectra coefficients
        fid = params[:,self.index['metabolites']].unsqueeze(-1).unsqueeze(-1)# * self.fitting_basis_fids
        fid = torch.cat([fid.mul(self.fitting_basis_fids[:,:,0,:].unsqueeze(2)),
                         fid.mul(self.fitting_basis_fids[:,:,1,:].unsqueeze(2))], dim=2)

        # Line broadening
        fid = self.lineshape_correction(fid, params[:,self.index['d']], 
                                             params[:,self.index['g']], 'fit')
        
        if not individual: fid = fid.sum(dim=-3)

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
