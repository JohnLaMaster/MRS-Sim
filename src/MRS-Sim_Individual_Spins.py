'''
This is a template for sampling parameters for deep learning applications. It is
annotated so that it can be used as a template for further customization. Otherwise
it can be used as is to generate datasets for deep learning applications. 

To date, most DL applications work with spectra in the frequency domain. The desired
crop range in ppm can be specified in the config.json file. Should time-domain FIDs
be required, that can also be specified in the config file.

Written by: Ing. John T LaMaster, 2023
'''
import argparse
import json
import os
import sys

import numpy as np
import scipy.io as io
import torch
from aux import normalize_old, load_default_values
from mainFcns import _save, prepare, simulate
from types import SimpleNamespace

sys.path.append('../')



'''
Todo: 
- Need to make sure the individual shifts min/max are minimal
- Otherwise, I think this is finished. Just need to make sure the basis
set is finished and the matlab code also works for IndividualSpins
'''

def sample(inputs):
    config, resWater_cfg, baseline_cfg, pm, l, ind, p, totalEntries = inputs

    # Sample parameters
    # This script assumes uniform distribution, however, other distributions
    # can be defined after this.
    params = torch.ones((totalEntries, ind['overall'][-1]+1)).uniform_(0,1)
    params = normalize_old(params, dims=-1) 
    # normalize converts the range from [0,1) to [0,1].
    
    # # Load updated default values for spins and metabolites
    T2_min, T2_max = load_default_values(config.default_values, pm=pm, quant="T2", spins=False)
    print('T2_min: {}'.format(T2_min))
    print('T2_max: {}'.format(T2_max))
    conc_min, conc_max = load_default_values(config.default_values, pm=pm, quant="Conc", spins=False)
    pm.set_parameter_constraints({"d": [1/(T2_max/1e3), 1/(T2_min/1e3)]})
    print(conc_min, conc_min.shape, len(pm._metab))
    print(pm._metab)
    print({met: [conc_min[i], conc_max[i]] for i, met in enumerate(pm._metab)})
    pm.set_parameter_constraints({met: [conc_min[i], conc_max[i]] for i, met in enumerate(pm._metab)})

    # Quantify parameters
    params = pm.quantify_params(params)
    # print(params[:,ind['metabolites']])
    
    '''
    >>>>>>>>>>
    This next section of code will need to be customized for your own implementations.
    The quantities have already been set, but can now be refined. 

    Hint: This is where the covariance matrix should be implemented if using it.
    >>>>>>>>>>
    '''

    # # All metabolite values are ratios wrt creatine or tCre. Therefore, Cr is always 1.0
    # params[:,ind['cr']].fill_(1.0)
    # if 'PCr' in config.metabolites: params[:,ind['pcr']].fill_(1.0)

    '''
    If you want to use a covariance matrix for sampling metabolite amplitudes, this is where covmat and loc 
    should be defined. 
    Use the ind variable to move the sampled parameters to the correct indices. The exact implementation will
    depend on the variables and order of variables that are included in your covariance matrix.
    '''
    # if config.use_covmat:
    #     # _, mtb_ind = pm.basis_metab
    #     # print('mtb_ind: ',mtb_ind)
    #     # covmat = torch.as_tensor(config.covmat) # 2D matrix
    #     # loc = torch.as_tensor(config.loc) # 1D matrix
    #     # mets = torch.distributions.multivariate_normal.MultivariateNormal(loc=loc,
    #     #                                                                   covariance_matrix=covmat)
    #     # start, stop = mtb_ind[0], mtb_ind[-1]
    #     # params[:,start:stop+1] = mets.rsample([params.shape[0]])
    #     _, mtb_ind = pm.basis_metab
    #     print('mtb_ind: ',mtb_ind)
    #     covmat = torch.as_tensor(config.covmat) # 2D matrix
    #     loc = torch.as_tensor(config.loc) # 1D matrix
    #     mets = torch.distributions.multivariate_normal.MultivariateNormal(loc=loc,
    #                                                                       covariance_matrix=covmat)
    #     start, stop = mtb_ind[0], mtb_ind[-1]
    #     temp = mets.rsample([params.shape[0]])        
    #     params[:,start:stop+1] = torch.cat(temp[...,0], torch.ones_like(temp[...,0]), temp[...,-1], dim=-1)
    # print(params.shape)

    '''
    >>>>>>>>>>
    The next section of code is used to drop some parameters from each spectrum 
    for deep learning applications. Should you want to use different 
    distributions for some of the parameters, the following can be used as a 
    guide. Defining different distributions can be done before OR after 
    quantifying the parameters.
    >>>>>>>>>>
    '''
    
    print('>>> Line Broadening')
    keys, g = ind.keys(), 0
    for k in keys: g += 1 if 'mm' in k else 0
    for k in keys: g += 1 if 'lip' in k else 0

    
    '''
    load json with stored default values
    mn, mx = load_spin_values(path, pm)
    
    def load_default_values(path: str, pm: nn.Module, spins: bool=False):
        key = "spins" if spins else "metab"
        with open(path) as file:
            values = json.load(file)
        
        num_spins = pm._num_spins
        end = len(pm.spins) - 1
        
        minimum = torch.zeros(end)
        maximum = torch.zeros(end)
    
        metabs, _ = pm.metab
    
        for i, ind in enumerate(range(0,end,num_spins)):
            met = values[metabs[i]]["T2"][key]
            for n in range(0,len(met["min"])):
                minimum[:,ind+n] = met["min"][n]
                maximum[:,ind+n] = met["max"][n]
        return minimum, maximum
    
    T2_min, T2_max = load_default_values(config.default_values, pm=pm, quant="T2", spins=True)
    '''
    
    # Drop D from some metabolites
    if config.lineshape in ['voigt','lorentzian']:
        for n in ind['d']:
            sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
            params[sign,n].fill_(0.)
    else:
        for n in ind['d']: params[:,n].fill_(0.0)
        
    # Drop G from some spectra by group
    if config.lineshape in ['voigt','gaussian']:
        groups = [0]
        if g!=0: groups.append(int(l-g-1))
        for n in groups:
            sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
            params[sign,n].fill_(0.)
    else:
        for n in ind['g']: params[:,n].fill_(0.0)

    # One Gaussian value is used for metabolites and the other is used for 
    # MM/Lip - but only 2 values! Should an additional group be separated, this 
    # and the pm.initialize() code will need to be updated.
    # deltaG_min, deltaG_max = torch.from_numpy(np.sqrt(20)), torch.from_numpy(np.sqrt(100))
    deltaG_min, deltaG_max = torch.tensor(20), torch.tensor(100)
    if not config.b0:
        for n in ind['g']:
            jitter = torch.randint(low=deltaG_min, high=deltaG_max, size=(1,)).sqrt().squeeze() * torch.bernoulli(torch.tensor(p))
            if n>0 and n<l-g-1:
                params[:,n] = params[:,ind['g'][0]].clone() + jitter
            if n>l-g-1:
                params[:,n] = params[:,ind['g'][int(l-g-1)]].clone() + torch.randint(low=deltaG_min, high=deltaG_max, size=(1,)).sqrt().squeeze() * torch.bernoulli(torch.tensor(p))
    else:
        # The B0 field distortions are modeled instead of using a Gaussian term for the metabolties
        for n in ind['g']:
            if n>0 and n<l-g-1:
                params[:,n].fill_(0.)
            if n>l-g-1:
                jitter = torch.randint(low=deltaG_min, high=deltaG_max, size=(1,)).sqrt().squeeze() * torch.bernoulli(torch.tensor(p))
                params[:,n] = params[:,ind['g'][int(l-g-1)]].clone() + jitter

    
    # Randomly drop some metabolites and their line broadening and fshifts
    for _, n in enumerate(ind['metabolites']):
        if not n==ind['cr']: # Creatine
            sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
            params[sign,int(n)].fill_(0.) # amplitude
            params[sign,int(n+l)].fill_(0.) # If the lines are omitted, then the broadening is too
            params[sign,int(n+2*l)].fill_(0.) # gaussian
            params[sign,int(n+3*l+1)].fill_(0.) # fshift


    # Fully omit the macromolecular baseline
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    for n in ind['mac']:
        params[sign,n].fill_(0) # amplitude
        params[sign,n+l].fill_(0) # lorentzian
        params[sign,n+2*l].fill_(0) # gaussian
        params[sign,n+3*l+1].fill_(0) # fshift

    # Fully omit the lipid signal
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    for n in ind['lip']:
        params[sign,n].fill_(0) # amplitude
        params[sign,n+l].fill_(0) # lorentzian
        params[sign,n+2*l].fill_(0) # gaussian
        params[sign,n+3*l+1].fill_(0) # fshift

    # Fully omit both lipid and macromolecular signal
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    for n in ind['mac']:
        params[sign,n].fill_(0) # amplitude
        params[sign,n+l].fill_(0) # lorentzian
        params[sign,n+2*l].fill_(0) # gaussian
        params[sign,n+3*l+1].fill_(0) # fshift
    for n in ind['lip']:
        params[sign,n].fill_(0) # amplitude
        params[sign,n+l].fill_(0) # lorentzian
        params[sign,n+2*l].fill_(0) # gaussian
        params[sign,n+3*l+1].fill_(0) # fshift

    # Frequency Shift
    print('>>> Frequency Shift')
    # Drop the global frequency shift
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    params[sign,ind['f_shift']] = 0.0
    if 'f_shifts' in ind.keys():
        for i, n in enumerate(ind['f_shifts']):
            sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
            params[sign,n] = 0.0 # Hz
    

    # Noise
    print('>>> Noise')
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    params[:,ind['snr']].uniform_(config.snr[0],config.snr[1]) # dB
    params[sign,ind['snr']].fill_(100.) 
    # Noiseless data is assumed to have SNR=100dB - set arbitrarily


    # Phase Shift
    print('>>> Phase Shift')
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    params[sign,ind['phi0']].fill_(0.0)
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    params[sign,ind['phi1']].fill_(0.0)


    # Eddy Currents
    print('>>> Eddy Currents')
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    for i in ind['ecc']: params[sign,i].fill_(0.0)


    # B0 Inhomogeneities
    print('>>> B0 Inhomogeneities')
    if 'b0' in ind.keys():
        sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
        params[sign,ind['b0']].fill_(0.0)
        for i, n in enumerate(ind['b0_dir']):
            sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
            params[sign,n].fill_(0.5) # 1 Hz minimum


    if config.num_coils>1:
        print('>>> Transients')
        sign = torch.tensor([True if torch.rand([params[:,ind['coil_snr']].shape]) > p else False for _ in range(params.shape[0])])
        factors = torch.distributions.normal.Normal(1,0.25).sample(params[:,ind['coil_snr']].shape)
        factors[sign,ind['coil_snr']].fill_(1.0)
        params[:,ind['coil_snr']] = factors
        # Values are sampled from a Gaussian mu=1, min/max=~0/2
        # The linear SNR is calculated and scaled based on the number of transients
        # Then the linear SNR is scaled about 1.0 so mu = lin_snr
        
        if config.coil_sens:
            # drop_prob does not affect this parameter
            print('>>> Coil Sensitivities')
            params[:,ind['coil_sens']] = torch.distributions.normal.Normal(1,0.5).sample(params[:,ind['coil_sens']].shape).clamp(min=0.0,max=2.0)
        
        if config.coil_fshift:
            print('>>> Coil Frequency Drift')
            sign = torch.tensor([True if torch.rand([params[:,ind['coil_fshift']].shape]) > p else False for _ in range(params.shape[0])])
            factors = torch.distributions.normal.Normal(1,0.25).sample(params[:,ind['coil_fshift']].shape)
            factors[sign,ind['coil_fshift']].fill_(1.0)
            params[:,ind['coil_fshift']] = factors * params[:,ind['coil_fshift']][0]
        
        if config.coil_phi0:
            print('>>> Coil Phase Drift')
            sign = torch.tensor([True if torch.rand([params[:,ind['coil_phi0']].shape]) > p else False for _ in range(params.shape[0])])
            factors = torch.distributions.normal.Normal(1,0.25).sample(params[:,ind['coil_phi0']].shape)
            factors[sign,ind['coil_phi0']].fill_(1.0)
            params[:,ind['coil_phi0']] = factors * params[:,ind['coil_phi0']][0]


    
    '''
    >>>>>>>>>>
    If certain parts of the model are turned off, then their values should be zeroed out.
    >>>>>>>>>>
    '''
    if not config.b0:
        params[:,ind['b0']].fill_(0.0)
        for n in ind['b0_dir']: params[:,n].fill_(0.0)
    # Coil_sens is dealt with above
    # D is dealt with above
    if not config.eddy: 
        for n in ind['ecc']: params[:,n].fill_(0.0)
    if not config.fshift_g: params[:,ind['f_shift']].fill_(0.0)
    if not config.fshift_i:
        for n in ind['f_shifts']: 
            params[:,n].fill_(0.0)
    # G is dealt with above
    if not config.noise: params[:,ind['snr']].fill_(0.0)
    if not config.phi0: params[:,ind['phi0']].fill_(0.0)
    if not config.phi1: params[:,ind['phi1']].fill_(0.0)
    # Multi_coil is dealt with above

    return config, resWater_cfg, baseline_cfg, pm, l, ind, p, totalEntries, params



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default='./dataset/')
    parser.add_argument('--batchSize', type=int, default=10000)
    parser.add_argument('--stepSize', type=int, default=10000)
    parser.add_argument('--parameters', type=str, default=None, help='Path to .mat file with pre-sampled parameters')
    parser.add_argument('--config_file', type=str, default='./src/configurations/debug_new_init.json')

    args = parser.parse_args()
    
    os.makedirs(args.savedir, exist_ok=True)

    # Simulate
    if isinstance(args.parameters, str):
        from aux import load_parameters
        sampled = load_parameters(args.parameters, prepare(args.config_file))
    else:
        sampled = sample(prepare(args.config_file))

    simulate(sampled,args=args)
   
