import argparse
import json
import os
import sys

import numpy as np
import scipy.io as io
import torch
from aux import normalize
from main_fcns import _save, prepare, simulate
# from pm_v3 import PhysicsModel
from types import SimpleNamespace

sys.path.append('../')


def sample(inputs):
    config, resWater_cfg, baseline_cfg, pm, l, ind, p, totalEntries = inputs

    # Sample parameters
    # print(ind)
    params = torch.ones((totalEntries, ind['overall'][-1]+1)).uniform_(0,1)
    params = normalize(params, dims=-1) 
    # normalize converts the range from [0,1) to [0,1].
    # print(params[:,ind['metabolites']])

    for i in ind['metabolites']:
        params[:,i].fill_(1.0)


    # Quantify parameters
    params = pm.quantify_params(params)
    # print(params[:,ind['metabolites']])
    
    '''
    This next section of code will need to be customized for your own implementations.
    '''

    # All metabolite values are ratios wrt creatine. Therefore, Cr is always 1.0
    denom = 2 # 1
    
    try: params[:,ind['pcr']].fill_(1.0)
    except KeyError as E: pass
    try: params[:,ind['cr']].fill_(1.0)
    except KeyError as E: pass
    if 'PCr' in config.metabolites and 'Cr' in config.metabolites:
        params[:,ind['cr']].fill_(0.525/denom)
        params[:,ind['pcr']].fill_(0.475/denom)

    # print(ind)

    names = ["Asc", "Asp", "Ch", "Cr", "GABA", "GPC", "GSH", "Gln", "Glu", "mI", "Lac", "NAA", "NAAG", "PCh", "PCr", "PE", "sI", "Tau", "MM09", "MM12", "MM14", "MM17", "MM20", "Lip09", "Lip13", "Lip20"]
    amps = [0.0804920187119920, 0, 0.1081, 0.507854651524663, 0.455613005467203, 0.221068064161334,  0.0968670765788001, 0.209273371106189, 1.45415790831963, 0.449486907354451,  0.0164778245907856, 1.45445578745539, 0.320213124374504, 0.1130, 0.492145348475337, 0, 0, 0.108357885219082, 10.314097589906174, 10.402136461787936, 10.358101078275321, 10.452705224928629, 10.551475712699752, 10.719485302705014,  10.414165486070721, 0]
    for n, v in zip(names,amps):
        params[:,ind[n.lower()]].fill_(v)
    '''
    The next section of code is used to drop some parameters from each spectrum for deep learning applications.
    Should you want to use different distributions for some of the parameters, the following can be used as a 
    guide. Defining different distributions can be done before OR after quantifying the parameters.
    '''
    
    print('>>> Line Broadening')
    keys, g = ind.keys(), 0
    for k in keys: g += 1 if 'mm' in k else 0
    for k in keys: g += 1 if 'lip' in k else 0

    # Drop D from some metabolites
    if config.lineshape in ['voigt','lorentzian']:
        print(params.shape)
        print(ind['d'])
        # metabs = ["Asc", "Asp", "Cho", "Cr", "GABA", "Glc", "Gln", "Glu", "Gly", "GPC", "GSH", "Lac", "mI", "NAA", "NAAG", "PCh", "PCr", "PE", "sI", "Tau"]
        t2 = [105, 90, 213, 128, 75, 88, 99, 122, 81, 213, 72, 99, 229, 263, 107, 213, 128, 86, 107, 102, 59, 65, 17, 70, 48] # from Wyss 2018 "In Vivo Estimation of Transverse Relaxation Time..."
        for i, n in enumerate(ind['d']):
            # print(i)
            if i<len(t2):
                params[:,n].fill_(1000/t2[i]-1)
            # params[:,n].fill_(1.0)
    else:
        for n in ind['d']: params[:,n].fill_(0.0)
        
    # One Gaussian value is used for metabolites and the other is used for MM/Lip - but only 2 values!
    # Should an additional group be separated, this and the pm.initialize() code will need to be updated.
    for n in ind['g']:
        if n>0 and n<l-g-1:
            params[:,n] = params[:,ind['g'][0]].clone()
        if n>l-g-1:
            params[:,n] = params[:,ind['g'][int(l-g-1)]].clone()

    
#     if config.num_coils>1:
      print('>>> Transients')
      factors = torch.distributions.normal.Normal(1,0.25).sample(params[:,ind['coil_snr']].shape)
      params[:,ind['coil_snr']] = factors
      # Values are sampled from a Gaussian mu=1, min/max=0/2
      # The linear SNR is calculated and scaled based on the number of transients
      # Then the linear SNR is scaled about 1.0 so mu = lin_snr
      if config.coil_sens:
          print('>>> Coil Sensitivities')
          params[:,ind['coil_sens']] = torch.distributions.normal.Normal(1,0.5).sample(params[:,ind['coil_sens']].shape).clamp(min=0.0,max=2.0)

      if config.coil_fshift:
          print('>>> Coil Frequency Drift')
          factors = torch.distributions.normal.Normal(1,0.25).sample(params[:,ind['coil_fshift']].shape)
          params[:,ind['coil_fshift']] = factors * params[:,ind['coil_fshift']][0]

      if config.coil_phi0:
          print('>>> Coil Phase Drift')
          factors = torch.distributions.normal.Normal(1,0.25).sample(params[:,ind['coil_phi0']].shape)
          params[:,ind['coil_phi0']] = factors * params[:,ind['coil_phi0']][0]

    if config.B0_samples
        # Create pairs of samples with the same parameters - B0 inhomogeneities examples
        for i in [1,3,5]:
            params[i,:] = params[int(i-1),:].clone()

            if config.b0:
                params[i,ind['b0']].fill_(0.0)
                for n in ind['b0_dir']: params[i,n].fill_(1**10-6)
            for n in ind['g']:
                params[i,n].fill_(0.0)

    if config.EC_samples:
        # Create pairs of samples with the same parameters - B0 inhomogeneities examples
        for i in range(1,6): params[i,:] = params[0,:].clone()
        for i in [1,3,5]:
            # params[i,:] = params[int(i-1),:].clone()
            for ii, n in enumerate(ind['ecc']): 
                if ii==1: 
                    params[int(i-1),n].fill_(0.001)
                    params[i,ii].fill_(i)
    
    if config.phi_samples:
        # # Comparing phased, zero-, and first-order phase
        for i in range(1,6): params[i,:] = params[0,:].clone()
        for i in [0,2,4]:
            params[i,ind['phi0']].fill_(0.0)
            params[i,ind['phi1']].fill_(0.0)
        params[1,ind['phi0']].fill_(0.0)
        params[3,ind['phi0']].fill_(45.0)
        params[5,ind['phi0']].fill_(0.0)
        params[1,ind['phi1']].fill_(0.0)
        params[3,ind['phi1']].fill_(0.0)
        params[5,ind['phi1']].fill_(-20.0)
    

    '''
    If certain parts of the model are turned off, then their values should be zeroed out.
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
#     Multi_coil is dealt with above
    if config.num_coils<=1:
        params[:,ind['coil_snr']].fill_(0.0)
        params[:,ind['coil_sens']].fill_(0.0)
        params[:,ind['coil_fshift']].fill_(0.50)
        params[:,ind['coil_phi0']].fill_(0.0)
    
    params[:,ind['snr']].fill_(15)#8.6)

    return config, resWater_cfg, baseline_cfg, pm, l, ind, p, totalEntries, params


#~/In-Vivo-MRSI-Simulator/dataset/30ms_publication/dataset_spectra_sampled_parameters.mat
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default='./dataset/30ms_publication')
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

    path = simulate(sampled,args=args)

    io.savemat(path+'_sampled_parameters.mat', mdict={'params': sampled[-1]})




"""
$ python ./src/30ms_echo_publication.py --config_file './src/config/templates/B0_samples.json' --savedir './dataset/B0_samples'

$ python ./src/30ms_echo_publication.py --config_file './src/templates/EC_samples.json' --savedir './dataset/EC_samples'

$ python ./src/30ms_echo_publication.py --config_file './src/transient_samples.json' --savedir './dataset/transient_samples'

$ python ./src/30ms_echo_publication.py --config_file './src/coil_combined_clean.json' --savedir './dataset/CC_30ms_clean'

$ python ./src/30ms_echo_publication.py --config_file './src/coil_combined_with_artifacts.json' --savedir './dataset/CC_30ms_with_artifacts'

"""
