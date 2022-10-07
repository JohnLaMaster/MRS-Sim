import argparse
import json
import os
import sys

import numpy as np
import scipy.io as io
import torch

from aux import concat_dict, sort_parameters, torch2numpy
from pm_v2 import PhysicsModel
from types import SimpleNamespace


sys.path.append('../')



def simulate(config_file, args=None):
    print('config_file: ',config_file)
    with open(config_file) as file:
        config = json.load(file)


    confg_kys = config.keys()
    parameters = config["parameters"] if "parameters" in confg_kys else None
    resWater_cfg = config["resWater_cfg"] if "resWater_cfg" in confg_kys else None
    baseline_cfg = config["baseline_cfg"] if "baseline_cfg" in confg_kys else None


    config = SimpleNamespace(**config)
    totalEntries = config.totalEntries


    # Define and initialize the physics model
    pm = PhysicsModel(PM_basis_set=config.PM_basis_set)
    pm.initialize(metab=config.metabolites, 
                  cropRange=config.cropRange,
                  length=config.spectrum_length,
                  ppm_ref=config.ppm_ref,
                  num_coils=config.num_coils,
                  coil_sens=config.coil_sens,
                  spectral_resolution=config.spectral_resolution,
                  image_resolution=config.image_resolution,
                  lineshape=config.lineshape,
                  fshift_i=config.fshift_i,
                  spectralwidth=config.spectralwidth,
                  basisFcn_len=config.basis_fcn_length)


    if parameters: pm.set_parameter_constraints(parameters)
    ind = pm.index
    '''
    pm.index follows the template below. Please reference this when organizing
    the variables for a covariance matrix
    - Metabolites: alphabetical order with MM and then Lip in numerical order
    - Lorentzian line broadening factors - 1 per basis function
    - Gaussian line broadening factors - 1 per basis function
    - Global frequency shfit
    - Individual frequency shifts - 1 per basis function
    - SNR
    - Zero-order phase
    - First-order phase
    - B0 mean offset
    - B0_dir - 3 values - 1/2*[dx, dy, dz] across the entire voxel
    - Optional:
        - Multiple coils: 1 value per coil to scale the SNR
        - Coil sensitivities: 1 value per coil to weight the transients
    - All Metabolites + MM/Lip
    - All Artifacts (non-metabolite aplitude variables)
    - All variables
    '''
    l = len(ind['metabolites'])


    print('>>> Metabolite Quantities')
    for k, v in ind.items():
        if isinstance(v, tuple):
            ind[k] = list(ind[k])
            for i, val in enumerate(v): ind[k][i] = int(val)
            ind[k] = tuple(ind[k])
        else:
            ind[k] = int(v)

    ind['mac'] = tuple([v for k, v in ind.items() if 'mm' in k]) # (6,7,8,9,10)
    ind['lip'] = tuple([v for k, v in ind.items() if 'lip' in k]) # (11,12,13)


    # Sample parameters
    params = torch.zeros((totalEntries, ind['overall'][-1]+1)).uniform_(0,1+1e-6)
    params = params.clamp(0,1)
    # Adding eps and then clamping converts the range from [0,1) to [0,1].


    # Grouping the variables for the Gaussian lineshape variable
    # One Gaussian value is used for metabolites and the other is used for MM/Lip - but only 2 values!  
    keys, g = ind.keys(), 0
    for k in keys: g += 1 if 'mm' in k else 0
    for k in keys: g += 1 if 'lip' in k else 0
        
    for n in ind['g']:
        if n>0 and n<l-g:
            params[:,n] = params[:,ind['g'][0]].clone()
        if n>l-g:
            params[:,n] = params[:,ind['g'][int(l-g)]].clone()


    # Quantify parameters
    params = pm.quantify_params(params)

    
    '''
    This next section of code will need to be customized for your own implementations.

    Hint: This is where the covariance matrix should be implemented.
    '''
    # All metabolite values are ratios wrt creatine. Therefore, Cr is always 1.0
    params[:,ind['cr']].fill_(1.0)

    # Noise
    params[:,ind['snr']].uniform_(config.snr[0],config.snr[1]) # dB

 
    if config.num_coils>1:
        params[:,ind['multi_coil']] = torch.distributions.normal.Normal(1,0.25).sample(params[:,ind['multi_coil']].shape)
        # Values are sampled from a Gaussian mu=1, min/max=0/2
        # The linear SNR is calculated and scaled based on the number of transients
        # Then the linear SNR is scaled about 1.0 so mu = lin_snr
        if config.coil_sens:
            params[:,ind['coil_sens']] = torch.distributions.normal.Normal(1,0.25).sample(params[:,ind['coil_sens']].shape)


    '''
    If certain parts of the model are turned off, then their values should be zeroed out.
    '''
    if not config.b0:
        params[:,ind['b0']].fill_(0.0)
        for n in ind['b0_dir']: params[:,n].fill_(0.0)
    # Coil_sens is dealt with above
    # D is dealt with above
    if not config.fshift_g: params[:,ind['f_shift']].fill_(0.0)
    # Fshift_i is dealt with in pm.initialize()
    # G is dealt with above
    if not config.noise: params[:,ind['snr']].fill_(0.0)
    if not config.phi0: params[:,ind['phi0']].fill_(0.0)
    if not config.phi1: params[:,ind['phi1']].fill_(0.0)
    # Transients are dealth with above


    first = True
    step = 10000
    threshold = args.batchSize
    path = args.savedir + '/dataset_spectra'
    counter = 0
    for i in range(0,config.totalEntries,step):
        n = i+step if i+step<=params.shape[0] else i+(params.shape[0])
        outputs = pm.forward(params=params[i:n,:], 
                             diff_edit=None,
                             b0=config.b0,
                             gen=True, 
                             eddy=False,
                             fids=config.fids,
                             phi0=config.phi0,
                             phi1=config.phi1,
                             noise=config.noise, 
                             apodize=config.apodize,
                             offsets=True if baseline_cfg or resWater_cfg else False,
                             fshift_g=config.fshift_g,
                             fshift_i=config.fshift_i,
                             resample=config.resample,
                             baselines=sample_baselines(n-i, **baseline_cfg) if baseline_cfg else False,
                             coil_sens=config.coil_sens,
                             magnitude=config.magnitude,
                             multicoil=config.num_coils,
                             snr_combo=config.snr_combo,
                             zero_fill=config.zero_fill,
                             broadening=config.broadening,
                             residual_water=sample_resWater(n-i, **resWater_cfg) if resWater_cfg else False,
                             drop_prob=config.drop_prob)
        ppm = pm.ppm

        if first:
            spectra, fit, baseline, reswater, parameters, quantities = outputs
            first = False
        else:
            spectra = np.concatenate([spectra, outputs[0]], axis=0)
            fit = np.concatenate([fit, outputs[1]], axis=0)
            baseline = np.concatenate([baseline, outputs[2]], axis=0) if baseline else None
            reswater = np.concatenate([reswater, outputs[3]], axis=0) if reswater else None
            parameters = np.concatenate([parameters, outputs[4]], axis=0)
            quantities = concat_dict(quantities, outputs[5])
        if config.totalEntries>threshold and (i+step) % threshold==0:
            _save(path=path + '_{}'.format(counter), 
                  spectra=spectra, 
                  fits=fit,
                  baselines=baseline, 
                  reswater=reswater, 
                  parameters=sort_parameters(parameters, ind), 
                  quantities=quantities, 
                  cropRange=pm.cropRange, 
                  ppm=ppm.numpy())
            first = True
            counter += 1
            print('>>> ** {} ** <<<'.format(counter))
        elif ((i+step) >= params.shape[0]) or (config.totalEntries<=threshold):
            _save(path=path + '_{}'.format(counter), 
                  spectra=spectra, 
                  fits=fit,
                  baselines=baseline, 
                  reswater=reswater, 
                  parameters=sort_parameters(parameters, ind), 
                  quantities=quantities, 
                  cropRange=pm.cropRange, 
                  ppm=ppm.numpy())
        del spectra, fit, baseline, reswater, parameters, quantities



def _save(path, spectra, fits, baselines, reswater, parameters, quantities, cropRange, ppm):
    print('>>> Saving Spectra')
    base, _ = os.path.split(path)
    os.makedirs(base, exist_ok=True)

    mdict = {'spectra': spectra,
             'spectral_fit': fits,
             'baselines': baselines if not isinstance(baselines, type(None)) else [],
             'residual_water': reswater if not isinstance(reswater, type(None)) else [],
             'params': parameters,
             'quantities': quantities,
             'cropRange': cropRange,
             'ppm': ppm,
            }
    io.savemat(path + '.mat', do_compression=True, mdict=mdict)
    print(path + '.mat')




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default='./dataset/')
    parser.add_argument('--batchSize', type=int, default=10000)
    parser.add_argument('--config_file', type=str, default='./src/configurations/debug_new_init.json')#DL_PRESS_144_ge.json')
    parser.add_argument('--PM_basis_set', type=str, default='fitting_basis_ge_press144_1.mat', help='file name of the basis set')
    parser.add_argument('--simple', action='store_true', default=False, help='Use ISMRM basis functions and a simplfied physics model')

    args = parser.parse_args()

    # torch.distributed.init_process_group(backend='gloo', world_size=4)
    
    os.makedirs(args.savedir, exist_ok=True)

    simulate(config_file=args.config_file, args=args)
