import argparse
import json
import os
import sys

import numpy as np
import scipy.io as io
import torch

from aux import concat_dict, normalize, sort_parameters, torch2numpy
from pm_v2 import PhysicsModel
from types import SimpleNamespace


sys.path.append('../')


def prepare(sonfig_file):
    # Load the config file
    with open(config_file) as file:
        config = json.load(file)

    confg_kys = config.keys()
    parameters = config["parameters"] if "parameters" in confg_kys else None
    resWater_cfg = config["resWater_cfg"] if "resWater_cfg" in confg_kys else None
    baseline_cfg = config["baseline_cfg"] if "baseline_cfg" in confg_kys else None

    config = SimpleNamespace(**config)
    p = 1 - config.drop_prob #0.8 # probability of including a variable for a given data sample
    totalEntries = config.totalEntries

    # Define and initialize the physics model
    pm = PhysicsModel(PM_basis_set=config.PM_basis_set)
    pm.initialize(metab=config.metabolites, 
                  basisFcn_len=config.basis_fcn_length,
                  b0=config.b0,
                  coil_fshift=config.coil_fshift,
                  coil_phi0=config.coil_phi0,
                  coil_sens=config.coil_sens,
                  cropRange=config.cropRange,
                  difference_editing=False,
                  eddycurrents=config.eddy,
                  fshift_i=config.fshift_i,
                  image_resolution=config.image_resolution,
                  length=config.spectrum_length,
                  lineshape=config.lineshape,
                  num_coils=config.num_coils,
                  ppm_ref=config.ppm_ref,
                  spectral_resolution=config.spectral_resolution,
                  spectralwidth=config.spectralwidth)

    if parameters: pm.set_parameter_constraints(parameters)
    ind = pm.index
    l = len(ind['metabolites'])

    
    print('>>> Metabolite Quantities')
    for k, v in ind.items():
        if isinstance(v, tuple):
            ind[k] = list(ind[k])
            for i, val in enumerate(v): ind[k][i] = int(val)
            ind[k] = tuple(ind[k])
        else:
            ind[k] = int(v)

    ind['mac'] = tuple([v for k, v in ind.items() if 'mm' in k]) 
    ind['lip'] = tuple([v for k, v in ind.items() if 'lip' in k]) 

    return config, resWater_cfg, baseline_cfg, pm, l, ind, p, totalEntries    



def sample(config, resWater_cfg, baseline_cfg, pm, l, ind, p, totalEntries):
    # Sample parameters
    params = torch.zeros((totalEntries, ind['overall'][-1]+1)).uniform_(0,1)
    params = normalize(params, dims=-1) 
    # normalize converts the range from [0,1) to [0,1].


    # Quantify parameters
    params = pm.quantify_params(params)
    
    '''
    This next section of code will need to be customized for your own implementations.
    '''

    # All metabolite values are ratios wrt creatine. Therefore, Cr is always 1.0
    params[:,ind['cr']].fill_(1.0)
    if 'pcr' in config.metabolites:
        params[:,ind['cr']].fill_(0.5)
        params[:,ind['pcr']].fill_(0.5)

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

    # One Gaussian value is used for metabolites and the other is used for MM/Lip - but only 2 values!
    # Should an additional group be separated, this and the pm.initialize() code will need to be updated.
    for n in ind['g']:
        if n>0 and n<l-g-1:
            params[:,n] = params[:,ind['g'][0]].clone()
        if n>l-g-1:
            params[:,n] = params[:,ind['g'][int(l-g-1)]].clone()

    
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
    for i, n in enumerate(ind['f_shifts']):
        sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
        params[sign,n] = 0.0 # Hz
    
    # Noise
    print('>>> Noise')
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    params[:,ind['snr']].uniform_(config.snr[0],config.snr[1]) # dB
    params[sign,ind['snr']].fill_(100.) # Noiseless data is assumed to have SNR=100dB

    # Phase Shift
    print('>>> Phase Shift')
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    params[sign,ind['phi0']].fill_(0.0)
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    params[sign,ind['phi1']].fill_(0.0)

    # B0 Inhomogeneities
    print('>>> B0 Inhomogeneities')
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    params[sign,ind['b0']].fill_(0.0)
    for i, n in enumerate(ind['b0_dir']):
        sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
        params[sign,n].fill_(0.5) # 1 Hz minimum

    if config.num_coils>1:
        # drop_prob does not affect these parameters
        print('>>> Transients')
        params[:,ind['multi_coil']] = torch.distributions.normal.Normal(1,0.25).sample(params[:,ind['multi_coil']].shape)
        # Values are sampled from a Gaussian mu=1, min/max=0/2
        # The linear SNR is calculated and scaled based on the number of transients
        # Then the linear SNR is scaled about 1.0 so mu = lin_snr
        if config.coil_sens:
            print('>>> Coil Sensitivities')
            params[:,ind['coil_sens']] = torch.distributions.normal.Normal(1,0.25).sample(params[:,ind['coil_sens']].shape)
        if config.coil_fshift:
            print('>>> Coil Frequency Drift')
            params[:,ind['coil_fshift']] = torch.distributions.normal.Normal(0,0.25).sample(params[:,ind['coil_fshift']].shape) + params[:,ind['coil_fshift']][0]
        if config.coil_phi0:
            print('>>> Coil Phase Drift')
            params[:,ind['coil_phi0']] = torch.distributions.normal.Normal(0,0.25).sample(params[:,ind['coil_phi0']].shape) + params[:,ind['coil_phi0']][0]


    '''
    If certain parts of the model are turned off, then their values should be zeroed out.
    '''
    if not config.b0:
        params[:,ind['b0']].fill_(0.0)
        for n in ind['b0_dir']: params[:,n].fill_(0.0)
    # Coil_sens is dealt with above
    # D is dealt with above
    if not config.fshift_g: params[:,ind['f_shift']].fill_(0.0)
    if not config.fshift_i:
        for n in ind['f_shifts']: params[:,n].fill_(0.0)
    # G is dealt with above
    if not config.noise: params[:,ind['snr']].fill_(0.0)
    if not config.phi0: params[:,ind['phi0']].fill_(0.0)
    if not config.phi1: params[:,ind['phi1']].fill_(0.0)
    # Multi_coil is dealt with above

    return config, resWater_cfg, baseline_cfg, pm, l, ind, p, totalEntries, params
    
def simulate(config, resWater_cfg, baseline_cfg, pm, l, ind, p, totalEntries, params, args=None):
    '''
    Begin simulating and saving the spectra
    '''
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
                             eddy=config.eddy,
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
                             coil_phi0=config.coil_phi0,
                             coil_sens=config.coil_sens,
                             magnitude=config.magnitude,
                             multicoil=config.num_coils,
                             snr_combo=config.snr_combo,
                             wrt_metab=config.wrt_metab,
                             zero_fill=config.zero_fill,
                             broadening=config.broadening,
                             coil_fshift=config.coil_fshift,
                             residual_water=sample_resWater(n-i, **resWater_cfg) if resWater_cfg else False,
                             drop_prob=config.drop_prob)
        ppm = pm.ppm.numpy()

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
                  ppm=ppm)
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
                  ppm=ppm)
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

    args = parser.parse_args()

    # torch.distributed.init_process_group(backend='gloo', world_size=4)
    
    os.makedirs(args.savedir, exist_ok=True)

    # Simulate
    simulate(sample(prepare(config_file)),args=args)
   
