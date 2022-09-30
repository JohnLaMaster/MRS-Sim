import argparse
import copy
import json
import os
import sys

import numpy as np
import scipy.io as io
import torch
from aux import *
from pm_v2 import PhysicsModel
from types import SimpleNamespace

sys.path.append('../')


def simulate(config_file, args=None):
    with open(config_file) as file:
        config = json.load(file)

    confg_kys = config.keys()
    parameters = config["parameters"] if "parameters" in confg_kys else None
    resWater_cfg = config["resWater_cfg"] if "resWater_cfg" in confg_kys else None
    baseline_cfg = config["baseline_cfg"] if "baseline_cfg" in confg_kys else None


    config = SimpleNamespace(**config)
    p = 1 - config.drop_prob #0.8 # probability of including a variable for a given data sample
    totalEntries = config.totalEntries

    pm = PhysicsModel(PM_basis_set=config.PM_basis_set)
    pm.initialize(metab=config.metabolites, 
                  cropRange=config.cropRange,
                  length=config.spectrum_length,
                  ppm_ref=config.ppm_ref,
                  transients=config.transients,
                  spectral_resolution=config.spectral_resolution,
                  image_resolution=config.image_resolution,
                  lineshape=config.lineshape)

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

    # Sample parameters
    params = torch.zeros((totalEntries, ind['overall'][-1]+1)).uniform_(0,1+1e-6)
    params = params.clamp(0,1)

    # Quantify parameters
    params = pm.quantify_params(params)
    
    # Randomly drop some metabolites
    for n in ind['metabolites']:
        sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
        params[sign,n].fill_(0.)

    # All metabolite values are ratios wrt creatine. Therefore, Cr is always 1.0
    params[:,ind['cr']].fill_(1.0)

    '''
    If you want to use a covariance matrix for sampling metabolite amplitudes, this is where covmat and loc 
    should be defined. 
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

    print('>>> Line Broadening')
    keys, g = ind.keys(), 0
    for k in keys: g += 1 if 'mm' in k else 0
    for k in keys: g += 1 if 'lip' in k else 0

    # Drop D from some metabolites
    for n in ind['d']:
        sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
        params[sign,n].fill_(0.)
        
    # Drop G from some spectra
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    for n in ind['g']:
        params[sign,n].fill_(0.)  
        

    # One Gaussian value is used for metabolites and the other is used for MM/Lip - but only 2 values!  
    for n in ind['g']:
        if n>0 and n<l-g:
            params[:,n] = params[:,ind['g'][0]].clone()
        if n>l-g:
            params[:,n] = params[:,ind['g'][int(l-g)]].clone()


    # Zero out metabolites and their line broadening
    for n in range(l):
        if not n==ind['cr']: # Creatine
            sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
            params[sign,int(n)].fill_(0.)
            params[sign,int(n+l)].fill_(0.) # If the lines are omitted, then the broadening is too
            params[sign,int(n+2*l)].fill_(0.)

    # Fully omit the macromolecular baseline
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    for n in ind['mac']:
        params[sign,n].fill_(0)
        params[sign,n+l].fill_(0)

    # Fully omit the lipid signal
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    for n in ind['lip']:
        params[sign,n].fill_(0)
        params[sign,n+l].fill_(0)

    # Fully omit both lipid and macromolecular signal
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    for n in ind['mac']:
        params[sign,n].fill_(0)
        params[sign,n+l].fill_(0)
        params[sign,n+2*l].fill_(0)
    for n in ind['lip']:
        params[sign,n].fill_(0)
        params[sign,n+l].fill_(0)
        params[sign,n+2*l].fill_(0)

    # Frequency Shift
    print('>>> Frequency Shift')
    for i, n in enumerate(ind['f_shift']):
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

    if config.transients>1:
        # drop_prob does not affect these parameters
        print('>>> Transients')
        params[:,ind['transients']] = torch.distributions.normal.Normal(1,0.25).sample(params[:,ind['transients']].shape)
        # Values are sampled from a Gaussian mu=1, min/max=0/2
        # The linear SNR is calculated and scaled based on the number of transients
        # Then the linear SNR is scaled about 1.0 so mu = lin_snr

        print('>>> Coil Sensitivities')
        params[:,ind['coil_sens']] = torch.distributions.normal.Normal(1,0.25).sample(params[:,ind['coil_sens']].shape)



    first = True
    step = 10000
    threshold = args.batchSize
    path = args.savedir + '/dataset_spectra'
    counter = 0
    for i in range(0,config.totalEntries,step):
        n = i+step if i+step<=params.shape[0] else i+(params.shape[0])
        outputs = pm.forward(params=params[i:n,:], 
                             b0=config.b0,
                             fid=config.fid,
                             gen=True, 
                             eddy=False,
                             phi0=config.phi0,
                             phi1=config.phi1,
                             noise=config.noise, 
                             fshift=config.fshift,
                             apodize=config.apodize,
                             offsets=True if baseline_cfg or resWater_cfg else False,
                             resample=config.resample,
                             snr_both=config.snr_both,
                             baselines=sample_baselines(n-i, **baseline_cfg) if baseline_cfg else False,
                             magnitude=config.magnitude,
                             zero_fill=config.zero_fill,
                             broadening=config.broadening,
                             transients=config.transients,
                             residual_water=sample_resWater(n-i, **resWater_cfg) if resWater_cfg else False,
                             drop_prob=config.drop_prob)
        ppm = pm.get_ppm(cropped=True)

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
                  cropRange=config.cropRange, 
                  ppm=pm.get_ppm(cropped=True).numpy())
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
                  cropRange=config.cropRange, 
                  ppm=pm.get_ppm(cropped=True).numpy())
        del spectra, fit, baseline, reswater, parameters, quantities


def torch2numpy(input: dict):
    for k, v in input.items():
        if isinstance(input[k], dict):
            torch2numpy(input[k])
        elif torch.is_tensor(v):
            input[k] = v.numpy()
        else:
            pass
    return input


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
