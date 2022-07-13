import argparse
import copy
import json
import os
import sys

import numpy as np
import scipy.io as io
import torch
from aux import *
from physics_model import PhysicsModel

sys.path.append('../')
# import torch.nn as nn
# from scipy.interpolate import CubicSpline

'''
To-do:
Ger rid of all the excess stuff at the bottom. All spectra are cropped. All spectra are resampled. No need to make them optional.
Need to resample the basis spectra from 2048 to 8192 and flip them. They current run -ppm to +ppm while the basis sets run in the
opposite direction.
Confirmed that they need to be flipped. Something happened. I resamples using pchip from 2048 to 8192 and now instead of spanning
ppm [0,5], it's rather [3.5,4.9].

python generatev5.py --savedir './dataset/FinalPhase/dataset' --totalEntries 125000 --crop (1005, 1325) --resample 1024 --phase 'train'
python generate_supervised_fitting.py --savedir '/media/data_4T/john/dataset/ablation/' --totalEntries 125000 --test 10000 --crop '0.2,4.2' --resample 512
python generate_supervised_fitting.py --savedir '/media/data_4T/john/dataset/ablation/set2' --totalEntries 125000 --test 10000 --crop '0.2,4.2' --resample 512
# Decreased baseline!
python generate_supervised_fitting.py --savedir '/media/data_4T/john/dataset/ablation4' --totalEntries 125000 --test 10000 --crop '0.2,4.2' --resample 512 --metabolites 'PCh,Cr,NAA,Glx,Ins' --metab_as_is
python generate_supervised_fitting.py --savedir '/media/data_4T/john/dataset/ablation5' --totalEntries 125000 --test 10000 --crop '0.2,4.2' --resample 512 --metabolites 'PCh,Cr,NAA' --metab_as_is
'''

def simulate(config_file, args=None):
    with open(config_file) as file:
        config = json.load(file)

    parameters = config["parameters"]
    resWater_cfg = config["resWater_cfg"]
    baseline_cfg = config["baseline_cfg"]

    config = SimpleNamespace(**config)
    p = 1-config.drop_prob #0.8 # probability of including a variable for a given data sample
    totalEntries = config.totalEntries

    pm = PhysicsModel(PM_basis_set=config.PM_basis_set)
    pm.initialize(metab=config.metabolites, 
                  cropRange=config.cropRange,
                  length=config.spectrum_length,
                  ppm_ref=config.ppm_ref,
                  spectral_resolution=config.spectral_resolution,
                  image_resolution=config.image_resolution)
    print(parameters)
    if parameters: pm.set_parameter_constraints(parameters)
    ind = pm.index
    l = ind['metabolites'][-1] + 1


    print('>>> Metabolite Quantities')
    ind = pm.index
    for k, v in ind.items():
        if isinstance(v, tuple):
            ind[k] = list(ind[k])
            for i, val in enumerate(v): ind[k][i] = int(val)
            ind[k] = tuple(ind[k])
        else:
            ind[k] = int(v)
    ind['cre'] = tuple([v for k, v in ind.items() if 'cr' in k]) # (1,2,3)
    ind['naa'] = tuple([v for k, v in ind.items() if 'naa' in k]) # (4,5)
    ind['mac'] = tuple([v for k, v in ind.items() if 'mm' in k]) # (6,7,8,9,10)
    ind['lip'] = tuple([v for k, v in ind.items() if 'lip' in k]) # (11,12,13)


    params = torch.zeros((totalEntries, ind['overall'][-1]+1)).uniform_(0,1+1e-6)
    params = params.clamp(0,1)
#     print(params.shape)

    # All metabolite values are ratios wrt creatine. Therefore, Cr is always 1.0
    params[:,ind['cr']].fill_(1.0)
    '''
    If you want to use a covariance matrix for sampling metabolite amplitudes, this is where covmat and loc 
    should be defined. 
    '''
    if config.use_covmat:
        _, mtb_ind = pm.basis_metab
        print('mtb_ind: ',mtb_ind)
        covmat = torch.as_tensor(config.covmat) # 2D matrix
        loc = torch.as_tensor(config.loc) # 1D matrix
        mets = torch.distributions.multivariate_normal.MultivariateNormal(loc=loc,
                                                                          covariance_matrix=covmat)
        start, stop = mtb_ind[0], mtb_ind[-1]
        params[:,start:stop+1] = mets.rsample([params.shape[0]])
#     print(params.shape)

    print('>>> Line Broadening')
    # if not args.no_omit:
    keys, g = ind.keys(), 0
    for k in keys: g += 1 if 'mm' in k else 0
    for k in keys: g += 1 if 'lip' in k else 0

    for n in ind['d']:
        if n<l-g: params[:,n].div(10)
        sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
        params[sign,n].fill_(1.)
        
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    params[sign,ind['g'][0]].fill_(0.)
    if g>0:
        sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
        params[sign,ind['g'][l-g]].fill_(0.)    
        
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
            params[sign,int(n+l)].fill_(1.) # If the lines are omitted, then the broadening is too
            params[sign,int(n+2*l)].fill_(1.)

    # Fully omit the macromolecular baseline
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    for n in ind['mac']:
        params[sign,n].fill_(0)
        params[sign,n+l].fill_(1)

    # Fully omit the lipid signal
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    for n in ind['lip']:
        params[sign,n].fill_(0)
        params[sign,n+l].fill_(1)

    # Fully omit lipid and macromolecular signal
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    for n in ind['mac']:
        params[sign,n].fill_(0)
        params[sign,n+l].fill_(1)
        params[sign,n+2*l].fill_(1)
    for n in ind['lip']:
        params[sign,n].fill_(0)
        params[sign,n+l].fill_(1)
        params[sign,n+2*l].fill_(1)

    # Frequency Shift
    print('>>> Frequency Shift')
    base_shift = torch.ones([totalEntries, 1]).uniform_(0.1,0.9+1e-8)
    for i, n in enumerate(ind['f_shift']):
#         if len(in
        # MM/Lip fshift value must be larger (and +) than water shift
        if len(ind['f_shift'])==2 and i==1:
            params[:,n] = params[:,n] / 6 + 0.5 # Only uses [0, 1/2*max_range]
        sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
        params[sign,n] = 0.5
    
    # Scaling
    # This range needs to be symmetric, therefore zero scaling occurs at 0.5
    print('>>> Scale')
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    # params[:,ind['scale']].uniform_(0.1,0.1) # [-0.1,0.1]
    params[sign,ind['scale']].fill_(0.5)

    # Noise
    print('>>> Noise')
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    params[:,ind['snr']].uniform_(config.snr[0],config.snr[1]) # dB
    params[sign,ind['snr']].fill_(1.0)

    # Phase Shift
    # These ranges need to be symmetric, therefore zero phase occurs at 0.5
    print('>>> Phase Shift')
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    params[sign,ind['phi0']].fill_(0.5)
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    params[sign,ind['phi1']].fill_(0.5)

    # B0 Inhomogeneities
    # This range is asymmetric [-100, 200]. Therefore, the zero-point is at 1/3
    print('>>> B0 Inhomogeneities')
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    for i, n in enumerate(ind['b0']):
        if i==0: 
            params[sign,n].fill_(1/3)
        else: 
            params[:,n].uniform_(0, 10)
            params[sign,n].fill_(0)
            

    first = True
    step = 10000
    threshold = args.batchSize
    path = args.savedir + '/dataset_spectra'
    counter = 0
    for i in range(0,config.totalEntries,step):
        n = 1+step if 1+step<=params.shape[0] else params.shape[0]-i
        print('step: ',n)
        outputs = pm.forward(params=params[i:n,:], 
                             b0=config.b0,
                             gen=True, 
                             phi0=config.phi0,
                             phi1=config.phi1,
                             noise=config.noise, 
                             scale=config.scale,
                             fshift=config.fshift,
                             apodize=config.apodize,
                             offsets=True if baseline_cfg or resWater_cfg else False,
                             baselines=sample_baselines(n, **baseline_cfg),
                             magnitude=config.magnitude,
                             broadening=config.broadening,
                             transients=config.transients,
                             residual_water=sample_resWater(n, **resWater_cfg),
                             drop_prob=config.drop_prob)
        ppm = pm.get_ppm(cropped=True)
        print(type(ppm), ppm.min(), ppm.max())

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


def _save(path, spectra, fits, baselines, reswater, parameters, quantities, cropRange, ppm):
    print('>>> Saving Spectra')
    base, _ = os.path.split(path)
    os.makedirs(base, exist_ok=True)

    mdict = {'spectra': spectra,
             'spectral_fit': fits,
             'baselines': baselines if not isinstance(baselines, type(None)) else [],
             'residual_water': reswater if isinstance(reswater, type(None)) else [],
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
    parser.add_argument('--config_file', type=str, default='./src/configurations/DL_PRESS_144_ge.json')
    args = parser.parse_args()
    
    os.makedirs(args.savedir, exist_ok=True)
    
    simulate(config_file=args.config_file, args=args)




# def simulate(totalEntries=100000, crop=False, resample=False, args=None):
#     with open(args.config_file) as file:
#         config = json.load(file)

#     try:
#         parameters = config['parameters'], config.pop('parameters')
#     except Exception as e:
#         parameters = None

#     try:
#         resWater_cfg = config['resWater_cfg'], config.pop('resWater_cfg')
#     except Exception as e:
#         resWater_cfg = None

#     try:
#         baseline_cfg = config['baseline_cfg'], config.pop('baseline_cfg')
#     except Exception as e:
#         baseline_cfg = None

#     config = SimpleNamespace(config)
#     p = config.drop_prob #0.8 # probability of including a variable for a given data sample

#     pm = PhysicsModel(PM_basis_set=arg.PM_basis_set)
#     pm.initialize(metab=config.metabolites, 
#                   cropRange=config.cropRange,
#                   length=config.length,
#                   ppm_ref=config.ppm_ref)
#     if parameters: pm.set_parameter_constraints(config['parameters'])
#     ind = pm.index
#     l = ind['metabolites'][-1] + 1


#     print('>>> Metabolite Quantities')
#     ind = pm.index
#     for k, v in ind.items():
#         if isinstance(v, tuple):
#             ind[k] = list(ind[k])
#             for i, val in enumerate(v): ind[k][i] = int(val)
#             ind[k] = tuple(ind[k])
#         else:
#             ind[k] = int(v)
#     ind['cre'] = tuple([v for k, v in ind.items() if 'cr' in k]) # (1,2,3)
#     ind['naa'] = tuple([v for k, v in ind.items() if 'naa' in k]) # (4,5)
#     ind['mac'] = tuple([v for k, v in ind.items() if 'mm' in k]) # (6,7,8,9,10)
#     ind['lip'] = tuple([v for k, v in ind.items() if 'lip' in k]) # (11,12,13)


#     params = torch.zeros((totalEntries, l)).normal_(0,1)
#     min_val = params.amin(dim=(0,1), keepdims=True)
#     params = (params - min_val) / (params.amax(dim=(0,1), keepdims=True) - min_val)
#     # params = params.clamp(0,1)

#     # All metabolite values are ratios wrt creatine. Therefore, Cr is always 1.0
#     params[:,ind['cr']].fill_(1.0)

#     '''
#     If you want to use a covariance matrix for sampling metabolite amplitudes, this is where covmat and loc 
#     should be defined. 
#     '''
#     if args.use_covmat:
#         covmat = torch.as_tensor(config.covmat) # 2D matrix
#         loc = torch.as_tensor(config.loc) # 1D matrix
#         mets = torch.distributions.multivariate_normal.MultivariateNormal(loc=loc,
#                                                                           covariance_matrix=covmat)
#         start, stop = ind['metabolites'][0], ind['metabolites'][-1]
#         params[:,start:stop] = mets.rsample(params[:,start:stop].shape)


#     print('>>> Line Broadening')
#     # if not args.no_omit:
#     keys, g = ind.keys(), 0
#     for k in keys: g += 1 if 'mm' in k else 0
#     for k in keys: g += 1 if 'lip' in k else 0

#     for n in ind['d']:
#         if n<l-g: params[:,n].div(10)
#         sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
#         params[sign,n].fill_(1.)
        
#     sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
#     params[sign,ind['g'][0]].fill_(0.)
#     if g>0:
#         sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
#         params[sign,ind['g'][l-g]].fill_(0.)    
        
#     for n in ind['g']:
#         if n>0 and n<l-g:
#             params[:,n] = params[:,ind['g'][0]].clone()
#         if n>l-g:
#             params[:,n] = params[:,ind['g'][int(l-g)]].clone()
            

#     # Zero out metabolites and their line broadening
#     for n in range(l):
#         if not n==ind['cr']: # Creatine
#             sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
#             params[sign,int(n)].fill_(0.)
#             params[sign,int(n+l)].fill_(1.) # If the lines are omitted, then the broadening is too
#             params[sign,int(n+2*l)].fill_(1.)

#     # Fully omit the macromolecular baseline
#     sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
#     for n in ind['mac']:
#         params[sign,n].fill_(0)
#         params[sign,n+l].fill_(1)

#     # Fully omit the lipid signal
#     sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
#     for n in ind['lip']:
#         params[sign,n].fill_(0)
#         params[sign,n+l].fill_(1)

#     # Fully omit lipid and macromolecular signal
#     sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
#     for n in ind['mac']:
#         params[sign,n].fill_(0)
#         params[sign,n+l].fill_(1)
#         params[sign,n+2*l].fill_(1)
#     for n in ind['lip']:
#         params[sign,n].fill_(0)
#         params[sign,n+l].fill_(1)
#         params[sign,n+2*l].fill_(1)

#     # Frequency Shift
#     print('>>> Frequency Shift')
#     for i, n in enumerate(ind['f_shift']):
#         params[:,n] /= i
#         sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
#         params[sign,n] = 0.5

#     # Scaling
#     # This range needs to be symmetric, therefore zero scaling occurs at 0.5
#     print('>>> Scale')
#     sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
#     # params[:,ind['scale']].uniform_(0.1,0.1) # [-0.1,0.1]
#     params[sign,ind['scale']].fill_(0.5)

#     # Noise
#     print('>>> Noise')
#     sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
#     params[:,ind['snr']].uniform_(config.snr[0],config.snr[1]) # dB
#     params[sign,ind['snr']].fill_(1.0)

#     # Phase Shift
#     # These ranges need to be symmetric, therefore zero phase occurs at 0.5
#     print('>>> Phase Shift')
#     sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
#     params[sign,ind['phi0']].fill_(0.5)
#     sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
#     params[sign,ind['phi1']].fill_(0.5)

#     # B0 Inhomogeneities
#     # This range is asymmetric [-100, 200]. Therefore, the zero-point is at 1/3
#     print('>>> B0 Inhomogeneities')
#     sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
#     params[sign,ind['b0']].fill_(1/3)
