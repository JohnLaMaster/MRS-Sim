import copy
import json
import math
import os
from collections import OrderedDict

import numpy as np
from NIfTI-MRS import Mat2NIfTI_MRS
from physics_model import PhysicsModel
from scipy.io import savemat as io_savemat
from types import SimpleNamespace

__all__ = ['prepare', '_save', 'simulate']


def prepare(config_file):
    # Load the config file
    with open(config_file) as file:
        config = json.load(file)

    confg_kys = config.keys()
    parameters = config["parameters"] if "parameters" in confg_kys else None
    resWater_cfg = config["resWater_cfg"] if "resWater_cfg" in confg_kys else None
    baseline_cfg = config["baseline_cfg"] if "baseline_cfg" in confg_kys else None

    config = SimpleNamespace(**config)
    p = 1 - config.drop_prob # probability of including a variable for a given data sample
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
    config.header = pm.header

    # print(pm.index)
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


def _save(path: str, 
          spectra: np.ndarray, 
          fits: np.ndarray, 
          baselines: np.ndarray, 
          reswater: np.ndarray, 
          parameters: np.ndarray, 
          quantities: dict, 
          cropRange: list, 
          ppm: np.ndarray,
          header: dict):
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
             'header': header
            }
    io_savemat(path + '.mat', do_compression=True, mdict=mdict)
    print(path + '.mat')
    return path + '.mat'



def simulate(inputs, args=None):
    config, resWater_cfg, baseline_cfg, pm, l, ind, p, totalEntries, params = inputs
    '''
    Begin simulating and saving the spectra
    '''
    first = True
    step = args.stepSize
    threshold = args.batchSize
    path = args.savedir + '/dataset_spectra'
    if config.NIfTIMRS: 
        save2nifti = Mat2NIfTI_MRS(combine_complex=True, test_output=True)
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
            new_path = _save(path=path + '_{}'.format(counter), 
                             spectra=spectra, 
                             fits=fit,
                             baselines=baseline, 
                             reswater=reswater, 
                             parameters=sort_parameters(parameters, ind), 
                             quantities=quantities, 
                             cropRange=pm.cropRange, 
                             ppm=ppm,
                             header=config.header)
            if config.NIfTIMRS:
                save2nifti(datapath=new_path)
            first = True
            counter += 1
            print('>>> ** {} ** <<<'.format(counter))
        elif ((i+step) >= params.shape[0]) or (config.totalEntries<=threshold):
            new_path = _save(path=path + '_{}'.format(counter), 
                             spectra=spectra, 
                             fits=fit,
                             baselines=baseline, 
                             reswater=reswater, 
                             parameters=sort_parameters(parameters, ind), 
                             quantities=quantities, 
                             cropRange=pm.cropRange, 
                             ppm=ppm,
                             header=config.header)
            if config.NIfTIMRS:
                save2nifti(datapath=new_path)
        del spectra, fit, baseline, reswater, parameters, quantities
