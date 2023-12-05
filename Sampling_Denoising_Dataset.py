import argparse
import json
import os
import random
import sys

import numpy as np
import scipy.io as io
import torch
from src.aux import normalize
from src.mainFcns import _save, prepare, simulate
from types import SimpleNamespace

sys.path.append('../')


def sample(inputs):
    config, resWater_cfg, baseline_cfg, pm, l, ind, p, totalEntries = inputs

    # Sample parameters
    params = torch.ones((totalEntries, ind['overall'][-1]+1)).uniform_(0,1)
    params, _ = normalize(params, noisy=-1, denom=None) 
    # normalization converts the range from [0,1) to [0,1].
    

    # Quantify parameters
    params = pm.quantify_params(params)
    
    '''
    This next section of code will need to be customized for your own implementations.
    '''

    # All metabolite values are ratios wrt creatine. Therefore, Cr is always 1.0
    # denom = 1
    
    # try: params[:,ind['pcr']].fill_(1.0)
    # except KeyError as E: pass
    # try: params[:,ind['cr']].fill_(1.0)
    # except KeyError as E: pass
    # if 'PCr' in config.metabolites and 'Cr' in config.metabolites:
    #     params[:,ind['cr']].fill_(0.525/denom)
    #     params[:,ind['pcr']].fill_(0.475/denom)

    # # print(ind)
    # Metabolite values: Gudmundson et al. 2023
    #                    Meta-analysis and Open-source Database for In Vivo Brain Magnetic Resonance Spectroscopy in Health and Disease
    # MM/Lip values come from Osprey example; STD is ~10%
    names = ["Asc", "Asp", "Ch", 
             "Cr", "GABA", "GPC", 
             "GSH", "Gln", "Glu", 
             "mI", "Lac", "NAA", 
             "NAAG", "PCh", "PCr", 
             "sI", "Tau", 
             "MM09", "MM12", "MM14", 
             "MM17", "MM20", "Lip09", 
             "Lip13", "Lip20"
             ]
    amps = [0.08202179866, 0.02271305968;
            0.2277019638, 0.1036748435;
            0.1693652297, 0.02823696723;

            0.5853935312, 0.0260751858;
            0.2088882558, 0.06536990668;
            0.168192897, 0.07404373306;

            0.1408989742, 0.06259323849;
            0.2167542566, 0.09484093279;
            0.865008085, 0.2360240584;

            0.5732830022, 0.1841692379;
            0.0635885279, 0.03917125776;
            1.054923016, 0.2303316518;

            0.1648531131, 0.06860956658;
            0.08468261485, 0.02823696723;
            0.4146064688, 0;

            0.01767595665, 0.01020771789;
            0.123438892, 0.07545435854;

            10.314097589906174, 1;
            10.402136461787936, 1;
            0.358101078275321, 0.035;
            10.452705224928629, 1;
            10.551475712699752, 1;
            10.719485302705014, 1;
            10.414165486070721, 1;
            0.3, 0.03;
            ]
    for n, v in zip(names,amps):
        mu, std = v[0]; v[1]*2.5;
        params[:,ind[n.lower()]].uniform_(mu-std,mu+std)
        # params[:,ind[n.lower()]].fill_(mu+std*(random.randrange(-100, 100)/100))
    params[:,ind['pcr']] = 1 - params[:,ind['cr']]
    params[:,tuple(ind['metabolites'])].clamp(min=0.0)
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
        T2 = {}
        # pregenual anterior cingulate cortex (pACC)
        metabs = ["Asc"; "Asp";  "Ch"; "Cr"; "GABA"; "Glc"; "Gln"; "Glu"; "Gly"; "GPC"; "GSH"; "Lac", "mI", "NAA", "NAAG", "PCh", "PCr", "PE", "sI", "Tau", "MM09", "MM12", "MM14", "MM17", "MM20", "Lip09", "Lip13", "Lip20"]
        t2     = [  125,   111,   257,  141,    102,   117,   122,   135,   102,   257,    99,   110,  244,   253,    128,   243,   148,  119,  125,   123,    100,    100,    100,    100,    100,     100,     100,     100] # from Wyss 2018 "In Vivo Estimation of Transverse Relaxation Time..."
        t2_std = [   19,    20,    57,   18,     19,    22,    19,    28,    22,    57,    16,    24,   61,    64,     18,    51,    22,   25,   19,    23,      5,      5,      5,      5,      5,       5,       5,       5]
        for m, t, s in zip(metabs, t2, t2_std):
            if m in names: T2[m] = [t, s]
        for i, n in enumerate(ind['d'], names):
            params[:,i].uniform_(1000/(T2[n][0]+T2[n][1]),1000/(T2[n][0]-T2[n][1]))
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
    print('>>> Coil Sensitivities')
    params[:,ind['coil_sens']] = torch.distributions.normal.Normal(1,0.5).sample(params[:,ind['coil_sens']].shape).clamp(min=0.0,max=2.0)

    # Values were taken from the DL model performance in Tapper 2021
    print('>>> Coil Frequency Drift')
    # mu = 0.0 +/- 0.4 Hz => MUST convert from Hz to PPM
    params[:,ind['coil_fshift']].uniform_(-0.4,0.4)*10e6 / (42.577478518*3)

    print('>>> Coil Phase Drift')
    # mu = -0.11 +/- 0.25 degrees
    params[:,ind['coil_phi0']].uniform_(-0.36,0.14)

    

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
        params[:,ind['coil_fshift']].fill_(0.0)
        params[:,ind['coil_phi0']].fill_(0.0)
    

    # presim = {'baseline': baseline, 'reswater': res_water}

    return config, resWater_cfg, baseline_cfg, pm, l, ind, p, totalEntries, params, presim

def _save(path: str, 
          spectra: np.ndarray, 
          transients: np.ndarray, 
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
             'transients': transients,
             'noise': spectra[:,0:2,...] - spectra[:,2:4,...],
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
    config, resWater_cfg, baseline_cfg, pm, l, ind, p, totalEntries, params, presim = inputs
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
                             zero_fill=config.zero_fill,
                             broadening=config.broadening,
                             coil_fshift=config.coil_fshift,
                             residual_water=sample_resWater(n-i, **resWater_cfg) if resWater_cfg else False,
                             drop_prob=config.drop_prob,
                             presim=presim)
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
                             transients=spectra.mean(axis=-3),
                             fits=fit,
                             baselines=baseline, 
                             reswater=reswater, 
                             parameters=sort_parameters(parameters, ind), 
                             quantities=quantities, 
                             cropRange=pm.cropRange, 
                             ppm=ppm,
                             header=config.header)
            if config.NIfTIMRS:
                save2nifti.forward(datapath=new_path)
            first = True
            counter += 1
            print('>>> ** {} ** <<<'.format(counter))
        elif ((i+step) >= params.shape[0]) or (config.totalEntries<=threshold):
            new_path = _save(path=path + '_{}'.format(counter), 
                             spectra=spectra, 
                             transients=spectra.mean(axis=-3),
                             fits=fit,
                             baselines=baseline, 
                             reswater=reswater, 
                             parameters=sort_parameters(parameters, ind), 
                             quantities=quantities, 
                             cropRange=pm.cropRange, 
                             ppm=ppm,
                             header=config.header)
            if config.NIfTIMRS:
                save2nifti.forward(datapath=new_path)
        del spectra, fit, baseline, reswater, parameters, quantities
    return path

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


'''
    "baseline_cfg": {
        "start":           [    -1,     1],
        "end":             [    -1,     1],
        "upper":           [            1],
        "lower":           [           -1],
        "std":             [  0.05,  0.20],
        "window":          [  0.15,   0.3],
        "pt_density":          128,
        "ppm_range":       [  -1.6,   8.5],
        "scale":           [   0.1,   1.0],
        "drop_prob":           0.0
    },
'''
