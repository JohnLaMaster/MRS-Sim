import argparse
import json
import os
import random
import sys

import numpy as np
import scipy.io as io
import torch
from src.aux import normalize, load_parameters
from src.mainFcns import _save, prepare, simulate
from src.aux.sample_from_fitted_dist import populate_params_from_distributions, sample_from_copula
from types import SimpleNamespace

sys.path.append('../')


def sample(inputs):
    config, resWater_cfg, baseline_cfg, pm, l, ind, p, totalEntries = inputs

    # Sample parameters
    params = torch.ones((totalEntries, ind['overall'][-1]+1)).uniform_(0,1)
    # params = torch.ones((27, ind['overall'][-1]+1)).uniform_(0,1)
    params, _ = normalize(params, noisy=-1, denom=None) 
    # normalization converts the range from [0,1) to [0,1].
    baselines, res_water = None, None
    
    # print("ind: {}".format(ind))
    
    # # totalEntries += 1
    # print('totalEntries: ',totalEntries)
    
    # for i in ind['metabolites']:
    #     params[:,i].fill_(1.0)


    # Quantify parameters
    # params = pm.quantify_params(params)
    # for target_key in ["ampl", "lorentzLB", "freqShift", "d", "g", "f_shifts", "ph0", "ph1"]:
    name_map = {
        "Cr_SNR": "snr",
        "CrCH2": "cr391",
        "Cr": "crno391",
        "PCh": "cho",
        "PCr": "pcr393",
    }
    suffix_to_ind = {
        "lorentzLB": "d",
        "freqShift": "f_shifts",
        "ph0": "phi0",
        "ph1": "phi1",
    }
    global_param_map = {
        "g": "gaussLB",
        "f_shift": "global_freqShift",
    }
    # for target_key in ["ampl", "lorentzLB", "freqShift", "ph0", "ph1", "gaussLB"]:
        # params = populate_params_from_distributions(
        #             ind=ind,
        #             params=params,
        #             metabolites=config.metabolites,
        #             dist_json_path=config.param_distributions, 
        #             target_key=target_key,
        #             name_map=name_map,
        #             global_param_map=global_param_map,
        #             suffix_to_ind=suffix_to_ind,
        #             )
    params = sample_from_copula(
        ind=ind,
        params=params,
        dist_json_path=config.param_distributions,
        corr_matrix=config.corr_matrix,
        name_map=name_map,
        global_param_map=global_param_map,
        suffix_to_ind=suffix_to_ind,
    )
    # params,
    # dist_json_path: str,
    # corr_matrix: np.ndarray,
    # name_map: Optional[Mapping[str, str]] = None,
    # global_param_map: Optional[Mapping[str, str]] = None,
    # suffix_to_ind: Optional[Mapping[str, str]] = None,
    # seed: Optional[int] = None,

    # print('ind: ',ind)
    # x = breaking
    
    '''
    This next section of code will need to be customized for your own implementations.
    '''
    
    '''
    The next section of code is used to drop some parameters from each spectrum for deep learning applications.
    Should you want to use different distributions for some of the parameters, the following can be used as a 
    guide. Defining different distributions can be done before OR after quantifying the parameters.
    '''
    

    params[:,ind['ecc'][0]] = 0
    params[:,ind['ecc'][1]] = 0

    '''
    If certain parts of the model are turned off, then their values should be zeroed out.
    '''
    params[:,ind['b0']].fill_(0.0)
    for n in ind['b0_dir']: params[:,n].fill_(0.0)
#     Multi_coil is dealt with above
    if config.num_coils<=1:
        params[:,ind['coil_snr']].fill_(0.0)
        params[:,ind['coil_sens']].fill_(0.0)
        params[:,ind['coil_fshift']].fill_(0.0)
        params[:,ind['coil_phi0']].fill_(0.0)
        
        
    # params[0,:].fill_(0.0)
    # print("params\n",params[0:5,0:25])
    # print(params.shape)
    totalEntries = 1
    # params = params[0:2,:]
    
    # print(params)
# 
    return config, resWater_cfg, baseline_cfg, pm, l, ind, p, totalEntries, params, baselines, res_water


#~/In-Vivo-MRSI-Simulator/dataset/30ms_publication/dataset_spectra_sampled_parameters.mat
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default='./dataset/COWS')
    parser.add_argument('--batchSize', type=int, default=1000)
    parser.add_argument('--stepSize', type=int, default=10000)
    parser.add_argument('--parameters', type=str, default=None, help='Path to .mat file with pre-sampled parameters')
    parser.add_argument('--config_file', type=str, default='./src/configurations/debug_new_init.json')
    # parser.add_argument('--bypass', action='store_true', default=False, help='Allows uploading a parameter mat file for the simulations.')

    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    # Simulate
    if isinstance(args.parameters, str):
        sampled = load_parameters(args.parameters, prepare(args.config_file))
    else:
        sampled = sample(prepare(args.config_file))

    path = simulate(sampled,args=args)

    io.savemat(path+'_sampled_parameters.mat', mdict={'params': sampled[-3]})




"""
$ python ./src/30ms_echo_publication.py --config_file './src/config/templates/B0_samples.json' --savedir './dataset/B0_samples'

$ python ./src/30ms_echo_publication.py --config_file './src/templates/EC_samples.json' --savedir './dataset/EC_samples'

$ python ./src/30ms_echo_publication.py --config_file './src/transient_samples.json' --savedir './dataset/transient_samples'

$ python ./src/30ms_echo_publication.py --config_file './src/coil_combined_clean.json' --savedir './dataset/CC_30ms_clean'

$ python ./src/30ms_echo_publication.py --config_file './src/coil_combined_with_artifacts.json' --savedir './dataset/CC_30ms_with_artifacts'

$ python PRESS_30ms_sample_for_Kelley.py --config_file './src/config/kelley.json' --savedir './dataset/VERI_sample'

$ python sim_COWS.py --config_file './src/config/cows.json' --savedir './dataset/COWS'

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



# if config.samples=="by_artifact": 
#     '''
#     Let's do 6 examples of each: frequency shifts, lorentzian lineshapes, gaussian lineshapes, SNRs, zero-order phases, first-order phases, eddy currents
#     Each can have random 
#     '''
    
"""
$ python -m src.process_basis_functions --spin_path './src/basis_sets/references/raw_basis_functions' --save_subdir 'references/raw_basis_functions' --save_name_prefix 'raw' --pulse_sequence 'COWS7_sLASER' --vendor 'Siemens' --centerFreq 4.65 --sim_software 'MARSS'
$ python ./src/findParamDist.py --paramPath ~/Documents/Repositories/Augmentrum/ignore/AUGMENTRUM_DATA_FILES/ --savedir ~/Documents/Repositories/Augmentrum/ignore/ --commonDist --bootstrapping '50,1000'
$ python sim_COWS.py --config_file './src/config/cows.json' --savedir './dataset/COWS'
$ python ./src/aux/plot_mrs.py --data '/home/john/Documents/Repositories/MRS-Sim/dataset/COWS/dataset_spectra_0.mat' --basis_set '/home/john/Documents/Repositories/MRS-Sim/src/basis_sets/references/raw_COWS7_sLASER_30_Siemens_3000.mat' --savedir '/home/john/Documents/Repositories/MRS-Sim/dataset/COWS/' --ind '1,2,3,4,5,6,7,8,9,10' --ppm '7,-2' --met_labels "Asc,Asp,Cho,Cr391,CrNo391,GABA,Gln,Glu,Gly,GPC,GSH,Lac,mI,NAA,NAAG,PCr393,PCrNo393,PE,sI,Tau,MM09"


$ python -m src.process_basis_functions --spin_path './src/basis_sets/references/raw_basis_functions' --save_subdir 'references/raw_basis_functions' --save_name_prefix 'raw' --pulse_sequence 'COWS7_sLASER' --vendor 'Siemens' --centerFreq 4.65 --sim_software 'MARSS'
$ python ./src/findParamDist.py --paramPath ~/Documents/Repositories/Augmentrum/ignore/AUGMENTRUM_DATA_FILES/ --savedir ~/Documents/Repositories/Augmentrum/ignore/ --commonDist --bootstrapping '50,100'
$ python sim_COWS.py --config_file './src/config/cows.json' --savedir './dataset/COWS'
$ python ./src/aux/plot_mrs.py --data '/home/john/Documents/Repositories/MRS-Sim/dataset/COWS/dataset_spectra_0.mat' --basis_set '/home/john/Documents/Repositories/MRS-Sim/src/basis_sets/references/raw_COWS7_sLASER_30_Siemens_3000.mat' --savedir '/home/john/Documents/Repositories/MRS-Sim/dataset/COWS/' --ind '1,2,3,4,5,6,7,8,9,10' --ppm '7,-2' --met_labels "Asc,Asp,Cho,Cr391,CrNo391,GABA,Gln,Glu,Gly,GPC,GSH,Lac,mI,NAA,NAAG,PCr393,PCrNo393,PE,sI,Tau"
"""
# python sim_COWS.py --config_file './src/config/cows.json' --savedir './dataset/COWS'; python ./src/aux/plot_mrs.py --data '/home/john/Documents/Repositories/MRS-Sim/dataset/COWS/dataset_spectra_0.mat' --basis_set '/home/john/Documents/Repositories/MRS-Sim/src/basis_sets/references/raw_COWS7_sLASER_30_Siemens_3000.mat' --savedir '/home/john/Documents/Repositories/MRS-Sim/dataset/COWS/' --ind '1,2,3,4,5,6,7,8,9,10' --ppm '7,-2' --met_labels "Asc,Asp,Cho,Cr391,CrNo391,GABA,Gln,Glu,Gly,GPC,GSH,Lac,mI,NAA,NAAG,PCr393,PCrNo393,PE,sI,Tau"

"""
    "residual_water_ampl": {
        "weibull_min": {
            "c": 0.9473379398002264,
            "loc": 8.761077817235494e-08,
            "scale": 1.1235865805080556e-06
        }
    },


    "Cr_ampl": {
        "nakagami": {
            "nu": 19043.993251051666,
            "loc": -95.59464192903673,
            "scale": 97.56329437085617
        }
    },
"""