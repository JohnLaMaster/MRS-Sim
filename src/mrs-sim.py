import argparse
import copy
import os
import sys

import numpy as np
import scipy.io as io
import torch
from modules.physics_model.fitting_physics_model import \
    FittingPhysicsModel as PhysicsModel

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

def simulate(totalEntries=100000, crop=False, resample=False, args=None):
    p = 0.8 # probability of including a variable for a given data sample

    pm = PhysicsModel(opt=args, cropRange=crop, length=resample).to(device)
    pm.initialize(metab=args.metabolites, cropRange=crop, supervised=True, baselines=args.baselines, metab_as_is=args.metab_as_is)#,'Mac','Lip'], cropRange=crop, supervised=True)
    
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


    params = torch.zeros((totalEntries, ind['overall'][-1] + 1)).uniform_(0,1+1e-6)
    params = params.clamp(0,1)
    ind = pm.index
    l = ind['metabolites'][-1] + 1

    # All metabolite values are ratios wrt creatine. Therefore, Cr is always 1.0
    params[:,ind['cr']].fill_(1.0)

    '''
    If you want to use a covariance matrix for sampling metabolite amplitudes, this is where covmat and loc 
    should be defined. 
    '''
    # if len(args.metabolites)==3 and args.use_covmat:
    #     covmat = torch.as_tensor() # 2D matrix
    #     loc = torch.as_tensor() # 1D matrix
    #     mets = torch.distributions.multivariate_normal.MultivariateNormal(loc=loc,
    #                                                                       covariance_matrix=covmat)
    #     start, stop = ind['metabolites'][0], ind['metabolites'][-1]
    #     params[:,start:stop] = mets.rsample(params[:,start:stop].shape)


    print('>>> Line Broadening')
    if not args.no_omit:
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
        for n in ind['lip']:
            params[sign,n].fill_(0)
            params[sign,n+l].fill_(1)

        # Frequency Shift
        print('>>> Frequency Shift')
        for n in ind['f_shift']:
            sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
            params[sign,n] = 0.5

        # Scaling
        print('>>> Scale')
        sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
        params[:,ind['scale']].uniform_(0.05,0.30) # dB
        params[sign,ind['scale']].fill_(1.0)

        # Noise
        print('>>> Noise')
        sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
        params[:,ind['snr']].uniform_(0.05,0.30) # dB
        params[sign,ind['snr']].fill_(1.0)

        # Phase Shift
        print('>>> Phase Shift')
        sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
        params[sign,ind['phi0']].fill_(0.5)
        sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
        params[sign,ind['phi1']].fill_(0.5)

        # B0 Inhomogeneities
        print('>>> B0 Inhomogeneities')
        sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
        params[sign,ind['b0']].fill_(1/3)

    print('Too large: {}, Too Small: {}'.format((params>1).any(), (params<0).any()))   # Too large: True, Too Small: True
    print('params max {} min {}'.format(params.max(), params.min()))
    params = params.clamp(min=0, max=1)

    first = True
    step = 12500
    path = args.savedir + '/dataset'
    counter = 0
    for i in range(0,args.totalEntries,step):
        if first:
            spectra = pm.forward(params=params[i:i+step,:], 
                                 step=step,
                                 b0=True,
                                 gen=True, 
                                 phi0=True,
                                 phi1=True,
                                 noise=True, 
                                 scale=True,
                                 fshift=True,
                                 apodize=False,
                                 offsets=None,
                                 baselines=sample_baselines(baseline_cfg),
                                 magnitude=True,
                                 broadening=True,
                                 transients=False,
                                 residual_water=sample_resWater(resWater_cfg))

#             spectra = pm.fit(params[i:i+step,:], magnitude=True)#, phi1=True)
            first = False
        else:
            spectra = torch.cat([spectra, pm.forward(params[i:i+step,:], gen=True, noise=True, baselines=args.baselines, phi1=True)], dim=0)
            if args.totalEntries>=250000 and (i+step) % 100000==0:
                _save(path + '{}'.format(counter), spectra, params[i+step-spectra.shape[0]:i+step,:], args.crop)
                first = True
                counter += 1
                print('>>> ** {} ** <<<'.format(counter))
    return spectra, params, baselines_f, baselines
#     return pm.forward(params, gen=True), params

def torch2numpy(input: dict):
    for k, v in input.items():
        if isinstance(input[k], dict):
            torch2numpy(input[k])
        elif torch.is_tensor(v):
            input[k] = v.numpy()
        else:
            pass
    return input


def _save(path, spectra, parameters, ppm_range, test=None, bl=None, blf=None):
    print('>>> Saving Spectra')
    base, _ = os.path.split(path)
    os.makedirs(base, exist_ok=True)
    num_test = 0
    if test:
        spectra = torch.cat([spectra, test['spec']], dim=0)
        parameters = torch.cat([parameters,  test['params']], dim=0)
        num_test = test['num']
    dict = {'spectra': spectra.numpy(),
            'params': parameters.unsqueeze(-1).numpy(),
            'num_test': int(num_test),
            'ppm_range': ppm_range,
            'baselines': bl.numpy(),
            'baselines_f': blf.numpy()}
    io.savemat(path + '_spectra.mat', do_compression=True, mdict=dict)
    print(path + '_spectra.mat')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default='./dataset/')
    parser.add_argument('--totalEntries', type=int, default=120000)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--test', type=int, default=False)
    parser.add_argument('--metabolites', type=str, default='Cho,Cre,Naa,Glx,Ins')
    parser.add_argument('--metab_as_is', action='store_true', default=False)
    parser.add_argument('--snr', type=tuple,default=(0.80, 0.50, 0.30, 0.10, 0.03, 0.00))
    parser.add_argument('--crop', type=str, default='0.2,4.2', help='flag and range for cropping')#(1005,1325)
    parser.add_argument('--resample', type=int, default=512, help='desired size of the interpolated output')
    parser.add_argument('--G0', action='append_const', dest='gpu_ids', const=0, help='use GPU 0 in addition to the default GPU')
    parser.add_argument('--G1', action='append_const', dest='gpu_ids', const=1, help='use GPU 0 in addition to the default GPU')
    parser.add_argument('--G2', action='append_const', dest='gpu_ids', const=2, help='use GPU 0 in addition to the default GPU')
    parser.add_argument('--PM_basis_set', type=str, default='fitting_basis_ge_press144_1.mat', help='file name of the basis set')
    parser.add_argument('--spectrum_length', type=float, default=512, help='length of the input spectra')
    parser.add_argument('--baselines', action='store_false', default=True, help='Add baselines to the spectra')
    parser.add_argument('--use_covmat', action='store_true', default=False, help='Use the metabolite covariance matrix for sampling quantities')
    parser.add_argument('--no_omit', action='store_false', default=True, help='Add baselines to the spectra')
    parser.add_argument('--test_p', type=str, default='0.8', help='Keep_prob fro omitting variables')
    parser.add_argument('--test_use_covmat', action='store_true', default=False, help='Use the metabolite covariance matrix for sampling quantities')
    parser.add_argument('--test_no_omit', action='store_false', default=True, help='Add baselines to the spectra')
    parser.add_argument('--test_no_artifacts', action='store_true', default=False, help='Add baselines to the spectra')

    args = parser.parse_args()
    
    os.makedirs(args.savedir, exist_ok=True)
    args.metabolites = list([x for x in args.metabolites.split(',')])
    print('Metabolites: ',args.metabolites)
    
    args.test_p = [float(p) for p in args.test_p.split(',')]
    if len(args.test_p)==1: args.test_p *= 2

    args.gpu_ids = []
    device = 'cpu'

    if args.crop:
        crop = args.crop.split(',')
        args.crop = tuple([float(crop[0]),float(crop[1])])
    
    if args.phase=='train':
#         spectra, parameters = train(totalEntries=args.totalEntries, crop=args.crop, resample=args.resample, args=args)
        spectra, parameters, blf, bl = train(totalEntries=args.totalEntries, crop=args.crop, resample=args.resample, args=args)
#     spectra, parameters = simulate(totalEntries=25000, crop=args.crop, resample=args.resample, args=args)
#     for _ in range(4):
#         s, p = train(totalEntries=25000, crop=args.crop, resample=args.resample, args=args)
#         spectra, parameters = torch.cat([spectra, s], dim=0), torch.cat([parameters, p], dim=0)

    
    out = None
    if args.test:
        out = {}
        out['spec'], out['params'] = test(totalEntries=args.test, crop=args.crop, resample=args.resample, args=args)
        out['num'] = args.test
        #         test['spec'], test['params'], test['num'] = out[0], out[1], args.test
        if args.phase=='test':
            spectra, parameters = out['spec'], out['params']
            out = None

    args.savedir += '/dataset'
    _save(args.savedir, spectra, parameters, args.crop, out, bl, blf)
