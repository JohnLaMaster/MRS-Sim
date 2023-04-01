import argparse
import copy
import json
import os
import sys
from collections import OrderedDict

import matplotlib.pyplot.close as plt_close
import matplotlib.pyplot.savefig as plt_savefig
import numpy as np
import scipy.io.loadmat as ioloadmat
from fitter import Fitter as Fitter
from fitter import get_common_distributions, get_distributions

sys.path.append('../')


def loadParams(parameters_path: str, ):
    '''
    Load the osp_esportParams mat file. 
    If multiple paths are specified, concatenate the data along
    the batch size, dim=0.
    '''
    paths = parameters_path.split(',')
    params = None
    for path in paths:
        with open(path, 'rb') as file:
            temp = ioloadmat(file, struct_as_record=True)
            temp.pop('__globals__')
            temp.pop('__header__')
            temp.pop('__version__')
            try: temp.pop('header')
            except Exception: pass
        if isinstance(params, type(None)):
            params = copy.copy(temp)
        else:
            for k, v in temp.items():
                params[k] = np.concatenate(params[k], v, axis=0)
        

    return params


def findDistribution(v: np.ndarray,
                     comprehensive: bool=False):
    # Iterate through the num of fitted metabolites per variable
    dist = get_distributions() if comprehensive else get_common_distributions()
    for i in range(v.shape[-1]):
        f = Fitter(data=v[...,i], distributions=dist)
        f.fit(progress=True)
        yield f


def main(args):
    # Load the parameters to find their distributions
    params = loadParams(args.paramPath)
    distributions, best = OrderedDict(), OrderedDict()
    dist_names = ['']

    # Iterate through the parameters
    for k, v in params.items():
        for i, f in enumerate(findDistribution(v, args.comprehensiveDist)):
            # Identify the top-k best fitting distributions
            distributions.update({'{}_{}'.format(k,i): 
                                  f.summary(Nbest=args.Nbest, method=args.selectionMetric).to_dict()})
            with open(args.savedir + 'parameter_distribution_fits.json', 'w') as file:
                json.dump(distributions, file, indent=4, separators=(',', ': '))

            # Save the best fitting distribution and its parameters
            best.update({'{}_{}'.format(k,i): f.get_best()})
            with open(args.savedir + 'best_parameter_distributions.json', 'w') as file:
                json.dump(best, file, indent=4, separators=(',', ': '))

            # Save the top-k best fitting distribution plots
            plt_savefig(args.savedir + '{}_{}.png'.format(k,i), dpi=140)
            plt_savefig(args.savedir + '{}_{}.eps'.format(k,i), dpi=140)
            plt_close()

    print('Saved fitting parameter distributions at: {}'.format(args.savedir + 'parameter_distributions.json'))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paramPath', type=str, help='Path to .mat file exported after Osprey fitting')
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--selectionMetric', type=str, default='sumsquare_error')
    parser.add_argument('--Nbest', type=int, default=5)
    parser.add_argument('--comprehensiveDist', type=bool, default=False)
    

    args = parser.parse_args()

    # Create the save directory
    os.makedirs(args.savedir, exist_ok=True)
    
    # Specifying a single file
    if os.path.isfile(args.paramPath):
        assert os.path.splitext(args.paramPath)[-1].lower() in ['.mat']

    # Specifying a list of files
    if isinstance(args.paramPath, str):
        args.paramPath = [path for path in args.paramPath.split(',')]

    # Specifying a directory of files
    if os.isdir(args.paramPath):
        args.paramPath = [path for path in os.listdir(args.paramPath) if os.path.splitext(path)[-1].lower() in ['.mat']]
    
    main(args)


'''
  Reference for Fitter:
    f = Fitter(data) # f = Fitter(data, distributions=["gamma", "rayleigh", "uniform"])
    f.fit(progress=False, n_jobs=-1, max_workers=-1)
    f.summary(Nbest=5, lw=2, plot=True, method='sumsquare_error', clf=True)
    f.get_best(method='sumsquare_error')
    f.hist()
    f.plot_pdf(names=None, Nbest=5, lw=2, method='sumsquare_error')
'''
