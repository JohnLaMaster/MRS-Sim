import argparse
import copy
import json
import os
import sys
import glob
from collections import OrderedDict

import matplotlib.pyplot.close as plt_close
import matplotlib.pyplot.savefig as plt_savefig
import numpy as np
import scipy.io.loadmat as ioloadmat
from fitter import Fitter as Fitter
from fitter import get_common_distributions, get_distributions

from types import Callable, GeneratorType, Tuple, Optional


sys.path.append('../')


def loadParams(parameters_path: Optional[str, list], ) -> Tuple:
    '''
    Load the osp_esportParams mat file. 
    If multiple paths are specified, concatenate the data along
    the batch size, dim=0, yielding shape [batchSize, num_params].
    Expects each file's arrays to be stored column-wise: [num_params, n_subjects].
    '''
    if isinstance(parameters_path, str):
        paths = parameters_path.split(',')
        
    params = None

    for path in paths:
        with open(path, 'rb') as file:
            temp = ioloadmat(file, struct_as_record=True)
        for key in ['__globals__', '__header__', '__version__', 'ECC', 'J', 'beta_j', 'lineShape']:
            temp.pop(key, None)
            
        names = temp['header'].names
        temp.popt('header', None)

        # Validate column-wise storage and transpose to [n_subjects, num_params]
        for k, v in temp.items():
            assert v.ndim == 2, (
                f"Expected 2D array for key '{k}' in {path}, got shape {v.shape}. "
                f"Data must be stored column-wise: [num_params, n_subjects]."
            )
            temp[k] = v.T  # [num_params, n_subjects] -> [n_subjects, num_params]

        if params is None:
            params = copy.copy(temp)
        else:
            for k, v in temp.items():
                assert params[k].shape[1] == v.shape[1], (
                    f"Mismatch in num_params for key '{k}': "
                    f"{params[k].shape[1]} vs {v.shape[1]}."
                )
                params[k] = np.concatenate([params[k], v], axis=0)

    sample_size = next(iter(params.values())).shape[0]
    
    return params, sample_size, names


def findDistribution(v: np.ndarray,
                     ind: np.ndarray,
                     args: argparse.Namespace,
                    ) -> GeneratorType:
    # Unpack args
    if not args.distribuions: 
        dist = get_common_distributions() if args.commonDist else get_distributions()
    else:
        dist = args.distribuions
        
    # Iterate through the num of fitted metabolites per variable
    for i in range(v.shape[-1]):
        f = Fitter(data=v[ind,i], distributions=dist, timeout=args.timeout)
        f.fit(progress=True)
        yield f


def main(args):
    # Load the parameters to find their distributions
    params, sample_size, names = loadParams(args.paramPath)
    args.nx = 1 if args.n==-1 else args.nx
    args.n = sample_size if args.n==-1 else int(args.n*sample_size)
    ind = np.zeros(sample_size, dtype='bool')
    ind[0:args.n] = True
    num_res = params['ampl'].shape[-1]
    
    if not args.paramKeys:
        args.paramKeys = params.keys()

    for ct in range(args.nx + 1 if args.nx!=1 else args.nx):
        # If multiple rounds, then the last round should analyze the entire dataset 
        # for reference. If not multiple rounds, then it will do a single pass
        savedir = os.path.join(args.savedir,'split_{}_of_{}/'.format(ct,args.nx))
        np.shuffle(ind) # Shuffles which samples are analyzed in each round
        if ct==args.nx or args.nx==1:
            # When doing multiple rounds, the results for the entire dataset are 
            # stored in the parent directory for reference
            savedir = args.savedir
            ind = np.ones(sample_size, dtype='bool')
    
        distributions, best = OrderedDict(), OrderedDict()
        # dist_names = ['']
        
        
        # Iterate through the parameters
        for k, v in params.items():
            # Check if the parameter name was selected for analysis
            if k in args.paramKeys:
                num_vars = v.shape[1]
                if k in 'ampl': k = '{}_ampl'.format(k)
                if k in 'lorentzLB': k = '{}_lorentzLB'.format(k)
                if k in 'freqShift': k = '{}_freqShift'.format(k)
                for j in range(num_vars):
                    for i, f in enumerate(findDistribution(v=v[...,:j], ind=ind, args=args)):
                        # Identify the top-k best fitting distributions
                        distributions.update({'{}_{}'.format(k,i): 
                                            f.summary(Nbest=args.Nbest, method=args.selectionMetric).to_dict()})
                        
                        # Save the best fitting distribution and its parameters
                        best.update({'{}_{}'.format(k,i): f.get_best()})
                        
                        # Save the top-k best fitting distribution plots
                        plt_savefig(savedir + '{}_{}.png'.format(k,i), dpi=140)
                        plt_savefig(savedir + '{}_{}.eps'.format(k,i), dpi=140)
                        plt_close()
                        
                        # with open(savedir + 'summary_of_{}_best_fits.json'.format(args.Nbest), 'w') as file:
                        #     json.dump(distributions, file, indent=4, separators=(',', ': '))

                        # with open(savedir + 'best_fit.json', 'w') as file:
                        #     json.dump(best, file, indent=4, separators=(',', ': '))
                        
                    # Append-by-merge: load existing content, update, and write back
                    summary_path = savedir + 'summary_of_{}_best_fits.json'.format(args.Nbest)
                    if os.path.isfile(summary_path):
                        with open(summary_path, 'r') as file:
                            existing = json.load(file)
                        existing.update(distributions)
                        distributions = existing
                    with open(summary_path, 'w') as file:
                        json.dump(distributions, file, indent=4, separators=(',', ': '))

                    best_path = savedir + 'parameter_distributions_best_fit.json'
                    if os.path.isfile(best_path):
                        with open(best_path, 'r') as file:
                            existing = json.load(file)
                        existing.update(best)
                        best = existing
                    with open(best_path, 'w') as file:
                        json.dump(best, file, indent=4, separators=(',', ': '))

    print('Saved fitting parameter distributions at: {}'.format(args.savedir + 'parameter_distributions_best_fit.json'))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paramPath', type=str, help='Path to .mat file exported after Osprey fitting')
    parser.add_argument('--savedir', type=str, help='Should be a directory path')
    parser.add_argument('--selectionMetric', type=str, default='sumsquare_error', help='Options are: sumsquare_error, aic, bic, kl_div, ks_statistic, ks_pvalue')
    parser.add_argument('--Nbest', type=int, default=5, help='Sets the number of best performing fits to include in the results for each variable.')
    parser.add_argument('--commonDist', type=bool, default=True, help='Restricts the analysis to 10 common distributions.')
    parser.add_argument('--comprehensiveDist', type=bool, action="store_false", dest='commonDist', help='Evaluate the comprehensive list of available distributions.')
    parser.add_argument('--distributions', type=str, default=None, help='Specify specific distribuions to test for. Possible distributions can be found in the scipy.stats documentation.')
    parser.add_argument('--paramKeys', type=str, default=None, help='CAn be used to select specific parameters for analysis.')
    parser.add_argument('--n', type=float, default=-1, help='Percentage of the sample size to test. Results will converge when the sample size is sufficiently large.')
    parser.add_argument('--nx', type=int, default=5, help='The number of times to resample the dataset and retest the distribution.')
    parser.add_argument('--timeout', type=float, default=30, help='Time limit (sec) for testing a given distribution. Used to cap runtimes.')
    
    args = parser.parse_args()


    # Create the save directory
    os.makedirs(args.savedir, exist_ok=True)
    
    # Specifying a single file
    if os.path.isfile(args.paramPath):
        assert os.path.splitext(args.paramPath)[-1].lower() in ['.mat']
        args.paramPath = [args.paramPath]
    elif os.path.isdir(args.paramPath):
        args.paramPath = sorted(glob.glob(os.path.join(args.paramPath, '*fitting_parameters*.mat')))
        assert len(args.paramPath) > 0, f"No '*fitting_parameters*.mat' files found in: {args.paramPath}"

    # Specifying a list of files
    if isinstance(args.paramPath, str):
        args.paramPath = [path for path in args.paramPath.split(',')]

    # Specifying a directory of files
    # if os.isdir(args.paramPath):
    #     args.paramPath = [path for path in os.listdir(args.paramPath) if os.path.splitext(path)[-1].lower() in ['.mat']]
    
    # Limit analysis to specific distributions
    if args.distributions:
        args.distributions = [dist for dist in args.distributions.split(',')]
        args.commonDist = False
        
    # Limit analysis to specific variables
    if not args.paramKeys:
        args.paramKeys = ''
    args.paramKeys = [k.lower() for k in args.paramKeys.split(',')]
    
    
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
