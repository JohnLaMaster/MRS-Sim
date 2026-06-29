import argparse
import copy
import json
import os
import sys
import glob

import numpy as np
import pandas as pd

from pathlib import Path
from collections import OrderedDict
from matplotlib.pyplot import close as plt_close
from matplotlib.pyplot import savefig as plt_savefig
from scipy.io import loadmat as ioloadmat
# from copy import deepcopy, copy

from fitter import Fitter as Fitter
from fitter import get_common_distributions, get_distributions

from types import GeneratorType
from typing import Tuple, Optional


sys.path.append('../')


NONNEGATIVE_DIST = [
    "gamma", "lognorm", "weibull_min", "nakagami", "truncexpon", "truncnorm", "truncweibull_min",
    "expon", "chi2",
    "halfcauchy", "halflogistic", "halfnorm", "halfgennorm", #"beta"
]
TAGS = ['ampl','lorentzLB','freqShift','SNR','ph0','ph1']

def loadParams(parameters_path: Optional[str | list | None], ) -> Tuple:
    '''
    Load the osp_esportParams mat file. 
    If multiple paths are specified, concatenate the data along
    the batch size, dim=0, yielding shape [batchSize, num_params].
    Expects each file's arrays to be stored column-wise: [num_params, n_subjects].
    '''
    if isinstance(parameters_path, str):
        paths = parameters_path.split(',')
    else:
        paths = parameters_path
        
    params = None

    for i, path in enumerate(paths):
        with open(path, 'rb') as file:
            temp = ioloadmat(file, struct_as_record=True, simplify_cells=True)
        for key in ['__globals__', '__header__', '__version__', 'ECC', 'J', 'beta_j', 'lineShape', 'seq']:
            temp.pop(key, None)
            
        if i==0:
            names = temp['header']['names']
        temp.pop('header', None)
        
        # print(f'temp.keys(): {temp.keys()}')
        
        # Only fit the targeted variables
        # Helps with the simulation part because entries don't need to be deleted manually
        # keys = list(temp.keys())
        # for k in keys:
        #     keep = False
        #     for tag in TAGS:
        #         print(f'k: {k}, tag: {tag}')
        #         if tag in k:
        #             keep = True
        #             break  # stop checking tags
        #         if not tag in k:
        #             temp.pop(k, None)
        for k in list(temp.keys()):
            if not any(tag in k for tag in TAGS):
                # print(f'Popping key {k} from the loaded parameters.')
                temp.pop(k, None)

        # print('temp: ',temp)

        # Validate column-wise storage and transpose to [n_subjects, num_params]
        for k, v in temp.items():
            # assert v.ndim == 2, (
            #     f"Expected 2D array for key '{k}' in {path}, got shape {v.shape}. "
            #     f"Data must be stored column-wise: [num_params, n_subjects]."
            # )
            # print('key: ', k)
            if isinstance(v, float): v = np.array([v])
            if v.ndim==1: v = np.expand_dims(v, axis=1)
            temp[k] = v.T  # [num_params, n_subjects] -> [n_subjects, num_params]
            # temp[k] = np.concatenate([temp[k], v.T], axis=0)  # [num_params, n_subjects] -> [n_subjects, num_params]


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
    # Unpack distributions
    if not args.distributions:
        dist = get_common_distributions() if args.commonDist else get_distributions()
    elif "nonnegative" in args.distributions:
        dist = NONNEGATIVE_DIST
    else:
        dist = args.distributions

    # Normalize to a list of distribution names
    if isinstance(dist, str):
        dist = [dist]
    else:
        dist = list(dist)

    # Bootstrap settings
    boot = args.bootstrapping
    metric = getattr(args, "bootstrap_metric", "sumsquare_error")
    top_k = int(getattr(args, "top_k", 5))

    # Iterate through the number of fitted metabolites per variable
    for i in range(v.shape[-1]):
        data_full = np.asarray(v[ind, i]).ravel()

        # No bootstrapping: original behavior
        if not boot:
            f = Fitter(data=data_full, distributions=dist, timeout=args.timeout)
            f.fit(progress=True)
            yield f
            continue

        # Parse bootstrapping config
        if isinstance(boot, int):
            sample_size = len(data_full)
            n_reps = int(boot)
        elif isinstance(boot, (list, tuple)) and len(boot) == 2:
            sample_size = int(boot[0])
            n_reps = int(boot[1])
        else:
            raise ValueError(
                "args.bootstrapping must be False, an int, or a list/tuple of "
                "[sample_size, n_reps]"
            )

        if sample_size <= 0:
            raise ValueError("bootstrap sample_size must be positive")
        if n_reps <= 0:
            raise ValueError("bootstrap n_reps must be positive")

        rng = np.random.default_rng()

        # Collect bootstrap ranks
        rank_sum = pd.Series(0.0, index=dist, dtype=float)
        seen_count = pd.Series(0, index=dist, dtype=int)

        for _ in range(n_reps):
            sample = rng.choice(data_full, size=sample_size, replace=True)

            fb = Fitter(data=sample, distributions=dist, timeout=args.timeout)
            fb.fit(progress=False)

            # df_errors is the score table; lower is better for the usual methods
            scores = fb.df_errors[metric].reindex(dist)

            # Make failed fits rank worst
            scores = scores.replace([np.inf, -np.inf], np.nan).fillna(np.inf)

            ranks = scores.rank(method="average", ascending=True)

            rank_sum = rank_sum.add(ranks, fill_value=0.0)
            seen_count = seen_count.add(scores.notna().astype(int), fill_value=0)

        # Mean rank across bootstrap repetitions
        mean_rank = rank_sum / n_reps

        # Pick top-k distributions by aggregated rank
        top_dists = mean_rank.sort_values().index[:top_k].tolist()

        # Final refit on the full data using only the selected distributions
        final_sample = rng.choice(data_full, size=sample_size * n_reps, replace=True)
        f_final = Fitter(data=final_sample, distributions=top_dists, timeout=args.timeout)
        f_final.fit(progress=False)

        # Optional: attach bootstrap selection info for later inspection
        f_final.bootstrap_rank_mean = mean_rank
        f_final.bootstrap_top_distributions = top_dists
        f_final.bootstrap_metric = metric
        f_final.bootstrap_n_reps = n_reps
        f_final.bootstrap_sample_size = sample_size

        yield f_final


def main(args):
    # Load the parameters to find their distributions
    params, sample_size, names = loadParams(args.paramPath)
    set_distributions = args.distributions

    args.nx = 1 if args.n == -1 else args.nx
    if args.n == -1:
        args.n = sample_size
    else:
        assert isinstance(args.n, (float, int)), (
            "--n must be a float or int but got type {}".format(type(args.n))
        )
        if isinstance(args.n, int):
            assert args.n == 1, "If --n is an integer, it must be equal to 1"
        elif isinstance(args.n, float):
            assert (args.n > 0) & (args.n < 1), "if --n is a float, it must be between (0,1)"
            args.n = int(args.n * sample_size)

    ind = np.zeros(sample_size, dtype=bool)
    ind[0:args.n] = True

    # Normalize param keys to a set for exact membership testing
    if not args.paramKeys:
        print("Redfining paramKeys")
        param_keys = [k for k in params.keys()]
    else:
        print("Using predefined paramKeys")
        param_keys = list(args.paramKeys)
        
    # print('param_keys = ',param_keys)
    
    # Check if files already exist and delete
    summary_path = os.path.join(args.savedir, f'summary_of_{args.Nbest}_best_fits.json')
    best_path = os.path.join(args.savedir, 'parameter_distributions_best_fit.json')
    if os.path.exists(summary_path): os.remove(summary_path)
    if os.path.exists(best_path): os.remove(best_path)
    
    def pretty_name(k, met):
        if k == 'ampl':
            return f'{met}_ampl'
        if k == 'lorentzLB':
            return f'{met}_lorentzLB'
        if k == 'freqShift':
            return f'{met}_freqShift'
        return k

    for ct in range(args.nx + 1 if args.nx != 1 else args.nx):
        # If multiple rounds, then the last round should analyze the entire dataset
        savedir = os.path.join(args.savedir, f'split_{ct}_of_{args.nx}/')
        np.random.shuffle(ind)

        if ct == args.nx or args.nx == 1:
            savedir = args.savedir
            ind = np.ones(sample_size, dtype=bool)

        distributions, best = OrderedDict(), OrderedDict()

        for k, v in params.items():
            if k not in param_keys:
                print(f'key {k} is not in args.paramKeys')
                continue

            v = np.asarray(v)
            if v.ndim == 1:
                v = v[:, None]

            num_vars = v.shape[1]
            # base_key = pretty_name(k)
            

            for j in range(num_vars):
                # print(f'params_keys: {k}')
                if any(token.lower() in k.lower() for token in ["ampl", "lorentzLB", "gaussLB", "SNR"]):
                    print(f'Key {k} needs to be nonnegative.')
                    args.distributions = "nonnegative"
                else:
                    print(f'Key {k} needs can be positive or negative.')
                    args.distributions = set_distributions
                
                # keep a 2D slice so findDistribution still sees one variable
                if num_vars>1:
                    met_name = names[j] if j < len(names) else f"met_{j}"
                else:
                    met_name = None
                base_key = pretty_name(k, met_name)

                vj = v[:, j:j+1]

                for i, f in enumerate(findDistribution(v=vj, ind=ind, args=args)):
                    key = base_key

                    distributions.update({
                        key: f.summary(Nbest=args.Nbest, method=args.selectionMetric).to_dict()
                    })

                    best.update({
                        key: f.get_best()
                    })

                    # base_name = f'{met_name}_{base_key}' if met_name else f'{base_key}'
                    plt_savefig(savedir + f'{key}.png', dpi=140)
                    plt_savefig(savedir + f'{key}.eps', dpi=140)
                    plt_close()

                    summary_path = os.path.join(
                        savedir, f'summary_of_{args.Nbest}_best_fits.json'
                    )
                    if os.path.isfile(summary_path):
                        with open(summary_path, 'r') as file:
                            existing = json.load(file)
                        existing.update(distributions)
                        distributions = existing
                    with open(summary_path, 'w') as file:
                        json.dump(distributions, file, indent=4, separators=(',', ': '))

                    best_path = os.path.join(savedir, 'parameter_distributions_best_fit.json')
                    if os.path.isfile(best_path):
                        with open(best_path, 'r') as file:
                            existing = json.load(file)
                        existing.update(best)
                        best = existing
                    with open(best_path, 'w') as file:
                        json.dump(best, file, indent=4, separators=(',', ': '))

                    print(f'Saved summary of fitting parameter distributions at: {summary_path}')
                    print(f'Saved fitting parameter distributions at: {best_path}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paramPath', type=str, help='Path to .mat file exported after Osprey fitting')
    parser.add_argument('--savedir', type=str, help='Should be a directory path')
    parser.add_argument('--selectionMetric', type=str, default='sumsquare_error', help='Options are: sumsquare_error, aic, bic, kl_div, ks_statistic, ks_pvalue')
    parser.add_argument('--Nbest', type=int, default=5, help='Sets the number of best performing fits to include in the results for each variable.')
    parser.add_argument('--commonDist', action="store_true", default=True, help='Restricts the analysis to 10 common distributions.')
    parser.add_argument('--comprehensiveDist', action="store_false", dest='commonDist', help='Evaluate the comprehensive list of available distributions.')
    parser.add_argument('--distributions', type=str, default=None, help='Specify specific distribuions to test for. Possible distributions can be found in the scipy.stats documentation.')
    parser.add_argument('--paramKeys', type=str, default=None, help='Can be used to select specific parameters for analysis.')
    parser.add_argument('--n', type=float, default=-1, help='Percentage of the sample size to test. Results will converge when the sample size is sufficiently large.')
    parser.add_argument('--nx', type=int, default=5, help='The number of times to resample the dataset and retest the distribution.')
    parser.add_argument('--timeout', type=float, default=30, help='Time limit (sec) for testing a given distribution. Used to cap runtimes.')
    parser.add_argument('--bootstrapping', type=str, default=False, help='Uses bootstrapping to test the distributions.')
    
    args = parser.parse_args()


    # Create the save directory
    os.makedirs(args.savedir, exist_ok=True)
    
    # Specifying a single file
    if os.path.isfile(args.paramPath):
        print('paramPath is file')
        assert os.path.splitext(args.paramPath)[-1].lower() in ['.mat']
        args.paramPath = [args.paramPath]
    elif os.path.isdir(args.paramPath):
        print('paramPath is folder')
        # args.paramPath = sorted(glob.glob(os.path.join(args.paramPath, '*fitting_parameters*.mat')))
        args.paramPath = [str(f) for f in sorted(Path(args.paramPath).rglob('fitting_parameters.mat'))]
        assert len(args.paramPath) > 0, f"No '*fitting_parameters*.mat' files found in: {args.paramPath}"

    # Specifying a list of files
    if isinstance(args.paramPath, str):
        args.paramPath = [path for path in args.paramPath.split(',')]
        
    if isinstance(args.bootstrapping, str):
        args.bootstrapping = args.bootstrapping.split(",")

    # Specifying a directory of files
    # if os.isdir(args.paramPath):
    #     args.paramPath = [path for path in os.listdir(args.paramPath) if os.path.splitext(path)[-1].lower() in ['.mat']]
    
    # Limit analysis to specific distributions
    if args.distributions:
        args.distributions = [dist for dist in args.distributions.split(',')]
        args.commonDist = False
        
    # Limit analysis to specific variables
    if not args.paramKeys:
        args.paramKeys = 'ampl,ph0,ph1,gaussLB,lorentzLB,freqShift,Cr_SNR,residual_water_ampl,global_freqShift'
        # args.paramKeys = 'ampl,ph0,ph1,gaussLB,lorentzLB,freqShift,Res,refShift,refFWHM,Cr_SNR,Cr_FWHM,residual_water_ampl,global_freqShift,relResA'
        # args.paramKeys = 'gaussLB'#,lorentzLB,freqShift,Res,refShift,refFWHM,Cr_SNR,Cr_FWHM,global_freqShift,relResA'
    args.paramKeys = [k for k in args.paramKeys.split(',')]
    
    # print(args.paramKeys)
    
    
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
