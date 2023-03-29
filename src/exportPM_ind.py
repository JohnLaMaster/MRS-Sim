import argparse
import json
import os
import sys

import scipy.io as io
from aux import _save, prepare

sys.path.append('../')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, help='Where to save the PM index')
    parser.add_argument('--config_file', type=str, default='./src/configurations/debug_new_init.json', help='Config file path for defining the PM')

    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    # Simulate
    _, _, _, _, _, ind, _, _ = prepare(args.config_file)
   
    path = os.path.join(args.savedir, 'physics_model_index.mat')
    io.savemat(path, do_compression=True, mdict={"ind": ind})
    print('Saved physics model index at: {}'.format(path))
