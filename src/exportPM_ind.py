import argparse
import json
import os
import sys

from main_fcns import _save, prepare

sys.path.append('../')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--name', type=str, default=None, help='Specify custom file name for saved index without a file extension')
    parser.add_argument('--config_file', type=str, default='./src/configurations/debug_new_init.json')#DL_PRESS_144_ge.json')

    args = parser.parse_args()

    # Create the save directory and construct the file name
    os.makedirs(args.savedir, exist_ok=True)
    if isinstance(args.name, type(None)):
        path = os.path.join(args.savedir,'mrs-sim_PM_ind.json')
    else:
        name, _ = os.path.splitext(args.name)
        path = os.path.join(args.savedir, name + '.json')

    # Define the index
    _, _, _, _, _, ind, _, _ = prepare(args.config_file)

    # Export the PM index
    with open(path, 'w+') as file:
        json.dump(ind, file, indent=4, separators=(',', ': '))
    print('Saved physics model index at: {}'.format(path))
