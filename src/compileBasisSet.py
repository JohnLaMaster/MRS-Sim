import argparse
import json
import os

import numpy as np
import scipy.io as io


def main(cfg_path: str):
    with open(cfg_path, 'r') as file:
        config = json.load(file)

    # Load configuration parameters
    seq = config['pulse_sequence']
    vendor = config['vendor']
    te = config['TE']

    # Load template file
    template_data = io.loadmat(config['template_path'])
    metabolites = template_data['metabolites']
    header = template_data['header']
    artifacts = template_data['artifacts']

    # Loop over files in the new_path directory
    for i, filename in enumerate(os.listdir(config['new_path'])):
        if filename.endswith('.mat'):
            filepath = os.path.join(config['new_path'], filename)
            exptDat = io.loadmat(filepath)['exptDat']#[0][0]
            if i == 0:
                sw = exptDat['sw_h']
                dt = 1 / exptDat['sw_h']
                header['spectralwidth'] = sw
                header['carrier_frequency'] = exptDat['sf']
                header['Ns'] = exptDat['nspecC']
                header['t'] = np.arange(0, dt * header['Ns'], dt)
                header['centerFreq'] = config['centerFreq']
                header['B0'] = config['B0']
                header['TE'] = te
                header['pulse_sequence'] = seq
                header['vendor'] = vendor
                header['ppm'] = (-0.5 * sw + np.arange(0, header['Ns']) * \
                                    sw / (header['Ns'] - 1)) + config['centerFreq']
            a, metab, b = os.path.split(filepath)
            metab = metab.lower()
            metabolites[metab]['fid'] = np.stack([np.squeeze(exptDat['fid'].real), 
                                                  np.squeeze(exptDat['fid'].imag)], 
                                                 axis=0)

    # Store edited spectra if needed
    if config['edit_off_path']:
        for met in config['metabs_off']:
            filepath = os.path.join(config['edit_off_path'], met + '.mat')
            exptDat = io.loadmat(filepath)['exptDat']#[0][0]
            a, metab, b = os.path.split(filepath)
            metab = metab.lower()
            metabolites[metab]['fid_OFF'] = np.stack([np.squeeze(exptDat['fid'].real), 
                                                      np.squeeze(exptDat['fid'].imag)], 
                                                     axis=0)

    # Save output file
    if not config['save_name']:
        save_name = '{}_{}_{}_{}.mat'.format(seq,te,vendor,header['spectralwidth'])
    else:
        save_name = config['save_name']

    path, _, _ = os.path.split(config['template_path'])
    save_path = os.path.join(path, save_name)
    io.savemat(save_path, mdict={'metabolites': metabolites, 'header': header, 'artifacts': artifacts})
    print('Saved the {} basis set at: {}'.format(save_name,path))
    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', '--cfg', metavar='cfg', type=str, default='./src/configurations/debug_new_init.json')#DL_PRESS_144_ge.json')

    args = parser.parse_args()

    # Confirm the config file exists and is a json file
    assert os.isfile(args.config_file)
    assert os.path.splitext(args.config_file)[1] in ['.json']

    main(args.config_file)
