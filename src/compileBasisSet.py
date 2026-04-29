import argparse
import json
import os

import numpy as np
import scipy.io as io

import re
import copy
import matplotlib.pyplot as plt



def main(args: str):
    # Load template file
    template_data = loadmat_as_dict(args.template)
    metabolites = template_data['metabolites']
    header = template_data['header']
    del template_data
    
    config = loadmat_as_dict(args.marssinput)
    header['B0'] = config['B0']
    header['spectralwidth'] = config['bw']
    header['Ns'] = config['Npoints']
    try:
        header['centerFreq'] = config['referencePeak']
    except KeyError:
        header['centerFreq'] = 4.65

    header['TE'] = int(np.round(1e3*(np.sum(config['delays']) + np.sum(config['durations']) - 0.5*config['durations'][0])))
    header.update(pulseSequence=args.pulse_sequence)
    header['vendor'] = args.vendor

    # Loop over files in the new_path directory
    for i, filename in enumerate(os.listdir(args.spin_path)):
        if filename.endswith('.mat'):
            filepath = os.path.join(args.spin_path, filename)
            exptDat = loadmat_as_dict(filepath)['exptDat']
            if i == 0:
                sw = exptDat['sw_h']
                dt = 1 / exptDat['sw_h']
                header['carrier_frequency'] = exptDat['sf']
                header['t'] = np.arange(0, dt * header['Ns'], dt)
                header['ppm'] = np.expand_dims(np.linspace(
                    start=(-0.5 * sw) / exptDat['sf'], 
                    stop=(0.5 * sw) / exptDat['sf'], 
                    num=int(header['Ns'])
                ) + header['centerFreq'], axis=0)
            else:
                assert sw == exptDat['sw_h']
                assert header['carrier_frequency'] == exptDat['sf']
                assert header['Ns'] == exptDat['nspecC']
            

            _, metab = os.path.split(filepath)
            metab, _ = os.path.splitext(metab)
            metab = metab.lower()
            try:
                metabolites[metab]['fid'] = np.expand_dims(np.stack(
                    [np.squeeze(exptDat['fid'].real), 
                    np.squeeze(exptDat['fid'].imag)], 
                    axis=0
                ), axis=0)
            except KeyError:
                metabolites.update({metab: {
                    'fid': np.expand_dims(np.stack(
                    [np.squeeze(exptDat['fid'].real), 
                    np.squeeze(exptDat['fid'].imag)], 
                    axis=0
                ), axis=0),
                'min':0.0,
                'max':1.0}
                })
                
    metabolites = reorder_metabolite_struct(metabolites)

    # # Store edited spectra if needed
    # if config['edit_off_path']:
    #     for met in config['metabs_off']:
    #         filepath = os.path.join(config['edit_off_path'], met + '.mat')
    #         exptDat = io.loadmat(filepath)['exptDat']#[0][0]
    #         a, metab, b = os.path.split(filepath)
    #         metab = metab.lower()
    #         metabolites[metab]['fid_OFF'] = np.stack([np.squeeze(exptDat['fid'].real), 
    #                                                   np.squeeze(exptDat['fid'].imag)], 
    #                                                  axis=0)

    # Visually inspect the basis functions to ensure they appear correctly
    # at least in terms of chemical shift and directionality
    visual_inspection(metabolites, header['ppm'])

    # Load configuration parameters
    seq = args.pulse_sequence
    vendor = args.vendor
    te = header['TE']
    
    # Save output file
    if not args.save_name:
        save_name = '{}_{}ms_{}_{}'.format(seq,te,vendor,header['spectralwidth'])
        if args.save_name_prefix: save_name = '{}_{}'.format(args.save_name_prefix, save_name)
        if args.save_name_suffix: save_name = '{}_{}'.format(save_name, args.save_name_suffix)
        save_name = '{}.mat'.format(save_name)
    else:
        save_name = args.save_name

    path = os.path.join(os.getcwd(),'src/basis_sets')
    save_path = os.path.join(path, save_name)
    io.savemat(save_path, mdict={'metabolites': metabolites, 'header': header})#, 'artifacts': artifacts})
    print('Saved the {} basis set at: {}'.format(save_name,path))
    

def visual_inspection(metabolites, ppm):
    # --- Extract field names ---
    fn = list(metabolites.keys())
    num = len(fn)

    # --- Collect fid arrays ---
    tmp = [metabolites[f]['fid'] for f in fn]
    shape = metabolites[fn[0]]['fid'].shape

    # --- Try stacking directly ---
    try:
        stacked = np.concatenate(tmp, axis=0)
    except Exception:
        stacked = np.zeros((num, 2, shape[-1]), dtype=np.float64)

        for k, f in enumerate(fn):
            tmp = [np.reshape(metabolites[f]['fid'], (1, 2, shape[-1])) for f in fn]
            stacked = np.concatenate(tmp, axis=0)

    legend = [f for f in fn]

    # --- Convert to complex ---
    fid = stacked[:, 0, :] + 1j * stacked[:, 1, :]

    # --- FFT ---
    spec = np.fft.fftshift(np.fft.fft(fid, axis=1), axes=1)

    # =========================================================
    # Plot 1 (uses ppm directly)
    # =========================================================
    plt.figure()
    for i in range(num):
        plt.plot(ppm.squeeze(), np.real(spec[i, :]))
    plt.xlim([0, 5])
    plt.gca().invert_xaxis()
    plt.legend(legend)
    plt.show()

    # =========================================================
    # Plot 2 (reconstructed ppm axis like MATLAB colon)
    # =========================================================
    ppm_lin = np.linspace(np.min(ppm), np.max(ppm), shape[-1])

    plt.figure()
    for i in range(num):
        plt.plot(ppm_lin, np.real(spec[i, :]))
    plt.show()


def loadmat_as_dict(filename):
    """
    Load a MATLAB .mat file and recursively convert structs to Python dicts.
    """
    def _check_keys(d):
        for key in d:
            if isinstance(d[key], io.matlab.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _toarray(d[key])
        return d

    def _todict(matobj):
        d = {}
        for field in matobj._fieldnames:
            elem = getattr(matobj, field)
            if isinstance(elem, io.matlab.mat_struct):
                d[field] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[field] = _toarray(elem)
            else:
                d[field] = elem
        return d

    def _toarray(ndarray):
        # Convert arrays that may contain structs
        if ndarray.dtype == object:
            return [_todict(el) if isinstance(el, io.matlab.mat_struct)
                    else el for el in ndarray]
        else:
            return ndarray

    data = io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def reorder_metabolite_struct(S):
    """
    Recursively reorder a MATLAB-like struct (Python dict).

    Order:
    1. Alphabetical (non-MM, non-Lip)
    2. MM* sorted numerically by suffix
    3. Lip* sorted numerically by suffix
    """

    # --- Base cases ---
    if isinstance(S, list):
        return [reorder_metabolite_struct(x) for x in S]

    if not isinstance(S, dict):
        return S

    fn = list(S.keys())
    fn_lower = [f.lower() for f in fn]

    is_mm  = [f.startswith('mm') for f in fn_lower]
    is_lip = [f.startswith('lip') for f in fn_lower]
    is_other = [not (mm or lip) for mm, lip in zip(is_mm, is_lip)]

    # --- Split groups ---
    other_fields = sorted([f for f, keep in zip(fn, is_other) if keep])
    mm_fields    = sort_special_fields([f for f, keep in zip(fn, is_mm) if keep], 'mm')
    lip_fields   = sort_special_fields([f for f, keep in zip(fn, is_lip) if keep], 'lip')

    new_order = other_fields + mm_fields + lip_fields

    # --- Rebuild ordered dict ---
    S_ordered = {}
    for f in new_order:
        val = S[f]
        if isinstance(val, (dict, list)):
            val = reorder_metabolite_struct(val)
        S_ordered[f] = val

    return S_ordered


def sort_special_fields(fields, prefix):
    """
    Sort fields like:
        MM09, MM092, MM12, MM121

    Correctly distinguishes:
        MM09 ≠ MM092  (no prefix confusion)

    Sorting rule:
        1. numeric suffix (09 -> 9, 092 -> 92)
        2. then full string (stable tie-breaker)
    """

    if not fields:
        return fields

    prefix = prefix.lower()

    def extract_number(name):
        # strict match: prefix + digits ONLY at start
        match = re.match(rf'^{prefix}(\d+)$', name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return float('inf')  # non-matching go last

    # sort by numeric value, then by full string to avoid ambiguity
    return sorted(fields, key=lambda x: (extract_number(x), x))


if __name__=='__main__':
    # *NOTE*: Needs to be run from the MRS-Sim directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--spin_path', type=str, default='~/Documents/Repositories/MARSSCompiled/VERI_GE_PRESS_30ms/SummedSpins_for_MARSSinput')
    parser.add_argument('--marssinput', type=str, default='~/Documents/Repositories/MARSSCompiled/MARSSInput.mat')
    parser.add_argument('--template', type=str, default='~/Documents/Repositories/MRS-Sim/src/basis_sets/template.mat')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--save_name_prefix', type=str, default=None)
    parser.add_argument('--save_name_suffix', type=str, default=None)
    parser.add_argument('--pulse_sequence', type=str, default='unspecified_sequence')
    parser.add_argument('--vendor', type=str, default='unspecified_vendor')

    args = parser.parse_args()

    # Run checks on the input arguments
    # Expand ~ to full path
    spin_path = os.path.expanduser(args.spin_path)
    marssinput = os.path.expanduser(args.marssinput)
    template = os.path.expanduser(args.template)

    # --- spin_path: must be a directory ---
    assert os.path.isdir(spin_path), f"spin_path is not a valid directory: {spin_path}"

    # --- marssinput: must be .mat file ---
    assert os.path.isfile(marssinput), f"marssinput file not found: {marssinput}"
    assert os.path.splitext(marssinput)[1].lower() == '.mat', \
        f"marssinput must be a .mat file: {marssinput}"

    # --- template: must be .mat file ---
    assert os.path.isfile(template), f"template file not found: {template}"
    assert os.path.splitext(template)[1].lower() == '.mat', \
        f"template must be a .mat file: {template}"
    
    main(args)

'''
Usage
python ~/Documents/Repositories/MRS-Sim/src/compileBasisSet.py 
--spin_path '~/Documents/Repositories/MARSSCompiled/VERI_GE_PRESS_30ms/SummedSpins_for_MARSSinput' 
--marssinput '~/Documents/Repositories/MARSSCompiled/MARSSInput.mat' 
--template '~/Documents/Repositories/MRS-Sim/src/basis_sets/template.mat' 
--save_name_prefix 'VERI' 
--save_name_suffix 'wMM'
--pulse_sequence 'PRESS' 
--vendor 'GE'
'''
