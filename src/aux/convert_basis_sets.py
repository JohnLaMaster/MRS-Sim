import argparse
import json
import os
import re

import nibabel as nib
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

from collections import OrderedDict

from .aux import loadmat_as_dict, reorder_metabolite_struct, npfftshift, npifftshift
from .process_basis_functions import (gamma_nucleus, load_marss_mat, load_osprey_mat, 
                                      load_lcmodel_basis, load_raw_basis, load_nifti_mrs_basis, 
                                      load_fsl_mrs_basis_dir, mrscloud_correction, 
                                      build_header_fields, assign_fid, EXPORT_FORMATS, 
                                      visual_inspection, export_basis_set)


def main(config: dict):
    # Check if the nucleus was specified and if not, assume 1H
    # list of options are defined by gamma_nucleus above
    nuc = config.get('nucleus','1H')
    global gamma
    gamma = gamma_nucleus[nuc]

    seq    = config['pulse_sequence']
    vendor = config['vendor']
    te     = config['TE']
    
    global B0
    B0 = config['B0']
    global TE
    TE = config['TE']
    global DT 
    DT = config['dt']

    # # Load template
    header      = OrderedDict()
    metabolites = OrderedDict()

    header_set = False
    flip_spec = False

    # -- Process files / subdirectories in new_path --------------------------
    for filename in sorted(os.listdir(config['spin_path'])):
        filepath = os.path.join(config['spin_path'], filename)
        base, ext = os.path.splitext(filename)
        ext = ext.lower()

        # Handle .nii.gz double extension
        if filename.endswith('.nii.gz'):
            base = filename[:-7]
            ext  = '.nii.gz'

        print(f"Processing: {filename}")
               
        # ── MARSS & Osprey .mat (single-file, all metabolites) ────────────────────
        # Osprey .mat files contain a top-level 'BASIS' struct, whereas MARSS
        # .mat files contain 'exptDat'.  Try Osprey first if 'BASIS' is found.
        if ext == '.mat':
            raw_peek = io.loadmat(filepath, variable_names=['BASIS'])
            if 'BASIS' in raw_peek:
                hinfo, mets = load_osprey_mat(filepath)
            else:
                hinfo, mets = load_marss_mat(filepath, config['sim_config'])
            if not header_set:
                build_header_fields(header, hinfo, config, te_override=te)
                header_set = True
            for name, fid in mets.items():
                assign_fid(metabolites, name, fid)

        # -- LCModel .basis (multi-metabolite) --------------------------------
        elif ext == '.basis':
            hinfo, mets = load_lcmodel_basis(filepath)
            if not header_set:
                build_header_fields(header, hinfo, config, te_override=te)
                header_set = True
            for name, fid in mets.items():
                assign_fid(metabolites, name, fid)
            if config['MRSCloud']:
                metabolites = mrscloud_correction(metabolites)

        # -- LCModel .raw (single metabolite per file) ------------------------
        elif ext == '.raw':
            meta, fid = load_raw_basis(filepath)
            if fid.size == 0:
                print(f"  [warn] No FID data found in {filename}")
                continue
            if not header_set:
                # .raw files carry no sequence parameters; rely on config
                sw = config.get('spectralwidth',
                                1.0 / config.get('dt'))#calc_dt(fid)))
                hinfo = {
                    'sw': sw,
                    'sf': config.get('carrier_frequency', config['B0']*gamma),
                    'ns': fid.size,
                }
                build_header_fields(header, hinfo, config, te_override=te)
                header_set = True
            name = meta.get('ID', base).lower()
            assign_fid(metabolites, name, fid)
            flip_spec = True

        # -- NIfTI-MRS .nii / .nii.gz ----------------------------------------
        elif ext in ('.nii', '.nii.gz'):
            hinfo, mets = load_nifti_mrs_basis(filepath)
            if not header_set:
                build_header_fields(header, hinfo, config, te_override=te)
                header_set = True
            for name, fid in mets.items():
                assign_fid(metabolites, name, fid)

        # -- FSL-MRS JSON basis directory ------------------------------------
        # FSL-MRS outputs one .json file per metabolite into a folder.
        # If new_path contains an FSL-MRS basis folder as a subdirectory it
        # will be detected here automatically.  Alternatively, point new_path
        # directly at the FSL-MRS basis directory.
        elif os.path.isdir(filepath):
            try:
                hinfo, mets = load_fsl_mrs_basis_dir(filepath)
            except (FileNotFoundError, ValueError) as e:
                print(f"  [skip] {filename}/: {e}")
                continue
            if not header_set:
                # sf may be 0 if basis_hzperppm was absent; fall back to config
                if hinfo.get('sf', 0) == 0:
                    hinfo['sf'] = config.get('carrier_frequency', config['B0']*gamma)
                build_header_fields(header, hinfo, config, te_override=te)
                header_set = True
            for name, fid in mets.items():
                assign_fid(metabolites, name, fid)

        else:
            print(f"  [skip] Unsupported extension '{ext}'")


    # -- Visual Inspection ---------------------------------------------------
    # Visually inspect the basis functions to ensure they appear correctly
    # at least in terms of chemical shift and directionality
    visual_inspection(metabolites, header['ppm'], flip_spec, config['debug'])

    # -- Save ----------------------------------------------------------------
    # Reorder metabolites
    metabolites = reorder_metabolite_struct(metabolites)    

    seq    = config['pulse_sequence']
    vendor = config['vendor']
    te     = header['TE']
    
    if not config.get('save_name'):
        save_name = '{}_{}_{}_{}'.format(
            seq, te, vendor, round(header['spectralwidth']))
        if config.get('save_name_prefix'):
            save_name = '{}_{}'.format(config['save_name_prefix'],save_name)
        if config.get('save_name_suffix'):
            save_name = '{}_{}'.format(save_name,config['save_name_suffix'])
        save_name = '{}.mat'.format(save_name)
    else:
        save_name = config['save_name']

    # save_dir  = os.path.dirname(config['template_path'])
    save_dir  = os.path.dirname(__file__)
    save_dir  = os.path.join(save_dir,'basis_sets')
    if config.get('save_subdir') and not isinstance(config['save_subdir'], type(None)): 
        save_dir = os.path.join(save_dir, config['save_subdir'])
    save_dir  = os.path.dirname(save_dir)

    # -- Export to additional formats ----------------------------------------
    export_formats = config.get('export_formats', '')
    export_formats = [f.strip() for f in export_formats.split(',') if f.strip()]

    if export_formats:
        # Derive a clean base name (strip .mat if present) for single-file exports
        export_base = os.path.splitext(save_name)[0]
        export_root = os.path.join(save_dir, 'converted', export_base)

        print(f"\nExporting to {len(export_formats)} additional format(s):")
        for fmt in export_formats:
            fmt = fmt.strip()
            if fmt not in EXPORT_FORMATS:
                print(f"  [warn] Unknown export format '{fmt}'; skipping. "
                      f"Valid options: {', '.join(EXPORT_FORMATS)}")
                continue

            print(f"  → {fmt}")
            exported_paths = export_basis_set(
                metabolites=metabolites,
                header=header,
                config=config,
                export_format=fmt,
                output_dir=export_root,
                base_name=export_base,
            )

            # Early validation on the first output of multi-file formats.
            # If the first written file fails structural validation we abort
            # the rest of the export immediately so that a systematic error is
            # caught before writing hundreds of files.
            if len(exported_paths) > 1 and exported_paths:
                first = exported_paths[0]
                try:
                    from .validate_basis_sets import validate_single_file
                    ok, msg = validate_single_file(first, fmt)
                    if not ok:
                        print(f"\n  [ERROR] Early validation FAILED on first "
                              f"exported file: {first}")
                        print(f"  Reason: {msg}")
                        print(f"  Aborting export of remaining files in '{fmt}'.")
                        for p in exported_paths:
                            if os.path.isfile(p):
                                os.remove(p)
                        continue
                except ImportError:
                    pass   # Validator not yet installed; skip early check


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process MRS basis functions into simulation-ready .mat files.')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        metavar='PATH',
        type=str,
        default=None,#'./src/configurations/debug_new_init.json',
        help='Path to the JSON configuration file. Must include the arguments in this parser.')
    parser.add_argument('--spin_path', type=str, default='~/Documents/Repositories/MARSSCompiled/VERI_GE_PRESS_30ms/SummedSpins_for_MARSSinput')
    parser.add_argument('--sim_config', type=str, default=None, help='Basis set config file, if available.')
    parser.add_argument('--save_subdir', type=str, default=None, help='Specify a subdir inside the basis_sets dir for storing this compiled basis set.')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--save_name_prefix', type=str, default=None)
    parser.add_argument('--save_name_suffix', type=str, default=None)
    parser.add_argument('--pulse_sequence', type=str, default='unspecified_sequence')
    parser.add_argument('--vendor', type=str, default='unspecified_vendor')
    parser.add_argument('--centerFreq', type=float, default=3.65)
    parser.add_argument('--sim_software', type=str, default='MARSS')
    parser.add_argument('--TE', type=float, default=30)
    parser.add_argument('--B0', type=float, default=3)
    parser.add_argument('--dt', type=float, default=0.00025)
    parser.add_argument('--MRSCloud', action='store_true', default=False, help='Accounts for MRSCloud exporting basis sets in the spectral domain.')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument(
        '--export_formats',
        type=str,
        default=None,
        help=(
            'Comma-separated list of additional output formats to export after '
            'the internal MARSS .mat is written. '
            'Supported values: ' + ', '.join(EXPORT_FORMATS) + '. '
            'Example: --export_formats lcmodel_basis,fsl_mrs_json,nifti_mrs'
        ),
    )
    # parser.add_argument('--carrier_frequency', type=float, default=127.7)
    args = parser.parse_args()

    if not isinstance(args.config_file, type(None)):
        if not os.path.isfile(args.config_file):
            raise FileNotFoundError(
                f"Config file not found: {args.config_file}\n",
                f"Using command line arguments")
        args.config_file = args.config_file.split(",")
        if os.path.splitext(args.config_file)[1] != '.json':
            raise ValueError("Config file must have a .json extension")

        for config in args.config_file:
            with open(config,'r') as file:
                config = json.load(file)
                main(config)
    else: 
        main(vars(args))

