'''
This file converts the data simulated for the 2024 Fitting Challenge from Matlab
to NIfTI-MRS. It exports the T1 anatomical images, B0 field map, a segmentation mask,
metabolite maps, metabolite FIDs, nuisance FIDs, and the final FIDs.

Dr. Brian Soher & Ing, John LaMaster, 2024

EXAMPLE USAGE:
    > python ./Downloads/save_nifti_mrs.py '/home/john/Downloads/example_all.mat'

DEPENDENCIES: 
    numpy   - install via conda or PyPI
    scipy   - install via conda or PyPI
    nibabel - install via conda or PyPI

    nifti_mrs - install via conda(-forge) or PyPI
        >conda install -c conda-forge nifti-mrs
        or 
        >pip install nifti-mrs
        
        NB. Source code and more information available at:
        https://github.com/wtclarke/nifti_mrs_tools
    
   
'''
# Python modules
import argparse, os, h5py
from datetime import datetime

# Third party modules
import numpy as np
import nibabel as nib
from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
from nifti_mrs.hdr_ext import Hdr_Ext



def save_nifti_mrs(data, fname, b0, dwell, nucleus='1H', affine=None, no_conj=False):
    """
    data    numpy.array, complex at least 4D (max 7) (nz, ny, nx, nt)
    fname   string, name for data file output
    b0      float, center frequency in MHz (ex. 127.8)
    dwell   float, spectral (4th dimension) dwell time in seconds
    nucleus keyword, string, default '1H' 
    affine  keyword, numpy.array, float, 4x4 orientation/position affine, default=None 
    no_conj keyword, bool, If true stops conjugation of data on creation, default=False
     
    Note. below we call the nifti-mrs.gen_nifti_mrs_hdr_ext() method rather than
      the nifti-mrs.gen_nifti_mrs() method so we can create our own header and 
      populate it with some standard and user-defined entries ourselves.      
     
    """
    
    # data array has to be at least 4D - reshape if needed
    while data.ndim < 4:
        tmp = list(data.shape)
        tmp.insert(data.ndim,1)
        data.shape = tmp
        
    if not np.iscomplex(data).any():
        data = data + 1j*0
    

    # We create our own header extension object here so we can
    # populate it with our own standard and user-defined values
    #
    # Header has two required arguments (frequency and nucleus)
    meta = Hdr_Ext(spec_frequency=b0, resonant_nucleus=nucleus)

    # (OPTIONAL) example of standard NIfTI-MRS header entries
    meta.set_standard_def('ConversionMethod', 'My Program name here')
    conversion_time = datetime.now().isoformat(sep='T', timespec='milliseconds')
    meta.set_standard_def('ConversionTime', conversion_time)

    # (OPTIONAL) example of user-defined header entries
    #    - Useful for knowing what data is in this file later
    meta.set_user_def('Comment', 
                      'comment line 1 \n comment line 2 \n',
                      'Useful description about the saved data.')

    # Define a 4x4 affine array or use an existing one. Setting affine=None 
    # will force a default affine to be created with FOV vals of 10000.
    #
    # Here I create a default affine as an example
    
    affine = np.diag(np.array([1.0000, 1.0000, 1.0000, 1.0000]))
    nifti_orientation = NIFTIOrient(affine)

    no_conj = True #if not np.iscomplex(data).any() else False # set empirically whether conj() should be applied to data before save
    
    img = gen_nifti_mrs_hdr_ext(data, dwell, meta, affine=nifti_orientation.Q44, no_conj=no_conj)
    img.save(fname)


def save_nifti(data, fname, affine=None):
    """
    Save a 3D matrix as a NIfTI file.
    
    Parameters:
    data   : numpy.array, 3D matrix (nx, ny, nz)
    fname  : str, name for data file output
    affine : numpy.array, float, 4x4 orientation/position affine, default=None
    """
    # Ensure data is 3D
    if len(data.shape) != 3:
        raise ValueError("Data must be a 3D matrix")

    # Define a default 4x4 affine array if not provided
    if affine is None:
        affine = np.eye(4)

    # Create NIfTI image
    nifti_img = nib.Nifti1Image(data, affine)

    # Save NIfTI image to file
    nib.save(nifti_img, fname+'.nii.gz')



#------------------------------------------------------------------------------
# The following code is from spec2nii 
#
#  Author: William Clarke <william.clarke@ndcn.ox.ac.uk>
#  Copyright (C) 2020 University of Oxford

class NIFTIOrient:
    def __init__(self, affine):
        self.Q44 = affine
        qb, qc, qd, qx, qy, qz, dx, dy, dz, qfac = nifti_mat44_to_quatern(affine)
        self.qb = qb
        self.qc = qc
        self.qd = qd
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.qfac = qfac


def nifti_mat44_to_quatern(R):
    """4x4 affine to quaternion representation."""
    # offset outputs are read out of input matrix
    qx = R[0, 3]
    qy = R[1, 3]
    qz = R[2, 3]

    # load 3x3 matrix into local variables
    r11 = R[0, 0]
    r12 = R[0, 1]
    r13 = R[0, 2]
    r21 = R[1, 0]
    r22 = R[1, 1]
    r23 = R[1, 2]
    r31 = R[2, 0]
    r32 = R[2, 1]
    r33 = R[2, 2]

    # compute lengths of each column; these determine grid spacings
    xd = np.sqrt(r11 * r11 + r21 * r21 + r31 * r31)
    yd = np.sqrt(r12 * r12 + r22 * r22 + r32 * r32)
    zd = np.sqrt(r13 * r13 + r23 * r23 + r33 * r33)

    # if a column length is zero, patch the trouble
    if xd == 0.0:
        r11 = 1.0
        r21 = 0.0
        r31 = 0.0
        xd = 1.0
    if yd == 0.0:
        r22 = 1.0
        r12 = 0.0
        r32 = 0.0
        yd = 1.0
    if zd == 0.0:
        r33 = 1.0
        r13 = 0.0
        r23 = 0.0
        zd = 1.0

    # assign the output lengths
    dx = xd
    dy = yd
    dz = zd

    # normalize the columns
    r11 /= xd
    r21 /= xd
    r31 /= xd
    r12 /= yd
    r22 /= yd
    r32 /= yd
    r13 /= zd
    r23 /= zd
    r33 /= zd

    zd = r11 * r22 * r33\
        - r11 * r32 * r23\
        - r21 * r12 * r33\
        + r21 * r32 * r13\
        + r31 * r12 * r23\
        - r31 * r22 * r13
    # zd should be -1 or 1

    if zd > 0:  # proper
        qfac = 1.0
    else:  # improper ==> flip 3rd column
        qfac = -1.0
        r13 *= -1.0
        r23 *= -1.0
        r33 *= -1.0

    # now, compute quaternion parameters
    a = r11 + r22 + r33 + 1.0
    if a > 0.5:  # simplest case
        a = 0.5 * np.sqrt(a)
        b = 0.25 * (r32 - r23) / a
        c = 0.25 * (r13 - r31) / a
        d = 0.25 * (r21 - r12) / a
    else:  # trickier case
        xd = 1.0 + r11 - (r22 + r33)  # 4*b*b
        yd = 1.0 + r22 - (r11 + r33)  # 4*c*c
        zd = 1.0 + r33 - (r11 + r22)  # 4*d*d
        if xd > 1.0:
            b = 0.5 * np.sqrt(xd)
            c = 0.25 * (r12 + r21) / b
            d = 0.25 * (r13 + r31) / b
            a = 0.25 * (r32 - r23) / b
        elif yd > 1.0:
            c = 0.5 * np.sqrt(yd)
            b = 0.25 * (r12 + r21) / c
            d = 0.25 * (r23 + r32) / c
            a = 0.25 * (r13 - r31) / c
        else:
            d = 0.5 * np.sqrt(zd)
            b = 0.25 * (r13 + r31) / d
            c = 0.25 * (r23 + r32) / d
            a = 0.25 * (r21 - r12) / d

        if a < 0.0:
            b = -b
            c = -c
            d = -d

    qb = b
    qc = c
    qd = d
    return qb, qc, qd, qx, qy, qz, dx, dy, dz, qfac	


def load_mat_file(file_path):
    """
    Loads a .mat file and converts all variables stored as arrays to NumPy arrays.
    
    Parameters:
    file_path (str): The path to the .mat file.
    
    Returns:
    dict: A dictionary where the keys are variable names and the values are NumPy arrays.
    """
    data = {}
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    
    # Open the .mat file
    with h5py.File(file_path, 'r') as mat_file:
        # Iterate over each item in the file
        for key in mat_file.keys():
            # Get the dataset
            item = mat_file[key]
            # Check if the item is a dataset
            if isinstance(item, h5py.Dataset):
                data[key] = convert_to_numpy(item)
            elif isinstance(item, h5py.Group):
                # If it's a group, recursively load the data
                data[key] = load_group(item)
    
    return data


def load_group(group):
    """
    Recursively loads a group from the .mat file and converts all datasets to NumPy arrays.
    
    Parameters:
    group (h5py.Group): The HDF5 group to load.
    
    Returns:
    dict: A dictionary where the keys are variable names and the values are NumPy arrays or sub-dictionaries.
    """
    data = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            data[key] = convert_to_numpy(item)
        elif isinstance(item, h5py.Group):
            data[key] = load_group(item)
    
    return data


def convert_to_numpy(item):
    """
    Converts an HDF5 dataset to a NumPy array, handling structured arrays with complex numbers.
    
    Parameters:
    item (h5py.Dataset): The HDF5 dataset to convert.
    
    Returns:
    np.ndarray: The converted NumPy array.
    """
    array_data = np.array(item)
    
    if array_data.dtype.names is not None:
        # Handle structured arrays with fields like 'real' and 'imag'
        if 'real' in array_data.dtype.names and 'imag' in array_data.dtype.names:
            array_data = array_data['real'] + 1j * array_data['imag']
            
    # Handle the conversion from MATLAB's column-major order to NumPy's row-major order
    if array_data.ndim == 2:
        array_data = array_data.T
    elif array_data.ndim == 3:
        array_data = array_data.transpose(2, 1, 0)
    elif array_data.ndim == 4:
        array_data = array_data.transpose(3, 2, 1, 0)
    
    return array_data
	
def get_absolute_paths(directory):
    """
    Get the absolute paths of all files in a directory.

    Parameters:
    directory (str): The path to the directory.

    Returns:
    list: A list of absolute file paths.
    """
    absolute_paths = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            absolute_paths.append(os.path.abspath(os.path.join(dirpath, filename)))
    return absolute_paths
 
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', type=str, help='Path to the example file or folder.')
    parser.add_argument('--data', type=str, default='other', help='Used to differentiate example data from the other samples')
    parser.add_argument('--metmaps', action='store_true', default=False, help='Flag to export metabolite maps')
    parser.add_argument('--parentdir', action='store_true', default=False, help='Flag a parent directory containing the matfiles instead of individual file paths')
    args = parser.parse_args()
    

    if args.parentdir:
        import pathlib
        desktop = pathlib.Path(args.pth)
        lst = ''
        for item in desktop.rglob('*_all.mat'):
            item = str(item)
            if item.endswith('_all.mat'):
                if lst=='': lst = item
                else: lst += ','+item
        args.pth = lst
        
    args.pth = [pth for pth in args.pth.split(',')]
    # print(args.pth)
    if len(args.pth)==1:
        basedir = args.pth[0]
        if os.path.isdir(args.pth[0]):
            args.pth = get_absolute_paths(args.pth[0])
    else:
        for x in args.pth: print(x)
        basedir = os.path.commonpath(args.pth)
    
    for pth in args.pth:
        # assert(os.path.exists(pth))
        if pth.endswith('_all.mat'):
            print('pth: ',pth)
            base, _ = os.path.splitext(pth)
            savedir = basedir # os.path.join(basedir, base)
#             os.makedirs(savedir, exist_ok=True)
            
            # print('pth: ',pth) 
            file = load_mat_file(pth)
            
            # Time and frequency parameters
            t = file['t'].flatten()
            dim = file['xtAll'].shape # file['xtMeta'].shape
            dwell = t[1] - t[0]
            b0 = 123.23 

            fieldnames = ['Iref', 'brainMask', 'B0map']
            output_names = [base+'_'+x for x in ['mri_t1w_mpr', 'mri_brain_mask', 'mri_b0_map']]
            for i, (name, fname) in enumerate(zip(fieldnames, output_names)):
                img = np.flip(np.transpose(np.flip(file[name],axis=2), axes=[1, 0, 2]), axis=1)
                save_nifti(img, os.path.join(savedir,fname))

            fieldnames = ['xtMeta', 'xtNuisance', 'xtAll']
            output_names = [base+x for x in ['_mrs_fids_metabolites', '_mrs_fids_nuisance', '_mrs_fids_si_data']]
            placeholder = 2 if args.data=='test' else 0
            for i, (name, fname)  in enumerate(zip(fieldnames, output_names)):
                if i>=placeholder:
                    img = np.flip(np.transpose(np.flip(file[name],axis=2), axes=[1, 0, 2, 3]), axis=1)
                    img = np.ifft(np.fft.ifftshift(np.flip(np.fft.fftshift(np.fft(img, axis=-1), axis=-1), axis=-1), axis=-1), axis=-1)
                    img = np.flip(img, axis=-1) # flip the FID to decay left to right
                    save_nifti_mrs(img, os.path.join(savedir,fname), b0, dwell)
                    
            if args.metmaps:        
                # Metabolite maps
                # 'Asp';'GABA';'Gln';'Glu';'Lac';'NAA';'NAAG';'PE';'Tau';'mIns';'sIno';'tCho';'tCr';'tCr39'
                if not args.data=='example': mets = ['Asp', 'GABA', 'Gln', 'Glu', 'Lac', 'Lac41', 'NAA', 'NAAG', 'PE', 'PE40', 'Tau', 'mIns', 'sIno', 'tCho', 'tCr', 'tCr39']; # All other data
                else: mets = ['Asp', 'GABA', 'Gln', 'Glu', 'Lac', 'NAA', 'NAAG', 'PE', 'Tau', 'mIns', 'sIno', 'tCho', 'tCr', 'tCr39']; # Example data
                fieldnames = 'metaMap'
                output_names = base+'_mrs_metab_result_'
                for i, met  in enumerate(mets):
                    img = np.flip(np.transpose(np.flip(file[fieldnames][...,i],axis=2), axes=[1, 0, 2]), axis=1)
                    save_nifti(img, os.path.join(savedir,output_names+mets[i].lower()))

#/mnt/c/Users/LaMaster/Downloads/testing_data\testing_data\TestSub1\TestSub1_all.mat
#python3 /mnt/c/Users/LaMaster/Downloads/save_nifti_mrs.py '/mnt/c/Users/LaMaster/Downloads/testing_data/testing_data/TestSub1/TestSub1_all.mat,/mnt/c/Users/LaMaster/Downloads/testing_data/testing_data/TestSub2/TestSub2_all.mat,/mnt/c/Users/LaMaster/Downloads/testing_data/testing_data/TestSub3/TestSub3_all.mat,/mnt/c/Users/LaMaster/Downloads/testing_data/testing_data/TestSub4/TestSub4_all.mat,/mnt/c/Users/LaMaster/Downloads/testing_data/testing_data/TestSub5/TestSub5_all.mat'
#python3 C:Users\LaMaster\Downloads\save_nifti_mrs.py 'C:Users\LaMaster\Downloads\testing_data\testing_data\TestSub1\TestSub1_all.mat,C:Users\LaMaster\Downloads\testing_data\testing_data\TestSub2\TestSub2_all.mat,C:Users\LaMaster\Downloads\testing_data\testing_data\TestSub3\TestSub3_all.mat,C:Users\LaMaster\Downloads\testing_data\testing_data\TestSub4\TestSub4_all.mat,C:\Users\LaMaster\Downloads\testing_data\testing_data\TestSub5\TestSub5_all.mat'

# Updated commands:
# python .\save_nifti_mrs.py --pth 'C:\Users\LaMaster\Downloads\example_data\example_data\example_all.mat' --data 'example' --metmaps
# python .\save_nifti_mrs.py 'C:\Users\LaMaster\Downloads\testing_data\TestSub1\TestSub1_all.mat,C:\Users\LaMaster\Downloads\testing_data\TestSub2\TestSub2_all.mat,C:\Users\LaMaster\Downloads\testing_data\TestSub3\TestSub3_all.mat,C:\Users\LaMaster\Downloads\testing_data\TestSub4\TestSub4_all.mat,C:\Users\LaMaster\Downloads\testing_data\TestSub5\TestSub5_all.mat' --data 'test' 
# python .\save_nifti_mrs.py 'C:\Users\LaMaster\Downloads\contest_data\' --data 'contest' --metmaps --parentdir


