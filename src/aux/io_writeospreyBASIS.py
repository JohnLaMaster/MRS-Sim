import numpy as np
import scipy.io as io
import argparse
import os

from collections import OrderedDict


gamma = 42.577478518 # MHz/T

centerFreq = 4.65 # ppm - Water
bandwidths = [[2.95, 7866.7], [3.00, 8000.0], [3.05, 8133.3]]

flags = {}
flags['writtentostruct']    = 1
flags['gotparams']          = 1
flags['leftshifted']        = 0
flags['filtered']           = 0
flags['zeropadded']         = 0
flags['freqcorrected']      = 0
flags['phasecorrected']     = 0
flags['averaged']           = 1
flags['addedrcvrs']         = 1
flags['subtracted']         = 1
flags['writtentotext']      = 0
flags['downsampled']        = 0
flags['isISIS']             = 0
flags['freqranged']         = 1

dims = {}
dims['t']        = 1
dims['coils']    = 0
dims['averages'] = 0
dims['subspecs'] = 0
dims['extras']   = 0

def collect_filenames(directory: str):
    filenames, metabolites = [], []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            name, ext = os.path.splitext(filename)
            if 'mat' in ext: 
                metabolites.append(name)
                filenames.append(filename)
    return filenames, metabolites

def move_lip_mm_to_end(files):
    # Initialize two lists to store filenames containing "Lip" or "MM" and other filenames
    lip_mm_files = []
    other_files = []

    # Iterate through the sorted list of filenames
    for file in files:
        # Check if the filename contains "Lip" or "MM" (case-insensitive)
        if "lip" in file.lower() or "mm" in file.lower():
            lip_mm_files.append(file)  # Add to the list of Lip/MM filenames
        else:
            other_files.append(file)  # Add to the list of other filenames

    # Concatenate the two lists to move Lip/MM filenames to the end
    sorted_files = other_files + lip_mm_files

    return sorted_files, len(lip_mm_files)

def load_and_compile_mat_files(directory: str):
    # Get a list of all .mat files in the directory
    mat_files = [file for file in os.listdir(directory) if file.endswith('.mat')]
    mat_files, num_mm = move_lip_mm_to_end(mat_files.sort())  # Sort the files alphabetically
    
    # Initialize an empty list to store the fid matrices
    fid_matrices = []

    # Iterate through each .mat file
    for file in mat_files:
        # Load the .mat file
        mat_data = io.loadmat(os.path.join(directory, file), variable_names='exptDat')
        # Extract the fid variable
        fid = mat_data['fid']
        # Transpose the fid matrix
        fid = np.transpose(fid)
        # Append the fid matrix to the list
        fid_matrices.append(fid)

    # Find the maximum number of rows and columns among all fid matrices
    max_rows = max([fid.shape[0] for fid in fid_matrices])
    max_cols = max([fid.shape[1] for fid in fid_matrices])

    # Initialize an empty array to store the compiled matrix
    compiled_matrix = np.zeros((len(fid_matrices), max_rows, max_cols))

    # Iterate through each fid matrix and pad or truncate it to match the maximum size
    for i, fid in enumerate(fid_matrices):
        num_rows, num_cols = fid.shape
        compiled_matrix[i, :num_rows, :num_cols] = fid

    return compiled_matrix, num_mm [name for ]



def prepare_measured_MM(fids, t):
    MM = fids[-1,...] # 2D row matrix
    assert t.shape[-1]==MNM.shape[-1]
    mMM_params_path = '/home/john/Documents/Research/Workspace/generation/GE/BasisSets/mMM_parameters.mat'
    mMM_params = io.loadmat(mMM_params_path)
    mu = mMM_params['MM_mu'].t()
    names = mMM_params['names'] # ampl, lorentzian, fshift, gaussian
    indices = mMM_params['start']
    ind = [] 
    for i in mMM_params['indices']: ind.append(i['index'])
    params = OrderedDict({'ampl': None, 'lorentzian': None, 'fshift': None})#, 'gauss': None})
    for i, k in enumerate(params.keys()):
        params[k] = np.expand_dims(np.ndarray(mu[indices[i]:indices[i+1],:]).squeeze(), axis=-1)
    params['gaussian'] = np.expand_dims(np.ndarray(mu[indices[-1]]).squeeze(), axis=-1)
    
    MM *= params['ampl'] # Amplitdue scaling
    MM *= np.exp((-params['lorentzian'] - params['gaussian']*t)*t) # Lineshape
    MM *= np.exp(+1j*params['fshift']*t) # Frequency shift
    return MM.sum(axis=0, keepdims=True)
    



def unpack_naming(inputdir: str):
    # GE_PRESS_144ms_2.95T.7z
    # GE_MEGAPRESS_68ms_3.00T_GABA_A.7z
    # GE_HERMES_80ms_3.00T_GABA+GSH_A.7z
    # description = 'GE_PRESS_144ms_2.95T'
    description = os.path.basename(os.path.normpath(inputdir))
    vendor, sequence, te_label, b0_label = description.split("_", 4)
    if "_" in bo_label: 
        bo_label, met, edit = bo_label.split("_", 3)
    else:
        met = None
        edit = None
    te = int(te_label.strip("ms"))
    b0 = int(b0_label.strip("T"))
    if 'press' in sequence.lower():
        seq = 'PRESS'
    elif 'megapress' in sequence.lower():
        assert met
        seq = 'megapress'
    elif 'hermes' in sequence.lower():
        seq = 'HERMES'
    savename = vendor + "_" + sequence + "_" + te_label + "_" + bo_label
    if met: savename += "_{}_{}".format(met, edit)
    
    return vendor, sequence, seq, te_label, te, bo_label, b0, met, savename


def writeospreyBASIS(sw, basis_set, b0, seq, te, centerFreq, 
                     flags, num_mm, metabList, dims, savedir,
                     scale, savename):
    # Assume the MM/Lip nonsense has been taken care of for now
    BASIS = {}
    BASIS['spectralwidth'] = sw
    BASIS['dwelltime']     = 1 / sw
    BASIS['n']             = basis_set.shape[-1]
    BASIS['linewidth']     = 1
    BASIS['Bo']            = b0
    BASIS['seq']           = [seq]
    BASIS['te']            = te
    BASIS['centerFreq']    = centerFreq
    BASIS['t']             = t
    BASIS['flags']         = flags
    BASIS['nMM']           = num_mm+1
    BASIS['fids']          = basis_set.t()
    BASIS['name']          = metabList
    BASIS['dims']          = dims
    BASIS['nMets']         = len(metabList)
    BASIS['sz']            = [basis_set.shape[-1], len(metabList)]
    BASIS['scale']         = scale # I don't know
    savename = os.path.join(savedir,savename+"_{0:.1f}Hz".format(sw)))
    io.savemat(list_of_osprey_basis_files[-1], mdict=BASIS, do_compression=True)

    return list_of_osprey_basis_files

def osprey_main(args):
    # Define directory with basis set simulations
    input_dir = args.inputdir
    output_dir = args.savedir if not isinstance(args.savedir, type(None)) else args.inputdir
    list_of_osprey_basis_files = []
    
    # Unpack directory name into variables
    vendor, sequence, seq, te_label, te, b0_label, b0, met, savename = unpack_naming(input_dir)
    
    # load basis set files
    basis_set_names, metabList = collect_filenames(input_dir)
    basis_functions, num_mm = load_and_compile_mat_files(input_dir)
    
    # Define the bandwidth to establish dwelltime
    specLength = basis_functions.shape[-1]
    for b in bandwidths:
        if b0 in b[0]: bw = b[1]
    dt = 1/bw
    T = np.linspace(start=0, stop=dt*specLength, num=specLength)
    # carrierFreq = 
    
    # prepare the measured MM from Helge's work
    MMmeas = prepare_measured_MM(basis_functions, T)
    mm_bf = basis_functions[-1,...]
    basis_functions = basis_functions[0:-2,...]
    n,s,l = basis_functions.shape
    zeros = np.zeros([s,l])
    for i in range(mm_bf.shape[1]):
        # temp = zeros
        zeros[0,:] += mm_bf[i,:]
        basis_functions = np.concatenate(basis_functions, zeros, axis=0)
    
    zeros[0,:] += MMmeas
    basis_functions = np.concatenate(basis_functions, zeros, axis=0)

    for i in range(args.bw_subsample):
        basis_set = basis_functions
        sw = bw
        t = T
        if i>>0:
            sw = int(bw/(i+1))
            basis_set = basis_functions[:,:,0::(i+1)]
            t = np.expand_dims(T[0,0::(i+1)], axis=0)
            
        '''
        Need to interpolate and resample the spectra after downsampling
        '''
            
        # Assume the MM/Lip nonsense has been taken care of for now
        BASIS = {}
        BASIS['spectralwidth'] = sw
        BASIS['dwelltime']     = 1 / sw
        BASIS['n']             = basis_set.shape[-1]
        BASIS['linewidth']     = 1
        BASIS['Bo']            = b0
        BASIS['seq']           = [seq]
        BASIS['te']            = te
        BASIS['centerFreq']    = centerFreq
        BASIS['t']             = t
        BASIS['flags']         = flags
        BASIS['nMM']           = num_mm+1
        BASIS['fids']          = basis_set.t()
        BASIS['name']          = metabList
        BASIS['dims']          = dims
        BASIS['nMets']         = len(metabList)
        BASIS['sz']            = [basis_set.shape[-1], len(metabList)]
        BASIS['scale']         = 5.4184e+03 # I don't know
        list_of_osprey_basis_files.append(os.path.join(args.savedir,savename+"_{0:.1f}Hz".format(sw)))
        io.savemat(list_of_osprey_basis_files[-1], mdict=BASIS, do_compression=True)
    
    
    # rearrange into LCM format
    
    # Export
    return list_of_osprey_basis_files
    




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', type=str, default='./dataset/')
    parser.add_argument('--savedir', type=str, default=None)
    parser.add_argument('--name', type=str, default=10000)
    parser.add_argument('--bw_subsample', type=int, default=9)
    
    args = parser.parse_args()
    
    savedir = args.savedir
    save_file_list = []
    
    for directory in os.listdir(args.inputdir):
        if os.path.isdir(directory):
            inputdir = directory
            args.savedir = savedir if not isinstance(savedir, type(None)) else inputdir
            os.makedirs(args.savedir, exist_ok=True)
    
            save_file_list.append(osprey_main(args))
            
    save_file_list = save_file_list.extend()
    
    
    
    
    
    
'''
Osprey can do Osprey and LCM
FID-A can do jMRUI and LCM


MARSS simulations
exptDat.sf      :: 127.7324
exptDat.sw_h    :: 2000
exptDat.nspecC  :: 8192
exptDat.fid     :: 8192 x num_spins

dwelltime = 1 / exptDat.sw_h
# Bo = round(exptDat.sf / gamma * 100) / 100
centerFreq = 4.65

GE_PRESS_144ms_2.95T.7z
filename = 'GE_PRESS_144ms_2.95T'
vendor, sequence, te_label, bo_label = filename.split("_",4)
if "_" in bo_label: bo_label, MET = bo_label.split("_")
te = int(te_label.strip("ms"))
b0 = int(bo_label.strip("T"))
switch sequence
    case 'PRESS'
        seq = 'PRESS'
    case 'MEGAPRESS'
        assert MET
        seq = 'megapress'
    case 'HERMES'
        seq = 'HERMES'
savename = vendor + "_"

INSPECTOR
lcmBasis.data     :: 1xnum_metab cell
lcmBasis.sw_h     :: 2000
lcmBasis.sf       :: 127.7324
lcmBasis.ppmCalib :: 4.6500

lcmBasis.data{1:num_metab}{1:5} :: 'name'; 1.500; 0.0250; 8192x1 complex double; '';

Osprey - Unedited
BASIS.spectralwidth :: 2500
BASIS.dwelltime     :: 4.0000e-.04
BASIS.n             :: 2048
BASIS.linewidth     :: 1
BASIS.Bo            :: 3
BASIS.seq           :: 1x1 cell :: {'PRESS'}
BASIS.te            :: 30
BASIS.centerFreq    :: 3
BASIS.t             :: 1x2048 double
BASIS.flags         :: 1x1 struct
BASIS.flags.writtentostruct     :: 1
BASIS.flags.gotparams           :: 1
BASIS.flags.leftshifted         :: 0
BASIS.flags.filtered            :: 0
BASIS.flags.zeropadded          :: 0
BASIS.flags.freqcorrected       :: 0
BASIS.flags.phasecorrected      :: 0
BASIS.flags.averaged            :: 1
BASIS.flags.addedrcvrs          :: 1
BASIS.flags.subtracted          :: 1
BASIS.flags.writtentotext       :: 0
BASIS.flags.downsampled         :: 0
BASIS.flags.isISIS              :: 0
BASIS.flags.freqranged          :: 1
BASIS.nMM           :: 8
BASIS.fids          :: 2048xnum_metabs
BASIS.name          :: 1xnum_metabs cell
BASIS.dims          :: 1x1 struct
BASIS.nMets         :: num_metabs
BASIS.sz            :: [2048,num_metabs]
BASIS.scale         :: 5.4184e+03

Osprey - MEGA-PRESS GABA 68
BASIS.spectralwidth :: 4000
BASIS.dwelltime     :: 2.5000e-.04
BASIS.n             :: 8192
BASIS.linewidth     :: 2
BASIS.Bo            :: 3
BASIS.seq           :: 1x1 cell :: {'megapress'}
BASIS.te            :: 68
BASIS.centerFreq    :: 3
BASIS.t             :: 1x8192 double
BASIS.flags         :: 1x1 struct
BASIS.flags.writtentostruct     :: 1
BASIS.flags.gotparams           :: 1
BASIS.flags.leftshifted         :: 0
BASIS.flags.filtered            :: 0
BASIS.flags.zeropadded          :: 0
BASIS.flags.freqcorrected       :: 0
BASIS.flags.phasecorrected      :: 0
BASIS.flags.averaged            :: 1
BASIS.flags.addedrcvrs          :: 1
BASIS.flags.subtracted          :: 1
BASIS.flags.writtentotext       :: 0
BASIS.flags.downsampled         :: 0
BASIS.flags.isISIS              :: 0
BASIS.flags.freqranged          :: 1
BASIS.nMM           :: 8
BASIS.fids          :: 8192 x num_metabs x 4
BASIS.name          :: 1 x num_metabs cell
BASIS.dims          :: 1x1 struct
BASIS.dims.t        :: 1
BASIS.dims.coils    :: 0
BASIS.dims.averages :: 0
BASIS.dims.subspecs :: 0
BASIS.dims.extras   :: 0
BASIS.nMets         :: num_metabs
BASIS.sz            :: [8192,num_metabs,4]
BASIS.scale         :: 9.7295e+03

Osprey - HERMES
BASIS.spectralwidth :: 4000
BASIS.dwelltime     :: 2.5000e-.04
BASIS.n             :: 8192
BASIS.linewidth     :: 1
BASIS.Bo            :: 3
BASIS.seq           :: 1x1 cell :: {'HERMES'}
BASIS.te            :: 80
BASIS.centerFreq    :: 3
BASIS.t             :: 1x8192 double
BASIS.flags         :: 1x1 struct
BASIS.flags.writtentostruct     :: 1
BASIS.flags.gotparams           :: 1
BASIS.flags.leftshifted         :: 0
BASIS.flags.filtered            :: 0
BASIS.flags.zeropadded          :: 0
BASIS.flags.freqcorrected       :: 0
BASIS.flags.phasecorrected      :: 0
BASIS.flags.averaged            :: 1
BASIS.flags.addedrcvrs          :: 1
BASIS.flags.subtracted          :: 1
BASIS.flags.writtentotext       :: 0
BASIS.flags.downsampled         :: 0
BASIS.flags.isISIS              :: 0
BASIS.flags.freqranged          :: 1
BASIS.nMM           :: 8
BASIS.fids          :: 8192 x num_metabs x 7
BASIS.name          :: 1 x num_metabs cell
BASIS.dims          :: 1x1 struct
BASIS.dims.t        :: 1
BASIS.dims.coils    :: 0
BASIS.dims.averages :: 0
BASIS.dims.subspecs :: 0
BASIS.dims.extras   :: 0
BASIS.nMets         :: num_metabs
BASIS.sz            :: [8192,num_metabs,7]
BASIS.scale         :: 3.5851e+04
'''