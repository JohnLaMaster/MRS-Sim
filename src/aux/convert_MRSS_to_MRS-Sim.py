import torch
import numpy as np
import scipy.io as io
import argparse
import os

from .io_writelcmBASIS import *         # writelcmBASIS(in_data, outfile, vendor, SEQ)
from .io_writeospreyBASIS import *      # osprey_main
from aux.aux import Fourier_Transform, inv_Fourier_Transform
'''I need to refactor aux.py to be in the aux/ folder'''



def main(args):
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
        t = np.linspace(np.min(t), np.max(t), args.resolution)
            
        '''
        Need to interpolate and resample the spectra after downsampling
        '''
        ref = 0.5*sw / (b0 * gamma)
        ppm_orig = torch.linspace(-ref, ref, basis_set.shape[-1].data)
        ppm_new = torch.linspace(-ref, ref, args.resolution)
        basis_set = Fourier_Transform(torch.from_numpy(basis_set))
        chp = CubicHermiteMAkima(xaxis=ppm_orig, signal=basis_set)
        basis_set = inv_Fourier_Transform(chp.interp(xs=ppm_new)).numpy()
        
        
        
        scale = np.amax(basis_set[:,0,:]) # calculate the max real value of the entire basis set
        basis_set /= scale # Normalize the basis set
        list_of_osprey_basis_files.extend(
            writeospreyBASIS(sw=sw, basis_set=basis_set, b0=b0, seq=seq, te=te, centerFreq=centerFreq, 
                            flags=flags, num_mm=num_mm, metabList=metabList, dims=dims, savedir=savedir,
                            scale=scale, savename=savename, vendor=vendor)
        )
        writelcmBASIS(io.loadmat(list_of_osprey_basis_files[-1]), savename, vendor)

    
    
    # rearrange into LCM format
    
    # Export
    return list_of_osprey_basis_files, [vendor for _ in range(len(list_of_osprey_basis_files))]


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', type=str, default='./dataset/')
    parser.add_argument('--savedir', type=str, default=None)
    parser.add_argument('--name', type=str, default=10000)
    parser.add_argument('--bw_subsample', type=int, default=9)
    parser.add_argument('--resolution', type=int, default=8192)
    
    args = parser.parse_args()
    
    savedir = args.savedir
    
    for directory in os.listdir(args.inputdir):
        if os.path.isdir(directory):
            inputdir = directory
            args.savedir = savedir if not isinstance(savedir, type(None)) else inputdir
            os.makedirs(args.savedir, exist_ok=True)
    
    
            output_namae, vendors = osprey_main(args)
            save_file_list.extend(output_namae)
            save_vendor_list.extend(vendors)
            
    
    for filename, vend in zip(save_file_list, save_vendor_list):
        name, ext = os.path.splitext(os.path.basename(os.path.normpath(filename)))
        in_data = io.loadmat(filename)
        writelcmBASIS(in_data, name, vend, in_data['seq'])
    
    