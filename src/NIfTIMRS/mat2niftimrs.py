import datetime
import json
import os
import pickle

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.io as io
from pathlib import Path

from src.aux import loadmat_as_dict

# from torch.nn import Module

__all__ = ['Mat2NIfTI_MRS']


class Mat2NIfTI_MRS():
    '''
    This class is used to convert the simulated data in .mat files from MRS-Sim to
    NIfTI-MRS format. 
    '''
    def __init__(self,
                 combine_complex: bool=True,
                 test_output: bool=True,
                 ):
        NIFTIOrient = './src/NIfTIMRS/NIFTIOrient_object.pkl'
        with open(NIFTIOrient,'rb') as file:
            self.currNiftiOrientation = pickle.load(file)

        self.combine_complex = combine_complex
        self.test = test_output


    def forward(self, 
                datapath: str,
                reshape: list=None,
                label: str=None
               ):

        # Load the simualted data
        with open(datapath, 'rb') as file:
            # print('datapath: ',datapath)
            data = loadmat_as_dict(file)
            specDataCmplx = data['spectra']
            header = data['header']

        if isinstance(label,type(None)) and specDataCmplx.shape[1]>>1:
            specDataCmplx = np.expand_dims(specDataCmplx[:,0,...], axis=1)
        elif isinstance(label,str) and specDataCmplx.shape[1]>=2:
            if "noise_free" in label.lower():
                specDataCmplx = np.expand_dims(specDataCmplx[:,1,...], axis=1)
            elif "filtered" in label.lower():
                specDataCmplx = np.expand_dims(specDataCmplx[:,2,...], axis=1)
                
        # Combine real and imaginary to store as complex-valued
        if self.combine_complex:
            # print("specDataCmplx.shape before: ",specDataCmplx.shape)
            specDataCmplx = specDataCmplx[...,0,:] + 1j*specDataCmplx[...,1,:]
            # print("specDataCmplx.shape after: ",specDataCmplx.shape)
            # specDataCmplx = np.squeeze(specDataCmplx,axis=-2)

        # Reshape data to fit into the container
        x, y = reshape if not isinstance(reshape, type(None)) else 1, 1
        z = int(specDataCmplx.shape[0] / (x*y))
        s = int(specDataCmplx.shape[-1])
        specDataCmplx = specDataCmplx.reshape(x,y,z,s)

        path, name = os.path.split(datapath)
        save_name = os.path.splitext(name)[0]
        if label: save_name = os.path.join(save_name, label)

        # Define the metadata
        json_full, meta_dict = self.set_metadata(name, header)

        # Write the NIfTI files
        self.write_NIfTIMRS(specDataCmplx, self.currNiftiOrientation, header, 
                            json_full, path, save_name, meta_dict)


    def write_NIfTIMRS(self, 
                       specDataCmplx, 
                       currNiftiOrientation, 
                       header, 
                       json_full,
                       save_path,
                       save_name,
                       meta_dict
                       ):
        # Write the NIfTI-MRS file
        newobj = nib.nifti2.Nifti2Image(specDataCmplx, currNiftiOrientation.Q44)

        # Write new header
        pixDim = newobj.header['pixdim']
        pixDim[4] = header['dwelltime']
        newobj.header['pixdim'] = pixDim

        # Set q_form >0
        newobj.header.set_qform(currNiftiOrientation.Q44)

        # Set version information 
        newobj.header['intent_name'] = b'MRS-Sim_v0_0'

        # Write extension with encode 44
        extension = nib.nifti1.Nifti1Extension(44, json_full.encode('UTF-8'))
        newobj.header.extensions.append(extension)

        # # From nii obj and write    
        nib.save(newobj, os.path.join(save_path, save_name + '.nii.gz'))

        if self.test:
            # Check the integrity of the stored data and header information
            self.test_output(save_path, save_name, specDataCmplx, json_full, meta_dict)


    def set_metadata(self, path: str, header):
        spectrometer_frequency_hz = header['carrier_frequency']
        nucleus_str = '1H'

        echo_time_s = header['TE'] / 1000
        repetition_time_s = 'NA'

        DeviceSerialNumber = 'NA'
        Manufacturer = 'MRS-Sim'
        ManufacturersModelName = 'NA'
        SoftwareVersions = 'NA'

        PatientDoB = 'NA'
        PatientName = 'NA'
        PatientPosition = 'HFS'
        PatientSex = 'NA'
        PatientWeight = 0

        conversion_method = 'Manual'
        conversion_time = datetime.datetime.now().isoformat(sep='T',timespec='milliseconds')
        original_file = [path]

        dim_dict = {}

        meta_dict = {**dim_dict,
                    'SpectrometerFrequency':[spectrometer_frequency_hz,],
                    'ResonantNucleus':[nucleus_str,],
                    'EchoTime':echo_time_s,
                    'RepetitionTime':repetition_time_s,             
                    'DeviceSerialNumber':DeviceSerialNumber,
                    'Manufacturer':Manufacturer,
                    'ManufacturersModelName':ManufacturersModelName,
                    'SoftwareVersions':SoftwareVersions,
                    'PatientDoB':PatientDoB,
                    'PatientName':PatientName,
                    'PatientPosition':PatientPosition,
                    'PatientSex':PatientSex,
                    'PatientWeight':PatientWeight,
                    'ConversionMethod':conversion_method,
                    'ConversionTime':conversion_time,
                    'OriginalFile':original_file}

        return json.dumps(meta_dict), meta_dict


    def test_output(self, save_path, save_name, specDataCmplx, json_full, meta_dict):
        # Write hdf5 to check against using original information
        base_name = os.path.join(save_path, save_name)
        with h5py.File(base_name +'.h5', 'w') as h5f:
            h5f.create_dataset(save_name+'_data', data=specDataCmplx)
            h5f.create_dataset(save_name+'_header_ext', data=json_full)

        # Load NIfTI file
        check_nifti = nib.load(base_name + '.nii.gz')

        # Load h5py file
        with h5py.File(base_name + '.h5','r') as h5f:
            check_hdf5 = h5f[save_name+'_data'][:]
            
        # Compare header and data from NIfTI and h5py
        assert np.allclose(check_nifti.get_fdata(dtype=np.complex64),specDataCmplx)
        assert np.allclose(check_nifti.get_fdata(dtype=np.complex64),check_hdf5)

        loaded_he_content = json.loads(check_nifti.header.extensions[0].get_content())

        assert loaded_he_content == meta_dict

        # Delete testing data
        os.remove(base_name +'.h5')
        del loaded_he_content
