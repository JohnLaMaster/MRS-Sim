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
from .convention_testing import test_nifti_mrs_conventions

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
                 ppm_reference: float=4.65,
                 expected_peak_ppm: float=None,
                 ):
        NIFTIOrient = './src/NIfTIMRS/NIFTIOrient_object.pkl'
        with open(NIFTIOrient,'rb') as file:
            self.currNiftiOrientation = pickle.load(file)

        self.combine_complex = combine_complex
        self.test = test_output
        # v2.0: the water-reference peak location used both to reference the
        # exported ppm axis and (optionally) to sanity-check where the
        # dominant simulated peak actually lands was previously hardcoded to
        # 4.65 in test_output(). Exposed here so callers whose dominant peak
        # isn't water at 4.65ppm (or who use a different water-ppm
        # convention) can override it; defaults preserve prior behavior.
        self.ppm_reference = ppm_reference
        self.expected_peak_ppm = expected_peak_ppm if expected_peak_ppm is not None else ppm_reference


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

        # BUGFIX (v2.0): loadmat_as_dict() calls scipy.io.loadmat(...,
        # squeeze_me=True), which silently removes the noisy/clean axis
        # entirely whenever it has size 1 -- i.e. whenever the dataset was
        # generated with noise=False. Confirmed directly: a dataset's
        # saved spectra shape (batch, noisy/clean, channels, length), e.g.
        # (2, 1, 2, 2048), loads back as (2, 2, 2048) in that case. Every
        # index below assumes axis 1 is noisy/clean, so this silently
        # shifted every subsequent axis (making axis 1 the channels axis
        # instead), corrupting the label-based branch selection and then
        # crashing the real/imaginary combining step further down (that
        # step's own 0/1 indexing is correct -- it targets the channels
        # axis on purpose; it was just operating on a misaligned array).
        # Restore the squeezed-away axis explicitly rather than changing
        # loadmat_as_dict()'s shared default behavior.
        if specDataCmplx.ndim == 3:
            specDataCmplx = np.expand_dims(specDataCmplx, axis=1)

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

            # BUGFIX (v2.0): MRS-Sim's internal ppm axis is built as
            # ppm = +f_Hz/sf + centerFreq (ascending with array index --
            # confirmed directly against src/aux/process_basis_functions.py
            # and empirically against real basis sets: NAA's known 2.0ppm
            # singlet lands at the index this formula predicts). The
            # NIfTI-MRS / Levitt convention (confirmed directly against the
            # spec text) is the mirror image: ppm = -f_Hz/f0 + ref. Fed the
            # raw FID as-is, a spec-compliant reader places that same NAA
            # peak at 7.29ppm instead of 2.0ppm. Confirmed empirically that
            # a pure left-right reorder (in either the time or frequency
            # domain) also "fixes" the peak location but reverses the FID's
            # decay direction (violates the spec's separate "increasing
            # time" requirement) -- conjugation is the only operation that
            # corrects the frequency-sign convention while leaving the
            # time-domain decay envelope exactly unchanged (|conj(fid)| ==
            # |fid| at every timepoint). Applied at export time only --
            # the internal simulation/plotting convention (and plot_mrs.py's
            # own display-only invert_xaxis() correction) is untouched.
            specDataCmplx = specDataCmplx.conj()

        # Reshape data to fit into the container
        x, y = reshape if not isinstance(reshape, type(None)) else 1, 1
        z = int(specDataCmplx.shape[0] / (x*y))
        s = int(specDataCmplx.shape[-1])
        specDataCmplx = specDataCmplx.reshape(x,y,z,s)

        path, name = os.path.split(datapath)
        save_name = os.path.splitext(name)[0]
        # BUGFIX (v2.0): os.path.join(save_name, label) built a *subdirectory*
        # path (e.g. "dataset_spectra_0/noise_free") rather than a sibling
        # filename -- nothing ever creates that subdirectory, so nib.save()
        # would raise FileNotFoundError the first time this (previously
        # unreachable -- see mainFcns.simulate()) branch actually ran. A
        # plain suffix keeps the noise-free export next to, and named after,
        # its corresponding simulated-data file.
        if label: save_name = f"{save_name}_{label}"

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
        # BUGFIX (v2.0): header['dwelltime'] does not exist anywhere in a
        # simulated dataset's saved header -- PhysicsModel.header (loaded
        # via aux.convertdict()) only ever has spectralwidth,
        # carrier_frequency, Ns, t, centerFreq, B0, TE,
        # basis_set_software, ppm (confirmed directly). This was an
        # unconditional KeyError crash for every simulated dataset, with
        # or without noise. dwelltime = 1/spectralwidth (the standard
        # relationship, also asserted directly in
        # aux/validate_basis_sets.py's own basis-set validation).
        dwelltime = header['dwelltime'] if 'dwelltime' in header else 1.0 / header['spectralwidth']
        pixDim[4] = dwelltime
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
        # BUGFIX (v2.0): the NIfTI-MRS spec (wtclarke/mrs_nifti_standard,
        # confirmed directly) requires RepetitionTime to be a *number*
        # (seconds) when present; it is optional and the spec explicitly
        # permits JSON `null` for an unknown value. The string 'NA' used
        # here was a genuine type violation for any strict NIfTI-MRS
        # consumer. TR is not tracked anywhere in MRS-Sim (see
        # docs/v2/architecture_v1_audit.md / progress_log.md) -- None
        # (-> JSON null) is the honest, spec-compliant representation of
        # "genuinely unknown", not a fabricated numeric value.
        repetition_time_s = None

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
        
        # ---- NIfTI-MRS format convention tests ----
        test_nifti_mrs_conventions(
            save_path,
            save_name,
            expected_peak_ppm=self.expected_peak_ppm,
            ppm_reference=self.ppm_reference,
            allow_unlocalised=True,   # True for simulations
        )

        # Delete testing data
        os.remove(base_name +'.h5')
        del loaded_he_content
