import numpy as np
import scipy.io as io
import argparse
import os

from listValidBasisFunctionNames import *


def main(args):
    # Define directory with basis set simulations
    
    # load basis set files
    
    # rearrange into LCM format
    
    # Export
    vendor = 'GE'
    for filename in directory:
        name, ext = os.path.splitext(os.path.basename(os.path.normpath(filename)))
        in_data = io.loadmat(filename)
        writelcmBASIS(in_data, name, vendor, in_data['seq'])
    
## io_writelcmBASIS
#   This function creates a LCM usable .BASIS file from an Osprey Basisset.
#   MMs are not included in the output
#
#
#   USAGE:
#       RF=io_writelcmBASIS(in,outfile,vendor,resample);
#
#   INPUT:      in      = Osprey BASIS file
#               outfile = path and name of the LCModel .BASIS file
#               vendor  = String with Vendor name for consistent naming 
#               SEQ     = name of the sequence
#
#   OUTPUT:     RF is unused, but .BASIS file is created
#
#
#   AUTHORS:
#       Dr. Helge Zoellner (Johns Hopkins University, 2020-01-16)
#       hzoelln2@jhmi.edu
#
#   CREDITS:
#       This code is based on numerous functions from the FID-A toolbox by
#       Dr. Jamie Near (McGill University)
#       https://github.com/CIC-methods/FID-A
#       Simpson et al., Magn Reson Med 77:23-33 (2017)
#
#   HISTORY:
#       2020-02-11: First version of the code.
def writelcmBASIS(in_data, outfile, vendor):
    SEQ = in_data['seq']
    metabList = fit_createMetabList(['full'])

    # Add basis spectra (if they were removed to reduce the file size)
    if 'specs' not in in_data:
        in_data = osp_recalculate_basis_specs(in_data)

    # Create the modified basis set without macro molecules 
    basisSet = fit_selectMetabs(in_data, metabList, 0)

    Bo = basisSet['Bo']
    HZPPPM = 42.577 * Bo
    FWHMBA = basisSet['linewidth'] / HZPPPM
    ECHOT = basisSet['te']
    BADELT = basisSet['dwelltime']
    NDATAB = basisSet['sz'][0]

    XTRASH = 0

    # Write to txt file
    with open(outfile+'.BASIS', 'w') as fid:
        fid.write(' $SEQPAR\n')
        fid.write(' FWHMBA = #5.6f,\n' % FWHMBA)
        fid.write(' HZPPPM = #5.6f,\n' % HZPPPM)
        fid.write(' ECHOT = #2.2f,\n' % ECHOT)
        fid.write(' SEQ = \'#s\'\n' % SEQ)
        fid.write(' $END\n')
        fid.write(' $BASIS1\n')
        fid.write(' IDBASI = \'#s\',' % (vendor + ' ' + SEQ + ' ' + '{}'.format(ECHOT) + ' ms Osprey'))
        fid.write(' FMTBAS = \'(2E15.6)\',\n')
        fid.write(' BADELT = #5.6f,\n' % BADELT)
        fid.write(' NDATAB = #i\n' % NDATAB)
        fid.write(' $END\n')

        for i in range(basisSet['nMets']):
            if basisSet['name'][i] != 'CrCH2' and basisSet['name'][i] != 'H2O':
                RF = shift_centerFreq(basisSet, i)
                fid.write(' $NMUSED\n')
                fid.write(' XTRASH = #2.2f\n' # XTRASH)
                fid.write(' $END\n')
                fid.write(' $BASIS\n')
                fid.write(' ID = \'#s\',' % basisSet['name'][i])
                fid.write(' METABO = \'#s\',' % basisSet['name'][i])
                fid.write(' CONC = 1.,\n')
                fid.write(' TRAMP = 1.,\n')
                fid.write(' VOLUME = 1.,\n')
                fid.write(' ISHIFT = 0\n')
                fid.write(' $END\n')
                np.savetxt(fid, RF, fmt='#7.6e  #7.6e')

def shift_centerFreq(data_struct, idx):
    t = np.tile(data_struct['t'], (1, data_struct['sz'][1]))
    hzpppm = data_struct['Bo'] * 42.577
    f = (4.68 - data_struct['centerFreq']) * hzpppm
    fids = data_struct['fids'][:, idx]
    fids = fids * np.exp(-1j * t * f * 2 * np.pi)
    # Take the complex conjugate because the sense of rotation in LCModel seems to
    # be opposite to that used in FID-A.
    fids = np.conj(fids)
    specs = np.fft.fft(fids, axis=data_struct['dims']['t'])
    RF = np.zeros((len(specs.ravel()), 2))
    RF[:, 0] = np.real(specs.ravel())
    RF[:, 1] = np.imag(specs.ravel())
    return RF

def osp_recalculate_basis_specs(basisSet):
    # This function recalculates the basis spectra and ppm-axis of the
    # basis set

    basisSet['specs'] = np.fft.fftshift(np.fft.fft(basisSet['fids'], axis=0), axes=0)

    # Calculate ppm-axis
    f = np.linspace(-basisSet['spectralwidth']/2 + basisSet['spectralwidth']/(2 * basisSet['sz'][0]),
                    basisSet['spectralwidth']/2 - basisSet['spectralwidth']/(2 * basisSet['sz'][0]),
                    basisSet['sz'][0])
    basisSet['ppm'] = f / (basisSet['Bo'] * 42.577)
    basisSet['ppm'] = basisSet['ppm'] + basisSet['centerFreq']
    return basisSet



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', type=str, default='./dataset/')
    parser.add_argument('--savedir', type=int, default=10000)
    parser.add_argument('--name', type=int, default=10000)
    
    args = parser.parse_args()
    
    os.makedirs(args.savedir, exist_ok=True)
    
    
    
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