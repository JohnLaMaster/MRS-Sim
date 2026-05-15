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


# # Gyromagnetic Ratios for common nuclei
gamma_nucleus = {
    '1H'    : 42.5769e6, # Hz/T
    '2H'    :  6.53566e6,
    '13C'   : 10.7084e6,
    "15N"   : -4.316e6,
    '17O'   : -5.7720e6,
    '19F'   : 40.076e6,
    '23Na'  : 11.2619e6,
    '31P'   : 17.253e6,
    '129Xe' : -7.4521e6,
}

# ─────────────────────────────────────────────────────────────────────────────
# Format-specific loaders
# Each returns (header_info, metabolites) where:
#   header_info : dict with keys sw (Hz), sf (Hz), ns (int), and optionally te (ms)
#   metabolites : {name (str) → complex FID (1-D np.ndarray)}
# ─────────────────────────────────────────────────────────────────────────────

def load_marss_mat(filepath: str, config: str=None):
    """
    Load a single MARSS .mat basis file.

    MARSS stores each metabolite as a separate .mat file containing an
    'exptDat' struct with fields:
        sw_h   : spectral width in Hz
        sf     : spectrometer frequency in MHz
        nspecC : number of complex spectral points
        fid    : complex FID array

    Returns
    -------
    header_info : dict  -  keys: sw (Hz), sf (Hz), ns
    metabolites : {name: complex_fid}
    """
    mat = io.loadmat(filepath) #TODO: does this need to use loadmat_as_dict?
    if 'exptDat' not in mat:
        raise ValueError(f"No 'exptDat' variable found in {filepath}")

    exptDat = mat['exptDat']
    sw  = float(np.squeeze(exptDat['sw_h']))
    sf  = float(np.squeeze(exptDat['sf']))      # MHz -> stored as Hz below
    ns  = int(np.squeeze(exptDat['nspecC']))
    lw = float(1.0) # default basis functions are broadened by 1.0 Hz in MARSS
    
    try:
        fid = np.squeeze(exptDat['fid']).astype(complex)
    except TypeError:
        fid = np.squeeze(exptDat['fid'][0][0]).astype(complex)
    name = os.path.splitext(os.path.basename(filepath))[0].lower()

    # sf from MARSS is in MHz; convert to Hz so all loaders use consistent units
    header_info = {'sw': sw, 'sf': sf * 1e6, 'ns': ns, 'lw': lw}

    if config:
        config = io.loadmat(config)
        header_info['b0'] = float(config.B0)
        header_info['centerFreq'] = float(config.referencePeak)
        
    return header_info, {name: fid}


def load_osprey_mat(filepath: str):#, gamma: float=gamma_nucleus['1H']):
    """
    Load an Osprey basis set from a single .mat file.

    Osprey stores all metabolites in a single MATLAB struct called BASIS
    inside the .mat file. The struct layout is:

        BASIS.fids        : (n x num_mets) complex array  – FIDs for all
                            metabolites, MMs, and lipids concatenated
        BASIS.name        : 1 x num_mets cell of strings  – resonance names
        BASIS.nMets       : int  – number of metabolites (excludes MM/Lip)
        BASIS.nMM         : int  – number of MM/Lip resonances
        BASIS.spectralwidth : float  – spectral width in Hz
        BASIS.dwelltime   : float  – dwell time in seconds
        BASIS.n           : int   – number of time-domain points (= BASIS.sz[0])
        BASIS.Bo          : float – field strength in Tesla
        BASIS.te          : float – echo time in ms
        BASIS.centerFreq  : float – centre frequency in ppm (typically 4.65)
        BASIS.scale       : float – normalization factor applied to fids at
                            creation time (see below)

    Scaling convention
    ------------------
    When Osprey creates a basis set (fit_makeBasis / io_LCMBasis), it divides
    all FIDs by a global scale factor so that max(real(FFT(fids))) ≈ 1 across
    the whole basis set. This factor is stored as BASIS.scale.  At fit time,
    Osprey independently computes a per-dataset runtime scale:
        MRSCont.fit.scale = max(real(data.specs)) / max(real(basis.specs))
    which maps the normalised basis onto each subject's data amplitude.

    To recover the original simulation-amplitude FIDs:
        fid_original = fid_stored * BASIS.scale

    scipy.io.loadmat notes
    ----------------------
    MATLAB structs are loaded as numpy void/recarray objects. Cell arrays
    of strings surface as object arrays of arrays of uint16 characters.
    All squeeze/flatten calls below handle the typical 1-element wrapper
    arrays that loadmat introduces for scalars and 1xN cell arrays.

    Parameters
    ----------
    filepath : str
        Path to the Osprey .mat basis file (e.g. BASIS_MM.mat).

    Returns
    -------
    header_info : dict  –  keys: sw (Hz), sf (Hz), ns, te (ms)
    metabolites : {name (str): complex FID (1-D ndarray) at ORIGINAL scale}
    """
    raw = io.loadmat(filepath, squeeze_me=True, struct_as_record=False)

    if 'BASIS' not in raw:
        raise ValueError(f"No 'BASIS' variable found in {filepath}")

    B = raw['BASIS']   # MATLAB struct → MatlabObject / SimpleNamespace

    # ── Helper: squeeze a scalar out of a numpy scalar-wrapped value ──────
    def _scalar(x):
        """Unwrap MATLAB scalar to Python float/int."""
        x = np.squeeze(x)
        if x.ndim == 0:
            return x.item()
        return float(x.flat[0])

    def _cellstr(x):
        """
        Convert a MATLAB cell array of strings to a Python list of str.
        scipy.io.loadmat + squeeze_me renders a 1-D cell of strings as a
        numpy object array whose elements are either str (simple cases) or
        numpy char arrays.
        """
        x = np.atleast_1d(np.squeeze(x))
        names = []
        for elem in x.flat:
            if isinstance(elem, str):
                names.append(elem)
            elif isinstance(elem, np.ndarray):
                # char array → join characters
                names.append(''.join(chr(c) for c in elem.flat if c))
            else:
                names.append(str(elem))
        return names

    # ── Sequence / acquisition parameters ────────────────────────────────
    sw          = _scalar(B.spectralwidth)        # Hz
    dt          = _scalar(B.dwelltime)            # s  (cross-check: 1/sw)
    ns          = int(_scalar(B.n))
    bo          = _scalar(B.Bo)                   # Tesla
    te          = _scalar(B.te)                   # ms
    center_freq = _scalar(B.centerFreq)           # ppm

    # Osprey stores Bo in Tesla; carrier freq in Hz = Bo * gamma_H in MHz * 1e6
    # gamma_H = 42.577 MHz/T
    sf_hz = bo * gamma

    # ── Scaling factor ────────────────────────────────────────────────────
    # BASIS.scale is the divisor applied to all FIDs during basis creation,
    # chosen so that max(real(FFT(BASIS.fids))) ≈ 1 across the whole basis.
    # Multiply by this value to recover simulation-amplitude FIDs.
    scale = float(np.squeeze(B.scale))

    # ── FIDs and names ────────────────────────────────────────────────────
    # fids shape after squeeze_me: (ns, num_mets)  – complex
    fids  = np.atleast_2d(np.squeeze(B.fids))           # (ns, num_mets)
    if fids.shape[0] != ns:
        fids = fids.T                                    # ensure (ns, num_mets)

    names = _cellstr(B.name)                            # list of str

    n_mets = int(_scalar(B.nMets))
    n_mm   = int(_scalar(B.nMM))
    total  = n_mets + n_mm

    if len(names) != fids.shape[1]:
        print(f"  [warn] Osprey: name count ({len(names)}) ≠ fid columns "
              f"({fids.shape[1]}); some names may be missing")

    # ── Unscale and build metabolites dict ────────────────────────────────
    # Undo the basis-creation normalisation to restore simulation amplitude.
    # After this, fid_original = fid_stored * scale, which means
    # max(real(FFT(fid_original))) = max(real(FFT(fid_stored))) * scale ≈ scale.
    metabolites = {}
    for i in range(min(total, fids.shape[1])):
        name = names[i].lower() if i < len(names) else f'metabolite_{i}'
        fid_unscaled = fids[:, i] * scale          # restore original amplitude
        metabolites[name] = fid_unscaled

    if len(metabolites) == 0:
        raise ValueError(f"No valid FIDs extracted from {filepath}")

    header_info = {'sw': sw, 'sf': sf_hz, 'ns': ns, 'te': te}

    print(f"  Loaded {n_mets} metabolite(s) + {n_mm} MM/Lip resonance(s) "
          f"from Osprey basis (scale={scale:.4g})")
    return header_info, metabolites


# -- LCModel helpers ----------------------------------------------------------

def _parse_namelist_block(text: str) -> dict:
    """
    Parse a Fortran-style namelist block into a flat dict.
    Handles lines like:  KEY = value  or  KEY = 'value'
    """
    params = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('!'):
            continue
        if '=' in line:
            key, _, val = line.partition('=')
            params[key.strip().upper()] = val.strip().strip("'\"")
    return params


def _parse_fid_interleaved(text: str) -> np.ndarray:
    """
    Parse all whitespace-delimited floats from a text block, then pair them
    as (real, imag) complex values.

    LCModel stores FID data as interleaved real/imag values with up to 3
    complex pairs per line, e.g.:
        1.23456E-01 -2.34567E-02  3.45678E-01 -4.56789E-02  5.67890E-01  6.78901E-02

    A simple two-value-per-line parser would silently discard most data.
    """
    tokens = []
    for line in text.splitlines():
        parts = line.split()
        for p in parts:
            try:
                tokens.append(float(p))
            except ValueError:
                pass  # skip any stray non-numeric text

    if len(tokens) % 2 != 0:
        tokens = tokens[:-1]   # drop an unpaired trailing token if present

    it = iter(tokens)
    return np.array([complex(r, i) for r, i in zip(it, it)], dtype=complex)


def load_lcmodel_basis(filepath: str):
    """
    Parse an LCModel .BASIS file (may contain multiple metabolites).

    File structure (from LCModel manual section 8.6.6):

        $SEQPAR
          ECHOT  = 30.0          ! echo time in ms
          SEQ    = 'PRESS'
        $END
        $BASIS1                  ! appears once; holds global acquisition params
          HZPPPM = 123.234       ! Hz per ppm = spectrometer freq in MHz
          BADELT = 1.66693E-04   ! dwell time in seconds
          NDATAB = 2048          ! number of complex points per spectrum
        $END
        $NMUSED ...  $END        ! documents original input; not used by LCModel
        $BASIS
          ID     = 'NAA'
          METABO = 'NAA'
          ...
        $END
        <re> <im> <re> <im> ...  ! up to 3 complex pairs per line, interleaved
        ...

    Note: HZPPPM and NDATAB live in BASIS1, not SEQPAR.
    Note: Data is interleaved (re, im, re, im, ...) with multiple pairs per line.

    Returns
    -------
    header_info : dict  -  keys: sw (Hz), sf (Hz), ns, te (ms)
    metabolites : {name: complex_fid}
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # -- SEQPAR: echo time ---------------------------------------------------
    seq_params = {}
    m = re.search(r'\$SEQPAR(.*?)\$END', content, re.DOTALL | re.IGNORECASE)
    if m:
        seq_params = _parse_namelist_block(m.group(1))
    # te = float(seq_params.get('ECHOT', 0.0))
    te = float(seq_params.get('ECHOT', TE).strip().rstrip(','))

    # -- BASIS1: global acquisition parameters --------------------------------
    # HZPPPM = Hz per ppm = spectrometer frequency in MHz (e.g. 123.234 at 3T)
    # BADELT = dwell time in seconds
    # NDATAB = number of complex points
    basis1_params = {}
    m1 = re.search(r'\$BASIS1(.*?)\$END', content, re.DOTALL | re.IGNORECASE)
    if m1:
        basis1_params = _parse_namelist_block(m1.group(1))

    hzperppm = float(basis1_params.get('HZPPPM', seq_params.get('HZPPPM', 
                            gamma*B0)).strip().rstrip(','))
    dt_str   = basis1_params.get('BADELT', basis1_params.get('DELTAT',
               seq_params.get('BADELT', seq_params.get('DELTAT', ''))))
    dt  = float(dt_str.strip().rstrip(',')) if dt_str else DT
    sw  = 1.0 / dt
    # HZPPPM is the spectrometer frequency in MHz (= Hz per ppm at the given
    # field). Convert to Hz for consistency with all other loaders.
    sf  = hzperppm * 1e6
    ns_global = int(basis1_params.get('NDATAB', basis1_params.get('NUNFIL', 0)))
    tramp = basis1_params.get('TRAMP', 1.0)
    volume = basis1_params.get('VOLUME', 1.0)
    conc = basis1_params.get('CONC', 1.0)
    scale = tramp / (volume * conc)
    
    # -- Individual BASIS blocks ----------------------------------------------
    metabolites = {}

    # Match each $BASIS (but NOT $BASIS1) block header and the data that
    # follows it, up to the next $BASIS or end of file.
    basis_pattern = re.compile(
        r'\$(BASIS)(?!1)\b(.*?)\$END(.*?)(?=\$BASIS(?!1)\b|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    for block in basis_pattern.finditer(content):
        header_text = block.group(2)
        data_text   = block.group(3)

        meta = _parse_namelist_block(header_text)
        fid  = _parse_fid_interleaved(data_text)

        if fid.size == 0:
            continue

        name = meta.get('METABO', meta.get('ID', 'unknown')).lower()
        metabolites[name] = fid #* scale

    # Use first metabolite length as ns if BASIS1 did not specify it
    if ns_global == 0 and metabolites:
        ns_global = next(iter(metabolites.values())).size

    header_info = {'sw': sw, 'sf': sf, 'ns': ns_global, 'te': te}
    return header_info, metabolites


def load_raw_basis(filepath: str):
    """
    Parse a single-metabolite LCModel .raw basis file.

    File structure (LCModel .RAW / MakeBasis input format):

        $NMID
          ID     = 'NAA'
          FMTDAT = '(2E15.6)'
          TRAMP  = 1.0
          ISHIFT = 0
        $END
         1.234E+00 -2.345E-01
         ...

    Sequence parameters (sw, sf) are NOT embedded in .raw files; they must
    come from the config or a companion .basis file.

    Returns
    -------
    meta : dict   -  NMID block fields (includes 'ID')
    fid  : complex ndarray
    """
    with open(filepath, 'r') as f:
        content = f.read()
        
    name = os.path.splitext(os.path.basename(filepath))[0]

    meta = {}
    m = re.search(r'\$NMID(.*?)\$END', content, re.DOTALL | re.IGNORECASE)
    if m:
        meta = _parse_namelist_block(m.group(1))
        # print('meta: ',meta)
    
    # Name check
    if name.lower() not in meta['ID']: meta['ID'] = name.lower()

    # FID data follows the last $END
    data_section = re.split(r'\$END', content, flags=re.IGNORECASE)[-1]
    fid = _parse_fid_interleaved(data_section)
    # print('load_raw_basis :: fid.shape: ',fid.shape)

    return meta, fid


def load_nifti_mrs_basis(filepath: str):
    """
    Load a NIfTI-MRS basis set (.nii or .nii.gz).

    NIfTI-MRS format (Clarke et al., MRM 2022):
      - Complex time-domain data stored as shape (x, y, z, t [, d5, d6, d7])
        For SVS: x=y=z=1, so shape is (1, 1, 1, ns) or (1, 1, 1, ns, n_mets)
      - Dwell time (seconds) stored in pixdim[4]; units encoded in xyzt_units
      - JSON header extension (ecode=44) contains:
          SpectrometerFrequency : [float, ...]   MHz  - ALWAYS an array per spec
          ResonantNucleus       : [str, ...]     e.g. ["1H"]
          EchoTime              : float          seconds  (optional)
          dim_5                 : str            e.g. "DIM_BASIS"
          dim_5_header          : dict           e.g. {"Basis": ["NAA", "Cr", ...]}

    Note: SpectralWidth is NOT a standard NIfTI-MRS header field. Dwell time
    must be read from pixdim[4] and converted using xyzt_units.

    Returns
    -------
    header_info : dict  -  keys: sw (Hz), sf (Hz), ns, te (ms)
    metabolites : {name: complex_fid}
    """
    img  = nib.load(filepath)
    data = np.asarray(img.dataobj)   # complex64 or complex128

    # -- Dwell time from NIfTI pixdim[4] with units --------------------------
    hdr    = img.header
    dt_raw = float(hdr['pixdim'][4])

    # xyzt_units bits 3-5 encode time unit: 8=sec, 16=msec, 24=usec
    try:
        time_unit = hdr.get_xyzt_units()[1]   # e.g. 'sec', 'msec', 'usec'
    except Exception:
        time_unit = 'sec'

    if time_unit == 'msec':
        dt = dt_raw * 1e-3
    elif time_unit == 'usec':
        dt = dt_raw * 1e-6
    else:
        dt = dt_raw   # assume seconds

    sw = 1.0 / dt if dt > 0 else 0.0

    # -- JSON header extension (ecode 44) ------------------------------------
    hdr_ext = {}
    for ext in img.header.extensions:
        if ext.get_code() == 44:
            try:
                hdr_ext = json.loads(ext.get_content().decode('utf-8'))
            except Exception:
                pass
            break

    # SpectrometerFrequency is ALWAYS a JSON array per the NIfTI-MRS spec,
    # even for a single nucleus.  Units are MHz.
    sf_field = hdr_ext.get('SpectrometerFrequency', [0])
    if isinstance(sf_field, list):
        sf_mhz = float(sf_field[0]) if sf_field else 0.0
    else:
        sf_mhz = float(sf_field)
    sf_hz = sf_mhz * 1e6   # convert MHz -> Hz

    # EchoTime in the NIfTI-MRS standard is stored in seconds
    echo_time_s = hdr_ext.get('EchoTime', None)
    te_ms = float(echo_time_s) * 1e3 if echo_time_s is not None else 0.0

    # -- Data and metabolite names -------------------------------------------
    # Collapse spatial dims: data[0,0,0] -> shape (ns,) or (ns, n_mets, ...)
    data = data[0, 0, 0]
    ns   = data.shape[0]

    dim5_hdr    = hdr_ext.get('dim_5_header', {})
    metab_names = dim5_hdr.get('Basis', []) if isinstance(dim5_hdr, dict) else []

    metabolites = {}
    if data.ndim == 1:
        name = (str(metab_names[0]).lower() if metab_names
                else hdr_ext.get('dim_5', 'unknown'))
        if isinstance(name, list):
            name = str(name[0]).lower()
        metabolites[name] = data
    else:
        for i in range(data.shape[1]):
            name = (str(metab_names[i]).lower() if i < len(metab_names)
                    else f'metabolite_{i}')
            metabolites[name] = data[:, i]

    header_info = {'sw': sw, 'sf': sf_hz, 'ns': ns, 'te': te_ms}
    return header_info, metabolites


def load_fsl_mrs_basis_dir(dirpath: str):
    """
    Load an FSL-MRS basis set from a directory of per-metabolite .json files.

    FSL-MRS (fsl_mrs_sim) outputs one JSON file per metabolite. Each file
    contains a full description of the simulated FID, the sequence used, and
    metadata (Clarke et al., FSL-MRS documentation).

    Confirmed fields (from official FSL-MRS docs):
        basis_name   : str            metabolite name
        basis_re     : [float, ...]   real part of the FID
        basis_im     : [float, ...]   imaginary part of the FID
        basis_dwell  : float          dwell time in seconds
        basis_centre : float          receiver centre frequency in ppm
        basis_width  : float          Gaussian linewidth in Hz (MM spectra only)
        echotime     : float          echo time in ms (set with -e flag)

    The field 'basis_hzperppm' (Hz/ppm = spectrometer frequency in MHz) is
    read when present; if absent, the value falls back to config key
    'carrier_frequency' (expected in Hz) or 0.

    Each JSON also embeds the full sequence description used to generate it
    (so basis files double as sequence description files for new simulations).
    Non-basis JSON files (e.g. sequence files) are skipped if they lack
    basis_re / basis_im.

    Parameters
    ----------
    dirpath : str
        Path to the FSL-MRS basis directory.

    Returns
    -------
    header_info : dict  -  keys: sw (Hz), sf (Hz), ns, te (ms)
    metabolites : {name: complex_fid}
    """
    dirpath = os.path.abspath(dirpath)
    if not os.path.isdir(dirpath):
        raise NotADirectoryError(
            f"FSL-MRS basis path is not a directory: {dirpath}")

    metabolites = {}
    header_info = None

    json_files = sorted(
        f for f in os.listdir(dirpath) if f.lower().endswith('.json')
    )
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {dirpath}")

    for filename in json_files:
        filepath = os.path.join(dirpath, filename)
        with open(filepath, 'r') as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as e:
                print(f"  [warn] Could not parse {filename}: {e}")
                continue

        # Skip sequence-description files that lack FID arrays
        if 'basis_re' not in data or 'basis_im' not in data:
            continue

        fid_re = np.asarray(data['basis']['basis_re'], dtype=float)
        fid_im = np.asarray(data['basis']['basis_im'], dtype=float)
        if fid_re.size == 0:
            print(f"  [warn] Empty FID in {filename}, skipping.")
            continue

        fid  = fid_re + 1j * fid_im
        name = str(data.get('basis_name',
                             os.path.splitext(filename)[0])).lower()
        metabolites[name] = fid

        # Populate header_info from the first valid file only
        if header_info is None:
            dt  = float(data['basis'].get('basis_dwell'))   # seconds
            sw  = 1.0 / dt if dt > 0 else 0.0
            # basis_hzperppm is an optional field (Hz/ppm = spectrometer freq
            # in MHz). It is not enumerated in the official FSL-MRS docs but
            # may be present in simulator output. Falls back to config.
            sf_hzperppm = float(data.get('basis_centre'))
            # Hz/ppm * 1e6 = Hz (spectrometer frequency)
            sf  = sf_hzperppm * 1e6 if sf_hzperppm else 0.0
            ns  = fid.size
            # te  = float(data.get('echotime'))   # ms
            b0 = float(data['seq']['B0'])
            lw = float(data['seq']['Rx_LW'])
            header_info = {'sw': sw, 'sf': sf, 'ns': ns, 'te': te, 'B0': b0, 'centerFreq': float(data['seq']['centralShift']), 'lw': lw}

    if not metabolites:
        raise ValueError(
            f"No valid FSL-MRS basis spectra found in {dirpath}")

    print(f"  Loaded {len(metabolites)} FSL-MRS metabolite(s) from "
          f"{os.path.basename(dirpath)}/")
    return header_info, metabolites


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

def fid_to_stack(fid: np.ndarray) -> np.ndarray:
    """Return a (2, N) array with rows [real, imag]."""
    fid = np.squeeze(fid)
    return np.expand_dims(np.stack([fid.real, fid.imag], axis=0), axis=0)


def build_header_fields(header, header_info: dict, config: dict,
                        te_override=None):
    """
    Populate the output header struct from a format-specific header_info dict.

    Parameters
    ----------
    header      : MATLAB-loaded struct (numpy recarray or dict)
    header_info : dict with keys sw (Hz), sf (Hz), ns, and optionally te (ms)
    config      : full config dict (used for centerFreq, B0, etc.)
    te_override : if given, uses this TE value instead of header_info['te']
    """
    sw  = float(header_info['sw'])
    # sf from all loaders is in Hz; fall back to config if loader could not
    # determine it (e.g. .raw files have no embedded sequence parameters)
    sf  = float(header_info.get('sf') or config.get('carrier_frequency', 
                                                    config['B0']*gamma))
    if abs(sf) > 10e3: sf /= 1e6
    ns  = int(header_info['ns'])
    dt  = 1.0 / sw if sw else 1e-4
    te  = (te_override if te_override is not None
           else header_info.get('te', config['TE']))
    # ppm = np.expand_dims(-0.5 * sw + np.arange(ns) * sw / (ns - 1),
    #         axis=0) + config['centerFreq']
    ppm = np.expand_dims(np.linspace(start=-0.5 * sw / sf,
                                     stop=0.5 * sw / sf,
                                     num=ns) + config['centerFreq'],
                         axis=0)

    header['spectralwidth']      = sw
    header['carrier_frequency']  = sf
    header['Ns']                 = ns
    header['t']                  = np.arange(0, dt * ns, dt)
    header['centerFreq']         = config['centerFreq']
    header['B0']                 = config['B0']
    header['TE']                 = te
    header['pulse_sequence']     = config['pulse_sequence']
    header['vendor']             = config['vendor']
    header['basis_set_software'] = config['sim_software'] if config.get('sim_software') else 'unspecified'
    header['ppm']                = ppm
    
    return header


def assign_fid(metabolites, name: str, fid: np.ndarray):
    """
    Write a [2, N] real/imag FID into metabolites[name]['fid'] when the
    metabolite name exists in the template.
    """
    try:
        _ = metabolites[name]
        metabolites[name]['fid'] = fid_to_stack(fid)
    except (KeyError, ValueError):
        metabolites.update({name: {
            'fid': fid_to_stack(fid),
            'min': 0.0,
            'max': 1.0,
        }})
        # print(f"  [skip] '{name}' not found in template metabolites")

def mrscloud_correction(metabolites):
    for k in metabolites.keys():
        tmp = np.fliplr(npfftshift(metabolites[k]['fid'], axis=1))
        tmp = tmp[:,0,:] + 1j*tmp[:,1,:]
        tmp = np.fft.ifft(npifftshift(tmp, axes=-1), axis=-1)
        metabolites[k]['fid'] = fit_to_stack(tmp)
    return metabolites

# def calc_dt(fid: np.ndarray) -> float:
#     a = np.diff(fid)
#     print('a[0:9] = ',a[0:9])
#     b = np.mean(a)
#     print('b = ',b)
#     return np.mean(fid[...,:2] - fid[...,:1])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

# def main(cfg_path: str):
    # with open(cfg_path, 'r') as f:
    #     config = json.load(f)
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
    # template_data = io.loadmat(config['template_path'])
    # metabolites   = template_data['metabolites']
    # header        = template_data['header']
    # artifacts     = template_data['artifacts']
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

        # -- MARSS .mat ------------------------------------------------------
        # if ext == '.mat':
        #     hinfo, mets = load_marss_mat(filepath)
        #     if not header_set:
        #         build_header_fields(header, hinfo, config, te_override=te)
        #         header_set = True
        #     for name, fid in mets.items():
        #         assign_fid(metabolites, name, fid)
                
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

    # # -- Edited spectra (OFF condition) --------------------------------------
    # if config.get('edit_off_path'):
    #     for met in config.get('metabs_off', []):
    #         loaded = False

    #         # Try each format in order of likelihood
    #         candidates = [
    #             (met + '.mat',   lambda p: np.squeeze(
    #                 io.loadmat(p)['exptDat']['fid']).astype(complex)),
    #             (met + '.raw',   lambda p: load_raw_basis(p)[1]),
    #             (met + '.basis', lambda p: list(
    #                 load_lcmodel_basis(p)[1].values())[0]),
    #         ]
    #         for fname, loader in candidates:
    #             candidate = os.path.join(config['edit_off_path'], fname)
    #             if os.path.isfile(candidate):
    #                 try:
    #                     fid = loader(candidate)
    #                     metabolites[met.lower()]['fid_OFF'] = fid_to_stack(fid)
    #                     loaded = True
    #                     break
    #                 except Exception as e:
    #                     print(f"  [warn] Could not load OFF file "
    #                           f"{candidate}: {e}")

    #         if not loaded:
    #             print(f"  [warn] No OFF-condition file found for '{met}'")


    # -- Visual Inspection ---------------------------------------------------
    # Visually inspect the basis functions to ensure they appear correctly
    # at least in terms of chemical shift and directionality
    # print("header['spectralwidth'] = {}; header['carrier_frequency'] = {}".format(header['spectralwidth'],header['carrier_frequency']))
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
    save_path = os.path.join(save_dir, save_name)
    io.savemat(save_path,
               mdict={'metabolites': metabolites,
                      'header':      header,
                    #   'artifacts':   artifacts
                      })
    print(f"\nSaved basis set '{save_name}' to: {save_dir}")

    # -- Export to additional formats ----------------------------------------
    export_formats = config.get('export_formats', [])
    if isinstance(export_formats, str):
        export_formats = [f.strip() for f in export_formats.split(',') if f.strip()]

    if export_formats:
        # Derive a clean base name (strip .mat if present) for single-file exports
        export_base = os.path.splitext(save_name)[0]
        export_root = os.path.join(save_dir, 'exported', export_base)

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
                    from validate_basis_sets import validate_single_file
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


# =============================================================================
# Export helpers and format-specific writers
# =============================================================================

#: All supported export format identifiers.
EXPORT_FORMATS = [
    'lcmodel_raw',    # One .raw file per metabolite (no sequence params embedded)
    'lcmodel_basis',  # Single multi-metabolite .basis file
    'fsl_mrs_json',   # One .json file per metabolite (FSL-MRS directory layout)
    'osprey_mat',     # Single .mat with BASIS struct (Osprey convention)
    'marss_native',   # One .mat per metabolite with exptDat struct (MARSS native)
    'nifti_mrs',      # Single .nii with shape (1,1,1,ns,n_mets) and ecode-44 JSON
]


def _alert(fmt: str, msg: str):
    """Print a clearly labelled terminal alert for missing or defaulted export info."""
    print(f"\n  [ALERT] [{fmt}] {msg}")


def _extract_complex_fid(met_dict: dict) -> np.ndarray:
    """
    Extract a 1-D complex FID from the internal (1, 2, N) real/imag stack
    produced by fid_to_stack().

    Handles both the canonical (1, 2, N) shape and the legacy (2, N) shape,
    as well as arrays that are already complex.
    """
    fid_data = np.squeeze(met_dict['fid'])       # → (2, N) or (N,)
    if fid_data.ndim == 2 and fid_data.shape[0] == 2:
        return fid_data[0].astype(complex) + 1j * fid_data[1].astype(complex)
    if np.iscomplexobj(fid_data):
        return fid_data.flatten()
    raise ValueError(
        f"Unexpected FID shape {fid_data.shape}; cannot extract complex data."
    )


# ─────────────────────────────────────────────────────────────────────────────
# LCModel .raw  (one file per metabolite)
# ─────────────────────────────────────────────────────────────────────────────

def export_lcmodel_raw(metabolites: dict, header: dict, output_dir: str,
                       config: dict) -> list:
    """
    Write each metabolite as a separate LCModel .raw file.

    .raw files carry no sequence parameters; these must be supplied externally
    (via a companion .basis file or config) when re-importing into LCModel.
    TRAMP, VOLUME, and CONC are set to 1.0 (unscaled simulation amplitude).

    Parameters
    ----------
    metabolites : Internal metabolites dict {name: {'fid': (1,2,N), ...}}.
    header      : Internal header dict.
    output_dir  : Directory in which to write the .raw files.
    config      : Full config dict.

    Returns
    -------
    list of str : Paths to the written files, in metabolite order.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    _alert('lcmodel_raw',
           'Sequence parameters (sw, sf) are NOT embedded in .raw files. '
           'Supply these via a companion .basis file or the config '
           'spectralwidth / carrier_frequency keys when re-importing.')

    for name, met in metabolites.items():
        fid      = _extract_complex_fid(met)
        out_path = os.path.join(output_dir, f"{name}.raw")

        with open(out_path, 'w') as f:
            f.write(' $NMID\n')
            f.write(f"  ID     = '{name.upper()}'\n")
            f.write(f"  FMTDAT = '(2E15.6)'\n")
            f.write(f"  TRAMP  = 1.0\n")
            f.write(f"  ISHIFT = 0\n")
            f.write(' $END\n')
            for pt in fid:
                f.write(f" {pt.real:15.6E} {pt.imag:15.6E}\n")

        paths.append(out_path)

    print(f"  Exported {len(paths)} LCModel .raw file(s) to: {output_dir}")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# LCModel .basis  (single multi-metabolite file)
# ─────────────────────────────────────────────────────────────────────────────

def export_lcmodel_basis(metabolites: dict, header: dict, output_path: str,
                         config: dict) -> str:
    """
    Write all metabolites into a single LCModel .basis file.

    Global acquisition parameters are taken from the internal header.
    Per-metabolite TRAMP, VOLUME, and CONC are all set to 1.0.

    Returns
    -------
    str : Path to the written file.
    """
    sw       = float(header['spectralwidth'])           # Hz
    # header['carrier_frequency'] is in MHz (see build_header_fields)
    sf_mhz   = float(header['carrier_frequency'])       # MHz = Hz/ppm at this field
    ns       = int(header['Ns'])
    te       = float(header.get('TE', config.get('TE', 0.0)))
    dt       = 1.0 / sw if sw else 1e-4                 # dwell time in seconds
    seq      = config.get('pulse_sequence', 'unspecified_sequence')

    if seq in ('unspecified_sequence', '', None):
        _alert('lcmodel_basis',
               "Pulse sequence name is missing or 'unspecified_sequence'. "
               "Set 'pulse_sequence' in config for a meaningful SEQ entry.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'w') as f:
        # SEQPAR block
        f.write(' $SEQPAR\n')
        f.write(f"  ECHOT  = {te:.4f}\n")
        f.write(f"  SEQ    = '{seq}'\n")
        f.write(' $END\n')

        # BASIS1 block  (global acquisition parameters)
        f.write(' $BASIS1\n')
        f.write(f"  HZPPPM = {sf_mhz:.6f}\n")   # Hz/ppm = spectrometer freq in MHz
        f.write(f"  BADELT = {dt:.8E}\n")
        f.write(f"  NDATAB = {ns}\n")
        f.write(' $END\n')

        # One $BASIS block per metabolite followed by interleaved FID data
        for name, met in metabolites.items():
            fid        = _extract_complex_fid(met)
            upper_name = name.upper()

            f.write(' $BASIS\n')
            f.write(f"  ID     = '{upper_name}'\n")
            f.write(f"  METABO = '{upper_name}'\n")
            f.write(f"  TRAMP  = 1.0\n")
            f.write(f"  VOLUME = 1.0\n")
            f.write(f"  CONC   = 1.0\n")
            f.write(' $END\n')
            for pt in fid:
                f.write(f" {pt.real:15.6E} {pt.imag:15.6E}\n")

    print(f"  Exported LCModel .basis to: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# FSL-MRS JSON  (one file per metabolite)
# ─────────────────────────────────────────────────────────────────────────────

def export_fsl_mrs_json(metabolites: dict, header: dict, output_dir: str,
                        config: dict) -> list:
    """
    Write each metabolite as a separate FSL-MRS .json file.

    The 'seq' sub-dict that FSL-MRS uses to describe the spin system and pulse
    sequence is not recoverable from basis set data alone and is therefore set
    to null.  Re-simulating in FSL-MRS requires restoring this information.

    Returns
    -------
    list of str : Paths to the written files, in metabolite order.
    """
    os.makedirs(output_dir, exist_ok=True)

    sw   = float(header['spectralwidth'])
    dt   = 1.0 / sw if sw else 1e-4
    te   = float(header.get('TE',         config.get('TE',         0.0)))
    cf   = float(header.get('centerFreq', config.get('centerFreq', 4.65)))  # ppm
    lw   = float(header.get('lw',         2.0))   # linewidth in Hz; default if absent

    _alert('fsl_mrs_json',
           "Spin system and pulse sequence descriptor ('seq') cannot be "
           "recovered from basis set data alone. All exported files will have "
           "'seq': null. Re-simulation in FSL-MRS will not be possible without "
           "restoring this field.")
    _alert('fsl_mrs_json',
           "basis_width is set to the internal linewidth value "
           f"({lw:.2f} Hz). This field is documented for MM spectra only; "
           "it may be irrelevant for metabolite basis functions.")

    paths = []
    for name, met in metabolites.items():
        fid      = _extract_complex_fid(met)
        entry    = {
            'basis_name': name,
            'basis': {
                'basis_re':     fid.real.tolist(),
                'basis_im':     fid.imag.tolist(),
                'basis_dwell':  dt,
                'basis_centre': cf,
                'basis_width':  lw,
            },
            'echotime': te,
            'seq':      None,
        }
        out_path = os.path.join(output_dir, f"{name}.json")
        with open(out_path, 'w') as f:
            json.dump(entry, f, indent=2)
        paths.append(out_path)

    print(f"  Exported {len(paths)} FSL-MRS .json file(s) to: {output_dir}")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Osprey .mat  (single file, BASIS struct)
# ─────────────────────────────────────────────────────────────────────────────

def export_osprey_mat(metabolites: dict, header: dict, output_path: str,
                      config: dict) -> str:
    """
    Write all metabolites into an Osprey-compatible .mat file.

    FIDs are stored at original simulation amplitude with BASIS.scale = 1.0
    (i.e. no normalisation has been applied).  Osprey will compute its own
    runtime scale factor per dataset during fitting, so a stored scale of 1.0
    is the correct starting point.

    BASIS.nMM is set to 0.  If the metabolite dict contains MM or lipid
    entries they will be stored as metabolites; adjust nMets/nMM manually
    after import if Osprey needs the distinction.

    Returns
    -------
    str : Path to the written file.
    """
    sw     = float(header['spectralwidth'])
    sf_mhz = float(header['carrier_frequency'])   # MHz (see build_header_fields)
    ns     = int(header['Ns'])
    dt     = 1.0 / sw if sw else 1e-4
    te     = float(header.get('TE',         config.get('TE',         0.0)))
    b0     = float(header.get('B0',         config.get('B0',         3.0)))
    cf     = float(header.get('centerFreq', config.get('centerFreq', 4.65)))

    names  = list(metabolites.keys())
    n_mets = len(names)

    fids_array = np.zeros((ns, n_mets), dtype=complex)
    for i, name in enumerate(names):
        fid = _extract_complex_fid(metabolites[name])
        n   = min(len(fid), ns)
        fids_array[:n, i] = fid[:n]

    # MATLAB cell array of strings for BASIS.name
    name_cell = np.zeros((1, n_mets), dtype=object)
    for i, n in enumerate(names):
        name_cell[0, i] = n

    _alert('osprey_mat',
           'BASIS.scale = 1.0 (FIDs stored at original simulation amplitude; '
           'no normalisation applied). BASIS.nMM = 0; all entries are treated '
           'as metabolites. If MM/Lip resonances are present, adjust nMets and '
           'nMM manually after import.')
    _alert('osprey_mat',
           'BASIS.Bo is derived from header[\'B0\']. If the source basis set '
           'was not produced at exactly that field, verify Bo after import.')

    t_axis = np.arange(0, dt * ns, dt)[:ns].reshape(1, -1)

    basis_struct = {
        'fids':          fids_array,
        'name':          name_cell,
        'nMets':         np.array([[n_mets]], dtype=np.float64),
        'nMM':           np.array([[0]],      dtype=np.float64),
        'spectralwidth': np.array([[sw]]),
        'dwelltime':     np.array([[dt]]),
        'n':             np.array([[ns]],     dtype=np.float64),
        'sz':            np.array([[ns, n_mets]], dtype=np.float64),
        'Bo':            np.array([[b0]]),
        'te':            np.array([[te]]),
        'centerFreq':    np.array([[cf]]),
        'scale':         np.array([[1.0]]),
        't':             t_axis,
        'ppm':           header.get('ppm', np.zeros((1, ns))),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    io.savemat(output_path, {'BASIS': basis_struct})
    print(f"  Exported Osprey .mat to: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# MARSS native .mat  (one file per metabolite, exptDat struct)
# ─────────────────────────────────────────────────────────────────────────────

def export_marss_native(metabolites: dict, header: dict, output_dir: str,
                        config: dict) -> list:
    """
    Write each metabolite as a separate MARSS-native .mat file.

    Each file contains an 'exptDat' struct with fields sw_h, sf (MHz),
    nspecC, and fid, matching the layout expected by load_marss_mat().
    This format is distinct from the simulator-ready single-.mat output
    written by main(); it is the native per-metabolite MARSS input format.

    Returns
    -------
    list of str : Paths to the written files, in metabolite order.
    """
    os.makedirs(output_dir, exist_ok=True)

    sw     = float(header['spectralwidth'])
    # header['carrier_frequency'] is in MHz after build_header_fields;
    # MARSS stores sf in MHz, so this value is passed through directly.
    sf_mhz = float(header['carrier_frequency'])
    ns     = int(header['Ns'])
    paths  = []

    for name, met in metabolites.items():
        fid = _extract_complex_fid(met)
        exptDat = {
            'sw_h':   np.array([[sw]]),
            'sf':     np.array([[sf_mhz]]),
            'nspecC': np.array([[ns]], dtype=np.float64),
            'fid':    fid.reshape(1, -1),
        }
        out_path = os.path.join(output_dir, f"{name}.mat")
        io.savemat(out_path, {'exptDat': exptDat})
        paths.append(out_path)

    print(f"  Exported {len(paths)} MARSS-native .mat file(s) to: {output_dir}")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# NIfTI-MRS .nii  (single file, shape 1×1×1×ns×n_mets)
# ─────────────────────────────────────────────────────────────────────────────

def export_nifti_mrs(metabolites: dict, header: dict, output_path: str,
                     config: dict) -> str:
    """
    Write all metabolites as a NIfTI-MRS .nii file.

    Data shape: (1, 1, 1, ns, n_mets).  The 4th dimension (t) is the FID;
    the 5th dimension is tagged as DIM_BASIS in the JSON extension.
    Dwell time is stored in pixdim[4] in seconds.  The JSON extension
    (ecode=44) encodes SpectrometerFrequency, EchoTime, and metabolite names
    following the NIfTI-MRS specification (Clarke et al., MRM 2022).

    Returns
    -------
    str : Path to the written file.
    """
    sw     = float(header['spectralwidth'])
    sf_mhz = float(header['carrier_frequency'])   # MHz
    ns     = int(header['Ns'])
    dt     = 1.0 / sw if sw else 1e-4
    te     = float(header.get('TE',         config.get('TE',         0.0)))
    te_s   = te / 1e3                              # ms → s for NIfTI-MRS spec
    nuc    = config.get('nucleus', '1H')

    names  = list(metabolites.keys())
    n_mets = len(names)

    data = np.zeros((1, 1, 1, ns, n_mets), dtype=np.complex64)
    for i, name in enumerate(names):
        fid = _extract_complex_fid(metabolites[name])
        n   = min(len(fid), ns)
        data[0, 0, 0, :n, i] = fid[:n].astype(np.complex64)

    nii_hdr = nib.Nifti1Header()
    nii_hdr.set_data_dtype(np.complex64)
    nii_hdr['dim'][0]    = 5        # number of active dimensions
    nii_hdr['dim'][5]    = n_mets
    nii_hdr['pixdim'][4] = dt       # dwell time in seconds
    nii_hdr.set_xyzt_units('mm', 'sec')

    img = nib.Nifti1Image(data, np.eye(4), header=nii_hdr)

    json_ext_dict = {
        'SpectrometerFrequency': [sf_mhz],
        'ResonantNucleus':       [nuc],
        'EchoTime':              te_s,
        'dim_5':                 'DIM_BASIS',
        'dim_5_header':          {'Basis': names},
    }
    ext_bytes = json.dumps(json_ext_dict, indent=2).encode('utf-8')
    img.header.extensions.append(nib.nifti1.Nifti1Extension(44, ext_bytes))

    _alert('nifti_mrs',
           'volume is set to 1.0 (default) and is not embedded in the file. '
           'Metabolite concentrations are not stored; update the JSON extension '
           'manually if quantitative volume information is available.')
    _alert('nifti_mrs',
           f'SpectrometerFrequency = [{sf_mhz:.4f}] MHz derived from '
           "header['carrier_frequency']. Verify this matches the actual "
           'spectrometer frequency of the acquisition.')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    nib.save(img, output_path)
    print(f"  Exported NIfTI-MRS to: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def export_basis_set(metabolites: dict, header: dict, config: dict,
                     export_format: str, output_dir: str,
                     base_name: str = 'basis_set') -> list:
    """
    Dispatch an export request to the appropriate format-specific writer.

    Multi-file formats (lcmodel_raw, fsl_mrs_json, marss_native) write one
    file per metabolite into ``<output_dir>/<format>/``.
    Single-file formats (lcmodel_basis, osprey_mat, nifti_mrs) write a single
    file into ``output_dir``.

    Parameters
    ----------
    metabolites   : Internal metabolites dict.
    header        : Internal header dict.
    config        : Full config dict.
    export_format : One of EXPORT_FORMATS.
    output_dir    : Root output directory.
    base_name     : Base filename (no extension) for single-file formats.

    Returns
    -------
    list of str : Path(s) to all written files.
    """
    fmt = export_format.lower()

    if fmt == 'lcmodel_raw':
        dest = os.path.join(output_dir, 'lcmodel_raw')
        result = export_lcmodel_raw(metabolites, header, dest, config)
        return result if isinstance(result, list) else [result]

    elif fmt == 'lcmodel_basis':
        dest = os.path.join(output_dir, f"{base_name}.basis")
        return [export_lcmodel_basis(metabolites, header, dest, config)]

    elif fmt == 'fsl_mrs_json':
        dest = os.path.join(output_dir, 'fsl_mrs_json')
        result = export_fsl_mrs_json(metabolites, header, dest, config)
        return result if isinstance(result, list) else [result]

    elif fmt == 'osprey_mat':
        dest = os.path.join(output_dir, f"{base_name}_osprey.mat")
        return [export_osprey_mat(metabolites, header, dest, config)]

    elif fmt == 'marss_native':
        dest = os.path.join(output_dir, 'marss_native')
        result = export_marss_native(metabolites, header, dest, config)
        return result if isinstance(result, list) else [result]

    elif fmt == 'nifti_mrs':
        dest = os.path.join(output_dir, f"{base_name}.nii")
        return [export_nifti_mrs(metabolites, header, dest, config)]

    else:
        print(f"  [warn] Unknown export format '{fmt}'. "
              f"Valid options: {', '.join(EXPORT_FORMATS)}")
        return []


def visual_inspection(metabolites, ppm, flip=False, debug=False):
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
    if not debug:
        spec = npfftshift(np.fft.fft(fid, axis=1), axs=1)
        if flip: spec = np.fliplr(spec)
    else:
         spec = np.fliplr(npfftshift(fid, axis=1))

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

'''
# .RAW - Successful
python -m src.process_basis_functions 
--spin_path './src/basis_sets/references/raw_basis_functions'
--save_subdir 'references/raw_basis_functions'
--save_name_prefix 'raw'
--pulse_sequence 'COWS7_sLASER'
--vendor 'Siemens'
--centerFreq 4.65
--sim_software 'MARSS'
$ python -m src.process_basis_functions --spin_path './src/basis_sets/references/raw_basis_functions' --save_subdir 'references/raw_basis_functions' --save_name_prefix 'raw' --pulse_sequence 'COWS7_sLASER' --vendor 'Siemens' --centerFreq 4.65 --sim_software 'MARSS'
TODO: Some of these formats will need json sidecar files or something
TODO: The .RAW basis functions are reversed spectrally

# .MAT - Successful
python -m src.process_basis_functions 
--spin_path './src/basis_sets/references/basis_functions_metab_mm_mat'
--save_subdir 'references/basis_functions_metab_mm_mat'
--save_name_prefix 'mat'
--pulse_sequence 'COWS7_sLASER'
--vendor 'Siemens'
--centerFreq 4.65
--sim_software 'MARSS'
$ python -m src.process_basis_functions --spin_path './src/basis_sets/references/basis_functions_metab_mm_mat' --save_subdir 'references/basis_functions_metab_mm_mat' --save_name_prefix 'mat' --pulse_sequence 'COWS7_sLASER' --vendor 'Siemens' --centerFreq 4.65 --sim_software 'MARSS'

# .BASIS - Successful but incomplete! Need to check standard .BASIS files
python -m src.process_basis_functions 
--spin_path './src/basis_sets/references/basis_basis_functions'
--save_subdir 'references/basis_basis_functions'
--save_name_prefix 'basis'
--pulse_sequence 'COWS7_sLASER'
--vendor 'Siemens'
--centerFreq 4.65
--sim_software 'MARSS'
$ python -m src.process_basis_functions --spin_path './src/basis_sets/references/basis_basis_functions' --save_subdir 'references/basis_basis_functions' --save_name_prefix 'basis' --pulse_sequence 'COWS7_sLASER' --vendor 'Siemens' --centerFreq 4.65 --sim_software 'MARSS'
TODO: .BASIS FIDs are actually stored in the frequency domain and flipped???
TODO: This just might be a quirk of MRSCloud

# .MAT - MARSS - Successful
python -m src.process_basis_functions 
--spin_path '/home/john/Documents/Repositories/MARSSCompiled/VERI_GE_PRESS_30ms/SummedSpins_for_MARSSinput'
--save_subdir 'references/VERI_MARSS'
--save_name_prefix 'MARSS_mat'
--pulse_sequence 'PRESS'
--vendor 'GE'
--centerFreq 4.65
--sim_software 'MARSS'
$ python -m src.process_basis_functions --spin_path '/home/john/Documents/Repositories/MARSSCompiled/VERI_GE_PRESS_30ms/SummedSpins_for_MARSSinput' --save_subdir 'references/VERI_MARSS' --save_name_prefix 'MARSS_mat' --pulse_sequence 'PRESS' --vendor 'GE' --centerFreq 4.65 --sim_software 'MARSS'
TODO: .BASIS FIDs are actually stored in the frequency domain and flipped???
TODO: This just might be a quirk of MRSCloud
'''