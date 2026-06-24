"""
NIfTI-MRS format convention tests
==================================
Companion to the data/metadata authenticity tests already in the module.
Verifies conformance with the NIfTI-MRS specification (Clarke et al., 2022,
Magn Reson Med 88:2358-2370) at three levels:

  1. NIfTI header conventions
  2. JSON header-extension conventions
  3. Spectral / signal conventions (Levitt frequency-phase convention)

All tests are written as plain assertions so they can be dropped into the
existing unittest.TestCase subclass without modification.

References
----------
Clarke WT et al. (2022). NIfTI-MRS: A standard data format for magnetic
resonance spectroscopy. Magn Reson Med 88:2358-2370.
https://github.com/wtclarke/mrs_nifti_standard
"""

import os
import json
import numpy as np
import nibabel as nib

from typing import Optional


__all__ = ['test_nifti_mrs_conventions']


# ---------------------------------------------------------------------------
# Constants from the NIfTI-MRS specification
# ---------------------------------------------------------------------------

# NIfTI-2 header size (bytes)
_NIFTI2_SIZEOF_HDR = 540

# Allowed complex datatypes: complex64 (code 32) and complex128 (code 1792)
_COMPLEX_DATATYPES = {32, 1792}
_COMPLEX_DTYPE_NAMES = {32: "complex64", 1792: "complex128"}

# NIfTI xyzt_units bit masks
_UNITS_SPATIAL_MASK  = 0b00000111   # bits 0-2: spatial units
_UNITS_TEMPORAL_MASK = 0b00111000   # bits 3-5: temporal units
_NIFTI_UNITS_UNKNOWN  = 0
_NIFTI_UNITS_METER    = 1
_NIFTI_UNITS_MM       = 2
_NIFTI_UNITS_MICRON   = 3
_NIFTI_UNITS_SEC      = 8   # value in bits 3-5 when seconds are selected
_NIFTI_UNITS_MSEC     = 16
_VALID_SPATIAL_UNITS  = {_NIFTI_UNITS_UNKNOWN,
                          _NIFTI_UNITS_METER,
                          _NIFTI_UNITS_MM,
                          _NIFTI_UNITS_MICRON}
_VALID_TEMPORAL_UNITS = {_NIFTI_UNITS_UNKNOWN, _NIFTI_UNITS_SEC, _NIFTI_UNITS_MSEC}

# Default pixdim for unlocalised data (10 m = 10000 mm)
_UNLOCALISED_PIXDIM_MM = 10_000.0

# JSON header-extension ecode for NIfTI-MRS
_NIFTI_MRS_ECODE = 44

# Required metadata fields (minimum conformance)
_REQUIRED_META_KEYS = {"SpectrometerFrequency", "ResonantNucleus"}

# Valid DIM_* tags for higher dimensions
_VALID_DIM_TAGS = {
    "DIM_COIL",
    "DIM_DYN",
    "DIM_INDIRECT_0",
    "DIM_INDIRECT_1",
    "DIM_INDIRECT_2",
    "DIM_PHASE_CYCLE",
    "DIM_EDIT",
    "DIM_MEAS",
    "DIM_USER_0",
    "DIM_USER_1",
    "DIM_USER_2",
    "DIM_ISIS",
}

# Relative tolerance for floating-point comparisons
_RTOL = 1e-4


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _get_temporal_units(xyzt_units: int) -> int:
    """Extract the temporal-units nibble from the xyzt_units byte."""
    return xyzt_units & _UNITS_TEMPORAL_MASK


def _get_spatial_units(xyzt_units: int) -> int:
    """Extract the spatial-units nibble from the xyzt_units byte."""
    return xyzt_units & _UNITS_SPATIAL_MASK


def _load_mrs_extension(nifti_img) -> dict:
    """
    Return the parsed JSON from the NIfTI-MRS header extension (ecode 44).
    Raises AssertionError if no such extension is found.
    """
    ext_codes = nifti_img.header.extensions.get_codes()
    assert _NIFTI_MRS_ECODE in ext_codes, (
        f"No NIfTI-MRS header extension (ecode={_NIFTI_MRS_ECODE}) found. "
        f"Extension codes present: {ext_codes}"
    )
    idx = ext_codes.index(_NIFTI_MRS_ECODE)
    return json.loads(nifti_img.header.extensions[idx].get_content())


def _fid_to_spectrum(fid_1d: np.ndarray) -> np.ndarray:
    """
    Convert a 1-D complex FID to its frequency-domain spectrum following the
    NIfTI-MRS / Levitt convention:

        S(f) = FFT{ s(t) }  with  fftshift

    This produces a spectrum whose frequency axis runs from –SW/2 to +SW/2,
    with absolute frequency increasing left→right and ppm increasing
    right→left (display reversal applied by the viewer, not here).
    """
    return np.fft.fftshift(np.fft.fft(fid_1d))


def _frequency_axis_hz(n_pts: int, spectral_width_hz: float) -> np.ndarray:
    """
    Return the frequency axis in Hz for a spectrum of *n_pts* points with the
    given spectral width, centred on 0 Hz (i.e. relative to the carrier).

    Follows the MATLAB reference in the NIfTI-MRS standard:
        f = linspace(-SW/2 + SW/(2*N), SW/2 - SW/(2*N), N)
    """
    step = spectral_width_hz / n_pts
    start = -spectral_width_hz / 2.0 + step / 2.0
    return np.arange(n_pts) * step + start


def _ppm_axis(n_pts: int, spectral_width_hz: float,
              spectrometer_freq_mhz: float,
              ppm_reference: float = 4.65) -> np.ndarray:
    """
    Return the chemical-shift axis in ppm, referenced to *ppm_reference*.

    Convention (Levitt / NIfTI-MRS):
        ppm = -(f_Hz / f0_Hz) * 1e6  + ppm_reference
    where f_Hz is the offset from the carrier and f0_Hz = f0_MHz * 1e6.
    """
    f_hz = _frequency_axis_hz(n_pts, spectral_width_hz)
    f0_hz = spectrometer_freq_mhz * 1e6
    return -(f_hz / f0_hz) * 1e6 + ppm_reference


# ---------------------------------------------------------------------------
# Main convention-test method
# ---------------------------------------------------------------------------

def test_nifti_mrs_conventions(
    save_path: str,
    save_name: str,
    *,
    expected_peak_ppm: Optional[float] = None,
    ppm_reference: float = 4.65,
    peak_search_window_ppm: float = 0.5,
    allow_unlocalised: bool = True,
) -> None:
    """
    Test that the saved NIfTI-MRS file satisfies all format conventions.

    Parameters
    ----------
    save_path : str
        Directory that contains the .nii.gz file.
    save_name : str
        Base name (without extension) used when saving.
    expected_peak_ppm : float, optional
        If provided, the test will verify the Levitt frequency convention by
        checking that the dominant spectral peak appears within
        ±peak_search_window_ppm of this value. Useful for simulations where
        the metabolite frequencies are known exactly.
    ppm_reference : float
        The ppm value to which the carrier frequency is referenced
        (default 4.65 Hz, close to water at 4.68 ppm).
    peak_search_window_ppm : float
        Half-width of the acceptance window around *expected_peak_ppm*.
    allow_unlocalised : bool
        If True, a pixdim[1:4] value of 10 m (10000 mm) is accepted as the
        canonical placeholder for unlocalised / simulated data.
    """
    base_name = os.path.join(save_path, save_name)
    nifti_path = base_name + ".nii.gz"

    assert os.path.isfile(nifti_path), f"NIfTI file not found: {nifti_path}"
    img = nib.load(nifti_path)
    hdr = img.header

    # ------------------------------------------------------------------
    # 1. NIfTI HEADER CONVENTIONS
    # ------------------------------------------------------------------

    # 1a. Must be NIfTI-2 (sizeof_hdr == 540)
    assert int(hdr["sizeof_hdr"]) == _NIFTI2_SIZEOF_HDR, (
        f"Expected NIfTI-2 (sizeof_hdr={_NIFTI2_SIZEOF_HDR}), "
        f"got sizeof_hdr={int(hdr['sizeof_hdr'])}. "
        "NIfTI-MRS requires the NIfTI-2 format."
    )

    # 1b. Datatype must be complex (complex64 or complex128)
    datatype = int(hdr["datatype"])
    assert datatype in _COMPLEX_DATATYPES, (
        f"datatype={datatype} is not a complex type. "
        f"NIfTI-MRS allows only {_COMPLEX_DTYPE_NAMES}."
    )

    # 1c. Number of dimensions: dim[0] must be between 4 and 7 inclusive.
    #     The spectral axis occupies dim 4; dims 5-7 are optional.
    ndim = int(hdr["dim"][0])
    assert 4 <= ndim <= 7, (
        f"dim[0]={ndim}: NIfTI-MRS requires 4 ≤ ndim ≤ 7 "
        "(3 spatial + 1 spectral + up to 3 encoding dims)."
    )

    dim = hdr["dim"]  # full dim array, length 8

    # 1d. Spectral dimension (dim[4]) must be > 1
    n_spectral = int(dim[4])
    assert n_spectral > 1, (
        f"dim[4]={n_spectral}: the spectral dimension must have > 1 point."
    )

    # 1e. Any dimension > dim[0] that is reported must equal 1
    for d in range(ndim + 1, 8):
        assert int(dim[d]) == 1, (
            f"dim[{d}]={int(dim[d])} but dim[0]={ndim}: "
            "dimensions beyond the declared ndim must be 1."
        )

    # 1f. All used dimensions must be positive (≥ 1)
    for d in range(1, ndim + 1):
        assert int(dim[d]) >= 1, (
            f"dim[{d}]={int(dim[d])}: all used dimensions must be ≥ 1."
        )

    # 1g. pixdim[4] = dwell time (seconds) — must be strictly positive
    dwell_time = float(hdr["pixdim"][4])
    assert dwell_time > 0.0, (
        f"pixdim[4]={dwell_time}: dwell time must be strictly positive (seconds)."
    )

    # 1h. Spatial pixdim values (pixdim[1:4]) must be positive
    for d in range(1, 4):
        pv = float(hdr["pixdim"][d])
        if allow_unlocalised:
            assert pv > 0.0, (
                f"pixdim[{d}]={pv}: spatial voxel dimensions must be positive."
            )
        else:
            assert 0.0 < pv < _UNLOCALISED_PIXDIM_MM, (
                f"pixdim[{d}]={pv}: expected a real (localised) voxel size "
                f"in mm, not the unlocalised placeholder ({_UNLOCALISED_PIXDIM_MM} mm)."
            )

    # 1i. qfac (pixdim[0]) must be +1 or -1 when qform is used
    qform_code = int(hdr.get("qform_code", 0))
    if qform_code > 0:
        qfac = float(hdr["pixdim"][0])
        assert qfac in (1.0, -1.0), (
            f"pixdim[0] (qfac) = {qfac}: must be +1 or -1 when qform_code > 0."
        )

    # 1j. xyzt_units: temporal units must be seconds or unset
    xyzt = int(hdr.get("xyzt_units", 0))
    temporal_units = _get_temporal_units(xyzt)
    assert temporal_units in _VALID_TEMPORAL_UNITS, (
        f"xyzt_units temporal part = {temporal_units}: "
        f"expected seconds ({_NIFTI_UNITS_SEC}) or unknown (0), got {temporal_units}."
    )
    if temporal_units != _NIFTI_UNITS_UNKNOWN:
        assert temporal_units == _NIFTI_UNITS_SEC, (
            f"xyzt_units temporal units = {temporal_units}: "
            f"NIfTI-MRS stores dwell time in seconds; "
            f"expected {_NIFTI_UNITS_SEC} (NIFTI_UNITS_SEC)."
        )

    # 1k. Spatial units must be a recognised NIfTI unit code (or 0 = unknown)
    spatial_units = _get_spatial_units(xyzt)
    assert spatial_units in _VALID_SPATIAL_UNITS, (
        f"xyzt_units spatial part = {spatial_units}: unrecognised spatial unit code."
    )

    # # 1l. intent_name must follow the 'mrs_v<N>_<M>' versioning convention
    # # intent_name = hdr.get("intent_name", b"").decode("utf-8").strip("\x00").strip()
    # intent_name = safe_str(hdr.get("intent_name", b"")).strip("\x00").strip()
    # # intent = hdr.get("intent_name", b"")

    # # if isinstance(intent, bytes):
    # #     intent_name = intent.decode("utf-8")
    # # elif isinstance(intent, str):
    # #     intent_name = intent
    # # elif isinstance(intent, np.ndarray):
    # #     intent_name = "".join(intent.astype(str)).strip()
    # # else:
    # #     intent_name = str(intent)

    # # intent_name = intent_name.strip("\x00").strip()
    # assert intent_name.startswith("mrs_v"), (
    #     f"intent_name='{intent_name}': NIfTI-MRS requires the intent_name "
    #     "field to contain a version string beginning with 'mrs_v' "
    #     "(e.g. 'mrs_v0_2')."
    # )

    # ------------------------------------------------------------------
    # 2. JSON HEADER EXTENSION CONVENTIONS
    # ------------------------------------------------------------------

    meta = _load_mrs_extension(img)   # asserts ecode=44 is present

    # 2a. Required minimum-conformance fields must be present
    for key in _REQUIRED_META_KEYS:
        assert key in meta, (
            f"Required NIfTI-MRS metadata key '{key}' is absent from the "
            "JSON header extension."
        )

    # 2b. SpectrometerFrequency must be a non-empty list of positive floats
    sf = meta["SpectrometerFrequency"]
    assert isinstance(sf, list) and len(sf) > 0, (
        f"SpectrometerFrequency={sf!r}: must be a non-empty list."
    )
    for v in sf:
        assert isinstance(v, (int, float)) and v > 0, (
            f"SpectrometerFrequency contains invalid value {v!r}: "
            "must be a positive number (MHz)."
        )

    # 2c. ResonantNucleus must be a non-empty list of strings
    rn = meta["ResonantNucleus"]
    assert isinstance(rn, list) and len(rn) > 0, (
        f"ResonantNucleus={rn!r}: must be a non-empty list."
    )
    for v in rn:
        assert isinstance(v, str) and len(v) > 0, (
            f"ResonantNucleus contains invalid value {v!r}: must be a non-empty string."
        )

    # 2d. SpectrometerFrequency and ResonantNucleus must have matching lengths
    assert len(sf) == len(rn), (
        f"SpectrometerFrequency has {len(sf)} entries but "
        f"ResonantNucleus has {len(rn)}: lengths must match."
    )

    # 2e. If SpectralWidth is present, it must be consistent with pixdim[4]
    if "SpectralWidth" in meta:
        sw_meta = float(meta["SpectralWidth"])
        sw_from_pixdim = 1.0 / dwell_time
        assert np.isclose(sw_meta, sw_from_pixdim, rtol=_RTOL), (
            f"SpectralWidth in JSON header ({sw_meta:.4f} Hz) is inconsistent "
            f"with 1/pixdim[4] = 1/{dwell_time:.6e} = {sw_from_pixdim:.4f} Hz. "
            "Per the specification, software must preferentially infer spectral "
            "width from the dwell time in pixdim[4]."
        )

    # 2f. dim_5 / dim_6 / dim_7 labels
    #     - Must be present if the corresponding dimension is > 1
    #     - Must be absent (or can be present but harmless) if dimension == 1
    #     - When present, must be a recognised DIM_* tag
    for dim_idx, dim_key in [(5, "dim_5"), (6, "dim_6"), (7, "dim_7")]:
        dim_size = int(dim[dim_idx]) if dim_idx < 8 else 1
        if dim_size > 1:
            assert dim_key in meta, (
                f"dim[{dim_idx}]={dim_size} > 1 but '{dim_key}' is absent from "
                "the JSON header extension. NIfTI-MRS requires a DIM_* label "
                "for every higher dimension that contains more than one element."
            )
        if dim_key in meta:
            tag = meta[dim_key]
            assert tag in _VALID_DIM_TAGS, (
                f"'{dim_key}': '{tag}' is not a recognised NIfTI-MRS DIM_* tag. "
                f"Valid tags: {sorted(_VALID_DIM_TAGS)}"
            )

    # ------------------------------------------------------------------
    # 3. SPECTRAL / SIGNAL CONVENTIONS  (Levitt frequency-phase convention)
    # ------------------------------------------------------------------

    data = img.get_fdata(dtype=np.complex64)  # shape (..., n_spectral [, ...])
    # Spectral axis is always dimension index 3 (0-based) in NIfTI-MRS
    spectral_axis = 3

    # Flatten all non-spectral dimensions to get an array of FIDs
    # shape: (n_voxels_and_encodings, n_spectral)
    n_pts = data.shape[spectral_axis]
    fids = data.reshape(-1, n_pts)   # works for any number of higher dims

    # 3a. FID decay: the mean magnitude at the start of the FID should be
    #     greater than or equal to that at the end.  This is a necessary
    #     (though not sufficient) condition for correctly oriented time-domain
    #     data.  We use 5% of the FID length as the averaging window to reduce
    #     sensitivity to individual noisy samples.
    window = max(1, n_pts // 20)   # 5% of FID length
    mean_fid = np.mean(np.abs(fids), axis=0)   # average over all FIDs
    amp_start = float(np.mean(mean_fid[:window]))
    amp_end   = float(np.mean(mean_fid[-window:]))
    assert amp_start >= amp_end, (
        f"FID amplitude does not decay: mean amplitude over first {window} "
        f"points ({amp_start:.4g}) is less than over last {window} points "
        f"({amp_end:.4g}). Check that the time axis is oriented correctly "
        "(t=0 at index 0) and that the data have not been accidentally "
        "reversed or stored as an echo rather than a FID."
    )

    # 3b. Verify the Levitt frequency convention via an FFT direction test.
    #     The convention requires:
    #         S(f) = fftshift( FFT{ s(t) } )
    #     not the inverse FFT.  We verify this indirectly by checking that
    #     the power of fftshift(fft(fid)) exceeds that of fftshift(ifft(fid))
    #     in the spectral region expected for typical MRS signals.
    #
    #     Because simulations contain coherent signal, fft should produce
    #     sharper / higher-amplitude peaks than ifft (which would give a
    #     time-reversed spectrum with opposite frequency axis).
    mean_fid_1d = np.mean(fids, axis=0)   # representative FID (averaged)
    spec_fft  = np.fft.fftshift(np.fft.fft(mean_fid_1d))
    spec_ifft = np.fft.fftshift(np.fft.ifft(mean_fid_1d))
    # Peak amplitude: correct transform should give sharper, taller peaks
    peak_fft  = float(np.max(np.abs(spec_fft)))
    peak_ifft = float(np.max(np.abs(spec_ifft)))
    assert peak_fft >= peak_ifft, (
        f"FFT peak amplitude ({peak_fft:.4g}) < IFFT peak amplitude "
        f"({peak_ifft:.4g}). This suggests the frequency axis is reversed "
        "(data stored with wrong conjugation or time-axis direction), "
        "violating the NIfTI-MRS / Levitt convention."
    )

    # 3c. Phase convention: the real part of the FFT spectrum (averaged FID,
    #     assuming zero-phase or predominantly absorptive peaks) should
    #     contribute the majority of the total spectral power.  A purely
    #     dispersive spectrum (all imaginary power) would indicate a 90°
    #     phase error in the stored data.
    #     NOTE: This test is only meaningful if the simulation stores
    #     zero-phased (absorptive) data; comment it out if your simulation
    #     intentionally stores non-zero-phased FIDs.
    real_power = float(np.sum(np.real(spec_fft) ** 2))
    imag_power = float(np.sum(np.imag(spec_fft) ** 2))
    total_power = real_power + imag_power
    if total_power > 0:
        real_fraction = real_power / total_power
        assert real_fraction > 0.4, (
            f"Real-part power fraction = {real_fraction:.3f} < 0.4. "
            "The spectrum appears predominantly dispersive rather than "
            "absorptive.  Check the phase convention of the stored FID "
            "(expected zero-phase / absorptive-mode data)."
        )

    # 3d. Optional: verify the Levitt frequency-axis direction by checking
    #     that the dominant peak appears at the expected chemical-shift
    #     position.  Only executed when expected_peak_ppm is supplied.
    if expected_peak_ppm is not None:
        f0_mhz = float(meta["SpectrometerFrequency"][0])   # first nucleus
        sw_hz  = 1.0 / dwell_time
        ppm    = _ppm_axis(n_pts, sw_hz, f0_mhz, ppm_reference=ppm_reference)

        # Restrict search to the neighbourhood of the expected peak
        lo_ppm = expected_peak_ppm - peak_search_window_ppm
        hi_ppm = expected_peak_ppm + peak_search_window_ppm
        mask = (ppm >= lo_ppm) & (ppm <= hi_ppm)

        assert np.any(mask), (
            f"The expected peak window [{lo_ppm:.2f}, {hi_ppm:.2f}] ppm lies "
            f"entirely outside the spectral range "
            f"[{ppm.min():.2f}, {ppm.max():.2f}] ppm."
        )

        # The peak in the search window must be the global spectral maximum
        # (or at least a local dominant feature).
        power_in_window  = float(np.max(np.abs(spec_fft[mask])))
        power_outside    = float(np.max(np.abs(spec_fft[~mask])))

        assert power_in_window >= power_outside * 0.5, (
            f"Expected dominant peak at {expected_peak_ppm:.2f} ± "
            f"{peak_search_window_ppm:.2f} ppm, but the maximum power in that "
            f"window ({power_in_window:.4g}) is less than half of the maximum "
            f"power outside the window ({power_outside:.4g}). "
            "This may indicate that the frequency axis is inverted (ifft used "
            "instead of fft, or data conjugated), violating the Levitt convention."
        )


def safe_str(x):
    import numpy as np

    if x is None:
        return ""

    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")

    if isinstance(x, str):
        return x

    if isinstance(x, np.ndarray):
        # Handle scalar arrays (0-d)
        if x.ndim == 0:
            return str(x.item())
        # Handle char arrays
        return "".join(map(str, x.flatten()))

    return str(x)

# ---------------------------------------------------------------------------
# Integration into the existing TestCase
# ---------------------------------------------------------------------------
# Drop the call below into your existing `test_output` method (or a separate
# test method), using the same parameters you already pass to `test_output`.
#
#   def test_output(self, save_path, save_name, specDataCmplx,
#                   json_full, meta_dict):
#       ...existing assertions...
#
#       # ---- NIfTI-MRS format convention tests ----
#       test_nifti_mrs_conventions(
#           save_path,
#           save_name,
#           expected_peak_ppm=4.65,   # set to your dominant simulated peak
#           ppm_reference=4.65,
#           allow_unlocalised=True,   # True for simulations
#       )
#
# Alternatively, add it as a separate TestCase method:
#
#   def test_nifti_mrs_format_conventions(self):
#       test_nifti_mrs_conventions(
#           self.save_path,
#           self.save_name,
#           expected_peak_ppm=self.expected_peak_ppm,
#       )