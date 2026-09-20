"""
Regression tests for src.NIfTIMRS.mat2niftimrs (v2.0 handover section 12).

Before this milestone, NIfTI-MRS export crashed unconditionally for every
simulated dataset: a KeyError on a 'dwelltime' field that never exists in
a simulated dataset's header, and (for noise=False datasets specifically)
an axis-misalignment caused by scipy.io.loadmat's squeeze_me=True silently
removing the noisy/clean axis whenever it has size 1. Both are exercised
here against a synthetic .mat fixture matching mainFcns._save()'s schema,
so this doesn't depend on a real basis set (see test_parameters.py's
module docstring for why). End-to-end verification against a real
simulated dataset is recorded in docs/v2/progress_log.md.
"""
import json

import numpy as np
import pytest

from src.NIfTIMRS import Mat2NIfTI_MRS


def _save_synthetic_single_tone_dataset(path, f_offset_hz, spectralwidth=2000.0,
                                         carrier_frequency=127.7, n_pts=4096, T2=0.2):
    """
    A synthetic, single-metabolite-like FID with a *known* offset frequency,
    built the same way a basis-set FID would need to look for
    src.aux.aux.Fourier_Transform (fft, then fftshift) to place its peak at
    f_offset_hz: fid(t) = exp(-t/T2) * exp(i*2*pi*f_offset_hz*t). No real
    basis set is needed -- see the module docstring for why committed tests
    avoid depending on one.

    Uses batch=2/noisy_clean_axis_size=2 (both entries identical copies of
    the same tone) rather than size-1 axes: a batch of 1 *and* a noisy/clean
    axis of 1 together hit a separate, pre-existing scipy.io.loadmat
    squeeze_me=True edge case (both size-1 axes get squeezed away at once,
    which mat2niftimrs.py's existing ndim==3 restoration doesn't cover) --
    out of scope for this test, tracked in docs/v2/progress_log.md instead.
    """
    from scipy.io import savemat
    dt = 1.0 / spectralwidth
    t = np.arange(n_pts) * dt
    fid = np.exp(-t / T2) * np.exp(1j * 2 * np.pi * f_offset_hz * t)
    fid_arr = np.stack([fid.real, fid.imag], axis=0).astype(np.float32)  # (2, N)
    # (batch=2, noisy/clean=2, channels=2, length=N)
    spectra = np.broadcast_to(fid_arr, (2, 2) + fid_arr.shape).copy()
    mdict = {
        'spectra': spectra,
        'spectral_fit': spectra.copy(),
        'baselines': [],
        'residual_water': [],
        'params': np.zeros((2, 3), dtype=np.float32),
        'quantities': {'coefficient': np.zeros(2, dtype=np.float32)},
        'SNR': {'power': np.zeros(2, dtype=np.float32)},
        'cropRange': [0.2, 4.2],
        'ppm': np.linspace(0.2, 4.2, n_pts).astype(np.float32),
        'header': {'carrier_frequency': carrier_frequency, 'TE': 30.0,
                   'spectralwidth': spectralwidth},
    }
    savemat(str(path), mdict=mdict)


def _save_synthetic_dataset(path, noisy_clean_axis_size, seed=0):
    from scipy.io import savemat
    rng = np.random.default_rng(seed)
    # (batch, noisy/clean, channels, length) -- matches mainFcns._save()'s
    # 'spectra' field shape. noisy_clean_axis_size=1 reproduces the
    # noise=False case that used to trigger the squeeze_me bug.
    spectra = rng.standard_normal((2, noisy_clean_axis_size, 2, 32)).astype(np.float32)
    mdict = {
        'spectra': spectra,
        'spectral_fit': spectra.copy(),
        'baselines': [],
        'residual_water': [],
        'params': np.zeros((2, 3), dtype=np.float32),
        'quantities': {'coefficient': np.zeros(2, dtype=np.float32)},
        'SNR': {'power': np.zeros(2, dtype=np.float32)},
        'cropRange': [0.2, 4.2],
        'ppm': np.linspace(0.2, 4.2, 32).astype(np.float32),
        # Deliberately no 'dwelltime' key -- matches a real simulated
        # dataset's header exactly (PhysicsModel.header never has one).
        'header': {'carrier_frequency': 127.7, 'TE': 30.0, 'spectralwidth': 2000.0},
    }
    savemat(str(path), mdict=mdict)
    return spectra


def _save_synthetic_dataset_distinct_noisy_clean(path, n_pts=64, seed=0):
    """
    Noisy (index 0) and clean/noise-free (index 1) entries with distinct,
    recognizable content, so a test can confirm which one actually got
    written to a given exported file.
    """
    from scipy.io import savemat
    rng = np.random.default_rng(seed)
    noisy = rng.standard_normal((2, 2, n_pts)).astype(np.float32)   # (batch, channels, length)
    clean = noisy * 0.0 + 5.0                                       # trivially distinct constant
    spectra = np.stack([noisy, clean], axis=1)  # (batch, noisy/clean=2, channels, length)
    mdict = {
        'spectra': spectra,
        'spectral_fit': spectra.copy(),
        'baselines': [],
        'residual_water': [],
        'params': np.zeros((2, 3), dtype=np.float32),
        'quantities': {'coefficient': np.zeros(2, dtype=np.float32)},
        'SNR': {'power': np.zeros(2, dtype=np.float32)},
        'cropRange': [0.2, 4.2],
        'ppm': np.linspace(0.2, 4.2, n_pts).astype(np.float32),
        'header': {'carrier_frequency': 127.7, 'TE': 30.0, 'spectralwidth': 2000.0},
    }
    savemat(str(path), mdict=mdict)
    return noisy, clean


def test_export_does_not_crash_with_noisy_clean_axis(tmp_path):
    path = tmp_path / 'dataset.mat'
    _save_synthetic_dataset(path, noisy_clean_axis_size=2)

    exporter = Mat2NIfTI_MRS(test_output=False)
    exporter.forward(datapath=str(path))

    nii_path = tmp_path / 'dataset.nii.gz'
    assert nii_path.exists()


def test_export_does_not_crash_when_noisy_clean_axis_is_squeezed_away(tmp_path):
    """
    Regression test for the squeeze_me=True bug: a size-1 noisy/clean axis
    (the noise=False case) used to be silently removed on load, shifting
    every subsequent axis and crashing the real/imaginary combining step.
    """
    path = tmp_path / 'dataset_noise_false.mat'
    _save_synthetic_dataset(path, noisy_clean_axis_size=1)

    exporter = Mat2NIfTI_MRS(test_output=False)
    exporter.forward(datapath=str(path))

    nii_path = tmp_path / 'dataset_noise_false.nii.gz'
    assert nii_path.exists()


def test_export_computes_dwelltime_from_spectralwidth_when_missing(tmp_path):
    """Regression test for the dwelltime KeyError: header never has a
    'dwelltime' field for a simulated dataset; it must be derived from
    spectralwidth (dwelltime = 1/spectralwidth) instead of crashing."""
    import nibabel as nib

    path = tmp_path / 'dataset.mat'
    _save_synthetic_dataset(path, noisy_clean_axis_size=2)

    exporter = Mat2NIfTI_MRS(test_output=False)
    exporter.forward(datapath=str(path))

    img = nib.load(str(tmp_path / 'dataset.nii.gz'))
    assert img.header['pixdim'][4] == pytest.approx(1.0 / 2000.0)


def test_export_corrects_nifti_mrs_frequency_sign_convention(tmp_path):
    """
    Regression test (v2.0 handover section 12, ppm-direction follow-up):
    MRS-Sim's internal ppm axis is ppm = +f_Hz/sf + centerFreq (ascending
    with array index -- confirmed directly against
    src/aux/process_basis_functions.py and empirically against real basis
    sets, see docs/v2/progress_log.md). The NIfTI-MRS / Levitt convention
    (confirmed directly against the spec text) is the mirror image:
    ppm = -f_Hz/f0 + ref. Fed the raw FID unmodified, a spec-compliant
    reader would place a known resonance at the wrong ppm (mirrored about
    the reference). Mat2NIfTI_MRS.forward() must conjugate the FID at
    export time to correct this -- verified here with a synthetic
    single-tone FID at a known offset frequency, checked against the
    spec's own S(f)=fftshift(fft(fid)) + ppm=-f_Hz/f0+ref formula (not
    MRS-Sim's own ppm convention, which would trivially agree with
    whatever MRS-Sim itself writes).
    """
    import nibabel as nib

    spectralwidth = 2000.0
    carrier_frequency = 127.7
    ppm_reference = 4.65
    f_offset_hz = -150.0
    n_pts = 4096
    target_ppm = ppm_reference + f_offset_hz / carrier_frequency

    path = tmp_path / 'dataset.mat'
    _save_synthetic_single_tone_dataset(path, f_offset_hz=f_offset_hz,
                                         spectralwidth=spectralwidth,
                                         carrier_frequency=carrier_frequency,
                                         n_pts=n_pts)

    exporter = Mat2NIfTI_MRS(test_output=False)
    exporter.forward(datapath=str(path))

    img = nib.load(str(tmp_path / 'dataset.nii.gz'))
    fid = np.asarray(img.get_fdata(dtype=np.complex64)).reshape(-1, n_pts)[0]

    # Independent, spec-compliant reconstruction: S(f) = fftshift(fft(fid)),
    # ppm = -f_Hz/f0 + ref (the exact formula in
    # src/NIfTIMRS/convention_testing.py's _ppm_axis()).
    spec = np.fft.fftshift(np.fft.fft(fid))
    step = spectralwidth / n_pts
    f_hz = np.arange(n_pts) * step + (-spectralwidth / 2.0 + step / 2.0)
    ppm_spec = -(f_hz / (carrier_frequency * 1e6)) * 1e6 + ppm_reference

    peak_idx = np.argmax(np.abs(spec))
    assert ppm_spec[peak_idx] == pytest.approx(target_ppm, abs=0.05)


def test_export_uses_configurable_reference_peak_not_hardcoded_4_65ppm(tmp_path):
    """
    Regression test: expected_peak_ppm/ppm_reference used to be hardcoded to
    4.65 inside test_output(), so the convention check would misfire for any
    dataset whose dominant peak isn't a 4.65ppm water reference. Both are now
    Mat2NIfTI_MRS constructor parameters (default 4.65, preserving prior
    behavior) so a caller can point the check at their actual data.
    """
    spectralwidth = 2000.0
    carrier_frequency = 127.7
    ppm_reference = 2.0   # e.g. a metabolite-only simulation, no water peak
    f_offset_hz = 0.0     # peak exactly at the reference for a simple check
    n_pts = 4096

    path = tmp_path / 'dataset.mat'
    _save_synthetic_single_tone_dataset(path, f_offset_hz=f_offset_hz,
                                         spectralwidth=spectralwidth,
                                         carrier_frequency=carrier_frequency,
                                         n_pts=n_pts)

    exporter = Mat2NIfTI_MRS(test_output=True, ppm_reference=ppm_reference)
    assert exporter.expected_peak_ppm == ppm_reference
    # Would fail convention_testing.py's peak-location assertion if the old
    # hardcoded 4.65 were still in effect (0.0 offset is 2.65ppm away from
    # 4.65, well outside the default 0.5ppm search window).
    exporter.forward(datapath=str(path))


def test_export_noise_free_label_writes_a_sibling_file_not_a_subdirectory(tmp_path):
    """
    Regression test: label='noise_free' used to build a save_name via
    os.path.join(save_name, label), which is a *subdirectory* path that
    nothing creates -- nib.save() would raise FileNotFoundError the first
    time this branch actually ran (it was never invoked anywhere in the
    codebase). It must instead write a sibling file next to, and named
    after, the corresponding simulated-data export, and it must contain
    the clean/noise-free entry (index 1), not the noisy one (index 0).
    """
    import nibabel as nib

    path = tmp_path / 'dataset.mat'
    noisy, clean = _save_synthetic_dataset_distinct_noisy_clean(path)

    exporter = Mat2NIfTI_MRS(test_output=False)
    exporter.forward(datapath=str(path))
    exporter.forward(datapath=str(path), label='noise_free')

    default_path = tmp_path / 'dataset.nii.gz'
    noise_free_path = tmp_path / 'dataset_noise_free.nii.gz'
    assert default_path.exists()
    assert noise_free_path.exists()
    # Must be a sibling file, not e.g. tmp_path/'dataset'/'noise_free.nii.gz'
    assert noise_free_path.parent == tmp_path

    expected_clean_complex = np.conj(clean[:, 0, :] + 1j * clean[:, 1, :])
    written = nib.load(str(noise_free_path)).get_fdata(dtype=np.complex64)
    np.testing.assert_allclose(written.reshape(-1, expected_clean_complex.shape[-1]),
                                expected_clean_complex, atol=1e-4)


def test_repetition_time_is_json_null_not_a_string(tmp_path):
    """
    Regression test: RepetitionTime used to be the string 'NA', which
    violates the NIfTI-MRS spec's requirement that it be a number (or
    JSON null) when present. TR isn't tracked anywhere in this codebase,
    so None (-> JSON null) is the correct, spec-compliant representation.
    """
    import nibabel as nib

    path = tmp_path / 'dataset.mat'
    _save_synthetic_dataset(path, noisy_clean_axis_size=2)

    exporter = Mat2NIfTI_MRS(test_output=False)
    exporter.forward(datapath=str(path))

    img = nib.load(str(tmp_path / 'dataset.nii.gz'))
    ext = json.loads(img.header.extensions[0].get_content())
    assert ext['RepetitionTime'] is None
    assert ext['EchoTime'] == pytest.approx(0.03)
