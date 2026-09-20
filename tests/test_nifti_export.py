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
