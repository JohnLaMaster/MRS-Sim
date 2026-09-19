"""
Unit tests for src.nuisance (v2.0 handover section 4).

remove_nuisance_from_saved() is tested against a synthetic .mat file built
to match mainFcns._save()'s exact schema, rather than a real simulated
dataset -- basis-set files aren't tracked in git (see
test_parameters.py's module docstring), so a real end-to-end round trip
was instead verified manually and is recorded in
docs/v2/progress_log.md (Milestone 5), including the known approximation
gap documented in src/nuisance.py's module docstring.
"""
import numpy as np
import pytest
import torch

from src.nuisance import remove_nuisance, remove_nuisance_from_saved


def test_remove_nuisance_subtracts_both_components_torch():
    noise_free_total = torch.full((3, 4), 10.0)
    baseline = torch.full((3, 4), 2.0)
    residual_water = torch.full((3, 4), 1.0)
    out = remove_nuisance(noise_free_total, baseline, residual_water)
    torch.testing.assert_close(out, torch.full((3, 4), 7.0))


def test_remove_nuisance_subtracts_only_provided_components_numpy():
    noise_free_total = np.full((3, 4), 10.0)
    out = remove_nuisance(noise_free_total, baseline=np.full((3, 4), 2.0))
    np.testing.assert_allclose(out, np.full((3, 4), 8.0))


def test_remove_nuisance_no_components_returns_input_unchanged():
    noise_free_total = np.full((2, 2), 5.0)
    out = remove_nuisance(noise_free_total)
    np.testing.assert_allclose(out, noise_free_total)


def _save_synthetic_dataset(path, baselines=None, residual_water=None, seed=0):
    from scipy.io import savemat
    rng = np.random.default_rng(seed)
    spectra = rng.standard_normal((4, 2, 2, 16)).astype(np.float32)
    mdict = {
        'spectra': spectra,
        'spectral_fit': spectra.copy(),
        'baselines': baselines if baselines is not None else [],
        'residual_water': residual_water if residual_water is not None else [],
        'params': np.zeros((4, 3), dtype=np.float32),
        'quantities': {'coefficient': np.zeros(4, dtype=np.float32)},
        'SNR': {'power': np.zeros(4, dtype=np.float32)},
        'cropRange': [0.2, 4.2],
        'ppm': np.linspace(0.2, 4.2, 16).astype(np.float32),
        'header': {'note': 'synthetic test fixture'},
    }
    savemat(str(path), mdict=mdict)
    return spectra


def test_remove_nuisance_from_saved_clean_branch(tmp_path):
    # Shape (batch, 1, channels, length), matching what
    # PhysicsModel.compile_outputs() actually saves -- scipy.io.loadmat
    # squeezes the size-1 axis back out on load (verified directly), so the
    # expected value is computed against the *squeezed* arrays.
    rng = np.random.default_rng(1)
    baselines = rng.standard_normal((4, 1, 2, 16)).astype(np.float32)
    residual_water = rng.standard_normal((4, 1, 2, 16)).astype(np.float32)
    path = tmp_path / 'dataset.mat'
    spectra = _save_synthetic_dataset(path, baselines=baselines, residual_water=residual_water)

    out = remove_nuisance_from_saved(str(path), branch='clean')
    expected = spectra[:, 1, ...] - baselines.squeeze(1) - residual_water.squeeze(1)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_remove_nuisance_from_saved_noisy_branch(tmp_path):
    rng = np.random.default_rng(2)
    baselines = rng.standard_normal((4, 1, 2, 16)).astype(np.float32)
    path = tmp_path / 'dataset.mat'
    spectra = _save_synthetic_dataset(path, baselines=baselines)

    out = remove_nuisance_from_saved(str(path), branch='noisy')
    expected = spectra[:, 0, ...] - baselines.squeeze(1)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_remove_nuisance_from_saved_invalid_branch_raises(tmp_path):
    path = tmp_path / 'dataset.mat'
    _save_synthetic_dataset(path, baselines=np.zeros((4, 1, 2, 16), dtype=np.float32))
    with pytest.raises(ValueError):
        remove_nuisance_from_saved(str(path), branch='bogus')


def test_remove_nuisance_from_saved_raises_when_nothing_to_remove(tmp_path):
    path = tmp_path / 'no_offsets.mat'
    _save_synthetic_dataset(path, baselines=None, residual_water=None)
    with pytest.raises(ValueError):
        remove_nuisance_from_saved(str(path))
