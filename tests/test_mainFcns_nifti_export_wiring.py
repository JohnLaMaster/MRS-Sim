"""
Regression tests for the NIfTI-MRS noise-free export wiring in
src.mainFcns.simulate() (v2.0 handover section 12, follow-up to
docs/v2/progress_log.md Milestone 15).

Before this fix, mainFcns.simulate() only ever called
Mat2NIfTI_MRS.forward() once, with no `label` -- the noise-free/clean
counterpart (always present alongside the noisy one whenever noise was
simulated) was silently never exported, and the `label='noise_free'` path
that would have exported it separately built a broken subdirectory path
(see test_nifti_export.py's
test_export_noise_free_label_writes_a_sibling_file_not_a_subdirectory).

Uses a fake PhysicsModel (no real basis set needed -- see
test_parameters.py's module docstring for why) so this exercises the
actual mainFcns.simulate() control flow rather than re-testing
Mat2NIfTI_MRS in isolation.
"""
import numpy as np
import torch
from types import SimpleNamespace

from src.mainFcns import simulate


class _FakePhysicsModel:
    """Stands in for PhysicsModel: same forward() contract simulate() relies
    on (7-tuple output, .ppm, .cropRange), fixed synthetic content."""

    def __init__(self, n_pts=64):
        self.ppm = torch.tensor(np.linspace(0.2, 4.2, n_pts))
        self.cropRange = [0.2, 4.2]
        self.n_pts = n_pts

    def forward(self, params, **kwargs):
        batch = params.shape[0]
        rng = np.random.default_rng(0)
        t = np.arange(self.n_pts) / self.n_pts
        # Decaying, zero-offset-frequency signal: a real exponential decay
        # has all its power at DC, which lands at ppm==ppm_reference after
        # export -- satisfies both convention_testing.py's decay check and
        # its default expected_peak_ppm=4.65 dominant-peak check.
        envelope = np.exp(-t / 0.2).astype(np.float32)
        real_clean = np.broadcast_to(envelope, (batch, self.n_pts)).copy()
        imag_clean = np.zeros_like(real_clean)
        noise = rng.standard_normal((batch, 2, self.n_pts)).astype(np.float32) * 0.01 * envelope
        noisy = np.stack([real_clean, imag_clean], axis=1) + noise
        clean = np.stack([real_clean, imag_clean], axis=1)
        spectra = np.stack([noisy, clean], axis=1)  # (batch, noisy/clean=2, ch=2, N)
        quantities = {'coefficient': np.zeros(batch, dtype=np.float32)}
        snr = {'power': np.zeros(batch, dtype=np.float32)}
        return spectra, spectra.copy(), None, None, params.copy(), quantities, snr


def _run_simulate(tmp_path, niftimrs_noise_free):
    n_entries = 2
    config = SimpleNamespace(
        NIfTIMRS=True, NIfTIMRS_noise_free=niftimrs_noise_free,
        totalEntries=n_entries, b0=False, eddy=False, fids=True,
        phi0=False, phi1=False, noise=True, apodize=False,
        fshift_g=False, fshift_i=False, resample=False,
        coil_phi0=False, coil_sens=False, magnitude=False, num_coils=1,
        zero_fill=False, broadening=False, coil_fshift=False,
        drop_prob=0.0, header={'carrier_frequency': 127.7, 'TE': 30.0,
                                'spectralwidth': 2000.0},
    )
    args = SimpleNamespace(stepSize=n_entries, batchSize=10, savedir=str(tmp_path))
    pm = _FakePhysicsModel()
    params = np.zeros((n_entries, 3), dtype=np.float32)
    ind = {'a': [0], 'b': [1], 'c': [2]}
    inputs = (config, None, None, pm, n_entries, ind, 1.0, n_entries, params, None, None)
    simulate(inputs, args=args)


def test_noise_free_export_not_written_by_default(tmp_path):
    _run_simulate(tmp_path, niftimrs_noise_free=False)
    assert (tmp_path / 'dataset_spectra_0.nii.gz').exists()
    assert not (tmp_path / 'dataset_spectra_0_noise_free.nii.gz').exists()


def test_noise_free_export_written_as_sibling_file_when_requested(tmp_path):
    _run_simulate(tmp_path, niftimrs_noise_free=True)
    assert (tmp_path / 'dataset_spectra_0.nii.gz').exists()
    assert (tmp_path / 'dataset_spectra_0_noise_free.nii.gz').exists()


def test_noise_free_export_defaults_to_off_when_key_absent(tmp_path):
    """getattr(config, 'NIfTIMRS_noise_free', False) -- existing configs
    that predate this flag must keep exporting only the noisy variant."""
    n_entries = 2
    config = SimpleNamespace(
        NIfTIMRS=True,  # deliberately no NIfTIMRS_noise_free attribute
        totalEntries=n_entries, b0=False, eddy=False, fids=True,
        phi0=False, phi1=False, noise=True, apodize=False,
        fshift_g=False, fshift_i=False, resample=False,
        coil_phi0=False, coil_sens=False, magnitude=False, num_coils=1,
        zero_fill=False, broadening=False, coil_fshift=False,
        drop_prob=0.0, header={'carrier_frequency': 127.7, 'TE': 30.0,
                                'spectralwidth': 2000.0},
    )
    args = SimpleNamespace(stepSize=n_entries, batchSize=10, savedir=str(tmp_path))
    pm = _FakePhysicsModel()
    params = np.zeros((n_entries, 3), dtype=np.float32)
    ind = {'a': [0], 'b': [1], 'c': [2]}
    inputs = (config, None, None, pm, n_entries, ind, 1.0, n_entries, params, None, None)
    simulate(inputs, args=args)

    assert (tmp_path / 'dataset_spectra_0.nii.gz').exists()
    assert not (tmp_path / 'dataset_spectra_0_noise_free.nii.gz').exists()
