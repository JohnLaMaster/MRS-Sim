"""
Unit tests for src.sampling (UniformRangeSampler / CopulaInVivoSampler).

Uses a small fake PhysicsModel-like object (not a real PhysicsModel) so
these tests stay fast and independent of any basis-set file -- see
tests/test_parameters.py's module docstring for the same rationale.
"""
import json
import warnings

import numpy as np
import pytest
import torch

from src.sampling import CopulaInVivoSampler, UniformRangeSampler

METABOLITE_NAMES = ['cr', 'naa']

INDEX = {
    'cr': 0,
    'naa': 1,
    'd': (2, 3),
    'g': (4, 5),
    'f_shift': 6,
    'f_shifts': (7, 8),
    'snr': 9,
    'phi0': 10,
    'phi1': 11,
    'b0': 12,
    'b0_dir': (13, 14, 15),
    'ecc': (16, 17),
    'coil_snr': (18,),
    'coil_sens': (19,),
    'coil_fshift': (20,),
    'coil_phi0': (21,),
    'metabolites': (0, 1),
    'parameters': tuple(range(2, 22)),
    'overall': tuple(range(0, 22)),
}
N_COLUMNS = 22


class FakePhysicsModel:
    """Minimal stand-in exposing exactly what ParameterRegistry/samplers need."""

    def __init__(self, index=INDEX, metab_names=METABOLITE_NAMES, min_ranges=None, max_ranges=None):
        self._index = index
        self._metab_names = metab_names
        n = len(index['overall'])
        self.min_ranges = torch.zeros(1, n) if min_ranges is None else min_ranges
        self.max_ranges = torch.ones(1, n) if max_ranges is None else max_ranges
        # SNR should be quantified to dB range [5, 30], not [0, 1], so tests
        # can distinguish "was quantified" from "was left raw".
        self.min_ranges[0, index['snr']] = 5.0
        self.max_ranges[0, index['snr']] = 30.0

    @property
    def index(self):
        return self._index

    @property
    def metab(self):
        return self._metab_names, list(range(len(self._metab_names)))

    def quantify_params(self, params):
        delta = self.max_ranges - self.min_ranges
        return params.mul(delta) + self.min_ranges.clone()


# ---------------------------------------------------------------------------
# UniformRangeSampler
# ---------------------------------------------------------------------------

def test_uniform_sampler_shape_and_ranges():
    pm = FakePhysicsModel()
    sampler = UniformRangeSampler(pm, seed=0)
    params = sampler.sample(batch_size=64)
    assert params.tensor.shape == (64, N_COLUMNS)
    snr = params['snr']
    assert (snr >= 5.0).all() and (snr <= 30.0).all()
    assert (params.tensor >= 0.0).all()  # all other columns quantified to [0, 1]


def test_uniform_sampler_is_reproducible_with_same_seed():
    pm = FakePhysicsModel()
    a = UniformRangeSampler(pm, seed=42).sample(batch_size=8)
    b = UniformRangeSampler(pm, seed=42).sample(batch_size=8)
    torch.testing.assert_close(a.tensor, b.tensor)


def test_uniform_sampler_differs_with_different_seed():
    pm = FakePhysicsModel()
    a = UniformRangeSampler(pm, seed=1).sample(batch_size=8)
    b = UniformRangeSampler(pm, seed=2).sample(batch_size=8)
    assert not torch.allclose(a.tensor, b.tensor)


def test_uniform_sampler_does_not_touch_global_rng_state():
    pm = FakePhysicsModel()
    torch.manual_seed(1234)
    before = torch.rand(5)

    torch.manual_seed(1234)
    UniformRangeSampler(pm, seed=99).sample(batch_size=8)
    after = torch.rand(5)

    torch.testing.assert_close(before, after)


def test_uniform_sampler_metadata_records_seed():
    pm = FakePhysicsModel()
    params = UniformRangeSampler(pm, seed=7).sample(batch_size=2)
    assert params.metadata['seed'] == 7


# ---------------------------------------------------------------------------
# UniformRangeSampler(explicit_ranges=...) -- v2.0 handover section 2
# follow-up, repo owner's request: "I want the sampler to be able to
# [sample] in the actual parameter space too. Both options should be
# preserved." explicit_ranges columns are sampled directly in real units,
# bypassing pm.min_ranges/max_ranges/quantify_params() entirely (like
# CopulaInVivoSampler already does for its covered columns); every other
# column keeps the default [0, 1) -> quantify_params() behavior.
# ---------------------------------------------------------------------------

def test_uniform_sampler_explicit_ranges_bypasses_quantify_params():
    """FakePhysicsModel's min/max_ranges are [0, 1] for every non-snr
    column, so a real-unit explicit_ranges value outside [0, 1] proves
    quantify_params() was not consulted for that column."""
    pm = FakePhysicsModel()
    sampler = UniformRangeSampler(pm, seed=0, explicit_ranges={'g': (50.0, 100.0)})
    params = sampler.sample(batch_size=200)

    g = params.tensor[:, list(INDEX['g'])]  # columns 4, 5
    assert (g >= 50.0).all() and (g <= 100.0).all()
    assert g.mean() > 1.0  # sanity: nowhere near the default [0,1] range


def test_uniform_sampler_explicit_ranges_leaves_other_columns_on_default_path():
    pm = FakePhysicsModel()
    sampler = UniformRangeSampler(pm, seed=0, explicit_ranges={'g': (50.0, 100.0)})
    params = sampler.sample(batch_size=64)

    # 'd' (columns 2, 3) was not named in explicit_ranges -> still quantified
    # to the default [0, 1] range via quantify_params().
    d = params.tensor[:, list(INDEX['d'])]
    assert (d >= 0.0).all() and (d <= 1.0).all()


def test_uniform_sampler_explicit_ranges_both_modes_reproducible_with_same_seed():
    pm = FakePhysicsModel()
    a = UniformRangeSampler(pm, seed=3, explicit_ranges={'g': (50.0, 100.0)}).sample(8)
    b = UniformRangeSampler(pm, seed=3, explicit_ranges={'g': (50.0, 100.0)}).sample(8)
    torch.testing.assert_close(a.tensor, b.tensor)


def test_uniform_sampler_explicit_ranges_records_metadata():
    pm = FakePhysicsModel()
    params = UniformRangeSampler(pm, seed=0, explicit_ranges={'g': (50.0, 100.0)}).sample(4)
    assert params.metadata['explicit_ranges'] == ['g']


def test_uniform_sampler_explicit_ranges_rejects_unknown_key():
    pm = FakePhysicsModel()
    with pytest.raises(KeyError):
        UniformRangeSampler(pm, seed=0, explicit_ranges={'not_a_real_param': (0.0, 1.0)})


# ---------------------------------------------------------------------------
# CopulaInVivoSampler
# ---------------------------------------------------------------------------

NAMES = ['cr_ampl', 'naa_ampl', 'gaussLB', 'snr']
# A valid (symmetric, PSD) 4x4 correlation matrix.
CORR = np.array([
    [1.0, 0.3, 0.0, 0.0],
    [0.3, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.2],
    [0.0, 0.0, 0.2, 1.0],
])
DIST_DB = {
    'cr_ampl': {'lognorm': {'s': 0.5, 'loc': 0.0, 'scale': 1.0}},
    'naa_ampl': {'lognorm': {'s': 0.5, 'loc': 0.0, 'scale': 1.0}},
    'gaussLB': {'norm': {'loc': 0.1, 'scale': 0.02}},
    'snr': {'norm': {'loc': 15.0, 'scale': 3.0}},
}
GLOBAL_PARAM_MAP = {'g': 'gaussLB'}


@pytest.fixture
def dist_json_path(tmp_path):
    path = tmp_path / 'dist.json'
    with open(path, 'w') as f:
        json.dump(DIST_DB, f)
    return str(path)


@pytest.fixture
def corr_matrix_path(tmp_path):
    from scipy.io import savemat
    path = tmp_path / 'corr.mat'
    savemat(str(path), mdict={'corr': CORR, 'names': NAMES})
    return str(path)


@pytest.fixture
def legacy_corr_matrix_path(tmp_path):
    """A correlation-matrix file with no 'names' array, as older files have."""
    from scipy.io import savemat
    path = tmp_path / 'legacy_corr.mat'
    savemat(str(path), mdict={'corr': CORR})
    return str(path)


def test_copula_sampler_overwrites_only_selected_columns(dist_json_path, corr_matrix_path):
    pm = FakePhysicsModel()
    sampler = CopulaInVivoSampler(
        pm, dist_json_path, corr_matrix_path,
        global_param_map=GLOBAL_PARAM_MAP, seed=0,
    )
    params = sampler.sample(batch_size=100)

    # cr_ampl/naa_ampl -> concentration columns 0, 1 (lognormal -> always > 0)
    assert (params['cr']['concentration'] > 0).all()
    assert (params['naa']['concentration'] > 0).all()
    # gaussLB is a global broadcast onto every 'g' column (positions 0 and 1)
    torch.testing.assert_close(params['cr']['gaussian'], params['naa']['gaussian'])
    # snr sampled from N(15, 3), not the fallback's [5, 30] uniform quantify
    assert params['snr'].std() < 10.0


def test_copula_sampler_warns_when_config_override_overlaps_covered_column(dist_json_path, corr_matrix_path):
    """
    v2.0 handover section 2 follow-up ("make sure the overlapping
    parameter-range mechanisms don't conflict with each other"):
    PhysicsModel.set_parameter_constraints() records which columns it
    touched in explicitly_configured_columns; CopulaInVivoSampler must
    warn (not silently proceed) when one of its own covered columns
    ('g' -> columns 4, 5 via GLOBAL_PARAM_MAP) overlaps.
    """
    pm = FakePhysicsModel()
    pm.explicitly_configured_columns = {4}  # overlaps 'g' column 0

    with pytest.warns(UserWarning, match="config 'parameters' block override"):
        CopulaInVivoSampler(
            pm, dist_json_path, corr_matrix_path,
            global_param_map=GLOBAL_PARAM_MAP, seed=0,
        )


def test_copula_sampler_no_warning_without_overlapping_override(dist_json_path, corr_matrix_path):
    pm = FakePhysicsModel()
    pm.explicitly_configured_columns = {10}  # 'phi0' column, not covered by this copula (cr/naa/g/snr)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CopulaInVivoSampler(
            pm, dist_json_path, corr_matrix_path,
            global_param_map=GLOBAL_PARAM_MAP, seed=0,
        )


def test_copula_sampler_fallback_still_quantifies_uncovered_columns(dist_json_path, corr_matrix_path):
    pm = FakePhysicsModel()
    sampler = CopulaInVivoSampler(
        pm, dist_json_path, corr_matrix_path,
        global_param_map=GLOBAL_PARAM_MAP, seed=0,
    )
    params = sampler.sample(batch_size=50)
    # phi0/phi1/b0/etc. are not in DIST_DB, so they must come from the
    # UniformRangeSampler fallback -- i.e. properly quantified to [0, 1],
    # never left as raw un-quantified noise.
    for key in ('phi0', 'phi1', 'b0'):
        col = params[key]
        assert (col >= 0.0).all() and (col <= 1.0).all()


def test_copula_sampler_include_params_by_suffix(dist_json_path, corr_matrix_path):
    pm = FakePhysicsModel()
    included = CopulaInVivoSampler(
        pm, dist_json_path, corr_matrix_path,
        global_param_map=GLOBAL_PARAM_MAP, seed=0, include_params=['ampl'],
    )
    assert set(included.names) == {'cr_ampl', 'naa_ampl'}


def test_copula_sampler_exclude_params_by_exact_key(dist_json_path, corr_matrix_path):
    pm = FakePhysicsModel()
    excluded = CopulaInVivoSampler(
        pm, dist_json_path, corr_matrix_path,
        global_param_map=GLOBAL_PARAM_MAP, seed=0, exclude_params=['gaussLB'],
    )
    assert 'gaussLB' not in excluded.names
    assert {'cr_ampl', 'naa_ampl', 'snr'}.issubset(set(excluded.names))


def test_copula_sampler_reproducible_with_same_seed(dist_json_path, corr_matrix_path):
    pm = FakePhysicsModel()
    a = CopulaInVivoSampler(pm, dist_json_path, corr_matrix_path, global_param_map=GLOBAL_PARAM_MAP, seed=5).sample(20)
    b = CopulaInVivoSampler(pm, dist_json_path, corr_matrix_path, global_param_map=GLOBAL_PARAM_MAP, seed=5).sample(20)
    torch.testing.assert_close(a.tensor, b.tensor)


def test_copula_sampler_legacy_file_without_names_uses_json_order(dist_json_path, legacy_corr_matrix_path):
    pm = FakePhysicsModel()
    # Should not raise: JSON has 4 entries, matrix is 4x4, so the legacy
    # positional-trust path is used silently (matching current sim_COWS.py
    # behavior for such files).
    sampler = CopulaInVivoSampler(
        pm, dist_json_path, legacy_corr_matrix_path,
        global_param_map=GLOBAL_PARAM_MAP, seed=0,
    )
    assert sampler.names == NAMES


def test_copula_sampler_legacy_file_shape_mismatch_raises_without_reorder(tmp_path, corr_matrix_path):
    # A dist JSON with a different number of entries than a legacy
    # (name-less) correlation matrix must raise a clear error rather than
    # silently misaligning columns.
    from scipy.io import savemat
    small_dist_path = tmp_path / 'small_dist.json'
    with open(small_dist_path, 'w') as f:
        json.dump({'snr': DIST_DB['snr']}, f)
    legacy_path = tmp_path / 'legacy_mismatched.mat'
    savemat(str(legacy_path), mdict={'corr': CORR})  # 4x4, but JSON has 1 entry

    pm = FakePhysicsModel()
    with pytest.raises(ValueError):
        CopulaInVivoSampler(pm, str(small_dist_path), str(legacy_path), seed=0)
