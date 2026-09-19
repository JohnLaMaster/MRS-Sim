"""
Unit tests for src.parameters (ParameterRegistry / MetaboliteParameterView /
SimulationParameters).

These use a small synthetic index dict shaped like PhysicsModel.index rather
than an actual PhysicsModel, so they stay fast and independent of any basis
set file (see docs/v2/architecture_v1_audit.md section 13 for why: no basis
sets are currently tracked in git, and instantiating a real PhysicsModel is
comparatively heavy). Coverage of the real PhysicsModel.index construction
belongs in a separate, slower integration test once a small test basis set is
available.
"""
import pytest
import torch

from src.parameters import ParameterRegistry, SimulationParameters


# Basis-function line order matches PhysicsModel's convention: non-MM
# metabolites sorted alphabetically, then MM/Lip lines. Two metabolites
# ("cr", "naa") + one MM line ("mm09") => 3 lines total.
METABOLITE_NAMES = ['cr', 'naa', 'mm09']

INDEX = {
    # concentration: one column per basis-function line, in line order
    'cr': 0,
    'naa': 1,
    'mm09': 2,
    # per-line families, one tuple entry per line, same order as above
    'd': (3, 4, 5),
    'g': (6, 7, 8),
    'f_shift': 9,           # global frequency shift (not per-line)
    'f_shifts': (10, 11, 12),
    'snr': 13,
    'phi0': 14,
    'phi1': 15,
    'b0': 16,
    'b0_dir': (17, 18, 19),
    'ecc': (20, 21),
    'coil_snr': (22,),
    'coil_sens': (23,),
    'coil_fshift': (24,),
    'coil_phi0': (25,),
    'metabolites': (0, 1, 2),
    'parameters': tuple(range(3, 26)),
    'overall': tuple(range(0, 26)),
}
N_COLUMNS = 26


def make_params(batch_size=4):
    tensor = torch.arange(batch_size * N_COLUMNS, dtype=torch.float32).reshape(batch_size, N_COLUMNS)
    registry = ParameterRegistry(index=INDEX, metabolite_names=METABOLITE_NAMES)
    return SimulationParameters(tensor=tensor, registry=registry)


def test_flat_access_non_metabolite_family():
    params = make_params()
    torch.testing.assert_close(params['snr'], params.tensor[:, 13])
    torch.testing.assert_close(params['phi0'], params.tensor[:, 14])
    torch.testing.assert_close(params['b0_dir'], params.tensor[:, (17, 18, 19)])


def test_flat_access_is_case_insensitive():
    params = make_params()
    torch.testing.assert_close(params['SNR'], params.tensor[:, 13])


def test_metabolite_concentration():
    params = make_params()
    torch.testing.assert_close(params['NAA']['concentration'], params.tensor[:, 1])
    torch.testing.assert_close(params['cr']['concentration'], params.tensor[:, 0])
    torch.testing.assert_close(params['mm09']['concentration'], params.tensor[:, 2])


def test_metabolite_per_line_families_use_line_position_not_column_offset():
    params = make_params()
    # cr is line position 0, naa is position 1, mm09 is position 2 --
    # verifies indexing is by *position among lines*, not by name lookup
    # into the 'd'/'g'/'f_shifts' tuples directly.
    torch.testing.assert_close(params['cr']['lorentzian'], params.tensor[:, 3])
    torch.testing.assert_close(params['naa']['lorentzian'], params.tensor[:, 4])
    torch.testing.assert_close(params['mm09']['lorentzian'], params.tensor[:, 5])

    torch.testing.assert_close(params['cr']['gaussian'], params.tensor[:, 6])
    torch.testing.assert_close(params['naa']['gaussian'], params.tensor[:, 7])
    torch.testing.assert_close(params['mm09']['gaussian'], params.tensor[:, 8])

    torch.testing.assert_close(params['cr']['frequency_shift'], params.tensor[:, 10])
    torch.testing.assert_close(params['naa']['frequency_shift'], params.tensor[:, 11])
    torch.testing.assert_close(params['mm09']['frequency_shift'], params.tensor[:, 12])


def test_metabolite_attribute_access_matches_item_access():
    params = make_params()
    view = params['naa']
    torch.testing.assert_close(view.concentration, view['concentration'])
    torch.testing.assert_close(view.lorentzian, view['lorentzian'])
    torch.testing.assert_close(view.gaussian, view['gaussian'])
    torch.testing.assert_close(view.frequency_shift, view['frequency_shift'])


def test_unknown_metabolite_parameter_raises_keyerror():
    params = make_params()
    with pytest.raises(KeyError):
        params['naa']['banana']


def test_unknown_flat_parameter_raises_keyerror():
    params = make_params()
    with pytest.raises(KeyError):
        params['not_a_real_parameter']


def test_not_yet_modeled_parameters_raise_notimplementederror():
    params = make_params()
    with pytest.raises(NotImplementedError):
        params['naa']['T1']
    with pytest.raises(NotImplementedError):
        params['naa']['t2']


def test_registry_index_access_does_not_copy_data():
    params = make_params()
    view = params['naa']
    conc = view.concentration
    conc[0] = -999.0
    assert params.tensor[0, 1] == -999.0


def test_labels_length_and_key_columns():
    params = make_params()
    labels = params.labels()
    assert len(labels) == N_COLUMNS
    assert labels[0] == 'cr.concentration'
    assert labels[1] == 'naa.concentration'
    assert labels[2] == 'mm09.concentration'
    assert labels[3] == 'cr.d'
    assert labels[4] == 'naa.d'
    assert labels[5] == 'mm09.d'
    assert labels[13] == 'snr'
    assert labels[14] == 'phi0'
    # aggregate keys (metabolites/parameters/overall) must not overwrite
    # the individual per-column labels
    assert 'metabolites' not in labels
    assert 'overall' not in labels


def test_clone_is_independent():
    params = make_params()
    clone = params.clone()
    clone.tensor[0, 0] = -1.0
    assert params.tensor[0, 0] != -1.0


def test_batch_size():
    params = make_params(batch_size=7)
    assert params.batch_size == 7


def test_contains():
    params = make_params()
    assert 'snr' in params
    assert 'NAA' in params
    assert 'not_a_real_parameter' not in params
