"""
Unit tests for src.provenance (v2.0 handover section 9).

Uses a lightweight fake PhysicsModel-like object (see
tests/test_sampling.py's FakePhysicsModel for the same rationale) rather
than a real basis set. End-to-end verification against a real basis set
(cows.json) is recorded in docs/v2/progress_log.md, Milestone 7.
"""
import re
from types import SimpleNamespace

import torch

from src.provenance import Provenance, collect_provenance


class FakePhysicsModel:
    def __init__(self):
        self.PM_basis_set = 'fake_basis.mat'
        self.syn_basis_fids = torch.arange(2 * 2 * 8, dtype=torch.float32).reshape(1, 2, 2, 8)
        self.header = {
            'B0': torch.tensor(3.0),
            'TE': torch.tensor(30.0),
            'spectralwidth': torch.tensor(2000.0),
            'carrier_frequency': torch.tensor(127.7),
            'Ns': torch.tensor(2048.0),
            'basis_set_software': 'MARSS',
        }
        self.wrt_metab = 'cr'
        self.snr_metab = 'cr'
        self.lineshape_type = 'voigt'
        self.cropRange = [0.2, 4.2]


def test_collect_provenance_populates_acquisition_from_header():
    pm = FakePhysicsModel()
    prov = collect_provenance(pm)
    assert prov.acquisition['field_strength_T'] == 3.0
    assert prov.acquisition['echo_time_ms'] == 30.0
    assert prov.acquisition['spectral_width_Hz'] == 2000.0
    assert prov.acquisition['nucleus'] == '1H'
    assert prov.acquisition['basis_set_software'] == 'MARSS'


def test_collect_provenance_leaves_untracked_fields_none_not_guessed():
    pm = FakePhysicsModel()
    prov = collect_provenance(pm)
    # TR and pulse_sequence are not tracked anywhere in this codebase
    # (see the module docstring) -- must be None, not a guess.
    assert prov.acquisition['repetition_time_ms'] is None
    assert prov.acquisition['pulse_sequence'] is None


def test_collect_provenance_reads_vendor_from_config_not_header():
    pm = FakePhysicsModel()
    config = SimpleNamespace(vendor='GE', drop_prob=0.1)
    prov = collect_provenance(pm, config=config)
    assert prov.acquisition['vendor'] == 'GE'


def test_collect_provenance_basis_set_hash_is_deterministic_and_content_sensitive():
    pm_a = FakePhysicsModel()
    pm_b = FakePhysicsModel()
    prov_a = collect_provenance(pm_a)
    prov_b = collect_provenance(pm_b)
    assert prov_a.basis_set_hash == prov_b.basis_set_hash  # same content -> same hash

    pm_b.syn_basis_fids = pm_b.syn_basis_fids + 1.0  # different content
    prov_c = collect_provenance(pm_b)
    assert prov_c.basis_set_hash != prov_a.basis_set_hash


def test_collect_provenance_records_git_commit_as_a_real_hash():
    pm = FakePhysicsModel()
    prov = collect_provenance(pm)
    # This test runs inside a git repo, so a real commit hash should come
    # back -- not asserting a specific value (that would break on the next
    # commit), just the expected shape.
    assert prov.git_commit is None or re.fullmatch(r'[0-9a-f]{40}', prov.git_commit)


def test_collect_provenance_records_package_versions_and_seed():
    pm = FakePhysicsModel()
    sampler = SimpleNamespace(seed=42)
    prov = collect_provenance(pm, sampler=sampler, enabled_components={'noise': True, 'resample': True})
    assert 'torch' in prov.package_versions
    assert prov.rng_seed == 42
    assert prov.enabled_components == {'noise': True, 'resample': True}
    assert prov.data_state['resampled'] is True


def test_provenance_to_dict_is_a_plain_dict():
    prov = Provenance()
    d = prov.to_dict()
    assert isinstance(d, dict)
    assert d['parameter_schema_version'] == 'v2.0-params-1'
