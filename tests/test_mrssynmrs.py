"""
Unit tests for src.mrssynmrs (v2.0 handover section 9).

Uses a lightweight fake PhysicsModel-like object, following the same
pattern as test_sampling.py/test_provenance.py. End-to-end verification
against a real basis set (cows.json) -- including that the fetched
MRSsynMRS field structure this module's output matches was read directly
from the actual spreadsheet, not invented -- is recorded in
docs/v2/progress_log.md, Milestone 7.
"""
from types import SimpleNamespace

import pytest
import torch

from src.mrssynmrs import export_mrssynmrs_table
from src.provenance import collect_provenance


class FakePhysicsModel:
    def __init__(self):
        self.PM_basis_set = 'fake_basis.mat'
        self.syn_basis_fids = torch.zeros(1, 2, 2, 8)
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
        self._metab_names = ['cr', 'naa']
        self.index = {'cr': 0, 'naa': 1}
        self.min_ranges = torch.tensor([[0.0, 0.1]])
        self.max_ranges = torch.tensor([[1.0, 1.6]])
        self.ranges = {
            'cr': {'Conc': {'min': [0.0], 'max': [1.0]}, 'T2': {'metab': {'min': [150], 'max': [250]}}},
            'naa': {'Conc': {'min': [0.1], 'max': [1.6]}, 'T2': {'metab': {'min': [242.7], 'max': [320.17]}}},
        }

    @property
    def metab(self):
        return self._metab_names, list(range(len(self._metab_names)))


def _table(sampler=None, config=None, extra_sections=None):
    pm = FakePhysicsModel()
    prov = collect_provenance(pm, config=config, enabled_components={'noise': True})
    return export_mrssynmrs_table(pm, prov, config=config, sampler=sampler, extra_sections=extra_sections)


def test_table_top_level_sections_match_the_fetched_sheet_structure():
    table = _table()
    assert set(table.keys()) == {'Experiment', 'Software', 'Pulse Sequence', 'Metabolites'}
    assert set(table['Experiment'].keys()) == {'ID', 'Nucleus'}
    assert set(table['Software'].keys()) == {'Basis Set Simulation', 'Signal Model', 'Additional Code'}


def test_acquisition_fields_are_populated_from_provenance():
    table = _table()
    assert table['Pulse Sequence']['Field Strength [T]'] == 3.0
    assert table['Pulse Sequence']['Sequence Timing(s)']['Echo Time [ms]'] == 30.0
    assert table['Pulse Sequence']['Spectral Width [Hz]'] == 2000.0


def test_untracked_scanner_level_fields_are_explicitly_none():
    table = _table()
    assert table['Pulse Sequence']['Voxel Size (X, Y, Z) [mm]'] is None
    assert table['Pulse Sequence']['Water Suppression'] == {'Method': None, 'Bandwidth [Hz]': None}
    assert table['Pulse Sequence']['Shimming Method'] is None
    assert table['Metabolites']['Population'] is None
    assert table['Metabolites']['Region of interest'] is None


def test_concentration_and_t2_ranges_reflect_effective_min_max_ranges():
    table = _table()
    conc = table['Metabolites']['Concentration Ranges']
    assert conc['cr'] == pytest.approx([0.0, 1.0])
    assert conc['naa'] == pytest.approx([0.1, 1.6])
    assert table['Metabolites']['T2 Relaxation Ranges'] == {'cr': [150, 250], 'naa': [242.7, 320.17]}


def test_distribution_reflects_sampler_type():
    UniformRangeSampler = type('UniformRangeSampler', (), {})
    CopulaInVivoSampler = type('CopulaInVivoSampler', (), {})

    table = _table(sampler=UniformRangeSampler())
    assert 'Uniform' in table['Metabolites']['Distribution']

    table2 = _table(sampler=CopulaInVivoSampler())
    assert 'copula' in table2['Metabolites']['Distribution'].lower()
    assert len(table2['Software']['Additional Code']) == 1


def test_no_sampler_leaves_distribution_none():
    table = _table(sampler=None)
    assert table['Metabolites']['Distribution'] is None
    assert table['Software']['Additional Code'] == []


def test_component_dropout_probability_from_config():
    config = SimpleNamespace(drop_prob=0.15)
    table = _table(config=config)
    assert table['Metabolites']['Component dropout probability'] == 0.15


def test_extra_sections_merge_into_pulse_sequence_without_removing_base_fields():
    table = _table(extra_sections={'Editing included': {'Type': 'MEGA-PRESS-like'}})
    assert table['Pulse Sequence']['Editing included'] == {'Type': 'MEGA-PRESS-like'}
    # base fields still present
    assert 'Field Strength [T]' in table['Pulse Sequence']
