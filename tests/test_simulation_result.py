"""
Unit tests for the parts of src.simulation_result.SimulationResult that
don't require a real PhysicsModel/basis set. Integration coverage (that
PhysicsModel.forward(..., return_components=True) actually populates every
field correctly) was verified manually against a real basis set and is
recorded in docs/v2/progress_log.md, Milestone 5.
"""
import torch

from src.simulation_result import SimulationResult


def test_spectrum_is_an_alias_for_noisy():
    noisy = torch.arange(4.0)
    result = SimulationResult(noisy=noisy)
    assert result.spectrum is result.noisy
    torch.testing.assert_close(result.spectrum, noisy)


def test_unset_fields_default_to_none():
    result = SimulationResult()
    assert result.noisy is None
    assert result.noise_free_total is None
    assert result.nuisance_free is None
    assert result.baseline is None
    assert result.baseline_fit is None
    assert result.spline_coefficients is None
    assert result.crlb is None
    assert result.fim is None
    assert result.spectrum is None


def test_provenance_defaults_to_empty_dict_not_shared_across_instances():
    a = SimulationResult()
    b = SimulationResult()
    a.provenance['x'] = 1
    assert b.provenance == {}
