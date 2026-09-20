"""
Regression tests for specific bugs found and fixed during the v2.0 audit
(docs/v2/architecture_v1_audit.md section 14 / docs/v2/progress_log.md).

These test PhysicsModel methods directly where they are pure/stateless
enough to not require constructing a full PhysicsModel (which needs a real
basis-set .mat file -- not tracked in git, see test_parameters.py's module
docstring). Coverage requiring a real basis set is verified manually and
recorded in docs/v2/progress_log.md instead.
"""
import math

import pytest
import torch

from src.metabolite_database import MoietyRangeError
from src.physics_model import PhysicsModel
from src.relaxation import t1_recovery, t1_star_recovery


def test_stack_noisy_clean_leaves_clean_branch_unmodified():
    """
    Regression test for the noisy/clean coupling bug (Milestone 4): the
    clean branch used to also have noise_vec added to it.
    """
    bS, channels, L = 2, 2, 4
    signal = torch.arange(bS * channels * L, dtype=torch.float32).reshape(bS, channels, L)
    noise_vec = torch.full((bS, channels, L), 100.0)

    stacked = PhysicsModel._stack_noisy_clean(signal, noise_vec, dim=-3, has_transients_axis=False)
    noisy_branch = stacked[:, 0, ...]
    clean_branch = stacked[:, 1, ...]

    torch.testing.assert_close(clean_branch, signal)
    torch.testing.assert_close(noisy_branch, signal + noise_vec)
    # The old (buggy) behavior added noise_vec to the clean branch too;
    # explicitly assert that is no longer the case.
    assert not torch.allclose(clean_branch, signal + noise_vec)


def test_stack_noisy_clean_no_transients_axis_never_mixes_batch_samples():
    """
    CRITICAL regression test (Milestone 8): without a transients axis
    (has_transients_axis=False, the common single-coil case), each
    sample's noisy branch must be built from *its own* clean signal, not
    batch sample 0's. Verified directly against a real basis set: before
    this fix, sample i's noisy output correlated 0.9999+ with sample 0's
    clean spectrum and only ~0.92 with its own -- i.e. every sample's
    "noisy" output in a batch was silently built from sample 0's signal
    whenever multicoil <= 1. See docs/v2/progress_log.md, Milestone 8.
    """
    bS, channels, L = 4, 2, 8
    # Distinct, non-degenerate signal per sample so cross-contamination
    # would be detectable (a shared/constant signal across samples would
    # hide the bug).
    signal = torch.randn(bS, channels, L)
    noise_vec = torch.randn(bS, channels, L) * 0.01  # small relative to signal

    stacked = PhysicsModel._stack_noisy_clean(signal, noise_vec, dim=-3, has_transients_axis=False)
    noisy, clean = stacked[:, 0, ...], stacked[:, 1, ...]

    for i in range(bS):
        # Own signal + own noise, exactly.
        torch.testing.assert_close(noisy[i], signal[i] + noise_vec[i])
        # Explicitly not sample 0's signal (for i > 0), guarding against
        # the exact regression that occurred.
        if i != 0:
            assert not torch.allclose(noisy[i] - noise_vec[i], signal[0], atol=1e-3)


def test_stack_noisy_clean_with_transients_axis_broadcasts_transient_zero_per_sample():
    """
    Multicoil case (has_transients_axis=True): signal has a real
    transients axis (from PhysicsModel.multicoil()); the noisy branch
    should broadcast *each sample's own* transient-0 line across all
    transients, adding each transient's own noise realization -- not
    sample 0's transient 0.
    """
    bS, transients, channels, L = 4, 3, 2, 8
    signal = torch.randn(bS, transients, channels, L)
    noise_vec = torch.randn(bS, transients, channels, L) * 0.01

    stacked = PhysicsModel._stack_noisy_clean(signal, noise_vec, dim=-4, has_transients_axis=True)
    noisy, clean = stacked[:, 0], stacked[:, 1]

    torch.testing.assert_close(clean, signal)
    expected_noisy = signal[:, 0:1, :, :].expand_as(signal) + noise_vec
    torch.testing.assert_close(noisy, expected_noisy)
    # Per-sample: sample i's noisy base is sample i's own transient 0, not sample 0's.
    for i in range(1, bS):
        assert not torch.allclose(noisy[i, 0] - noise_vec[i, 0], signal[0, 0], atol=1e-3)


def test_scale_snr_reference_single_coil_no_transients_axis():
    """
    Single-coil case (Milestone 11 regression): noise_std has no
    transients axis, so it should just broadcast against the
    per-metabolite reference directly (this always worked; pinning it so
    the multicoil fix below can't regress it).
    """
    bS, num_bF, channels = 2, 3, 2
    reference = torch.arange(1, bS * num_bF * channels + 1, dtype=torch.float32).reshape(bS, num_bF, channels, 1)
    noise_std = torch.full((bS, channels, 1), 2.0)

    out = PhysicsModel._scale_snr_reference(reference, noise_std, has_transients_axis=False)

    assert out.shape == (bS, num_bF, channels, 1)
    torch.testing.assert_close(out, reference / 2.0)


def test_scale_snr_reference_multicoil_broadcasts_per_transient():
    """
    CRITICAL regression test (Milestone 11): multicoil case, where
    noise_std carries its own transients axis. Before this fix, dividing
    a per-metabolite reference by this noise_std crashed with a
    broadcast-shape RuntimeError (the metabolite axis collided with the
    transients axis). Also checks each transient gets divided by *its
    own* noise_std value, not a shared/wrong one.
    """
    bS, num_bF, channels, transients = 2, 3, 2, 4
    reference = torch.ones(bS, num_bF, channels, 1)
    # Distinct noise_std per transient so incorrect broadcasting would be
    # detectable: transient t has std = t + 1.
    noise_std = torch.arange(1, transients + 1, dtype=torch.float32).view(1, transients, 1, 1)
    noise_std = noise_std.expand(bS, transients, channels, 1).contiguous()

    out = PhysicsModel._scale_snr_reference(reference, noise_std, has_transients_axis=True)

    assert out.shape == (bS, transients, num_bF, channels, 1)
    for t in range(transients):
        expected = 1.0 / (t + 1)
        torch.testing.assert_close(out[:, t], torch.full((bS, num_bF, channels, 1), expected))


# ---------------------------------------------------------------------------
# Opt-in T1/T1* config surface (handover section 10, repo-owner request).
# Not active with the shipped database (T1 is all schema placeholders --
# see src/metabolite_database.py's get_t1_range()); these test the parsing
# and amplitude-scaling logic directly, without constructing a full
# PhysicsModel (needs a real basis-set file).
# ---------------------------------------------------------------------------

def test_resolve_t1_config_disabled_by_default():
    enabled, TR, flip_angle, t1_values = PhysicsModel._resolve_t1_config(
        None, ['naa'], {'naa': {'T1': {'metab': {'min': None, 'max': None}}}})
    assert enabled is False
    assert TR is None
    assert flip_angle is None
    assert t1_values is None


def test_resolve_t1_config_raises_for_shipped_database_placeholders():
    """The shipped metabolites_database.json has T1: null for everything
    -- enabling t1_cfg against it must fail loudly, not silently proceed."""
    ranges = {'naa': {'T1': {'metab': {'min': None, 'max': None}}}}
    with pytest.raises(MoietyRangeError):
        PhysicsModel._resolve_t1_config({'enabled': True, 'TR': 2000.0},
                                        ['naa'], ranges)


def test_resolve_t1_config_enabled_without_tr_raises():
    ranges = {'naa': {'T1': {'metab': {'min': [1100], 'max': [1500]}}}}
    with pytest.raises(ValueError):
        PhysicsModel._resolve_t1_config({'enabled': True}, ['naa'], ranges)


def test_resolve_t1_config_uses_database_midpoint_once_populated():
    ranges = {
        'naa': {'T1': {'metab': {'min': [1100], 'max': [1500]}}},
        'cr': {'T1': {'metab': {'min': [1300], 'max': [1300]}}},
    }
    enabled, TR, flip_angle, t1_values = PhysicsModel._resolve_t1_config(
        {'enabled': True, 'TR': 2000.0}, ['naa', 'cr'], ranges)
    assert enabled is True
    assert TR == 2000.0
    assert flip_angle is None
    assert t1_values == [1300.0, 1300.0]  # naa midpoint=(1100+1500)/2=1300


def test_resolve_t1_config_parses_optional_flip_angle():
    ranges = {'naa': {'T1': {'metab': {'min': [1200], 'max': [1200]}}}}
    enabled, TR, flip_angle, t1_values = PhysicsModel._resolve_t1_config(
        {'enabled': True, 'TR': 2000.0, 'flip_angle': 90.0}, ['naa'], ranges)
    assert flip_angle == 90.0


def test_apply_t1_scaling_matches_plain_t1_recovery():
    amp = torch.tensor([[1.0, 2.0]])
    t1_values_ms = torch.tensor([1000.0, 1500.0])
    TR_ms = 2000.0

    out = PhysicsModel._apply_t1_scaling(amp, t1_values_ms, TR_ms)

    expected_factor = t1_recovery(TR=TR_ms / 1000.0, T1=t1_values_ms / 1000.0)
    torch.testing.assert_close(out, amp * expected_factor)


def test_apply_t1_scaling_uses_t1_star_when_flip_angle_given():
    amp = torch.tensor([[1.0, 2.0]])
    t1_values_ms = torch.tensor([1000.0, 1500.0])
    TR_ms = 2000.0
    flip_angle = 60.0

    out = PhysicsModel._apply_t1_scaling(amp, t1_values_ms, TR_ms, flip_angle)

    expected_factor = t1_star_recovery(TR=TR_ms / 1000.0, T1=t1_values_ms / 1000.0,
                                       flip_angle=flip_angle)
    torch.testing.assert_close(out, amp * expected_factor)
    # Sanity: T1* and plain T1 recovery genuinely differ away from 90 degrees
    plain_factor = t1_recovery(TR=TR_ms / 1000.0, T1=t1_values_ms / 1000.0)
    assert not torch.allclose(expected_factor, plain_factor)
