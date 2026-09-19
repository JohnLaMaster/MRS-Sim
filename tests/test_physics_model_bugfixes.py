"""
Regression tests for specific bugs found and fixed during the v2.0 audit
(docs/v2/architecture_v1_audit.md section 14 / docs/v2/progress_log.md).

These test PhysicsModel methods directly where they are pure/stateless
enough to not require constructing a full PhysicsModel (which needs a real
basis-set .mat file -- not tracked in git, see test_parameters.py's module
docstring). Coverage requiring a real basis set is verified manually and
recorded in docs/v2/progress_log.md instead.
"""
import torch

from src.physics_model import PhysicsModel


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
