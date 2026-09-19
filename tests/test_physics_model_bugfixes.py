"""
Regression tests for specific bugs found and fixed during the v2.0 audit
(docs/v2/architecture_v1_audit.md section 14 / docs/v2/progress_log.md).

These test PhysicsModel methods directly where they are pure/stateless
enough to not require constructing a full PhysicsModel (which needs a real
basis-set .mat file -- not tracked in git, see test_parameters.py's module
docstring). Coverage requiring a real basis set is verified manually and
recorded in docs/v2/progress_log.md instead.

Only the dim=-3 (single-coil) shape is covered here; the dim=-4 variant
used when multicoil transients are active was not independently verified
against multicoil()'s actual output shape and is deliberately left
untested rather than encoding an unverified assumption as a passing test.
"""
import torch

from src.physics_model import PhysicsModel


def test_stack_noisy_clean_leaves_clean_branch_unmodified():
    """
    Regression test for the noisy/clean coupling bug: fidSum's "clean"
    branch used to also have noise_vec added to it. _stack_noisy_clean is
    now the single shared implementation both fidSum and spectral_fit use.
    """
    # Real call sites use shape (batch, X, C, L) where X (e.g. a
    # difference-editing ON/OFF axis) is size 1, so that
    # signal[...,0,:,:].unsqueeze(-3) reproduces signal's own shape -- a
    # precondition for torch.stack to accept both branches.
    signal = torch.arange(2 * 1 * 2 * 4, dtype=torch.float32).reshape(2, 1, 2, 4)
    noise_vec = torch.full((2, 1, 2, 4), 100.0)

    stacked = PhysicsModel._stack_noisy_clean(signal, noise_vec, dim=-3)

    noisy_branch = stacked[:, :, 0, ...]
    clean_branch = stacked[:, :, 1, ...]

    torch.testing.assert_close(clean_branch, signal)
    torch.testing.assert_close(noisy_branch, signal[..., 0, :, :].unsqueeze(-3) + noise_vec)
    # The old (buggy) behavior added noise_vec to the clean branch too;
    # explicitly assert that is no longer the case.
    assert not torch.allclose(clean_branch, signal + noise_vec)
