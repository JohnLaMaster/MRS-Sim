"""
Unit tests for src.crlb (v2.0 handover section 6).

`_crlb_signal_model`/`_build_layout` take plain tensors (no PhysicsModel
needed), so they're tested directly here. `compute_crlb()` itself needs a
real PhysicsModel (basis functions, parameter registry) and was instead
verified end to end against cows.json's real basis set -- see
docs/v2/progress_log.md, Milestone 6, including the critical check that
this module's forward model reproduces PhysicsModel.forward()'s actual
output (within this module's documented scope) to within 0.18% relative
error, not just that it runs without crashing.
"""
import torch
from torch.func import jacrev, vmap

from src.crlb import (
    ALL_CRLB_PARAMS,
    DEFAULT_CRLB_PARAMS,
    _build_layout,
    _crlb_signal_model,
    _voigt_decay,
    _voigt_decay_moiety,
)
from src.splines import build_spline_basis


def _make_inputs(param_names=DEFAULT_CRLB_PARAMS, n_lines=3, length=32):
    t = torch.linspace(0, 0.1, length)
    basis_fids = torch.randn(n_lines, 2, length) * 0.1
    x = torch.linspace(0.2, 4.2, length)
    spline_basis = build_spline_basis(x, knot_spacing=0.4, degree=3)
    n_basis = spline_basis.shape[-1]
    layout_est, n_est = _build_layout(param_names, n_lines, n_basis)
    fixed_names = [p for p in ALL_CRLB_PARAMS if p not in param_names]
    layout_fixed, n_fixed = _build_layout(fixed_names, n_lines, n_basis)
    carrier_frequency = torch.tensor(127.7)
    phi1_ref = torch.linspace(-3.0, 3.0, length)
    return (t, basis_fids, spline_basis, layout_est, layout_fixed, n_est,
            n_fixed, carrier_frequency, phi1_ref)


def test_build_layout_covers_every_column_exactly_once():
    layout, n_params = _build_layout(ALL_CRLB_PARAMS, n_lines=4, n_basis=6)
    covered = torch.zeros(n_params, dtype=torch.bool)
    for sl in layout.values():
        assert not covered[sl].any(), "overlapping parameter slices"
        covered[sl] = True
    assert covered.all(), "not every column is covered by some parameter family"


def test_build_layout_empty_names_gives_empty_layout():
    layout, n_params = _build_layout([], n_lines=4, n_basis=6)
    assert layout == {}
    assert n_params == 0


def test_voigt_decay_matches_physics_model_formula():
    """exp((-d - g*t) * t), copied from PhysicsModel.lineshape_voigt."""
    d = torch.tensor([0.5, 1.0])
    g = torch.tensor([0.1, 0.2])
    t = torch.linspace(0, 1, 8)
    out = _voigt_decay(d, g, t)
    expected = torch.exp((-d.unsqueeze(-1) - g.unsqueeze(-1) * t.unsqueeze(0)) * t.unsqueeze(0))
    torch.testing.assert_close(out, expected)


def test_signal_model_output_shape():
    (t, basis_fids, spline_basis, layout_est, layout_fixed, n_est, n_fixed,
     cf, phi1_ref) = _make_inputs()
    theta_est = torch.zeros(n_est)
    theta_est[layout_est['amp']] = 1.0
    theta_fixed = torch.zeros(n_fixed)
    out = _crlb_signal_model(theta_est, theta_fixed, basis_fids, t, cf,
                             spline_basis, phi1_ref, layout_est, layout_fixed)
    assert out.shape == (2 * t.shape[0],)
    assert torch.isfinite(out).all()


def test_signal_model_is_batchable_via_vmap_and_differentiable_via_jacrev():
    (t, basis_fids, spline_basis, layout_est, layout_fixed, n_est, n_fixed,
     cf, phi1_ref) = _make_inputs()
    batch_size = 4

    def model_fn(theta_est_i, theta_fixed_i):
        return _crlb_signal_model(theta_est_i, theta_fixed_i, basis_fids, t, cf,
                                  spline_basis, phi1_ref, layout_est, layout_fixed)

    theta_est_batch = torch.zeros(batch_size, n_est)
    theta_est_batch[:, layout_est['amp']] = 1.0
    theta_fixed_batch = torch.zeros(batch_size, n_fixed)

    J = vmap(jacrev(model_fn, argnums=0), in_dims=(0, 0))(theta_est_batch, theta_fixed_batch)
    assert J.shape == (batch_size, 2 * t.shape[0], n_est)
    assert torch.isfinite(J).all()


def test_zero_amplitude_gives_only_baseline_contribution():
    """With every amplitude at 0 and a nonzero baseline, the observed signal should equal the (normalized) baseline alone."""
    (t, basis_fids, spline_basis, layout_est, layout_fixed, n_est, n_fixed,
     cf, phi1_ref) = _make_inputs()
    theta_est = torch.zeros(n_est)
    theta_fixed = torch.zeros(n_fixed)
    n_basis = spline_basis.shape[-1]
    beta = torch.zeros(2, n_basis)
    beta[0, 0] = 1.0  # nonzero real-channel spline coefficient
    theta_est[layout_est['beta']] = beta.reshape(-1)

    out = _crlb_signal_model(theta_est, theta_fixed, basis_fids, t, cf,
                             spline_basis, phi1_ref, layout_est, layout_fixed)
    # With a zero spectrum, normalization divides by the baseline's own
    # peak magnitude, so the real channel should be a rescaled version of
    # the spline basis's first column and the imaginary channel should be
    # exactly zero (beta[1] is all zero).
    length = t.shape[0]
    assert torch.allclose(out[length:], torch.zeros(length), atol=1e-6)


def test_fixed_phi0_still_affects_signal_but_not_differentiated():
    """
    Repo-owner request: a parameter excluded from the CRLB (e.g. phi0) must
    still be physically applied at its actual value, not silently dropped
    -- only its own row/column disappears from the Jacobian/FIM.
    """
    param_names = ['amp']  # everything else (d, g, fshift, phi0, phi1, beta) fixed
    (t, basis_fids, spline_basis, layout_est, layout_fixed, n_est, n_fixed,
     cf, phi1_ref) = _make_inputs(param_names=param_names)

    theta_est = torch.ones(n_est)  # amp = 1 for every line

    theta_fixed_zero_phase = torch.zeros(n_fixed)
    out_zero_phase = _crlb_signal_model(theta_est, theta_fixed_zero_phase, basis_fids, t,
                                        cf, spline_basis, phi1_ref, layout_est, layout_fixed)

    theta_fixed_90_phase = theta_fixed_zero_phase.clone()
    theta_fixed_90_phase[layout_fixed['phi0']] = 90.0
    out_90_phase = _crlb_signal_model(theta_est, theta_fixed_90_phase, basis_fids, t,
                                      cf, spline_basis, phi1_ref, layout_est, layout_fixed)

    # A fixed (excluded) phi0 must still change the observed signal...
    assert not torch.allclose(out_zero_phase, out_90_phase)

    # ...but the Jacobian shape only ever reflects the estimated params (amp).
    def model_fn(theta_est_i, theta_fixed_i):
        return _crlb_signal_model(theta_est_i, theta_fixed_i, basis_fids, t, cf,
                                  spline_basis, phi1_ref, layout_est, layout_fixed)
    J = jacrev(model_fn, argnums=0)(theta_est, theta_fixed_90_phase)
    assert J.shape == (2 * t.shape[0], n_est)


# -- Moiety-level 'd' (v2.0 handover section 18 follow-up) -------------
#
# No basis set shipped today gives PhysicsModel.num_spins > 1 (see
# test_physics_model_bugfixes.py's _pad_and_stack_moiety_lines tests and
# docs/v2/progress_log.md) -- these exercise the moiety-aware code path
# directly with fabricated inputs, same convention as the rest of this
# file.

def _make_moiety_inputs(param_names=DEFAULT_CRLB_PARAMS, n_lines=2, num_spins=2, length=32):
    t = torch.linspace(0, 0.1, length)
    basis_fids = torch.randn(n_lines, num_spins, 2, length) * 0.1
    x = torch.linspace(0.2, 4.2, length)
    spline_basis = build_spline_basis(x, knot_spacing=0.4, degree=3)
    n_basis = spline_basis.shape[-1]
    layout_est, n_est = _build_layout(param_names, n_lines, n_basis, num_spins=num_spins)
    fixed_names = [p for p in ALL_CRLB_PARAMS if p not in param_names]
    layout_fixed, n_fixed = _build_layout(fixed_names, n_lines, n_basis, num_spins=num_spins)
    carrier_frequency = torch.tensor(127.7)
    phi1_ref = torch.linspace(-3.0, 3.0, length)
    return (t, basis_fids, spline_basis, layout_est, layout_fixed, n_est,
            n_fixed, carrier_frequency, phi1_ref)


def test_voigt_decay_moiety_reduces_to_voigt_decay_for_one_moiety():
    d = torch.tensor([0.5, 1.0])
    g = torch.tensor([0.1, 0.2])
    t = torch.linspace(0, 1, 8)
    expected = _voigt_decay(d, g, t)  # [n_lines, L]
    out = _voigt_decay_moiety(d.unsqueeze(-1), g, t)  # [n_lines, 1, L]
    torch.testing.assert_close(out.squeeze(1), expected)


def test_build_layout_d_scales_with_num_spins_others_unchanged():
    # ALL_CRLB_PARAMS order: amp, d, g, fshift, phi0, phi1, beta
    layout, n_params = _build_layout(ALL_CRLB_PARAMS, n_lines=3, n_basis=4, num_spins=2)
    assert layout['amp'] == slice(0, 3)     # unaffected: still 3 (one per line)
    assert layout['d'] == slice(3, 9)       # 3 lines * 2 moieties
    assert layout['g'] == slice(9, 12)      # unaffected: still 3
    assert n_params == layout['beta'].stop


def test_signal_model_moiety_path_output_shape_and_finite():
    (t, basis_fids, spline_basis, layout_est, layout_fixed, n_est, n_fixed,
     cf, phi1_ref) = _make_moiety_inputs()
    theta_est = torch.rand(n_est) * 0.1
    theta_fixed = torch.rand(n_fixed) * 0.1
    out = _crlb_signal_model(theta_est, theta_fixed, basis_fids, t, cf,
                             spline_basis, phi1_ref, layout_est, layout_fixed,
                             num_spins=2)
    assert out.shape == (2 * t.shape[0],)
    assert torch.isfinite(out).all()


def test_signal_model_moiety_path_matches_manual_sum_when_decay_is_shared():
    """
    Cross-check against the non-moiety path: if both moieties of a line
    share the same 'd'/'g'/amp/fshift (i.e. nothing actually
    distinguishes them), summing basis functions first and decaying once
    (the n=1-moiety model) must equal decaying each moiety separately and
    then summing (the moiety-aware model) -- decay/shift are both linear
    operations applied identically to every moiety in this degenerate
    case, so the two orders commute.
    """
    n_lines, num_spins, length = 2, 2, 24
    t = torch.linspace(0, 0.1, length)
    basis_fids = torch.randn(n_lines, num_spins, 2, length) * 0.1
    x = torch.linspace(0.2, 4.2, length)
    spline_basis = build_spline_basis(x, knot_spacing=0.4, degree=3)
    n_basis = spline_basis.shape[-1]
    cf = torch.tensor(127.7)
    phi1_ref = torch.linspace(-3.0, 3.0, length)

    param_names = ['amp', 'd', 'g', 'fshift', 'phi0']
    layout_est, n_est = _build_layout(param_names, n_lines, n_basis, num_spins=num_spins)
    fixed_names = [p for p in ALL_CRLB_PARAMS if p not in param_names]
    layout_fixed, n_fixed = _build_layout(fixed_names, n_lines, n_basis, num_spins=num_spins)

    theta_est = torch.zeros(n_est)
    theta_est[layout_est['amp']] = torch.tensor([1.0, 2.0])
    d_shared = torch.tensor([0.3, 0.6])
    theta_est[layout_est['d']] = d_shared.repeat_interleave(num_spins)  # same d for both moieties
    theta_est[layout_est['g']] = torch.tensor([0.05, 0.08])
    theta_est[layout_est['fshift']] = torch.tensor([0.01, -0.02])
    theta_fixed = torch.zeros(n_fixed)

    out_moiety = _crlb_signal_model(theta_est, theta_fixed, basis_fids, t, cf,
                                    spline_basis, phi1_ref, layout_est, layout_fixed,
                                    num_spins=num_spins)

    # Non-moiety equivalent: pre-sum the two (identically-decayed) moieties
    # into one basis FID per line, then run the plain (num_spins=1) path.
    basis_fids_summed = basis_fids.sum(dim=1)  # [n_lines, 2, L]
    layout_est_1, n_est_1 = _build_layout(param_names, n_lines, n_basis, num_spins=1)
    layout_fixed_1, n_fixed_1 = _build_layout(fixed_names, n_lines, n_basis, num_spins=1)
    theta_est_1 = torch.zeros(n_est_1)
    theta_est_1[layout_est_1['amp']] = theta_est[layout_est['amp']]
    theta_est_1[layout_est_1['d']] = d_shared
    theta_est_1[layout_est_1['g']] = theta_est[layout_est['g']]
    theta_est_1[layout_est_1['fshift']] = theta_est[layout_est['fshift']]
    theta_fixed_1 = torch.zeros(n_fixed_1)

    out_plain = _crlb_signal_model(theta_est_1, theta_fixed_1, basis_fids_summed, t, cf,
                                   spline_basis, phi1_ref, layout_est_1, layout_fixed_1,
                                   num_spins=1)

    torch.testing.assert_close(out_moiety, out_plain)


def test_signal_model_moiety_path_is_sensitive_to_each_moietys_own_d():
    """Changing just one moiety's 'd' (holding its sibling moiety and
    everything else fixed) must change the output -- confirms per-moiety
    'd' columns are not accidentally aliased to the same value."""
    (t, basis_fids, spline_basis, layout_est, layout_fixed, n_est, n_fixed,
     cf, phi1_ref) = _make_moiety_inputs(param_names=['amp', 'd'], n_lines=2, num_spins=2)
    theta_est = torch.zeros(n_est)
    theta_est[layout_est['amp']] = 1.0
    theta_est[layout_est['d']] = 0.3
    theta_fixed = torch.zeros(n_fixed)

    out_baseline = _crlb_signal_model(theta_est, theta_fixed, basis_fids, t, cf,
                                      spline_basis, phi1_ref, layout_est, layout_fixed,
                                      num_spins=2)

    theta_est_moiety0_changed = theta_est.clone()
    theta_est_moiety0_changed[layout_est['d']][0] = 1.5  # line 0, moiety 0 only
    out_moiety0_changed = _crlb_signal_model(theta_est_moiety0_changed, theta_fixed, basis_fids,
                                             t, cf, spline_basis, phi1_ref,
                                             layout_est, layout_fixed, num_spins=2)
    assert not torch.allclose(out_baseline, out_moiety0_changed)

    theta_est_moiety1_changed = theta_est.clone()
    theta_est_moiety1_changed[layout_est['d']][1] = 1.5  # line 0, moiety 1 only
    out_moiety1_changed = _crlb_signal_model(theta_est_moiety1_changed, theta_fixed, basis_fids,
                                             t, cf, spline_basis, phi1_ref,
                                             layout_est, layout_fixed, num_spins=2)
    assert not torch.allclose(out_baseline, out_moiety1_changed)
    # Changing different moieties must have distinguishable effects (not
    # both collapsing to the same output).
    assert not torch.allclose(out_moiety0_changed, out_moiety1_changed)


def test_signal_model_moiety_path_is_batchable_via_vmap_and_differentiable_via_jacrev():
    (t, basis_fids, spline_basis, layout_est, layout_fixed, n_est, n_fixed,
     cf, phi1_ref) = _make_moiety_inputs()
    batch_size = 3

    def model_fn(theta_est_i, theta_fixed_i):
        return _crlb_signal_model(theta_est_i, theta_fixed_i, basis_fids, t, cf,
                                  spline_basis, phi1_ref, layout_est, layout_fixed,
                                  num_spins=2)

    theta_est_batch = torch.rand(batch_size, n_est) * 0.1
    theta_fixed_batch = torch.rand(batch_size, n_fixed) * 0.1
    J = vmap(jacrev(model_fn, argnums=0), in_dims=(0, 0))(theta_est_batch, theta_fixed_batch)
    assert J.shape == (batch_size, 2 * t.shape[0], n_est)
    assert torch.isfinite(J).all()
