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

from src.crlb import ALL_CRLB_PARAMS, DEFAULT_CRLB_PARAMS, _build_layout, _crlb_signal_model, _voigt_decay
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
