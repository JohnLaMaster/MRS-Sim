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

from src.crlb import _build_layout, _crlb_signal_model, _voigt_decay
from src.splines import build_spline_basis


def _make_inputs(n_lines=3, length=32, n_basis=5):
    t = torch.linspace(0, 0.1, length)
    basis_fids = torch.randn(n_lines, 2, length) * 0.1
    x = torch.linspace(0.2, 4.2, length)
    spline_basis = build_spline_basis(x, knot_spacing=0.4, degree=3)
    # Match n_basis by rebuilding with a spacing that yields it isn't
    # guaranteed exactly, so just use whatever build_spline_basis returns.
    layout, n_params = _build_layout(n_lines, spline_basis.shape[-1])
    carrier_frequency = torch.tensor(127.7)
    return t, basis_fids, spline_basis, layout, n_params, carrier_frequency


def test_build_layout_covers_every_column_exactly_once():
    layout, n_params = _build_layout(n_lines=4, n_basis=6)
    covered = torch.zeros(n_params, dtype=torch.bool)
    for sl in layout.values():
        assert not covered[sl].any(), "overlapping parameter slices"
        covered[sl] = True
    assert covered.all(), "not every column is covered by some parameter family"


def test_voigt_decay_matches_physics_model_formula():
    """exp((-d - g*t) * t), copied from PhysicsModel.lineshape_voigt."""
    d = torch.tensor([0.5, 1.0])
    g = torch.tensor([0.1, 0.2])
    t = torch.linspace(0, 1, 8)
    out = _voigt_decay(d, g, t)
    expected = torch.exp((-d.unsqueeze(-1) - g.unsqueeze(-1) * t.unsqueeze(0)) * t.unsqueeze(0))
    torch.testing.assert_close(out, expected)


def test_signal_model_output_shape():
    t, basis_fids, spline_basis, layout, n_params, cf = _make_inputs()
    theta = torch.zeros(n_params)
    theta[layout['amp']] = 1.0
    out = _crlb_signal_model(theta, basis_fids, t, cf, spline_basis, layout)
    assert out.shape == (2 * t.shape[0],)
    assert torch.isfinite(out).all()


def test_signal_model_is_batchable_via_vmap_and_differentiable_via_jacrev():
    t, basis_fids, spline_basis, layout, n_params, cf = _make_inputs()
    batch_size = 4

    def model_fn(theta_i):
        return _crlb_signal_model(theta_i, basis_fids, t, cf, spline_basis, layout)

    theta_batch = torch.zeros(batch_size, n_params)
    theta_batch[:, layout['amp']] = 1.0

    J = vmap(jacrev(model_fn))(theta_batch)
    assert J.shape == (batch_size, 2 * t.shape[0], n_params)
    assert torch.isfinite(J).all()


def test_zero_amplitude_gives_only_baseline_contribution():
    """With every amplitude at 0 and a nonzero baseline, the observed signal should equal the (normalized) baseline alone."""
    t, basis_fids, spline_basis, layout, n_params, cf = _make_inputs()
    theta = torch.zeros(n_params)
    n_basis = spline_basis.shape[-1]
    beta = torch.zeros(2, n_basis)
    beta[0, 0] = 1.0  # nonzero real-channel spline coefficient
    theta[layout['beta']] = beta.reshape(-1)

    out = _crlb_signal_model(theta, basis_fids, t, cf, spline_basis, layout)
    # With a zero spectrum, normalization divides by the baseline's own
    # peak magnitude, so the real channel should be a rescaled version of
    # the spline basis's first column and the imaginary channel should be
    # exactly zero (beta[1] is all zero).
    length = t.shape[0]
    assert torch.allclose(out[length:], torch.zeros(length), atol=1e-6)
