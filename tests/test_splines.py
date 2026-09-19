"""
Unit tests for src.splines (v2.0 handover section 5).
"""
import numpy as np
import pytest
import torch

from src.splines import build_spline_basis, evaluate_spline, fit_baseline_spline, fit_spline


def test_basis_shape_and_no_nans():
    x = torch.linspace(0.2, 4.2, 512)
    basis = build_spline_basis(x, knot_spacing=0.4, degree=3)
    assert basis.shape[0] == 512
    assert basis.shape[1] > 1
    assert not torch.isnan(basis).any()


def test_fit_reconstructs_a_smooth_signal_closely():
    """A smooth, low-frequency signal should be fit closely by the spline."""
    x = torch.linspace(0.2, 4.2, 512)
    signal = torch.sin(x) + 0.5 * torch.cos(2 * x)  # smooth, well within spline capacity
    fit = fit_baseline_spline(signal, x, knot_spacing=0.4)
    max_err = (fit.fitted - signal).abs().max().item()
    assert max_err < 0.05, f"expected a close fit for a smooth signal, got max error {max_err}"


def test_fit_is_not_exact_ie_measurably_different_from_generated():
    """
    Per the handover doc: "the difference between generated and fitted
    baseline should be measurable/testable" -- i.e. the fit is a lossy
    approximation, not a copy, for a signal with structure finer than the
    spline can represent.
    """
    x = torch.linspace(0.2, 4.2, 512)
    rng = torch.Generator().manual_seed(0)
    noisy_structure = torch.sin(x * 20) * 0.1 + torch.randn(512, generator=rng) * 0.05
    fit = fit_baseline_spline(noisy_structure, x, knot_spacing=0.4)
    diff = (fit.fitted - noisy_structure).abs()
    assert diff.max().item() > 1e-4, "fit should not be an exact copy of a signal with fine structure"


def test_operates_on_arbitrary_leading_dimensions_without_loops():
    x = torch.linspace(0.2, 4.2, 64)
    batch = torch.randn(5, 3, 2, 64)  # e.g. [batch, extra, channels, length]
    fit = fit_baseline_spline(batch, x, knot_spacing=0.5)
    assert fit.coefficients.shape[:-1] == batch.shape[:-1]
    assert fit.fitted.shape == batch.shape


def test_batched_fit_matches_per_sample_fit():
    """Vectorized batch fitting must give identical results to fitting each sample alone."""
    x = torch.linspace(0.2, 4.2, 128)
    batch = torch.randn(6, 128)
    basis = build_spline_basis(x, knot_spacing=0.4)
    batched_coeffs = fit_spline(batch, basis)
    for i in range(batch.shape[0]):
        single_coeffs = fit_spline(batch[i], basis)
        torch.testing.assert_close(batched_coeffs[i], single_coeffs)


def test_fit_spline_does_not_require_grad():
    x = torch.linspace(0.2, 4.2, 64)
    basis = build_spline_basis(x)
    signal = torch.randn(64, requires_grad=True)
    coeffs = fit_spline(signal, basis)
    assert not coeffs.requires_grad


def test_evaluate_spline_is_differentiable_wrt_coefficients():
    x = torch.linspace(0.2, 4.2, 64)
    basis = build_spline_basis(x)
    coeffs = torch.zeros(basis.shape[1], requires_grad=True)
    fitted = evaluate_spline(coeffs, basis)
    loss = fitted.sum()
    loss.backward()
    assert coeffs.grad is not None
    assert torch.isfinite(coeffs.grad).all()


def test_mismatched_shapes_raise_clear_error():
    x = torch.linspace(0.2, 4.2, 64)
    basis = build_spline_basis(x)
    with pytest.raises(ValueError):
        fit_spline(torch.randn(10), basis)
