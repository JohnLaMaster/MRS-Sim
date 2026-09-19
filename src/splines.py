"""
Batched spline fitting for baseline signals (v2.0 handover, section 5).

Fits a cubic B-spline to an already-generated baseline signal, immediately
after generation and before noise/phase/other contamination -- without
replacing the baseline actually used in the forward simulation. Per the
handover doc:

- default knot spacing: 0.4 ppm
- must operate on arbitrary n-dimensional batched tensors, vectorized, no
  Python loops over samples
- the fit itself is not differentiated through (`fit_spline` runs under
  `torch.no_grad()`); the *evaluated* spline `B @ beta` is differentiable
  with respect to the coefficients `beta`, since that is what the CRLB
  work (section 6) needs: a local, differentiable parameterization of the
  baseline around the fitted coefficients.
- provides a clean API independent of the forward simulator: none of this
  module imports from physics_model.py, and it operates on any
  (signal, x-axis) pair, not specifically PhysicsModel's baseline.

The spline basis matrix `B` (shape `[n_points, n_basis]`) depends only on
the x-axis and knot spacing, not on the data, so it is built once (via
`scipy.interpolate.BSpline`, looping only over the small number of basis
functions -- not over samples) and reused for every sample in a batch via
a single batched matrix multiply.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from scipy.interpolate import BSpline

__all__ = ['build_spline_basis', 'fit_spline', 'evaluate_spline', 'SplineFit', 'fit_baseline_spline']

ArrayLike = Union[torch.Tensor, np.ndarray]


def build_spline_basis(
    x: ArrayLike,
    knot_spacing: float = 0.4,
    degree: int = 3,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build a clamped B-spline design matrix for x-axis `x`.

    Parameters
    ----------
    x :
        1-D x-axis (e.g. ppm values) the spline is evaluated on, shape `[n_points]`.
    knot_spacing :
        Spacing between interior knots, in the same units as `x`. Default
        0.4 (ppm), per the handover doc.
    degree :
        B-spline degree (3 = cubic).

    Returns
    -------
    torch.Tensor, shape `[n_points, n_basis]`.
    """
    x_np = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    if x_np.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x_np.shape}.")

    x_min, x_max = float(x_np.min()), float(x_np.max())
    span = x_max - x_min
    n_interior = max(int(round(span / knot_spacing)) - 1, 1)
    interior_knots = np.linspace(x_min, x_max, n_interior + 2)[1:-1]

    # Clamped knot vector: degree+1 repeats at each end so the spline
    # spans exactly [x_min, x_max] with no extrapolation region.
    knots = np.concatenate([
        np.repeat(x_min, degree + 1),
        interior_knots,
        np.repeat(x_max, degree + 1),
    ])
    n_basis = len(knots) - degree - 1

    B = np.zeros((len(x_np), n_basis), dtype=np.float64)
    for i in range(n_basis):
        coeff = np.zeros(n_basis)
        coeff[i] = 1.0
        basis_fn = BSpline(knots, coeff, degree, extrapolate=False)
        B[:, i] = np.nan_to_num(basis_fn(x_np))

    return torch.as_tensor(B, dtype=dtype)


def fit_spline(signal: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """
    Least-squares fit `signal`'s last dimension onto `basis` columns.

    `signal`: `[..., n_points]`, any number of leading (batch/channel/etc.)
    dimensions. `basis`: `[n_points, n_basis]`. Returns coefficients
    `[..., n_basis]`.

    Not differentiable by design (per the handover doc: "do not
    differentiate through the spline-fitting procedure itself") -- the
    pseudoinverse and the resulting coefficients are both detached.
    """
    if signal.shape[-1] != basis.shape[0]:
        raise ValueError(
            f"signal's last dimension ({signal.shape[-1]}) must match "
            f"basis's first dimension ({basis.shape[0]})."
        )
    with torch.no_grad():
        pinv = torch.linalg.pinv(basis.to(signal.dtype))
        beta = torch.matmul(signal, pinv.T)
    return beta.detach()


def evaluate_spline(coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """
    `basis @ coefficients`, i.e. the fitted spline curve(s).

    Differentiable with respect to `coefficients` (this is the local,
    differentiable baseline parameterization the CRLB work needs -- see
    the module docstring). `coefficients`: `[..., n_basis]`, `basis`:
    `[n_points, n_basis]`. Returns `[..., n_points]`.
    """
    return torch.matmul(coefficients, basis.to(coefficients.dtype).T)


@dataclass
class SplineFit:
    coefficients: torch.Tensor  # [..., n_basis], detached (see fit_spline)
    basis: torch.Tensor         # [n_points, n_basis]
    fitted: torch.Tensor        # [..., n_points] == evaluate_spline(coefficients, basis)


def fit_baseline_spline(
    baseline: torch.Tensor,
    x: ArrayLike,
    knot_spacing: float = 0.4,
    degree: int = 3,
) -> SplineFit:
    """
    Fit `baseline` (last dimension aligned with `x`) with a cubic B-spline.

    Convenience wrapper combining `build_spline_basis` + `fit_spline` +
    `evaluate_spline`. See the module docstring for the intended use: fit
    the already-generated baseline, without replacing it in the forward
    simulation.
    """
    basis = build_spline_basis(x, knot_spacing=knot_spacing, degree=degree, dtype=baseline.dtype)
    coefficients = fit_spline(baseline, basis)
    fitted = evaluate_spline(coefficients, basis)
    return SplineFit(coefficients=coefficients, basis=basis, fitted=fitted)
