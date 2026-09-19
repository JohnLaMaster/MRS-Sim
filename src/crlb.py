"""
CRLB / Fisher information (v2.0 handover, section 6).

Optional, PyTorch-based, disabled-by-default analysis quantity -- not part
of the normal forward/training path (`PhysicsModel.forward()` never calls
this on its own; see `_crlb_forward_model` usage in `compute_crlb()`).

Scope of the CRLB forward model (documented explicitly, since this is a
deliberately-limited first implementation, not the full simulator):
included are per-line complex amplitude, Voigt lineshape (Lorentzian `d` +
Gaussian `g`), per-line frequency shift, global zero-order phase, and the
baseline's spline coefficients as nuisance parameters. NOT included: B0
field inhomogeneity, eddy currents, multi-coil combination, first-order
phase, residual water, and resampling/cropping/zero-filling (the model
assumes the acquired grid needs no resampling, matching a config like
`cows.json`'s `resample: False`). These follow the classic MRS CRLB
parameterization (e.g. Cavassila et al. 2001) rather than the full
stochastic generative pipeline (bounded-random-walk baseline generation,
B0 field maps, etc. are not part of "the signal model" in the CRLB sense
even in the classic literature).

To keep the forward model `torch.func.vmap`-safe (PhysicsModel's own
`lineshape_correction`/`frequency_shift`/etc. branch on tensor rank in ways
that don't reliably survive vmap's tracing), the Voigt decay and amplitude
scaling are short, direct copies of the exact formulas in
`PhysicsModel.lineshape_voigt`/`modulate` (verified against
physics_model.py at the time of writing, not from memory), while frequency
shift, phase, and the FFT reuse `aux.py`'s generic `complex_exp`/
`Fourier_Transform` directly, since those have no rank-branching and are
shape-generic.

Observation model: per the handover doc, a complex spectrum is represented
as a real-valued vector by concatenating its real and imaginary parts.
Zero-filled points, when a `mask` is supplied, are excluded from the
observation vector so they never contribute spurious information (the
current zero_fill() pipeline in physics_model.py is not exercised by this
first implementation -- pass `mask` explicitly if using it).

Noise covariance: `sigma` is one real-valued standard deviation per
sample, consistent with `generate_noise()`'s i.i.d. Gaussian noise model
(same variance on both real and imaginary channels); the covariance is
`sigma**2 * I`. Pass the *realized* noise's empirical std
(`SimulationResult.noise.std(dim=-1)`) for consistency with the actual
simulated noise realization, per the handover doc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.func import jacrev, vmap

from .aux import Fourier_Transform, complex_exp
from .splines import evaluate_spline

__all__ = ['CRLBResult', 'compute_crlb']


@dataclass
class CRLBResult:
    crlb: torch.Tensor                 # [batch, n_params]
    fim: Optional[torch.Tensor] = None  # [batch, n_params, n_params]
    labels: Optional[List[str]] = None  # len == n_params


def _voigt_decay(d: torch.Tensor, g: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """exp((-d - g*t) * t), one-to-one with PhysicsModel.lineshape_voigt's formula. d, g: [n_lines]; t: [L]. Returns [n_lines, L]."""
    d = d.unsqueeze(-1)
    g = g.unsqueeze(-1)
    return torch.exp((-d - g * t.unsqueeze(0)) * t.unsqueeze(0))


def _crlb_signal_model(
    theta: torch.Tensor,
    basis_fids: torch.Tensor,
    t: torch.Tensor,
    carrier_frequency: torch.Tensor,
    spline_basis: torch.Tensor,
    layout: dict,
) -> torch.Tensor:
    """
    Single-sample (no batch dim), differentiable, real-valued observation
    model: `theta -> [real(spectrum); imag(spectrum)]`. Intended to be
    wrapped with `torch.func.vmap` for batched use -- see `compute_crlb`.

    `basis_fids`: [n_lines, 2, L] (PhysicsModel.syn_basis_fids with its
    leading batch=1 dimension squeezed out). `t`: [L]. `spline_basis`:
    [L, n_basis].
    """
    amp = theta[layout['amp']]
    d = theta[layout['d']]
    g = theta[layout['g']]
    fshift_ppm = theta[layout['fshift']]
    phi0 = theta[layout['phi0']]
    beta = theta[layout['beta']].view(2, -1)

    # modulate: amp * basis_fids (PhysicsModel.modulate's formula)
    fid = amp.unsqueeze(-1).unsqueeze(-1) * basis_fids  # [n_lines, 2, L]

    # lineshape_voigt: fid * exp((-d - g*t) * t)
    decay = _voigt_decay(d, g, t)  # [n_lines, L]
    fid = fid * decay.unsqueeze(1)  # broadcast over the real/imag channel dim

    # frequency_shift: fid * exp(i * fshift_hz * t), via complex_exp (reused
    # directly rather than re-derived, since it defines the exact
    # real/imag rotation convention the rest of the pipeline relies on)
    fshift_hz = fshift_ppm * carrier_frequency * 2 * torch.pi
    f_shift_angle = fshift_hz.unsqueeze(-1) * t.unsqueeze(0)  # [n_lines, L]
    fid = complex_exp(fid, f_shift_angle.unsqueeze(1))  # fid: [n_lines, 2, L]

    # line_summing: sum over lines
    fid_sum = fid.sum(dim=0)  # [2, L]

    # zero_order_phase: complex_exp(fid, -phi0_rad)
    phi0_rad = phi0 * torch.pi / 180.0
    fid_sum = complex_exp(fid_sum.unsqueeze(0), (-phi0_rad).view(1, 1, 1)).squeeze(0)  # [2, L]

    # Fourier_Transform requires ndim>=3; add and drop a dummy leading dim.
    spectrum = Fourier_Transform(fid_sum.unsqueeze(0)).squeeze(0)  # [2, L]

    # baseline spline nuisance term, added directly in the frequency domain
    # (matching where baseline is actually added relative to the final
    # spectrum -- see docs/v2/architecture_v1_audit.md section 3)
    baseline = evaluate_spline(beta, spline_basis)  # [2, L]

    observed = spectrum + baseline  # [2, L]

    # normalize: matches aux.normalize()'s formula for a [2, L] (real,
    # imag) signal exactly (denom = max magnitude over L). The "final
    # spectrum" this section computes CRLBs from is the normalized one
    # PhysicsModel actually returns -- omitting this step produced outputs
    # off from the real pipeline's by several orders of magnitude when
    # this was verified (see docs/v2/progress_log.md, Milestone 6).
    magnitude = torch.sqrt(observed[0] ** 2 + observed[1] ** 2)
    denom = magnitude.amax()
    denom = torch.where(denom == 0, torch.ones_like(denom) * 1e-6, denom)
    observed = observed / denom

    return torch.cat([observed[0], observed[1]], dim=0)  # [2L]


def _build_layout(n_lines: int, n_basis: int) -> dict:
    i = 0
    layout = {}
    for name, size in [('amp', n_lines), ('d', n_lines), ('g', n_lines),
                        ('fshift', n_lines), ('phi0', 1), ('beta', 2 * n_basis)]:
        layout[name] = slice(i, i + size)
        i += size
    return layout, i


def compute_crlb(
    pm,
    params: torch.Tensor,
    spline_coefficients: torch.Tensor,
    spline_basis: torch.Tensor,
    sigma: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    return_fim: bool = False,
    rcond: float = 1e-10,
) -> CRLBResult:
    """
    Compute the Cramer-Rao Lower Bound for one batch of simulated samples.

    Parameters
    ----------
    pm :
        The PhysicsModel the samples came from (used for its basis
        functions, time axis, carrier frequency, and parameter registry --
        not called/differentiated through directly, see the module
        docstring for why).
    params :
        The full `[batch, n_columns]` sampled-parameter tensor (the same
        tensor passed to `forward()`) -- used as theta_hat, the point the
        CRLB is linearized around.
    spline_coefficients :
        `[batch, 1, 2, n_basis]` or `[batch, 2, n_basis]`
        (`SimulationResult.spline_coefficients`) -- the baseline's fitted
        spline coefficients, included as nuisance parameters per the
        handover doc.
    spline_basis :
        `[L, n_basis]`, the same basis `spline_coefficients` was fit
        against (`SimulationResult`'s spline fit uses `pm.ppm_cropped`).
    sigma :
        `[batch]` or `[batch, 1]`, one real-valued noise standard deviation
        per sample (same value for both real and imaginary channels),
        consistent with `generate_noise()`'s noise model. Pass
        `SimulationResult.noise.std(dim=-1)` (per real/imag channel,
        averaged or matched appropriately) for consistency with the actual
        realized noise.
    mask :
        Optional `[2L]` boolean mask selecting genuinely-acquired
        observations (see the module docstring re: zero-filling). Defaults
        to all `True` (nothing excluded).
    return_fim :
        Also return the full Fisher information matrix.
    rcond :
        Passed to `torch.linalg.pinv` when inverting the Fisher
        information matrix. MRS Fisher matrices are commonly
        ill-conditioned (highly overlapping/correlated basis lines,
        especially macromolecule/lipid groups); expect near-zero or
        slightly negative CRLB entries (floating-point noise around zero)
        for parameters the data barely constrain, rather than these being
        indicative of a bug. Treat such entries as "effectively unbounded
        precision" / poorly identified, not as literal negative variances.
        Inspect `torch.linalg.eigvalsh(fim)` / `torch.linalg.cond(fim)` if
        you need to diagnose which samples/parameters are affected.

    Returns
    -------
    CRLBResult
    """
    n_lines = len(pm.index['d'])
    n_basis = spline_basis.shape[-1]
    layout, n_params = _build_layout(n_lines, n_basis)

    amp = params[:, pm.index['metabolites']]
    d = params[:, pm.index['d']]
    g = params[:, pm.index['g']]
    fshift = params[:, pm.index['f_shifts']]
    phi0 = params[:, pm.index['phi0']].unsqueeze(-1) if params[:, pm.index['phi0']].ndim == 1 else params[:, pm.index['phi0']]
    beta = spline_coefficients.reshape(spline_coefficients.shape[0], -1)

    theta = torch.cat([amp, d, g, fshift, phi0, beta], dim=-1)  # [batch, n_params]

    basis_fids = pm.syn_basis_fids.squeeze(0)  # [n_lines, 2, L]
    t = pm.t.squeeze()
    carrier_frequency = pm.carrier_frequency

    def model_fn(theta_i):
        return _crlb_signal_model(theta_i, basis_fids, t, carrier_frequency, spline_basis, layout)

    jacobian_fn = vmap(jacrev(model_fn))
    J = jacobian_fn(theta)  # [batch, 2L, n_params]

    if mask is not None:
        J = J[:, mask, :]

    sigma = sigma.reshape(sigma.shape[0], *([1] * (J.ndim - 1)))
    weighted_J = J / sigma  # equivalent to Sigma^{-1/2} @ J for Sigma = sigma^2 * I

    fim = torch.matmul(weighted_J.transpose(-1, -2), weighted_J)  # [batch, n_params, n_params]

    fim_pinv = torch.linalg.pinv(fim, rcond=rcond, hermitian=True)
    crlb = torch.diagonal(fim_pinv, dim1=-2, dim2=-1)

    # Per the handover doc: "the parameter registry must identify each
    # CRLB/FIM dimension" -- label each line by its actual metabolite/MM
    # name (registry.metabolite_names is in the same order as pm.index['d']
    # /['g']/['f_shifts'], see ParameterRegistry's docstring) rather than a
    # generic "line{i}" placeholder.
    from .parameters import ParameterRegistry
    line_names = ParameterRegistry.from_physics_model(pm).metabolite_names
    labels = (
        [f'{name}.amplitude' for name in line_names]
        + [f'{name}.d' for name in line_names]
        + [f'{name}.g' for name in line_names]
        + [f'{name}.frequency_shift' for name in line_names]
        + ['phi0']
        + [f'baseline_spline.real[{i}]' for i in range(n_basis)]
        + [f'baseline_spline.imag[{i}]' for i in range(n_basis)]
    )

    return CRLBResult(crlb=crlb, fim=fim if return_fim else None, labels=labels)
