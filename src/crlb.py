"""
CRLB / Fisher information (v2.0 handover, section 6).

Optional, PyTorch-based, disabled-by-default analysis quantity -- not part
of the normal forward/training path (`PhysicsModel.forward()` never calls
this on its own; see `_crlb_forward_model` usage in `compute_crlb()`).

Scope of the CRLB forward model (documented explicitly, since this is a
deliberately-limited first implementation, not the full simulator):
included are per-line complex amplitude, Voigt lineshape (Lorentzian `d` +
Gaussian `g`), per-line frequency shift, global zero- and first-order
phase, and the baseline's spline coefficients as nuisance parameters. NOT
included: B0 field inhomogeneity, eddy currents, multi-coil combination,
the global frequency shift, residual water, and resampling/cropping/
zero-filling (the model assumes the acquired grid needs no resampling,
matching a config like `cows.json`'s `resample: False`). These follow the
classic MRS CRLB parameterization (e.g. Cavassila et al. 2001) rather than
the full stochastic generative pipeline (bounded-random-walk baseline
generation, B0 field maps, etc. are not part of "the signal model" in the
CRLB sense even in the classic literature).

**Which parameters are estimated vs. fixed** (repo-owner request,
docs/v2/progress_log.md): every parameter family above is always applied
in the forward model at its actual sampled/fitted value -- excluding a
family from the CRLB does not turn its physical effect off, it fixes it at
that value instead (the standard "nuisance parameter known exactly"
CRLB variant), matching how MRS fitting software commonly lets a user
decide whether e.g. first-order phase is estimated jointly with everything
else or held fixed. `compute_crlb(..., include_params=...,
exclude_params=...)` controls this -- see `ALL_CRLB_PARAMS`/
`DEFAULT_CRLB_PARAMS` below. Only the *included* families contribute rows/
columns to the returned `crlb`/`fim`/`labels`; excluded ones still shape
`fim` through the model's nonlinearity (a fixed nuisance value still
affects how identifiable the estimated parameters are), they just aren't
estimated themselves.

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

__all__ = ['CRLBResult', 'compute_crlb', 'ALL_CRLB_PARAMS', 'DEFAULT_CRLB_PARAMS']

# Canonical order -- also the order labels/columns appear in whenever more
# than one family is included, regardless of the order the caller passed.
ALL_CRLB_PARAMS = ('amp', 'd', 'g', 'fshift', 'phi0', 'phi1', 'beta')

# Matches this module's pre-existing behavior exactly (before
# include_params/exclude_params existed): phi1 was never modeled at all,
# so it stays opt-in rather than changing default output shape/labels.
DEFAULT_CRLB_PARAMS = ('amp', 'd', 'g', 'fshift', 'phi0', 'beta')


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
    theta_est: torch.Tensor,
    theta_fixed: torch.Tensor,
    basis_fids: torch.Tensor,
    t: torch.Tensor,
    carrier_frequency: torch.Tensor,
    spline_basis: torch.Tensor,
    phi1_ref: torch.Tensor,
    layout_est: dict,
    layout_fixed: dict,
) -> torch.Tensor:
    """
    Single-sample (no batch dim), differentiable, real-valued observation
    model: `(theta_est, theta_fixed) -> [real(spectrum); imag(spectrum)]`.
    Intended to be wrapped with `torch.func.vmap(torch.func.jacrev(...,
    argnums=0))` for batched use -- see `compute_crlb`.

    Every parameter family in `ALL_CRLB_PARAMS` is always physically
    applied; `layout_est`/`layout_fixed` (together covering exactly
    `ALL_CRLB_PARAMS`, see `compute_crlb`) just say whether a given
    family's value comes from `theta_est` (differentiated -> contributes
    to the CRLB/FIM) or `theta_fixed` (held constant at its actual sampled/
    fitted value -- still shapes the model's nonlinearity, just isn't
    itself estimated). See the module docstring's "which parameters are
    estimated vs. fixed" section.

    `basis_fids`: [n_lines, 2, L] (PhysicsModel.syn_basis_fids with its
    leading batch=1 dimension squeezed out). `t`, `phi1_ref`: [L].
    `spline_basis`: [L, n_basis].
    """
    def _get(name: str) -> torch.Tensor:
        if name in layout_est:
            return theta_est[layout_est[name]]
        return theta_fixed[layout_fixed[name]]

    amp = _get('amp')
    d = _get('d')
    g = _get('g')
    fshift_ppm = _get('fshift')
    phi0 = _get('phi0')
    phi1 = _get('phi1')
    beta = _get('beta').view(2, -1)

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

    # first_order_phase: PhysicsModel.first_order_phase() applies
    # complex_exp(FFT(fid), -phi1_ref*phi1_rad) then inverse-FFTs back to
    # time domain (so it can be called on, and return, a time-domain fid).
    # Nothing this model includes happens between that internal FFT and
    # the Fourier_Transform call directly above (the real pipeline's only
    # intervening step, the global frequency_shift, isn't modeled here
    # either -- see the module docstring's scope note), so applying the
    # same ramp directly to `spectrum` is exactly equivalent without the
    # redundant IFFT/FFT round trip.
    phi1_rad = phi1 * torch.pi / 180.0
    phase_ramp = (-1 * phi1_ref * phi1_rad).view(1, 1, -1)  # [1, 1, L]
    spectrum = complex_exp(spectrum.unsqueeze(0), phase_ramp).squeeze(0)  # [2, L]

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


def _build_layout(param_names, n_lines: int, n_basis: int) -> tuple:
    """
    Assign each name in `param_names` (an ordered subset/permutation of
    `ALL_CRLB_PARAMS`) a contiguous column slice, in the given order.
    Returns `(layout, n_params)`; `layout` is empty (and `n_params` is 0)
    for an empty `param_names`, which is valid (`compute_crlb` uses this
    for the "fixed" side when nothing is excluded).
    """
    sizes = {'amp': n_lines, 'd': n_lines, 'g': n_lines, 'fshift': n_lines,
             'phi0': 1, 'phi1': 1, 'beta': 2 * n_basis}
    i = 0
    layout = {}
    for name in param_names:
        layout[name] = slice(i, i + sizes[name])
        i += sizes[name]
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
    include_params: Optional[List[str]] = None,
    exclude_params: Optional[List[str]] = None,
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
    include_params :
        Which parameter families (from `ALL_CRLB_PARAMS`:
        `'amp'`, `'d'`, `'g'`, `'fshift'`, `'phi0'`, `'phi1'`, `'beta'`)
        to *estimate* -- i.e. differentiate w.r.t., so they get a row/
        column in `crlb`/`fim`/`labels`. Defaults to `DEFAULT_CRLB_PARAMS`
        (everything except `'phi1'`, matching this module's behavior
        before this option existed). Order in the input is irrelevant --
        output order always follows `ALL_CRLB_PARAMS`.
    exclude_params :
        Families to drop from whatever `include_params` (or the default)
        would otherwise estimate -- e.g. `exclude_params=['phi0']` to fix
        zero-order phase at its sampled value instead of estimating it.
        Families not being estimated are still applied in the forward
        model at their actual value (see the module docstring's "which
        parameters are estimated vs. fixed" section) -- they are never
        simply left out of the signal.

    Returns
    -------
    CRLBResult
    """
    include = list(include_params) if include_params is not None else list(DEFAULT_CRLB_PARAMS)
    if exclude_params:
        include = [p for p in include if p not in exclude_params]
    unknown = sorted(set(include) - set(ALL_CRLB_PARAMS))
    if unknown:
        raise ValueError(
            f"Unknown CRLB parameter name(s) {unknown}; valid names are {ALL_CRLB_PARAMS}.")
    if not include:
        raise ValueError("At least one parameter family must be included in the CRLB "
                          "(include_params/exclude_params left nothing to estimate).")
    # Canonical order regardless of what order the caller listed things in.
    include = [p for p in ALL_CRLB_PARAMS if p in include]
    exclude = [p for p in ALL_CRLB_PARAMS if p not in include]

    n_lines = len(pm.index['d'])
    n_basis = spline_basis.shape[-1]
    layout_est, n_est = _build_layout(include, n_lines, n_basis)
    layout_fixed, n_fixed = _build_layout(exclude, n_lines, n_basis)

    amp = params[:, pm.index['metabolites']]
    d = params[:, pm.index['d']]
    g = params[:, pm.index['g']]
    fshift = params[:, pm.index['f_shifts']]
    phi0 = params[:, pm.index['phi0']].unsqueeze(-1) if params[:, pm.index['phi0']].ndim == 1 else params[:, pm.index['phi0']]
    phi1 = params[:, pm.index['phi1']].unsqueeze(-1) if params[:, pm.index['phi1']].ndim == 1 else params[:, pm.index['phi1']]
    beta = spline_coefficients.reshape(spline_coefficients.shape[0], -1)

    values = {'amp': amp, 'd': d, 'g': g, 'fshift': fshift,
              'phi0': phi0, 'phi1': phi1, 'beta': beta}
    batch = amp.shape[0]

    theta_est = (torch.cat([values[name] for name in include], dim=-1)
                 if include else amp.new_zeros(batch, 0))
    theta_fixed = (torch.cat([values[name] for name in exclude], dim=-1)
                   if exclude else amp.new_zeros(batch, 0))

    basis_fids = pm.syn_basis_fids.squeeze(0)  # [n_lines, 2, L]
    t = pm.t.squeeze()
    carrier_frequency = pm.carrier_frequency
    phi1_ref = pm.phi1_ref.squeeze()

    def model_fn(theta_est_i, theta_fixed_i):
        return _crlb_signal_model(theta_est_i, theta_fixed_i, basis_fids, t,
                                  carrier_frequency, spline_basis, phi1_ref,
                                  layout_est, layout_fixed)

    jacobian_fn = vmap(jacrev(model_fn, argnums=0), in_dims=(0, 0))
    J = jacobian_fn(theta_est, theta_fixed)  # [batch, 2L, n_est]

    if mask is not None:
        J = J[:, mask, :]

    sigma = sigma.reshape(sigma.shape[0], *([1] * (J.ndim - 1)))
    weighted_J = J / sigma  # equivalent to Sigma^{-1/2} @ J for Sigma = sigma^2 * I

    fim = torch.matmul(weighted_J.transpose(-1, -2), weighted_J)  # [batch, n_est, n_est]

    fim_pinv = torch.linalg.pinv(fim, rcond=rcond, hermitian=True)
    crlb = torch.diagonal(fim_pinv, dim1=-2, dim2=-1)

    # Per the handover doc: "the parameter registry must identify each
    # CRLB/FIM dimension" -- label each line by its actual metabolite/MM
    # name (registry.metabolite_names is in the same order as pm.index['d']
    # /['g']/['f_shifts'], see ParameterRegistry's docstring) rather than a
    # generic "line{i}" placeholder.
    from .parameters import ParameterRegistry
    line_names = ParameterRegistry.from_physics_model(pm).metabolite_names
    label_map = {
        'amp': [f'{name}.amplitude' for name in line_names],
        'd': [f'{name}.d' for name in line_names],
        'g': [f'{name}.g' for name in line_names],
        'fshift': [f'{name}.frequency_shift' for name in line_names],
        'phi0': ['phi0'],
        'phi1': ['phi1'],
        'beta': ([f'baseline_spline.real[{i}]' for i in range(n_basis)]
                 + [f'baseline_spline.imag[{i}]' for i in range(n_basis)]),
    }
    labels = [label for name in include for label in label_map[name]]

    return CRLBResult(crlb=crlb, fim=fim if return_fim else None, labels=labels)
