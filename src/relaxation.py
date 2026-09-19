"""
Relaxation mechanisms (v2.0 handover, section 10).

The four equations below are transcribed directly from the repo owner
(first author of "Synthetic Data in MR Spectroscopy: Current Practices,
Applications, and Considerations", arXiv:2602.23463), Table 1 of that
paper's supplement, per their explicit request -- not derived from memory
(the paper's main text, which is publicly available, is a narrative review
of how existing tools handle relaxation, not a specification; the table
itself lives in a supplement not otherwise accessible in this
environment).

    T1  (longitudinal relaxation):
        M0 * (1 - exp(-TR / T1))
    T1* (apparent/steady-state longitudinal relaxation; Kaptein et al.
         1976, Taylor et al. 2016 -- the spoiled-gradient-echo/Ernst
         steady-state signal equation):
        M0 * (1 - exp(-TR / T1)) / (1 - cos(theta) * exp(-TR / T1)) * sin(theta)
    T2  (transverse relaxation):
        M0 * exp(-TE / T2)
    T2* (apparent transverse relaxation):
        M0 * exp(-TE / T2*)

Each function returns the scaling factor relative to `m0` (default 1.0),
since in MRS-Sim the quantity these factors multiply is the sampled
metabolite amplitude/concentration, not an absolute magnetization.

Inputs may be plain Python floats or torch.Tensors (broadcastable), and
the return type follows the inputs' type -- these are pure, stateless
functions, usable independently of PhysicsModel.

IMPORTANT -- not yet wired into PhysicsModel.forward(): these represent an
amplitude bias from the *timing* of the acquisition (how much of the
equilibrium magnetization is actually observed given this sequence's
TE/TR/flip angle), which is conceptually distinct from the existing
per-sample Lorentzian/Gaussian linewidth parameters ('d'/'g') that already
shape the FID's decay *during* the readout window. Whether 'd' already
encodes a T2-derived quantity in a way that would double-count with a T2*
amplitude scaling here has not been confirmed (the current definition
range for 'd' is read directly from metabolites_database.json's T2.metab
values in milliseconds with no explicit 1/T2 conversion visible in
PhysicsModel.define_parameter_ranges() -- see docs/v2/progress_log.md).
Wiring this into the forward pipeline is deferred until that's resolved,
so as not to reintroduce a new double-application bug while fixing the
old one (V1_0).
"""
from __future__ import annotations

import math
from typing import Union

import torch

__all__ = ['t1_recovery', 't1_star_recovery', 't2_decay', 't2_star_decay']

ArrayLike = Union[torch.Tensor, float]


def _is_tensor_input(*args) -> bool:
    return any(torch.is_tensor(a) for a in args)


def _exp(x, use_torch: bool):
    return torch.exp(x) if use_torch else math.exp(x)


def t1_recovery(TR: ArrayLike, T1: ArrayLike, m0: ArrayLike = 1.0) -> ArrayLike:
    """M0 * (1 - exp(-TR / T1)) -- longitudinal (T1) relaxation."""
    use_torch = _is_tensor_input(TR, T1, m0)
    return m0 * (1 - _exp(-TR / T1, use_torch))


def t1_star_recovery(
    TR: ArrayLike,
    T1: ArrayLike,
    flip_angle: ArrayLike,
    m0: ArrayLike = 1.0,
    degrees: bool = True,
) -> ArrayLike:
    """
    M0 * (1 - exp(-TR/T1)) / (1 - cos(theta) * exp(-TR/T1)) * sin(theta)

    Apparent/steady-state longitudinal relaxation (Kaptein et al. 1976,
    Taylor et al. 2016) -- the spoiled-gradient-echo/Ernst steady-state
    signal equation. `flip_angle` (theta) is in degrees by default
    (`degrees=False` for radians).
    """
    use_torch = _is_tensor_input(TR, T1, flip_angle, m0)
    if use_torch:
        theta = torch.deg2rad(torch.as_tensor(flip_angle)) if degrees else flip_angle
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    else:
        theta = math.radians(flip_angle) if degrees else flip_angle
        cos_t, sin_t = math.cos(theta), math.sin(theta)
    e1 = _exp(-TR / T1, use_torch)
    return m0 * (1 - e1) / (1 - cos_t * e1) * sin_t


def t2_decay(TE: ArrayLike, T2: ArrayLike, m0: ArrayLike = 1.0) -> ArrayLike:
    """M0 * exp(-TE / T2) -- transverse (T2) relaxation."""
    return m0 * _exp(-TE / T2, _is_tensor_input(TE, T2, m0))


def t2_star_decay(TE: ArrayLike, T2_star: ArrayLike, m0: ArrayLike = 1.0) -> ArrayLike:
    """M0 * exp(-TE / T2*) -- apparent transverse (T2*) relaxation."""
    return m0 * _exp(-TE / T2_star, _is_tensor_input(TE, T2_star, m0))
