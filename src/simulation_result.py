"""
Structured simulation output (v2.0 handover, section 3).

`PhysicsModel.forward(..., return_components=True)` returns a
`SimulationResult` instead of the legacy positional tuple
(`compile_outputs()`'s return value, which every existing caller --
`mainFcns.simulate()`, `sim_COWS.py` -- still uses unchanged). This is
purely additive: the legacy tuple path is untouched.

Terminology (matching the handover doc exactly):

- `noisy`: noise_free_total + noise. This is `result.spectrum`.
- `noise_free_total`: target signal + nuisance components (baseline,
  residual water), no noise.
- `nuisance_free`: nuisance components removed, physical transformations
  (relaxation/linewidth/phase/frequency-shift/etc.) retained. This is
  exactly `spectral_fit`'s existing "clean" branch -- PhysicsModel never
  adds baseline/residual-water to `spectral_fit`, so no new computation is
  needed to produce it; see docs/v2/architecture_v1_audit.md section 3.

T1/T2/relaxation and frequency shifts are *not* nuisance components (they
are physical model parameters and remain part of every one of the above),
per the handover doc's explicit terminology section.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from .parameters import SimulationParameters

__all__ = ['SimulationResult']


@dataclass
class SimulationResult:
    noisy: Optional[torch.Tensor] = None
    noise_free_total: Optional[torch.Tensor] = None
    nuisance_free: Optional[torch.Tensor] = None

    baseline: Optional[torch.Tensor] = None
    # Not yet implemented (handover section 5): fitting the generated
    # baseline with splines. Present as explicit None fields, not omitted,
    # so callers can tell "not computed" apart from "no such field".
    baseline_fit: Optional[torch.Tensor] = None
    spline_coefficients: Optional[torch.Tensor] = None

    residual_water: Optional[torch.Tensor] = None
    noise: Optional[torch.Tensor] = None

    parameters: Optional[SimulationParameters] = None
    target_snr: Optional[torch.Tensor] = None
    realized_snr: Optional[Dict[str, Any]] = None

    # Not yet implemented (handover section 6: CRLB/Fisher information).
    crlb: Optional[torch.Tensor] = None
    fim: Optional[torch.Tensor] = None

    quantities: Optional[Dict[str, Any]] = None

    # Not yet a full implementation of handover section 9; records what is
    # cheaply available now (which components were enabled for this call)
    # rather than the full provenance schema (git commit, RNG state,
    # basis-set hash, etc.), which is its own separate work item.
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def spectrum(self) -> Optional[torch.Tensor]:
        """Alias for `noisy`, matching the handover doc's `result.spectrum` sketch."""
        return self.noisy
