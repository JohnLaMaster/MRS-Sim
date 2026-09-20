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
from typing import Any, Dict, List, Optional

import torch

from .parameters import SimulationParameters

__all__ = ['SimulationResult']


@dataclass
class SimulationResult:
    noisy: Optional[torch.Tensor] = None
    noise_free_total: Optional[torch.Tensor] = None
    nuisance_free: Optional[torch.Tensor] = None

    baseline: Optional[torch.Tensor] = None
    # Handover section 5: the generated baseline immediately fit with a
    # spline (src/splines.py), without altering `baseline` itself. Both
    # are None whenever `baseline` is None (no baseline was generated).
    baseline_fit: Optional[torch.Tensor] = None
    spline_coefficients: Optional[torch.Tensor] = None

    residual_water: Optional[torch.Tensor] = None
    noise: Optional[torch.Tensor] = None

    parameters: Optional[SimulationParameters] = None

    # Handover section 7 audit (docs/v2/progress_log.md): target_snr and
    # realized_snr are not measured against the same signal, and
    # generate_noise() internally applies a decibel-style log conversion
    # to the sampled target_snr column even though SNR in MRS is a
    # unitless ratio (per the repo owner's expert-consensus correction --
    # see generate_noise()'s own comment/formula, flagged as an open
    # question about whether that internal conversion itself needs
    # revisiting, not settled here). Both facts are documented explicitly
    # here instead of left as an implicit trap for anyone comparing the
    # two directly.
    #
    # `target_snr` ([batch]): the sampled target SNR
    # (params[:, index['snr']]) used to derive the noise standard
    # deviation in generate_noise() -- referenced against the peak
    # amplitude of the metabolite line(s) named in `pm.snr_metab`
    # (defaults to `pm.wrt_metab`). SNR itself is unitless (a plain
    # ratio); see the note above re: generate_noise()'s internal handling.
    target_snr: Optional[torch.Tensor] = None

    # `realized_snr` (dict with 'power'/'spectral' keys, each
    # [batch, num_bF, channels, 1], unitless ratio): computed from the
    # *actual drawn* noise realization's measured standard deviation
    # (noise_vec.std()), divided into each individual basis-function
    # line's own clean-signal amplitude -- NOT measured from this
    # result's final `noisy`/`spectrum` output (that would additionally
    # reflect baseline/residual-water/multicoil-combination/phase/
    # frequency-shift/eddy-current/normalization, none of which feed into
    # this computation). 'spectral' is a frequency-domain peak height
    # (real channel only); 'power' is the FID's t=0 time-domain value
    # (real and imaginary channels both kept) -- despite the name, this
    # is closer to a total-signal/area quantity than squared power. The
    # generate_noise() docstring documents the original author's own
    # observation that this target/realized correspondence is close but
    # has real sample-to-sample variance, not floating-point noise.
    realized_snr: Optional[Dict[str, Any]] = None

    # Handover section 6: CRLB/Fisher information. Disabled by default
    # (forward(..., compute_crlb=False)); see src/crlb.py for the model's
    # documented scope. `crlb_labels` identifies each CRLB/FIM dimension
    # (registry.metabolite_names order for amplitude/d/g/frequency_shift,
    # then 'phi0', then baseline spline coefficients), per the handover
    # doc's "the parameter registry must identify each CRLB/FIM dimension".
    crlb: Optional[torch.Tensor] = None
    fim: Optional[torch.Tensor] = None
    crlb_labels: Optional[List[str]] = None

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
