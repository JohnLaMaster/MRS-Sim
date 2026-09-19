# V2.0 Refactor — Progress Log

Running log of milestones completed on the `V2.0` branch, kept up to date as
work proceeds (per the handover doc's "Final requirement" reporting checklist:
files changed, behavior changes, assumptions resolved, tests added/run,
remaining limitations). See `docs/v2/architecture_v1_audit.md` for the frozen
pre-refactor architecture baseline this work builds on.

## Milestone 1 — Architecture audit (commit `60025c4`)

- Read and documented the full v1 forward-simulation pipeline, parameter
  representation, SNR handling, baseline generation, config schema, and the
  in-vivo copula sampler.
- No code changes; no behavior changes.

## Milestone 2 — Parameter registry + `SimulationParameters` (commit `386c3ab`)

**Handover section addressed**: 1 (first-class simulation parameters).

**Files changed**:
- `src/parameters.py` (new) — `ParameterRegistry`, `MetaboliteParameterView`,
  `SimulationParameters`.
- `tests/test_parameters.py` (new) — 13 tests.
- `src/physics_model.py` — one-line bugfix (see below).

**Design**: wraps `PhysicsModel.index` rather than replacing it; no new
tensor storage layout. Gives `params["NAA"]["concentration"]`,
`params["NAA"]["lorentzian"]`, `params["NAA"]["frequency_shift"]`-style
access, plus `registry.labels()` for future CRLB/FIM axis naming.
`SimulationParameters` also carries `baseline`/`residual_water` sampling
dicts alongside the tensor (see architecture doc section 1 for why these
stay separate from the tensor rather than being forced into its column
layout).

**Behavior change (bugfix, not a physics change)**: `physics_model.py`'s
`initialize()` aliased `header, cnt = self._metab, counter(...)` — a later
loop then appended dozens of parameter-family name strings onto `header`,
which silently mutated `self._metab` in place (they were the same list
object). Fixed to `list(self._metab)`. This only affects the metadata
`pm.metab`/`pm._metab` expose (needed correct metabolite names to build the
registry); verified it does not change forward-pass numerics — the one
other place `len(self._metab)` was read (`line_summing()`) computes a value
that is never subsequently used. Verified against a real basis set
(`cows.json`): `pm.metab` now returns exactly the 28 configured names
instead of a polluted, much longer list.

**Assumptions resolved**: none open.

**Tests**: 13 unit tests (synthetic index fixture, no basis-set file
needed) + manual verification against a real `PhysicsModel` loaded from
`cows.json`/`VERI_PRESS_30ms_GE_2000_wMM.mat`.

**Remaining/follow-up**: T1/T2 accessors raise `NotImplementedError` until
relaxation modeling (handover section 10) adds real columns for them.

## Milestone 3 — Composable sampler (commit `4a532e7`)

**Handover section addressed**: 2 (composable parameter sampler).

**Files changed**:
- `src/sampling.py` (new) — `ParameterSampler`, `UniformRangeSampler`,
  `CopulaInVivoSampler`.
- `tests/test_sampling.py` (new) — 12 tests.
- `src/findParamDist.py` — `--findCorr` now also saves the ordered variable
  name list alongside the correlation matrix.

**Design**: `UniformRangeSampler` formalizes the plain
`torch.rand -> PhysicsModel.quantify_params` pattern (currently duplicated
in `mrs-sim_template.py`/`deep_learning_dataset_template.py`) as a reusable
backend. `CopulaInVivoSampler` generalizes `sim_COWS.py`'s Gaussian-copula,
in-vivo-fitting-derived sampling into a config-driven sampler, per the repo
owner's request to keep this sampling method available in v2.0, and adds
`include_params`/`exclude_params` to select/exclude specific fitting
parameters from the copula for a given run (also per request). Both
samplers own an explicit `torch.Generator`/`numpy.random.Generator` per
instance rather than touching global RNG state.

**Investigation outcome (root cause, not fully verified)**: the copula
sampler's unexplained magic-number correlation-matrix reordering
(`sample_from_fitted_dist.py:157-170`, flagged as an open item in the
architecture audit) most likely exists because `findParamDist.py`
accumulates its distributions JSON across multiple runs via
`dict.update()` (which keeps each key's original insertion position) while
`--findCorr` recomputes the correlation matrix fresh in the *current* run's
iteration order — so the two silently drift out of sync whenever the JSON
was built incrementally across more than one run. **I could not verify
this against the actual data** (`parameter_distributions_best_fit_recovered.json`
/ `correlation_matrix.mat`, referenced by `src/config/cows.json`): both
files are pointed at `/home/john/Documents/Repositories/Augmentrum/ignore/`
on a different machine and are not present in this environment. If this
diagnosis is wrong, `CopulaInVivoSampler`'s `legacy_reorder` parameter still
provides an explicit, documented escape hatch for the exact same situation.

**Behavior change**: none to any existing script. `sim_COWS.py` and
`src/aux/sample_from_fitted_dist.py` are untouched and continue to behave
exactly as before — `CopulaInVivoSampler` is a new, additive path, not a
replacement. Within the new sampler only: a column not covered by the
(possibly filtered) copula is now filled by `UniformRangeSampler` rather
than left as raw, never-quantified `[0, 1)` noise (the gap that exists
today in `sim_COWS.py` for any column its distributions JSON doesn't
mention and that isn't explicitly overridden afterward, e.g. phi0/phi1/eddy
current columns when `sample_from_copula` is used directly). This only
changes behavior for callers who adopt the new sampler.

**Assumptions resolved**: `findParamDist.py`'s `--findCorr` correlation
matrix now records its own variable order, removing the need to assume it
matches the distributions JSON's key order.

**Tests**: 12 unit tests, synthetic in-memory fixtures (no real basis set
or copula data files needed) covering both samplers' shapes, range
quantification, RNG reproducibility/isolation, copula column-overwrite
scope, include/exclude filtering, and both the name-aligned and legacy
(name-less) correlation-matrix loading paths.

**Remaining/follow-up**: `CopulaInVivoSampler`'s name-based alignment can
only be exercised end-to-end once a real (regenerated) distributions
JSON + correlation-matrix pair is available to test against — flagged for
whenever that data is accessible in this environment, or for the repo
owner to spot-check directly.

## Not yet started

Handover sections 3 (forward-sim component outputs/toggles), 4 (nuisance
removal), 5 (baseline spline fitting), 6 (CRLB/FIM), 7 (SNR audit/
formalization), 8 (parameter replay across basis sets), 9 (provenance), 10
(relaxation/TE/TR, including the agreed `V1_0` legacy-broadening flag), 11
(basis-set metadata / double-application audit), 12 (NIfTI-MRS export
audit), 13 (broader test-suite expansion beyond what's landed alongside
sections 1-2).
