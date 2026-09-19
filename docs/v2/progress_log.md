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

## Milestone 4 — Bugfixes surfaced while preparing sections 3-4 (commits `a3af34d`, `e244695`)

**Handover sections addressed**: prep work for 3 (forward-sim component
outputs) and 4 (nuisance-component removal); not the full sections yet.

**Important correction first**: before starting this milestone, I found
that `src/physics_model.py` had uncommitted, unrecognized edits already
sitting in the working tree (debug prints and commented-out code in
`add_offsets()`/`simulate_offsets()`/`forward()`'s offset-handling block).
The repo owner didn't recognize the diff either, and recalled it was from
debugging why residual water wasn't showing up as expected when simulating
the COWS dataset. At their request I discarded it (`git checkout --
src/physics_model.py`) before making further changes. **This means the
architecture audit's original bug #1 (section 14) -- a claimed
`simulate_offsets()`/`add_offsets()` crash on partial baseline/
residual-water config -- was actually reading that uncommitted debug code,
not the real committed codebase.** The committed version does not have
that crash. The audit doc has been corrected in place (section 14, item 1)
rather than silently left wrong.

**Investigating the residual-water report**: tracing the discarded diff
showed it had changed `sim_COWS.py`'s initial `baselines, res_water = None,
None` to `config.baseline, config.reswater` (i.e. `True, True`). Fed
through `forward()`'s `presim` reuse logic, boolean `True` values (neither
`None` nor a real tensor) produce a nonsensical offset tuple
(`True + True == 2`) instead of triggering fresh generation -- a real
design flaw (the mechanism does not validate its own inputs) that this
specific experiment happened to trigger. Empirically re-verified against
the real, clean, committed code (`cows.json` + `UniformRangeSampler`) that
residual water *does* generate and appear correctly in `forward()`'s output
when `presim` is left as `None, None` as the committed `sim_COWS.py` does.
The most likely explanation is that the reported "missing residual water"
was introduced by the debugging attempt itself, not a pre-existing bug in
the committed pipeline -- though a plotting-side issue (`src/aux/plot_mrs.py`,
also separately modified/uncommitted) hasn't been ruled out and wasn't
investigated further here.

**Bugs fixed**:

1. **`quantify_params()` shape mismatch** (commit `a3af34d`): a dead,
   never-fully-implemented `'temperature'` column was unconditionally added
   to the list sizing `min_ranges`/`max_ranges`, but never to `ind`, making
   them one column wider than the params tensor. Broke `quantify_params()`
   -- and therefore `UniformRangeSampler` -- for every basis set. Found
   while validating milestone 3's sampler end to end.
2. **Noise leaking into `fidSum`'s "clean" branch** (commit `e244695`):
   `fidSum` and `spectral_fit` each had their own hand-written
   `[noisy, clean]` stacking code, and the two had drifted apart --
   `fidSum`'s "clean" branch also had noise added to it. Extracted into one
   shared `PhysicsModel._stack_noisy_clean()` (a `staticmethod`, directly
   unit-testable without a basis set) used by both, so they cannot silently
   diverge again. Verified with an isolated unit test reproducing the exact
   old-vs-new formulas, plus an end-to-end run against the real basis set.

**Files changed**: `src/physics_model.py` (both fixes, isolated into their
own commits from the discarded WIP), `tests/test_physics_model_bugfixes.py`
(new, 1 test), `docs/v2/architecture_v1_audit.md` (corrected section 14).

**Behavior changes**: `quantify_params()` now works instead of always
raising `RuntimeError` (never usable before, so no prior caller could have
depended on the crash). The noise-leak fix changes the *numeric content* of
`fidSum`'s clean branch for any caller that reads it while `noise=True`
(it no longer contains noise) -- flagged per the reporting requirement,
though given the branch's entire documented purpose is to be noise-free,
no correct usage could have depended on the old behavior.

**Assumptions resolved**: none new (the presim type-validation improvement
mentioned above is deferred to when sections 3/4 are implemented properly,
not done in this milestone).

**Tests**: 1 new unit test (`test_physics_model_bugfixes.py`), full suite
still 26/26 passing. Manual end-to-end verification against `cows.json`'s
real basis set for both fixes (see above).

**Remaining/follow-up**: the `presim` mechanism (in `forward()`'s offset
block) still silently misbehaves on non-`None`, non-tensor input instead of
raising a clear error -- planned for when sections 3/4 are properly
implemented, per the "use clear error messages for incompatible
configurations" constraint. `compile_outputs()` also crashes when
`noise=False` because `SNR={'power': None, 'spectral': None}` still gets
`.any()` called on it -- found incidentally while verifying the noise-leak
fix, not yet fixed, tracked here for when sections 3/4 land.

## Milestone 5 — Structured component outputs + nuisance removal (commit `3f49448`)

**Handover sections addressed**: 3 (forward-sim component outputs) and 4
(nuisance-component removal).

**Files changed**: `src/simulation_result.py` (new), `src/nuisance.py`
(new), `src/physics_model.py` (additive `return_components=True` path plus
two more small bugfixes), `tests/test_simulation_result.py` (new, 3
tests), `tests/test_nuisance.py` (new, 7 tests).

**Design**: `forward(..., return_components=True)` is a fully additive,
opt-in alternative return path -- `compile_outputs()`'s legacy positional
tuple is completely unchanged and is still what `mainFcns.simulate()`/
`sim_COWS.py` use. The key simplification: `spectral_fit`'s "clean" branch
already *is* the nuisance-free signal (baseline/residual water are only
ever added to `fidSum`), so `nuisance_free` needed exposing under that
name, not new computation. `SimulationResult` bundles this with
`noisy`/`noise_free_total`/`baseline`/`residual_water`/`noise`/
`parameters` (a `SimulationParameters`, tying in Milestone 2's registry)/
`target_snr`/`realized_snr`/`quantities`, plus explicit `None` placeholders
for `baseline_fit`/`spline_coefficients` (section 5) and `crlb`/`fim`
(section 6) so callers can tell "not computed yet" from "field doesn't
exist".

**Axis resolution (important correction to my own reasoning, not to
committed code)**: while verifying the Milestone 4 noise-leak fix, an
end-to-end cross-seed test gave a confusing result suggesting the wrong
axis held the noisy/clean split. Root cause: with baseline/residual-water
enabled, `bounded_random_walk`'s internal randomness (and `rand_omit`'s,
even at `drop_prob=0`) also consumes torch's global RNG on every
`forward()` call, so changing `torch.manual_seed` changes more than just
the noise realization -- confounding that comparison. Re-tested with
baselines/residual-water fully disabled (noise as the only randomness
source) and got a clean, unambiguous result: axis 1, index 0 = noisy,
index 1 = clean, exactly matching `generate_noise()`'s `d` variable (which
Milestone 4's fix already used correctly). No code changed as a result of
this, only my own earlier uncertainty resolved -- noted here since it's a
good illustration of why the handover doc's "do not rely on global RNG
state" requirement matters even outside the sampler itself.

**Bugs fixed incidentally while building this**:
1. `forward()`'s no-noise branch never assigned `d` (only `generate_noise()`
   did, which isn't called when `noise=False`) -- added `d = -3` there,
   matching the hardcoded axis the unsqueeze already used.
2. `compile_outputs()` crashed whenever `noise=False`, since
   `SNR={'power': None, 'spectral': None}` still called `.any()` on `None`.
   Fixed to check for `None` first.

**Nuisance removal (section 4)**: `remove_nuisance()` (trivial subtraction,
used internally and directly usable on a live `SimulationResult`) and
`remove_nuisance_from_saved()` (path-based, from a `mainFcns._save()`
`.mat` file). The path-based function has a **measured, documented
limitation**: reconstructing nuisance-free by subtracting saved baseline/
residual-water from the saved spectrum does not exactly match the true
nuisance-free signal. Measured directly against `cows.json`'s real basis
set (identical seed/parameters, comparing to a live `SimulationResult`):
**max absolute error 0.64 against a signal with max absolute value 0.99 --
a ~65% relative error**, not numerical noise. Most likely cause (not fully
confirmed): `baseline_cfg`/`resWater_cfg` each resample onto their own
internal ppm range (`cows.json`: `[-1.6, 6]` / `[4.4, 4.85]`) rather than
the main spectrum's `cropRange` (`[0.2, 4.2]`), and it wasn't confirmed
these end up aligned by the time they're saved. This is a real, open
architectural question, not a bug in the new `nuisance.py` code itself
(which was verified to correctly read/reconstruct from the real saved
file format -- the discrepancy is inherent to what gets saved). Use
`SimulationResult.nuisance_free` (the "during simulation" path) whenever
exactness matters; it has no such gap.

**Also found, not fixed here (bug in my own first draft, fixed before
committing)**: `scipy.io.savemat`/`loadmat` silently squeezes a saved
array's size-1 axis on round-trip (verified directly: a `(4,1,2,16)` array
loads back as `(4,2,16)`). My first `remove_nuisance_from_saved()` draft
assumed the size-1 axis would still be there and indexed it explicitly,
which crashed against both my own test fixture and a real saved dataset.
Fixed to detect and handle both cases.

**Behavior changes**: none to any existing call path (`return_components`
defaults to `False`). Within the new path only: `compile_outputs()`'s
`noise=False` crash fix means that combination is now usable at all
(previously always raised).

**Assumptions resolved**: none new.

**Tests**: 10 new unit tests. Full suite: 36/36 passing. End-to-end
verified against `cows.json`'s real basis set: `return_components=True`
output shapes/values/types (including `result.parameters['naa']
['concentration']` nested access working end to end), the noisy/clean
axis resolution, and the exact `remove_nuisance_from_saved()` accuracy
numbers above (via a real `mainFcns._save()` round trip, not just the
synthetic test fixture).

**Remaining/follow-up**:
- The `presim` type-validation gap noted in Milestone 4 is still open
  (still silently misbehaves on non-`None`, non-tensor input).
- `return_components=True` does not yet support skipping individual
  components' computation to save memory/compute (the handover doc asks
  for this) -- everything it exposes was already being computed by the
  existing pipeline regardless, so this milestone only had to expose it,
  not add new optional compute paths. True skip-to-save-memory support is
  deferred.
- The baseline/residual-water ppm-grid alignment question above is a real
  open item, likely relevant to the section 5 (baseline spline fitting)
  and section 11 (basis-set/grid metadata audit) work.
- `SimulationResult` does not yet carry the `V1_0` legacy-broadening flag
  (still deferred to the relaxation/TE/TR milestone, section 10, as
  originally agreed).

## Not yet started

Handover sections 5 (baseline spline fitting), 6 (CRLB/FIM), 7 (SNR audit/
formalization), 8 (parameter replay across basis sets), 9 (provenance), 10
(relaxation/TE/TR, including the agreed `V1_0` legacy-broadening flag), 11
(basis-set metadata / double-application audit), 12 (NIfTI-MRS export
audit), 13 (broader test-suite expansion beyond what's landed alongside
sections 1-4).

Sections 3 (component outputs) and 4 (nuisance removal) are done for what
the current pipeline already computes (Milestone 5), but per-component
skip-to-save-memory/compute is deferred, and the `presim` validation gap
and baseline/residual-water ppm-grid alignment question (both noted above)
remain open.
