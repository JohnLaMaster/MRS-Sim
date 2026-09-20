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

## Milestone 6 — CRLB / Fisher information (commit `b464dbd`)

**Handover section addressed**: 6 (CRLB/FIM).

**Files changed**: `src/crlb.py` (new), `src/physics_model.py`
(`compute_crlb`/`return_fim` opt-in params on `forward()`, wired through
`_compile_result()`), `src/simulation_result.py` (`crlb`/`fim`/
`crlb_labels` fields), `tests/test_crlb.py` (new, 5 tests).

**Design and documented scope**: `compute_crlb()` builds a self-contained,
differentiable "CRLB forward model" -- a classic MRS-fitting-style
parametric signal model (per-line complex amplitude, Voigt lineshape,
per-line frequency shift, global zero-order phase, plus baseline spline
coefficients as nuisance parameters) -- rather than differentiating
through `PhysicsModel.forward()`'s entire stochastic generative pipeline.
This matches how CRLB is done in the MRS literature (e.g. Cavassila et
al. 2001): B0 field maps, eddy currents, multi-coil combination,
first-order phase, residual water, and resampling/zero-filling are **not**
part of this first model and are explicitly out of scope, documented in
`src/crlb.py`'s module docstring, not silently ignored.

To keep the forward model safe to batch with `torch.func.vmap` (needed for
an efficient batched Jacobian via `torch.func.jacrev`) -- `PhysicsModel`'s
own `lineshape_correction`/`frequency_shift`/etc. branch on tensor rank in
ways that don't reliably survive vmap's tracing -- the Voigt decay and
amplitude-scaling formulas are short, direct copies of the exact lines in
`PhysicsModel.lineshape_voigt`/`modulate` (copied from the source at the
time of writing, not from memory), while frequency shift, phase, and the
FFT reuse `aux.py`'s `complex_exp`/`Fourier_Transform` directly, since
those have no rank-branching and are shape-generic.

**Critical validation (not just "it runs")**: before trusting any CRLB
numbers, verified that this simplified forward model reproduces
`PhysicsModel.forward()`'s actual output under matching settings (no B0/
eddy/coil/resample/magnitude, Voigt lineshape, zero baseline). First
attempt was off by ~4 orders of magnitude; root cause was that the real
pipeline's returned spectrum is *normalized* (divided by peak magnitude)
and the first draft wasn't. After adding the same normalization (copied
from `aux.normalize()`'s formula for a `[2, L]` signal), the two matched to
**0.18% relative error** (max abs diff 0.0016 against a signal of scale
0.90) -- consistent with float32 precision, not a structural discrepancy.

**Fisher information / conditioning**: `compute_crlb()` returns `crlb`
(`[batch, n_params]`, diagonal of the pseudo-inverted FIM) and optionally
`fim` (`[batch, n_params, n_params]`), plus `crlb_labels` naming each
dimension using `ParameterRegistry.metabolite_names` (tying in Milestone
2), satisfying "the parameter registry must identify each CRLB/FIM
dimension". With cows.json's real 28-metabolite/MM basis set (139 total
CRLB parameters: 28 lines x 4 [amplitude, d, g, frequency_shift] + phi0 +
26 spline coefficients), Fisher matrices are, as expected for real
overlapping MRS spectra, often extremely ill-conditioned (observed
condition number ~6e19 for one sample) -- producing near-zero or slightly
negative CRLB entries (floating-point noise around zero, e.g. -1.4e-20)
for poorly-identified parameters, and occasionally large-magnitude
positive/negative values for strongly-correlated pairs (e.g. NAA/NAAG,
whose near-identical chemical shifts make them a textbook
poorly-separable pair in MRS fitting). This is expected/documented
behavior in the CRLB literature, not a bug -- `compute_crlb()`'s docstring
explains how to interpret it and exposes a `rcond` parameter for the
pseudo-inverse rather than silently clamping or hiding it.

**Wiring**: `forward(..., return_components=True, compute_crlb=True,
return_fim=False)` -- both new flags default to `False`, matching the
handover doc's "should be disabled by default for expensive DL training".
`compute_crlb=True` with `noise=False` raises a clear `ValueError` (CRLB
needs a noise covariance, which doesn't exist when noise is disabled)
rather than silently returning nonsense.

**Behavior changes**: none to any existing call path (both flags default
to `False`, and are only consulted inside the already-opt-in
`return_components=True` branch).

**Assumptions resolved**: none new.

**Tests**: 5 new unit tests (`test_crlb.py`) covering the parameter-layout
bookkeeping, the Voigt-decay formula matching its source, forward-model
output shape/finiteness, `vmap`+`jacrev` batching/differentiability, and a
zero-amplitude sanity case. `compute_crlb()` itself (needs a real
PhysicsModel) was verified end to end against `cows.json`'s real basis
set: the 0.18% forward-model validation above, correct output shapes,
FIM symmetry, registry-based labels resolving to real metabolite names,
and the `noise=False` error path.

**Remaining/follow-up**:
- The CRLB model's scope (no B0/eddy/coil/first-order-phase/residual-water/
  resampling) is a real, documented limitation, not a placeholder --
  extending it is future work, likely alongside whichever handover
  sections touch those components more deeply (10, 11, 12).
- No automatic zero-fill-aware masking is wired up yet (the `mask`
  parameter exists and is documented, but nothing currently populates it
  from a real zero-filled config) -- consistent with `zero_fill()` itself
  being separately flagged as broken in the architecture audit.
- CRLB is computed independently per forward() call; no caching/reuse
  across repeated calls with the same basis set (not needed yet at this
  scale, flagged in case it matters for larger basis sets later).

## Milestone 8 — CRITICAL: noisy output was built from the wrong batch sample (commit `f573f9c`)

**Found while working on**: section 8 (parameter replay) -- a
batch_size>1 replay test surfaced this; it is not itself a replay feature.

**The bug**: whenever `noise=True` and `multicoil<=1` (single coil -- the
common case for every config file in this repo), **every sample's "noisy"
spectrum in a batch was silently built from batch sample 0's clean
signal, not its own.** Verified directly against `cows.json`'s real basis
set with `batch_size=5`: for every sample `i != 0`, its noisy output
correlated 0.9999+ with sample 0's clean spectrum and only ~0.92 with its
own. This is about as severe a correctness bug as this codebase could
have -- it corrupts the primary output (the actual training spectrum)
for what is very likely the overwhelmingly common real usage pattern
(batched simulation, single coil, noise enabled).

**Who was actually affected**: unclear how far back this goes or which
existing datasets it touches -- flagged here rather than guessed.
`sim_COWS.py`'s committed version hardcodes `totalEntries = 1` right
before returning from `sample()`, which forces `batch_size=1` for any
COWS-generated dataset regardless of what `cows.json`'s own
`"totalEntries"` field says -- at `batch_size=1` there is no "sample 0
vs sample i" distinction, so COWS datasets generated through the
committed pipeline should not have been affected by this specific bug.
`mrs-sim_template.py`/`deep_learning_dataset_template.py`-style usage
with a real batch size would have been.

**Root cause**: the noise-stacking code
(`fidSum[...,0,:,:].clone().unsqueeze(-3) + noise_vec`, both in the
original code and in the shared `_stack_noisy_clean()` helper this
refactor extracted from it in Milestone 4/commit `e244695`) is only
correct when `signal` already has a transients axis at position -3
(shape `[bS, transients, channels, L]`, produced by `multicoil()` when
`multicoil > 1`) -- the ellipsis then consumes exactly the batch
dimension, and `,0,` picks transient 0 *per sample*. When no transients
axis exists (`multicoil <= 1`, `signal` is `[bS, channels, L]`), the same
indexing pattern has one dimension too few for the ellipsis to skip the
batch axis, so it silently indexed the **batch** axis instead of a
per-sample axis. This predates this refactor entirely; Milestone 4's own
verification of `_stack_noisy_clean()` happened to compare across noise
seeds on the same batch (not sample-to-sample), which is why it didn't
surface this.

**Fix**: `_stack_noisy_clean()` now takes an explicit
`has_transients_axis` argument (`multicoil > 1` in `forward()`) instead
of inferring the presence of a transients axis from `signal`'s shape --
shape-inference under-specified assumption was exactly the cause. When
false, noise is added directly to the sample's own full signal (no
slicing). Output tensor shapes are unchanged in both branches; only which
data feeds the noisy branch's per-sample content changed. The repo owner
explicitly considered and asked about a more sweeping alternative (always
unconditionally add a transients axis, everywhere) -- noted below as a
deliberate follow-up question, not adopted here because it would change
output tensor rank (5D even for single-coil) and break `mainFcns._save()`'s
format, NIfTI export, and `plot_mrs.py`'s shape assumptions; this fix was
kept minimal and shape-preserving instead.

**A related, smaller issue found in the same investigation**:
`SimulationResult.noise` was on a different numerical scale than
`.noisy`/`.noise_free_total` (captured before vs. after the final
`normalize()` call), so `noisy == noise_free_total + noise` didn't hold
even after the main fix, until `noise_vec` was also divided by the same
per-sample normalization `denom` before being returned. Fixed in the same
commit.

**Verification** (against `cows.json`'s real basis set unless noted):
- `batch_size=5`, single coil: every sample's noisy output now correlates
  ~1.0 with its own clean signal.
- `noisy == noise_free_total + noise` holds to ~6e-8 relative precision
  for every sample, including a near-zero-SNR sample that had previously
  shown spurious large errors from unrelated numerical instability
  (dividing by near-zero noise) during debugging -- worth remembering:
  ratio-based checks (`diff / noise`) are unreliable near-zero noise;
  absolute-difference-vs-signal-scale checks are not.
- The `has_transients_axis=True` (multicoil) branch was verified correct
  via direct synthetic-tensor testing rather than end-to-end, because
  `generate_noise()` itself has a **separate, pre-existing, unrelated**
  shape bug for `multicoil > 1` (crashes with a broadcast-shape
  `RuntimeError` in its SNR-per-transient scaling) that blocks exercising
  the real multicoil path at all right now. Documented as a new finding,
  not fixed (out of scope for this fix) -- tracked below.

**Tests**: 3 regression tests in `tests/test_physics_model_bugfixes.py`,
including one that explicitly reproduces the exact contamination pattern
(`batch_size=4`, asserts sample `i`'s noisy branch is never sample 0's
signal) and one covering the multicoil branch via synthetic tensors.
Full suite: 66/66 passing at this point.

**Open follow-up from the repo owner**: whether to go further and
*always* unconditionally add the transients (and possibly other) axes
regardless of whether that component is active, rather than
conditionally -- eliminating this whole class of shape-inference bugs at
the cost of a breaking output-shape change across the pipeline. Not
decided; needs a deliberate, separate discussion given the downstream
impact on `mainFcns._save()`, NIfTI export, and `plot_mrs.py`.

**Also newly found, not fixed**: `generate_noise()` crashes for
`multicoil > 1` (`RuntimeError: output with shape [4, 1, 1] doesn't match
the broadcast shape [4, 4, 1, 1]` in its per-transient SNR scaling,
`lin_snr /= s**0.5`). This means the multicoil path is currently unusable
end-to-end regardless of the bug above -- tracked here for whenever
multicoil support is revisited.

## Milestone 9 — Provenance, MRSsynMRS export, parameter replay (commit `9fd3da8`)

**Handover sections addressed**: 8 (parameter replay) and 9
(reproducibility/provenance).

**Files changed**: `src/provenance.py` (new), `src/mrssynmrs.py` (new),
`src/parameters.py` (`SimulationParameters.save()`/`.load()`),
`tests/test_provenance.py` (new, 7 tests), `tests/test_mrssynmrs.py` (new,
8 tests), `tests/test_parameters.py` (+2 tests).

**Provenance**: `collect_provenance()` gathers git commit, a content hash
of the actual basis functions used (`pm.syn_basis_fids`, not just the
filename -- catches two files sharing a name but differing in content),
acquisition metadata, enabled components, RNG seed, package versions, SNR
definitions, and data-processing state into a `Provenance` record.
**Real finding, not guessed**: `vendor`/`pulse_sequence` are written into
a compiled basis set's `.mat` header by
`process_basis_functions.py`'s `build_header_fields()` (e.g.
`--pulse_sequence 'COWS7_sLASER'`), but `aux.convertdict()` -- which runs
whenever `PhysicsModel` loads that `.mat` file -- unconditionally deletes
`'seq'`/`'vendor'`/`'pulse_sequence'`/`'pulseSequence'` keys. Verified
directly: `pm.header` never has them. This is a genuine basis-set-metadata
gap (relevant to the still-open section 11 audit, not just provenance) --
the information is written at compile time and silently discarded before
`PhysicsModel` ever sees it. Left `None`, not guessed, per the handover
doc's instruction; `vendor` is instead read from the JSON config (e.g.
`cows.json`'s `"vendor"` field) where available. Side note, not chased
further: `cows.json` sets `"vendor": "GE"` while its actual basis-set file
is named `raw_update_COWS7_sLASER_30_Siemens_3000.mat` (Siemens sLASER) --
possibly an inconsistency in that config, possibly intentional; not
investigated.

**MRSsynMRS export**: the handover doc names a specific Google Sheet as
the reporting standard and says to inspect it if accessible, and not to
invent its contents otherwise. **It was accessible from this
environment** -- `export_mrssynmrs_table()`'s field structure (section
names, field names, nesting) is transcribed directly from the actual
sheet, not recalled or invented. Per the repo owner (citing "Synthetic
Data in MR Spectroscopy: Current Practices, Applications, and
Considerations"): this table is meant to be tailored per dataset, not
treated as one fixed schema -- e.g. edited spectra need extra rows
(edited ppm, editing targets) unedited spectra don't. `extra_sections`
merges caller-supplied fields into the table's 'Pulse Sequence' section
for exactly that; nothing here auto-detects editing (MRS-Sim's
difference-editing support is itself flagged as largely unimplemented, so
there's nothing reliable to detect it from yet). Concentration/T2 ranges
are read from the *effective* `pm.min_ranges`/`max_ranges` (reflecting
any config override), not the raw `metabolites_database.json` values,
so the exported ranges match what a given dataset was actually sampled
from. Real scanner/sequence-level fields MRS-Sim does not model at all
(voxel size, water suppression, shimming, RF pulse shapes, patient
population) are left explicitly `None`.

**Parameter replay**: `SimulationParameters.save()`/`.load()` persist a
parameter tensor together with its own registry (index + metabolite
names), so a saved file can be reloaded and passed to any compatible
`PhysicsModel.forward()` without the original `PhysicsModel` instance.
Verified directly: two independently-constructed `PhysicsModel` instances
from the same config, given the identical `params.tensor`, produce
identical output (`nuisance_free` diff exactly 0.0) -- the tensor was
already portable by construction (no back-reference to any specific
model instance); `save()`/`load()` just make persisting it to disk
convenient. A genuinely incompatible layout (different metabolite
list/order) is expected to surface as an ordinary shape-mismatch error
inside `forward()`, not a silent misalignment -- not separately tested
here since it depends on which two basis sets are compared. Noise-
realization replay (the doc's other ask under section 8, "retain/replay
the actual noise realization") is satisfied by `SimulationResult.noise`
from Milestone 5, once its normalization-scale bug (Milestone 8, above)
was fixed: `noisy == noise_free_total + noise` now holds exactly, so
saving `.noise` alongside a result is sufficient for exact noisy-spectrum
reproduction without needing to reproduce `generate_noise()`'s own RNG
state.

**Behavior changes**: none -- all three modules are new, additive
utilities; nothing in the existing `forward()`/`compile_outputs()` path
was touched by this milestone (the noise-scale fix folded into Milestone
8's commit instead, since it was found and fixed together with that
critical bug).

**Assumptions resolved**: the MRSsynMRS table's field structure (now
known, not assumed); the vendor/pulse-sequence provenance gap (now
documented, not assumed away).

**Tests**: 17 new unit tests across the three areas, using lightweight
fake `PhysicsModel`-like objects per the established pattern. Full suite:
68/68 passing. End-to-end verified against `cows.json`'s real basis set
for provenance field population, the exported table's concentration/T2
ranges, and cross-instance parameter replay.

**Remaining/follow-up**:
- `Provenance`/`export_mrssynmrs_table()` are standalone utilities a
  caller invokes explicitly with `pm`/`config`/`sampler` -- they are not
  auto-populated into `SimulationResult.provenance` (which stays a
  cheap, always-on `{'noise_enabled', 'offsets_enabled'}` dict), since
  `forward()` doesn't receive `config`/`sampler` and threading them
  through would go against "keep the normal forward path lightweight".
- `mrs_sim_version` in `Provenance` is always `None` -- this repo has no
  package `__version__` yet.
- Full MRSsynMRS-table auto-population still needs per-dataset human
  input for fields MRS-Sim cannot know (Experiment ID, population/ROI
  description for whatever in-vivo data a copula sampler's parameters
  came from, any editing-specific rows).

## Open design questions raised by the repo owner (not yet acted on)

Three points raised mid-session that affect the sampler/config design
directly, recorded here so they aren't lost before a dedicated design
pass:

1. **Three separate, overlapping ways parameter ranges get defined**
   today: (a) `"parameters"` blocks in a config JSON
   (`set_parameter_constraints()`), (b) defaults baked into a basis set's
   `metabolites_database.json` (loaded automatically, overridden by (a)
   when present), and (c) bypassing ranges entirely by defining parameter
   *distributions* directly in already-quantified numerical space (the
   deep-learning-research convention: `[0, 1]` with `1` = the range's max
   and `0` = actually zero/omitted). The repo owner's own assessment:
   "probably excessive and very certainly redundant." Not simplified in
   this session -- flagged for a deliberate design pass, likely alongside
   generalizing (c) via `CopulaInVivoSampler`/point 2 below.
2. **Config-defined statistical distributions, not just min/max**: the
   repo owner is open to letting config files define per-parameter
   distributions the way `sim_COWS.py`'s copula/`findParamDist.py`
   pipeline already does, rather than being limited to `[min, max]`
   ranges. This would generalize `CopulaInVivoSampler`
   (`src/sampling.py`, Milestone 3) from "fit-to-in-vivo-data" specifically
   into a more general "config declares a distribution per parameter"
   mechanism, and probably also folds in point 1's redundancy cleanup.
3. **Confirmed**: the missing `parameter_distributions_best_fit_recovered.json`/
   `correlation_matrix.mat` files referenced by `cows.json` are on a
   different machine, not in this repo -- the repo owner will provide
   them later. Until then, `CopulaInVivoSampler`'s name-based alignment
   (Milestone 3) remains unverified against real data; the repo owner
   separately confirmed my root-cause guess for the historical magic-
   number reordering is plausible but not confirmed, and independently
   noted (re: `findParamDist.py`) that only one spectral-fitting
   software's export format is currently supported, with more planned.

## Milestone 10 — Two more multicoil/SNR bugs (repo-owner-requested re-verification) + metabolite database defaults (commits `b52f4cd`, `f6f5f97`)

**Context**: the repo owner pushed back on Milestone 8's critical bug
finding -- they had successfully generated large datasets with correctly
varying per-sample SNR, which seemed to contradict "the noisy output is
wrong". Re-verified with an airtight, dead-simple reproduction (two
samples with trivially distinguishable clean signals `1000`/`-1000` and
distinct noise `0.1`/`0.2`, run through the *exact* pre-fix code): sample
1's noisy output came out as `1000.2` (sample 0's clean signal + sample
1's own noise), not `-999.8` (its own). This also gave a precise way to
explain why the two observations don't conflict: noise *magnitude*
generation (`generate_noise()`) was always correct and independent per
sample -- only the *clean signal identity* the noise got added to was
wrong. The repo owner accepted this once demonstrated concretely, and
separately asked me to re-verify the multicoil crash claim too (also
confirmed reproducible, independently, with a fresh run).

**Two more bugs found and fixed while re-verifying, at the repo owner's
request to fix them before moving on**:

1. `generate_noise()` crashed unconditionally for `multicoil > 1`
   (`RuntimeError` in its per-transient SNR scaling, `lin_snr /= s**0.5`).
   Root cause: `zeros.unsqueeze(-1).unsqueeze(-1)` hardcoded two
   unsqueezes assuming `zeros` needed to go from 2-D to 4-D to match
   `param`, but `param` (and `fid`/fidSum at this point in the pipeline)
   is only ever 3-D here -- `multicoil()`'s transients-axis tiling hasn't
   happened yet. Fixed to match `param`'s own dimension-counting
   convention instead of a hardcoded count.
2. Fixing #1 surfaced a **third**, separate, deeper shape mismatch in the
   `pSNR`/`sSNR` (power/spectral SNR *reporting* metric) computation,
   also multicoil-only. **Not fixed** -- the multicoil path (`num_coils >
   1`) still isn't usable end to end. Deferred rather than open-endedly
   chasing what looks like a never-exercised code path; none of this
   repo's configs use `num_coils > 1`.

**Metabolite database defaults** (the repo owner's main ask this
milestone, prompted by the new handover doc section 18 -- see below):
`src/metabolite_database.py` provides validated access to
`metabolites_database.json`'s `Conc`/`T2.metab`/`T2.spins`/`omega`
entries and a `apply_range_overrides()` merge utility, wired into
`PhysicsModel.__init__()` (new `database_overrides` parameter) and
`mainFcns.prepare()` (new optional config field
`metabolite_database_overrides`). `Conc`/`T2.metab` were already used as
sampling-range defaults (confirmed by reading `define_parameter_ranges()`,
not assumed) -- not new. `T2.spins`/`omega` (per-moiety values) were
confirmed completely unused anywhere (zero grep hits) -- this commit makes
them loadable/validated/overridable, **not** wired into actual
simulation, since that requires finishing the separately-unfinished
individual-spin ("separated moieties") basis pipeline (see below).

**Real data-quality finding**: `validate_database()` (checks that
`omega`/`T2.spins.min`/`T2.spins.max` have matching lengths per
metabolite -- moiety ordering is a database contract) found 3 pre-existing
mismatches in the real database: `glc` (`T2.spins.max` has 15 entries vs.
14 for `omega`/`T2.spins.min`), `naa` (`T2.spins.max` has 6 vs. 5 -- the
extra value, `320.17`, exactly matches NAA's `T2.metab.max`, suggesting an
accidental append), and `try` (both `T2.spins.min`/`max` have 8 vs.
`omega`'s 7). Reported to the repo owner; **not** silently corrected
(guessing which array element is spurious would be fabricating data) and
pinned as a regression test (`test_validate_database_against_real_database_finds_known_issues`)
so the database's current, known-imperfect state is explicit rather than
silently passing or failing when it changes.

**Verified end to end** against `cows.json`'s real basis set: overriding
an unconstrained metabolite's (`cho`) concentration range via the new
config field changes `pm.min_ranges`/`max_ranges` as expected; a
metabolite the config's own `"parameters"` block already constrains
(`naa`) correctly keeps that more specific override (existing precedence,
confirmed unaffected -- my first test picked `naa` and initially looked
like the override "didn't work", which turned out to be exactly this
precedence rule, not a bug); omitting the new field entirely reproduces
the exact pre-existing baseline ranges.

**Handover doc section 18** (new, appended by the repo owner mid-session):
a substantially larger physiological-profile/pathology-modeling system --
separating base metabolite data from pathology profiles, multiplicative
concentration modifiers with retained descriptive statistics, integrating
the external Gudmundson MRS database
(https://github.com/agudmundson/mrs-database) via a curated-extraction
pipeline, field-strength metadata/extensibility, profile
inheritance/composition, and extensive provenance requirements. This is
explicitly **not** implemented in this milestone -- the repo owner asked
only for the scoped database-defaults wiring above, and separately asked
to be reminded to come back and finish both this section-18 system and
the individual-spin ("separated moieties") pipeline (see the reminder
below). Note: the doc's section 18 has a stray `print("I would append the
section above to the existing handover.")` line and an unmatched `"""`
right before "## Final requirement" -- flagged to the repo owner as
likely an accidental artifact from however that section was drafted, not
acted on (it's their document to edit).

**Behavior changes**: none to existing configs -- `database_overrides`/
`metabolite_database_overrides` both default to `None`/absent, reproducing
exactly the previous ranges.

**Tests**: 14 new (`test_metabolite_database.py`). No new test for the
`generate_noise()` multicoil fix yet (tracked as a follow-up once the
remaining pSNR/sSNR multicoil bug is also resolved, so multicoil can be
tested end-to-end in one pass, per the previous milestone's note). Full
suite: 82/82 passing.

**REMINDER (explicitly requested by the repo owner)**: come back and
finish (a) the full individual-spin ("separated moieties") simulation
pipeline -- per-spin frequency-shift application using `omega`, per-spin
linewidth sampling using `T2.spins`, fixing the existing unfinished/buggy
basis-loading branch, and extending the parameter registry for per-spin
columns -- and (b) handover doc section 18's physiological-profile/
pathology system in full (Gudmundson data curation, profile inheritance,
concentration multipliers, field-strength extensibility, provenance).
Neither is started beyond the database-access layer above.

## Milestone 11 — Multicoil path fully fixed (commit `d31bebf`)

Per the repo owner: previous datasets have used `num_coils>1`, so the
multicoil path being broken (Milestones 8/10) was a real regression to
fix, not a hypothetical low-priority path. Fixed the third and final
multicoil bug found this session: `pSNR`/`sSNR` (per-metabolite-line power/
spectral SNR reporting values, shape `[bS, num_bF, channels, 1]`, no
transients axis) were divided by `noise_vec.std(...).unsqueeze(1)`
unconditionally -- correct only when `noise_vec` has exactly one fewer
dimension than `pSNR`/`sSNR` (single-coil case). For multicoil,
`noise_vec` gains its own transients axis (same ndim as `pSNR`/`sSNR`), so
the per-metabolite axis collided with the transients axis during
broadcasting. Fixed by explicitly inserting a transients axis into
`pSNR`/`sSNR` only when `noise_vec` actually has one
(`has_transients_axis = noise_vec.ndim > 3`).

**Multicoil (`num_coils>1`) now works end to end** -- verified against
`cows.json` with `num_coils=3`: `forward()` completes successfully,
`noisy.shape` is `[batch, transients, channels, length]` as expected, and
critically, re-running the batch-contamination check from Milestone 8
confirms **no regression**: every sample's noisy output still correlates
1.000000 with its own clean signal in the multicoil case. Single-coil
path re-verified unaffected (identical realized-SNR shapes/values).

**Tests**: none added yet for this specific fix (time-constrained
session -- the repo owner needed to disconnect from the internet
shortly). Tracked as a follow-up: add a synthetic-tensor regression test
for the pSNR/sSNR multicoil shape handling, mirroring the pattern used
for `_stack_noisy_clean`'s `has_transients_axis` tests. Full suite:
82/82 passing (no new tests, but no regressions).

**Remaining known issue**: none identified in the multicoil path at this
point -- all three bugs found this session in it are now fixed. Worth a
broader/more thorough multicoil test pass later (edge cases: different
transient counts, combined with baseline/residual water, etc.) since this
path had apparently gone untested for a while.

## Milestone 12 — pSNR/sSNR regression tests + V1_0 flag (commits `5a5a0c2`, `1f1cdae`)

Quick session (repo owner had ~20 minutes before disconnecting).

**Tests for Milestone 11**: extracted the pSNR-scaling fix into a static,
directly-testable `PhysicsModel._scale_snr_reference()`. 2 new tests
(single-coil pinning existing behavior; multicoil with distinct
per-transient noise values, confirming correct per-transient division and
regression-testing the exact crash). Re-verified end to end against
`cows.json` with `num_coils=3` after the refactor.

**`V1_0` flag** (handover section 10 prep, agreed with the repo owner
early in this refactor, implemented now): `PhysicsModel.initialize(...,
V1_0=True)` gates the confirmed double-broadening bug
(architecture_v1_audit.md sections 6/11) -- default `True` preserves it
exactly for backward compatibility; `V1_0=False` omits it, with nothing
yet added in its place (the actual relaxation/TE/TR modeling that would
replace it is separate, unimplemented work). Wired into
`mainFcns.prepare()` via an optional config field, defaulting to `True`
via `getattr` when absent. Verified: `V1_0=True`/`False` produce
genuinely different basis FIDs (max abs diff ~14) against `cows.json`'s
real basis set; existing configs (no `V1_0` field) unaffected.

**New, minor, unrelated finding (not fixed)**: `order_metab()` mutates its
input metabolite list in place (`list.pop()`). Only matters if constructing
multiple `PhysicsModel`s from one shared list object directly --
`mainFcns.prepare()` is unaffected (loads a fresh list from JSON each
call).

Full suite: 84/84 passing. Committed to a clean state before the repo
owner disconnected (push is theirs to do, per their standing preference).

## Milestone 13 — Relaxation equations, a real 'd'/T2 units bug, T2* wiring (commits `84a981d`, `2c9486b`, `f6017fe`)

**Handover section addressed**: 10 (relaxation and acquisition parameters).

**Finding the actual equations**: the handover doc pointed at "Table 1" of
"Synthetic Data in MR Spectroscopy: Current Practices, Applications, and
Considerations" for the T1/T1*/T2/T2* equations, with an explicit "do not
rely on memory" instruction. Found the paper on arXiv (2602.23463) --
confirmed the repo owner is its first author. Fetched and read the full
100-page PDF directly (not just the abstract): its metadata states
**"Table Count: 0 (All 6 tables are in the supplement)"**, and the main-text
section that discusses relaxation (2.3.2) is a narrative review of how
*other* existing simulators handle T1/T2, not a specification. A full-text
search of all 100 pages found no table literally named "Table 1" -- only
Supplementary Tables S1-S4, none of which (based on surrounding context)
appear to be about relaxation equations. Reported this discrepancy to the
repo owner rather than guessing at a table I couldn't verify existed in
the form described; they provided Table 1's content directly (their own
paper's supplement, transcribed from memory during our conversation).

**The four equations** (`src/relaxation.py`, commit `84a981d`):
```
T1:  M0 * (1 - exp(-TR/T1))
T1*: M0 * (1 - exp(-TR/T1)) / (1 - cos(theta)*exp(-TR/T1)) * sin(theta)
     (Kaptein et al. 1976, Taylor et al. 2016 -- SPGR/Ernst steady-state)
T2:  M0 * exp(-TE/T2)
T2*: M0 * exp(-TE/T2*)
```
Pure, stateless, torch/scalar-compatible functions. 16 tests including
physical sanity checks (T1* at 90 degrees reduces exactly to plain T1
recovery; 0-degree flip angle gives 0 signal; TE/TR limiting behavior at
0 and infinity).

**A real, severe, independently-found bug** (commit `2c9486b`): while
assessing whether wiring in T2*-based amplitude scaling would double-count
with the existing sampled 'd' (Lorentzian decay rate) parameter, found
that `define_parameter_ranges()` sets 'd''s default sampling range
**directly from `T2.metab` values in milliseconds** (e.g. NAA:
242.7-320.17), used unconverted as a per-second decay rate in
`exp(-d * self.t)` -- but `self.t` is confirmed in **seconds** (range
matches `Ns/spectralwidth`, ~0.68s for `cows.json`). This made every
sampled 'd' from the default range ~1000x too large: `exp(-250*0.68) =
exp(-170)`, decaying every metabolite to numerically zero within
microseconds instead of producing a normal, visible lineshape. Only
affects the plain/uniform default sampler path
(`PhysicsModel.quantify_params()` / `UniformRangeSampler`) --
`sim_COWS.py`'s copula sampler sets 'd' directly from real fitted in-vivo
values, bypassing this broken range entirely, which is presumably why it
had gone unnoticed. Fixed by converting the T2 millisecond range into a
proper decay-rate range (`rate = 1000 / T2_ms`). Verified: NAA's 'd' range
is now `[3.12, 4.12]` (1/s) instead of `[242.7, 320.17]`, giving decay
factors of 6-12% at the end of the readout (a normal lineshape) instead
of ~0; a full `forward()` pass with `UniformRangeSampler` now produces a
real, nonzero spectrum.

**T2* wiring** (commit `f6017fe`): resolved the double-counting question
by recognizing the two effects are genuinely distinct -- the existing 'd'
(now correctly unitted) governs decay *during* the readout (t >= TE,
shaping linewidth), while the paper's T2/T2* equation governs signal lost
*before* TE (an amplitude bias the pipeline never modeled at all: basis
FIDs implicitly assume t=0 is the echo itself). Wired as
`amp *= exp(-TE * d)` (equivalently `t2_star_decay(TE, T2_star=1/d)`,
reusing the already-sampled 'd' rather than adding a new parameter),
applied to `params[:,ind['metabolites']]` before `modulate()`, gated by
`V1_0=False` -- `V1_0=True` (default) is completely unaffected.

**Another inconsistency found along the way, used but not separately
fixed**: `self.TE` (used for the T2* scaling) is a registered buffer
aliasing the basis set's own `header['TE']`, **not**
`PhysicsModel.__init__()`'s `TE` constructor argument, which is never
actually stored as an instance attribute at all. Confirmed directly for
`cows.json`: `config.TE = 26` but `pm.TE = 30` (the basis set's own
baked-in value). The config's TE is silently unused. Used the basis set's
own TE for the new scaling (the physically authoritative choice -- it's
the echo time the basis functions were actually simulated at), but the
`config.TE` field being dead weight is a separate, pre-existing issue not
fixed here.

**Verified end to end** against `cows.json`'s real basis set: `V1_0=True`
output is byte-identical to before this milestone; `V1_0=False` produces
a genuinely different spectrum (~11% relative difference -- real
per-metabolite reweighting, not just a global scale change that
normalization would hide); the exact scaling factor for a specific
metabolite (NAA: d=4.11/s, TE=30ms -> 0.884) matches the formula by hand
calculation.

**Behavior changes**: none for `V1_0=True` (default). `V1_0=False`'s
output changes meaningfully (as intended -- that's the corrected-physics
opt-in path). The 'd' units fix changes `UniformRangeSampler`-based
sampling's default 'd' range regardless of `V1_0` (this is an unambiguous
bug fix, not gated -- the old range was never usable for anything, always
decaying metabolites to zero).

**Tests**: 16 new (`test_relaxation.py`). No new test added for the T2*
wiring itself or the 'd' units fix (both verified manually against the
real basis set; both need a real `PhysicsModel` to exercise meaningfully,
consistent with the established pattern for basis-set-dependent changes).
Full suite: 100/100 passing throughout.

**Remaining/follow-up (section 10)**:
- T1/T1* wiring is not done: needs genuinely new config surface (TR,
  flip angle) and T1 database values that don't exist anywhere yet (no
  literature source available to populate them without fabricating data,
  per the "do not fabricate" principle carried over from section 18's
  discussion).
- `config.TE` being silently unused/inconsistent with the basis set's own
  TE is a separate, pre-existing issue, noted but not fixed.
- The paper's supplementary tables (S1-S4) and the rest of Table 1's
  context remain otherwise unverified/inaccessible from this environment
  -- only the specific equations the repo owner provided directly were
  used.

## Not yet started

Handover sections 7 (SNR audit/formalization), 8 (parameter replay across
10 (relaxation/TE/TR, including the agreed `V1_0` legacy-broadening flag),
11 (basis-set metadata / double-application audit), 12 (NIfTI-MRS export
audit), 13 (broader test-suite expansion beyond what's landed alongside
sections 1-9).

Sections 3 (component outputs) and 4 (nuisance removal) are done for what
the current pipeline already computes (Milestone 5), but per-component
skip-to-save-memory/compute is deferred, and the `presim` validation gap
and baseline/residual-water ppm-grid alignment question (both noted above)
remain open.

Sections 5 (baseline spline fitting) and 6 (CRLB/FIM) are done within the
documented scope noted above -- CRLB does not yet model B0/eddy currents/
multi-coil/first-order phase/residual water/resampling.

Sections 8 (parameter replay) and 9 (provenance) are done for what's
listed in Milestone 9 above; full MRSsynMRS auto-population still needs
per-dataset human input for fields MRS-Sim cannot know, and provenance is
a standalone opt-in utility rather than wired into SimulationResult.

**Update**: the multicoil (`num_coils>1`) path noted above as unusable is
now fully fixed as of Milestone 11 (commit `d31bebf`) -- see that
milestone for details. The repo owner's open question about
unconditionally adding transients (and other) axes everywhere, rather
than conditionally, remains unresolved/undecided (Milestone 8). Three
open design questions about parameter-range/distribution definition
(config `"parameters"` blocks vs. basis-set defaults vs. direct numerical-
space distributions, and generalizing config-declared distributions
beyond `CopulaInVivoSampler`) are recorded above, raised by the repo owner
but not yet acted on.

**Section 10 (relaxation/TE/TR)** is partially done as of Milestone 13:
the T1/T1*/T2/T2* equations are implemented and tested, the `V1_0` flag
exists, and T2* amplitude scaling is wired in behind `V1_0=False`. T1/T1*
wiring, TR/flip-angle config surface, and T1 database values remain
outstanding -- see Milestone 13's "Remaining/follow-up" for specifics.
