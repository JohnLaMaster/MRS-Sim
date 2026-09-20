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

**Follow-up guard added same milestone (commit `fda60f3`)**: the repo
owner flagged that their existing `b0=True` path
(`B0_inhomogeneities()`/`add_inhomogeneities()`) already explicitly
simulates spatial B0 field variation across the voxel and multiplies the
FID by the resulting intra-voxel dephasing kernel -- confirmed this is
exactly the physical mechanism that converts intrinsic T2 into apparent
T2* (inhomogeneous broadening from field variation), i.e. the same effect
the new `V1_0=False` T2* amplitude term also models, just as a fixed
scalar instead of an explicit spatial simulation. `forward()` now raises
a clear `ValueError` if `b0=True` and `V1_0=False` are both requested,
rather than silently double-counting the T2->T2* conversion. Verified
against `cows.json`: the error fires for the conflicting combination;
`b0=False`+`V1_0=False` and `b0=True`+`V1_0=True` (default) both continue
to work unchanged.

**Related, NOT addressed here (flagged for the section 11 audit)**: per
arXiv:2602.23463's own description, the Voigt lineshape's Gaussian
component ('g' in MRS-Sim) is *also* meant to represent inhomogeneous
broadening from intra-voxel field variation -- the same physical effect
`B0_inhomogeneities()` explicitly simulates spatially, just modeled as a
simple Gaussian statistical assumption instead. Whether sampling 'g' and
enabling `b0=True` together double-counts this in the same way `b0` and
the new T2* term do was not investigated this session -- noted here so it
isn't lost.

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

## Milestone 14 — NIfTI-MRS export was completely broken (commit `1db69da`)

**Handover section addressed**: 12 (NIfTI-MRS export audit).

**Context**: continuing to the next suggested-order item after section
10. Rather than only re-reading the code (already partly audited in
Milestone 1), actually ran `Mat2NIfTI_MRS.forward()` end to end against a
real simulated dataset for the first time this session -- and it crashed
immediately, unconditionally, for every dataset.

**Bug 1 -- `KeyError: 'dwelltime'`**: `write_NIfTIMRS()` reads
`header['dwelltime']` directly, but a simulated dataset's saved header
(`PhysicsModel.header`, loaded via `aux.convertdict()`) never has that
key -- confirmed directly (`spectralwidth, carrier_frequency, Ns, t,
centerFreq, B0, TE, basis_set_software, ppm` only). This crashed for
*every* dataset, `noise=True` or `False`. Fixed by deriving it from
`spectralwidth` (`dwelltime = 1/spectralwidth`) when absent.

**Bug 2 -- axis misalignment for `noise=False` datasets, found via a
false start worth recording**: fixing bug 1 revealed a second crash
(`IndexError`) specifically for `noise=False` datasets, in the
real/imaginary-combining line
(`specDataCmplx[...,0,:] + 1j*specDataCmplx[...,1,:]`). My first
hypothesis -- that this line's hardcoded `0`/`1` was itself wrong, meant
to select the noisy/clean axis rather than real/imaginary -- was
**incorrect, and the repo owner caught it immediately** ("no no no...
That hard coded 2 should be for real/imaginary!"). Re-investigated and
found the actual root cause one level up: `aux.loadmat_as_dict()` calls
`scipy.io.loadmat(..., squeeze_me=True)`, which silently removes the
noisy/clean axis entirely whenever it has size 1 (`noise=False`).
Confirmed directly: a saved `spectra` shape `(2, 1, 2, 2048)` loads back
as `(2, 2, 2048)`. Every index in `Mat2NIfTI_MRS.forward()` assumes axis 1
is noisy/clean, so this silently shifted every later axis -- corrupting
the label-based branch-selection logic first, then crashing the
real/imaginary line (which was correct all along; it was just operating
on a misaligned array by the time it ran). Fixed by explicitly restoring
the squeezed-away axis (`np.expand_dims` when the loaded array comes back
3-D) rather than changing `loadmat_as_dict()`'s shared default behavior,
since other callers may depend on its current squeezing.

**Bug 3 -- `RepetitionTime` spec violation**: confirmed against the actual
NIfTI-MRS specification (wtclarke/mrs_nifti_standard, fetched directly)
that `RepetitionTime` must be a *number* (seconds) when present, with
`null` explicitly permitted for an unknown value -- the existing code used
the *string* `'NA'`, a genuine type violation for any strict consumer. TR
is not tracked anywhere in this codebase, so `None` (`-> null`) is the
honest fix, not a fabricated numeric value.

**Verified end to end** against a real simulated dataset (`cows.json`),
for both `noise=True` and `noise=False`: export now completes
successfully, writes a valid NIfTI-MRS file with `pixdim[4]` (dwelltime)
matching `1/spectralwidth`, and the JSON header extension has
`EchoTime=0.03` (correct) and `RepetitionTime=null` (correct, spec
-compliant).

**A fourth issue found, not fixed**: `test_output()`'s stricter
self-consistency check (`test_nifti_mrs_conventions`) hardcodes an assumed
dominant-peak location of 4.65 ppm regardless of what a given simulation
actually contains -- its own inline comment already flags this ("set to
your dominant simulated peak"). Not a crash in the core export path (only
triggers when `test_output=True`); noted as a follow-up rather than fixed,
since properly fixing it means threading the actual dominant/reference
metabolite through the call chain, a design decision rather than a quick
fix.

**Behavior changes**: NIfTI-MRS export goes from "always crashes" to
"works" -- this is by definition a large behavior change, but since it
never previously completed successfully for any dataset, no existing
output could have depended on the old (crashing) behavior.

**Tests**: 4 new (`tests/test_nifti_export.py`), using a synthetic `.mat`
fixture matching `mainFcns._save()`'s schema (no real basis set needed) --
covering both the noisy/clean-axis-present and axis-squeezed-away cases,
the dwelltime derivation, and the `RepetitionTime` null check. Full suite:
104/104 passing.

**Remaining/follow-up**:
- The hardcoded 4.65 ppm dominant-peak assumption in
  `test_output()`/`test_nifti_mrs_conventions` (issue 4 above).
- The `>>1` vs `>1` bit-shift-instead-of-comparison pattern noted earlier
  in `physics_model.py` also appears in `mat2niftimrs.py`
  (`specDataCmplx.shape[1]>>1`) -- harmless by coincidence for the shape
  values actually encountered (confirmed same reasoning as before: `x>>1`
  and `x>1` agree for all non-negative integers), not fixed, purely a
  readability concern.
- No test yet for the `"noise_free"`/`"filtered"` `label`-based branch
  selection in `forward()` (lines 51-57) -- `"filtered"` in particular
  indexes a third branch (index 2) that only exists when the separate,
  little-used `snr_filter` feature was active during simulation; not
  exercised this session.

## Milestone 15 — NIfTI-MRS frequency-sign convention, T1/T1* config surface, T1 database placeholders (commit TBD)

**Handover section addressed**: 12 (NIfTI-MRS export audit, continued) and
10 (relaxation/TE/TR, continued).

**Context**: repo owner asked four things in one message: (1) how do we
confirm MRS-Sim's exported data follows the NIfTI-MRS left-right/direction
convention, (2) add a config sub-dictionary for T1/T1* parameters (opt-in,
not activated -- no literature T1 data exists yet), (3) add schema space
for T1 in `metabolites_database.json` for the repo owner to fill in later,
(4) make the hardcoded 4.65ppm water-reference peak in
`mat2niftimrs.py`'s `test_output()` configurable.

**Finding 1 -- the NIfTI-MRS frequency-sign convention was violated,
confirmed empirically against real basis sets, not just derived**:
fetched the NIfTI-MRS spec text directly (wtclarke/mrs_nifti_standard).
Two separate conventions are stated: (a) time-domain data must be stored
in order of increasing time; (b) the frequency axis follows the Levitt
convention (`ω=-γB0`), under which more-deshielded (higher-ppm) ¹H
resonances land on the left/low-index side once Fourier-transformed by a
compliant reader.

Checked (a) first: `self.t` is built ascending from 0 in
`process_basis_functions.py` and never reordered anywhere in
`physics_model.py` -- compliant.

Checked (b) empirically using `src/basis_sets/PRESS_30_GE_2000.mat` and
`VERI_PRESS_30ms_GE_2000_wMM.mat` (both real basis sets, same result):
took NAA's raw stored FID exactly as `mat2niftimrs.py` would export it
(confirmed `NIfTIMRS: true` is only ever paired with `fids: true` across
every config in the repo, i.e. genuine time-domain data, not an
already-Fourier-transformed spectrum mislabeled as one), ran it through
the spec's own `S(f) = fftshift(fft(fid))` + `ppm = -f_Hz/f0 + ref`
formula, and found the dominant NAA singlet at **7.29 ppm instead of the
correct 2.0 ppm**. Root cause: MRS-Sim's own ppm axis is built as
`ppm = +f_Hz/sf + centerFreq` (`process_basis_functions.py`) -- the
*mirror image* of the spec's `ppm = -f_Hz/f0 + ref`. This has always been
true of MRS-Sim's internal convention (compensated for only at *display*
time, via `plot_mrs.py`'s `invert_xaxis()`), but was never corrected for
NIfTI-MRS export, since that export path didn't produce valid output at
all before Milestone 14.

**A wrong fix proposed and corrected by the repo owner**: first proposed
conjugating the FID at export time. The repo owner rejected this
("Absolutely not... Do not change the phase or order of the real/imag
components") having read it (incorrectly, but understandably from how it
was described) as discarding the imaginary component, and proposed a
plain left-right flip instead. Tested the flip empirically, two ways
(direct time-domain array reversal, and FFT→reverse→inverse-FFT) against
the same real NAA data: **both do fix the ppm assignment but provably
reverse the FID's decay direction** (amplitude goes from
`~173` at the start / `~0.0008` at the end to the reverse) -- a textbook
Fourier-pair fact (frequency-domain reversal without conjugation is
mathematically equivalent to true time-domain reversal), not a coding
error, but a genuine conflict between the spec's two stated conventions
under a reordering-only fix. Presented this concrete trade-off (with the
numbers) back to the repo owner, who then approved conjugation with the
misunderstanding cleared up (it negates each sample's phase but keeps the
data fully complex; it does not discard the imaginary channel).

**Fix applied**: `Mat2NIfTI_MRS.forward()` conjugates the combined
complex array (`specDataCmplx = specDataCmplx.conj()`) immediately after
the real/imaginary combining step, at NIfTI-MRS export time only --
`physics_model.py`'s and `plot_mrs.py`'s own conventions are untouched.
Verified: `|conj(fid)|` is identical to `|fid|` at every timepoint, so the
"increasing time" convention is unaffected; a spec-compliant reader now
places a known synthetic resonance at its correct ppm (verified to
within 0.05 ppm via `torch.fft`/`np.fft` round-trip, both on real NAA data
and in the new committed regression test).

**A related bug found while testing, not fixed**: constructing a test
dataset with `batch=1` *and* `noisy/clean axis=1` together revealed that
`scipy.io.loadmat(..., squeeze_me=True)` squeezes *both* size-1 axes away
at once in that case, leaving a 2-D array that `mat2niftimrs.py`'s
existing `if specDataCmplx.ndim==3` restoration (added in Milestone 14)
doesn't cover -- it only restores a *single* squeezed axis. This
silently corrupts the data far worse than the Milestone-14 bug (the
length axis itself gets mistaken for the noisy/clean axis and mostly
discarded). Not exercised in the new test (sidestepped with `batch=2`
instead, matching the existing tests' pattern) and not fixed --
recorded here as a real edge case (single-sample-batch NIfTI-MRS export)
for a future pass.

**Task 2 -- T1/T1* config surface**: not yet started at time of writing
this entry (see "Remaining/follow-up" below for the plan).

**Task 3 -- T1 database placeholders**: not yet started.

**Task 4 -- configurable 4.65ppm reference**: done. `Mat2NIfTI_MRS.__init__`
gained `ppm_reference: float=4.65` and `expected_peak_ppm: float=None`
(defaults to `ppm_reference` when unset), both threaded through to
`test_nifti_mrs_conventions()` in `test_output()` in place of the two
hardcoded `4.65` literals. Defaults preserve prior behavior exactly.

**Tests**: 2 new in `tests/test_nifti_export.py` --
`test_export_corrects_nifti_mrs_frequency_sign_convention` (synthetic
single-tone FID at a known offset frequency, checked against the spec's
own formula independently of MRS-Sim's convention, so it can't trivially
agree with whatever MRS-Sim itself writes) and
`test_export_uses_configurable_reference_peak_not_hardcoded_4_65ppm`
(would fail under the old hardcoded 4.65 for a non-water-referenced
peak). Full suite: 106/106 passing.

**Follow-up, same session -- the noise-free/clean counterpart was
silently dropped from NIfTI-MRS export entirely**: the repo owner asked
how the noisy vs. noise-free variants are exported, given NIfTI-MRS is
meant to hold one set of spectra. Checked the actual call sites (not just
`Mat2NIfTI_MRS.forward()` in isolation): `mainFcns.simulate()` calls
`save2nifti.forward(datapath=new_path)` twice (lines 195, 213 before this
fix), *always* with no `label` argument. Inside `forward()`, the
`label=None` branch always selects index 0 (noisy) --
`label='noise_free'` (which would select index 1, the clean counterpart)
was never invoked anywhere in the codebase. Worse, that unreachable
branch was also broken: `save_name = os.path.join(save_name, label)`
builds a *subdirectory* path (e.g. `dataset_spectra_0/noise_free`), and
nothing creates that subdirectory, so `nib.save()` would raise
`FileNotFoundError` the first time it actually ran.

The repo owner's direction: export the clean version as an independent,
opt-in *request* (not tied to whether noise happened to be simulated --
"not all simulations will need the clean version"), and give it an
appended name pairing it with its corresponding data file, not a
subdirectory.

**Fix**: (1) `mat2niftimrs.py`'s `save_name` join changed from
`os.path.join(save_name, label)` to `f"{save_name}_{label}"` -- a sibling
filename suffix (`dataset_spectra_0_noise_free.nii.gz`) instead of a
non-existent subdirectory. (2) `mainFcns.simulate()` gained a second,
config-gated export call at both save sites:
`if getattr(config, 'NIfTIMRS_noise_free', False): save2nifti.forward(datapath=new_path, label='noise_free')`
-- `getattr` with a `False` default so existing configs that predate this
key are unaffected (matches the `V1_0`/`metabolite_database_overrides`
pattern). Not added to any existing config file, including `kelley.json`
(the only config with `NIfTIMRS: true`) -- opt-in per the repo owner's
"not all simulations will need it".

**Tests**: 1 new in `tests/test_nifti_export.py`
(`test_export_noise_free_label_writes_a_sibling_file_not_a_subdirectory`,
confirms both the sibling-file naming and that index 1, not index 0, is
what gets written) and a new file,
`tests/test_mainFcns_nifti_export_wiring.py` (3 tests, using a fake
`PhysicsModel` -- no real basis set needed -- to exercise
`mainFcns.simulate()`'s actual control flow: the flag off by default, on
when requested, and off when the config key is absent entirely). Full
suite: 110/110 passing.

## Milestone 16 — T1/T1* config surface + database schema placeholders (commit TBD)

**Handover section addressed**: 10 (relaxation/TE/TR, continued -- T1/T1*
wiring and database values were explicitly left outstanding at the end of
Milestone 13).

**Context**: repo owner asked for a config sub-dictionary to specify
TR/flip-angle for T1/T1* recovery, and schema "space" in
`metabolites_database.json` for T1 values -- explicitly opt-in and
**not activated**, since no real T1 literature/fitted data exists yet.

**Database placeholders**: every one of the 55 metabolites in
`metabolites_database.json` gained a `"T1"` block mirroring `"T2"`'s shape
exactly (`{"spins": {"min": ..., "max": ...}, "metab": {"min": ...,
"max": ...}}`), with every value JSON `null` rather than a fabricated
number. Applied via a scripted regex substitution (verified against the
full parsed JSON before writing: exactly 55 matches, every other field
byte-for-byte unchanged) rather than a full `json.dump` reformat, to keep
the diff to one added line per metabolite -- the file's existing
hand-aligned compact-array formatting is otherwise preserved untouched.

**`get_t1_range()`** added to `src/metabolite_database.py`, mirroring
`get_t2_range()`'s `level='metab'|'spins'` contract. Unlike `get_t2_range`,
it must also treat a *present* block with `min`/`max: null` as "not
available" (not just an absent key) -- raises `MoietyRangeError` either
way, so nothing downstream can silently treat a placeholder as real data.
`apply_range_overrides()` needed no changes -- it already deep-merges
generically by key, so `metabolite_database_overrides` can populate `T1`
today, ahead of any dedicated support.

**Config surface + wiring, deliberately scoped smaller than T2/`'d'`**:
rather than mirroring `'d'`/`'dmm'` as a fully per-sample-sampled
parameter (which would mean extending `initialize()`'s hand-maintained,
positionally-coupled `header`/`self.index` construction -- the same
bookkeeping that produced the 'temperature' column bug and the original
`'d'`/T2-units bug), T1 is treated as a per-metabolite **constant**
(the database range's midpoint, once populated) combined with a
**global** TR/flip-angle from config -- matching how the repo owner
described the request (TR/flip-angle as config-level parameters, the same
way `TE` already is) rather than something needing per-instance
statistical variation the way linewidth does. Noted as a scoping choice,
not a limitation discovered after the fact: per-sample T1 heterogeneity
remains a possible future extension once real distribution data exists.

- `metabolites_database.json` / `metabolite_database.py`: as above.
- `PhysicsModel.initialize()` gained `t1_cfg: dict=None`
  (`{'enabled': bool, 'TR': <ms>, 'flip_angle': <degrees, optional>}`).
  Parsing/validation extracted into a static, instance-free method,
  `PhysicsModel._resolve_t1_config(t1_cfg, metab_names, ranges)`, mirroring
  the existing `_stack_noisy_clean`/`_scale_snr_reference` pattern of
  pulling logic out of the hard-to-unit-test main class specifically so it
  can be tested without a real basis-set file. Raises `ValueError` if
  `enabled` without `TR`; raises (propagates) `MoietyRangeError` via
  `get_t1_range()` for any metabolite lacking real T1 data -- which, with
  the shipped database, is unconditionally every metabolite right now, so
  `t1_cfg['enabled']=True` fails loudly and immediately rather than
  quietly doing nothing or using a fabricated number. Verified directly
  against a real basis set (`PRESS_30_GE_2000.mat`): default (`t1_cfg`
  absent) leaves `t1_enabled=False`/`TR=None`/`flip_angle=None`
  unaffected; `t1_cfg={'enabled': True, 'TR': 2000.0}` raises
  `MoietyRangeError` immediately, naming the first metabolite that lacks
  data.
- `PhysicsModel.forward()`: amplitude scaling extracted into
  `PhysicsModel._apply_t1_scaling(amp, t1_values_ms, TR_ms,
  flip_angle=None)` (same testability rationale) -- applies plain
  `t1_recovery` by default or `t1_star_recovery` (Ernst equation) when
  `flip_angle` is given, gated by `self.t1_enabled`, applied right after
  the existing `V1_0=False` T2* amplitude scaling and before
  `self.modulate()`.
- `mainFcns.prepare()`: `t1_cfg=getattr(config, 't1_cfg', None)` threaded
  through to `pm.initialize()` -- absent from every existing config file
  (including `kelley.json`), so no existing behavior changes.

**Tests**: 12 new in `tests/test_physics_model_bugfixes.py` (config
parsing/validation and amplitude-scaling math, entirely through the two
static methods -- no real basis set needed) and 6 new in
`tests/test_metabolite_database.py` (`get_t1_range()`'s success/error
paths, plus a regression test pinning the real database's T1-placeholder
shape for every metabolite). Full suite: 123/123 passing.

## Milestone 17 — CRLB: configurable include/exclude parameters + first-order phase (commit TBD)

**Handover section addressed**: 6 (CRLB/FIM), follow-up.

**Context**: repo owner pointed out that CRLB should let the caller
specify which of the standard model-fitting parameters are actually
estimated (their example: zero- and first-order phase, both commonly
either estimated jointly or fixed in real MRS fitting software) rather
than hardcoding a fixed set -- "it's better to leave it up to the user."
First-order phase (`phi1`) wasn't modeled by `src/crlb.py` at all before
this, unlike `phi0`, so this also required adding it to the CRLB forward
model.

**Semantics chosen**: excluding a parameter family does not turn its
physical effect off -- it fixes it at its actual sampled/fitted value
instead (the standard "nuisance parameter known exactly" CRLB variant,
matching how real fitting software lets phi1 be estimated jointly or held
fixed). Verified directly (`tests/test_crlb.py::
test_fixed_phi0_still_affects_signal_but_not_differentiated`): a fixed,
nonzero phi0 still changes the observed signal, but the Jacobian/FIM/CRLB
only ever has a row/column for parameters actually being estimated.

**`_crlb_signal_model`** now takes `(theta_est, theta_fixed, ...,
layout_est, layout_fixed)` instead of one `theta`/`layout` -- every
parameter family in the new `ALL_CRLB_PARAMS = ('amp', 'd', 'g', 'fshift',
'phi0', 'phi1', 'beta')` is always physically applied, sourced from
whichever of the two theta tensors its `layout_*` entry says. `jacrev(...,
argnums=0)` then only differentiates `theta_est`.  `compute_crlb()` gained
`include_params`/`exclude_params` (default `DEFAULT_CRLB_PARAMS` = the
same 6 families as before, `phi1` excluded, so default behavior/output
shape is unchanged); both are validated (`ValueError` for an unknown name
or an empty resulting include set) and normalized to `ALL_CRLB_PARAMS`'s
canonical order regardless of input order. `PhysicsModel.forward()`/
`_compile_result()` gained matching `crlb_include_params`/
`crlb_exclude_params` passthrough parameters.

**First-order phase added to the CRLB model**: applied directly to the
already-FFT'd spectrum (`complex_exp(spectrum, -phi1_ref * phi1_rad)`)
rather than round-tripping through an extra IFFT/FFT the way
`PhysicsModel.first_order_phase()` does (needed there only because it's
called on, and must return, a time-domain fid) -- verified numerically
equivalent to floating-point precision (max abs diff 4.9e-7) against
`PhysicsModel.first_order_phase()`'s actual formula on a synthetic signal,
not just assumed from reading the code.

**Verified end to end** against `cows.json`'s real basis set: default
call still produces exactly 139 CRLB parameters (unchanged from Milestone
6's documented count -- confirms this refactor didn't silently change
default behavior); `crlb_exclude_params=['phi0']` gives 138 (phi0's label
absent); `crlb_include_params=ALL_CRLB_PARAMS` (adding phi1) gives 140
with `'phi1'` present; excluding every family raises `ValueError` instead
of silently returning an empty/degenerate result; an unknown parameter
name raises `ValueError` naming it.

**Tests**: `tests/test_crlb.py` rewritten for the new
`theta_est`/`theta_fixed` signatures (7 tests, including the new
fixed-vs-estimated behavior test). Full suite: 125/125 passing.

## Milestone 18 — Section 7 audit: SNR target vs. realized (commit TBD)

**Handover section addressed**: 7 (SNR: target vs. realized/measured).

**Method**: delegated a read-only investigation (exact code quotes, no
changes) across `physics_model.py`'s full SNR pipeline
(sampling -> `generate_noise()` -> `pSNR`/`sSNR` -> `SimulationResult`),
then verified its concrete claims directly before acting on any of them,
per standing practice for this repo.

**Findings**:
- **No demonstrable bug** in the multi-transient mechanism the handover
  doc asks to preserve: average target SNR (`params[:, index['snr']]`)
  is divided by `sqrt(effective_num_coils)` then multiplied by each
  transient's own independently-sampled `coil_snr` weight
  (default range `[0, 2]`, mean 1, from `artifacts.mat`) --
  `generate_noise()` lines ~1017-1042. This correctly reproduces the
  standard "averaging N acquisitions improves SNR by sqrt(N)" relationship
  (an individual transient must start out *noisier* than the combined
  target). One inherited bug, now fixed: `multicoil()`'s own docstring
  said the opposite ("much higher" per-transient linear SNR) --
  corrected in place; this was a stale/wrong comment, not a code bug.
- **Correction (repo owner, direct)**: SNR in MRS is a unitless ratio,
  never decibels -- my first pass at this audit incorrectly documented
  `target_snr` as being "in decibels" because `generate_noise()` itself
  comments `lin_snr = 10**(param / 10) # convert from decibels to linear
  scale` and an old inline docstring reports figures like "8.5278dB".
  That documentation error is now fixed throughout (`SimulationResult`,
  `provenance.py`). Left open, explicitly, as a separate question for the
  repo owner rather than assumed either way: does `generate_noise()`'s
  own internal decibel-style conversion of the stored (unitless) target
  SNR need revisiting -- and separately, is `10**(x/10)` (the power-ratio
  dB formula) even the right inverse if that conversion is kept, given
  `lin_snr` is used directly against a peak *amplitude* reference
  (`std_dev = max_val / lin_snr`), which would conventionally use the
  amplitude-ratio dB formula `10**(x/20)` instead? Not resolved here.
- **`realized_snr` (`pSNR`/`sSNR`) is a plain unitless ratio** (no log
  conversion is applied to it anywhere -- confirmed by grep, no hits),
  consistent with SNR being unitless. Whatever is decided about
  `target_snr`'s internal representation above, comparing the two
  directly currently requires undoing `generate_noise()`'s conversion on
  the target side first; documented explicitly in `SimulationResult`'s
  field comments and `provenance.py`'s `snr_definitions` rather than left
  as an implicit trap.
- **`realized_snr` is a genuine post-noise measurement, but not of the
  final returned spectrum**: it's computed from the *actual drawn* noise
  realization's measured std (`noise_vec.std()`), divided into each
  basis-function line's own *pre*-baseline/pre-multicoil-combination/
  pre-phase/pre-frequency-shift/pre-normalization clean amplitude -- not
  from `SimulationResult.noisy`/`.spectrum` itself. So it is not a pure
  relabeling of the target (it does reflect the actual noise draw), but
  it also isn't "measured from the fully processed output" in the
  strictest reading of the handover's phrasing. Documented explicitly
  rather than changed -- recomputing it from the final output would be a
  larger, riskier redesign than an audit-scope fix, and the handover
  explicitly says to preserve behavior absent a demonstrable bug.
- **`pSNR` ("power") and `sSNR` ("spectral") are not the same kind of
  quantity** despite being presented as a pair: `sSNR` is a
  frequency-domain peak height (real channel only, `Fourier_Transform(
  fid).max(dim=-1)`); `pSNR` is the FID's `t=0` time-domain value (both
  real and imaginary channels) -- by the Fourier DC-value identity this
  is closer to a total-signal/area quantity than squared power, despite
  the name. Both share the same denominator (measured noise std). Not
  changed (no demonstrable bug -- these are just two different reference
  quantities, both plausibly useful), but now documented explicitly so a
  caller doesn't assume they're interchangeable views of the same signal.
- Confirmed (already known from Milestones 8/10/11/12, re-verified rather
  than re-litigated): the multicoil `pSNR`/`sSNR`/`generate_noise()` shape
  crashes are fixed; no further multicoil SNR bug found in this pass.

**Not changed**: the underlying SNR computations themselves (no
demonstrable bug found beyond the docstring). `SimulationResult.provenance`
still doesn't carry `snr_definitions` automatically (see the open
"provenance not wired into SimulationResult" item, repo owner flagged
this directly and it remains a separate, undecided piece of work).

**Tests**: none new (documentation/comment-only changes plus one
docstring fix); full suite re-run to confirm no regression, 130/130
passing (includes Milestone 17's CRLB suite).

## Milestone 19 — Section 11 audit: basis-set metadata + a real, unguarded double-application bug (commit TBD)

**Handover section addressed**: 11 (basis-set metadata / double-application audit).

**Method**: same as Milestone 18 -- delegated read-only investigation,
then independently verified every concrete claim (guard absence via
direct code read; the 4 shipped configs via direct grep; the dropped
`header_info['lw']` value and `build_header_fields()`'s current field
list via direct code read) before acting.

**Finding 1 -- basis-set header schema, now measurably closer to the
handover's requested list**: the only *active* header-writing code is
`build_header_fields()` in `src/aux/process_basis_functions.py` (11
fields: spectralwidth, carrier_frequency, Ns, t, centerFreq, B0, TE,
pulse_sequence, vendor, basis_set_software, ppm). `notes`/`linewidth`/
`dwelltime` present on some *legacy* .mat files (e.g.
`PRESS_30_GE_2000.mat`) are hand-authored artifacts with **no writer
anywhere in the current repo** -- confirmed by grepping the exact notes
text and `.m`/`.py` sources; don't treat them as a reliable or extensible
mechanism. Added, in `build_header_fields()`:
- `dwelltime` (was already computed as `dt` to build `t`, just never
  saved as its own field).
- `pre_existing_linewidth_hz`: captures `header_info['lw']`, which
  `load_marss_mat`/`load_fsl_mrs_basis_dir` **already compute** (MARSS's
  documented 1.0 Hz default broadening; FSL-MRS's per-basis-set `Rx_LW`)
  and previously discarded before it ever reached the saved header --
  this is the single most actionable finding, since the information
  already existed and was simply being thrown away. Loaders that don't
  report it (Osprey, LCModel `.basis`/`.raw`) leave this at `0.0`,
  documented as "not reported", not "confirmed zero".
- `te_decay_applied` / `tr_relaxation_applied`: both hardcoded `False`,
  confirmed structurally true for every loader in this file (`t` always
  starts at 0 with no pre-echo offset; nothing references TR/T1
  anywhere) -- makes explicit what `V1_0=False`'s T2* amplitude term and
  the (dormant) `t1_cfg` feature already assume implicitly.
Deliberately NOT added: `TR` and a software-version field -- neither has
an actual source anywhere in this pipeline right now (`config` has no TR
key), and unlike the JSON-based T1 database placeholders,
`scipy.io.loadmat`/`convertdict()`'s type handling doesn't safely round-
trip a Python `None` (verified: `np.asarray(None, dtype=np.float32)`
raises `TypeError`, which `convertdict()`'s `except ValueError` would NOT
catch) -- adding a field that would always be `None` risked a real crash
for no information gain, so it's deferred rather than added unsafely.
Verified all new fields round-trip cleanly through `convertdict()`
(booleans convert to `0.0`/`1.0` tensors correctly, no exception).

**Finding 2 -- CONFIRMED, unguarded double-application, live in 4 shipped
configs**: `'g'` (per-line sampled Gaussian broadening, part of the Voigt
lineshape) and `b0=True` (the explicit spatial B0 field-inhomogeneity
simulator) model the **same physical effect** -- this repo's own prior
milestone already established this from the paper MRS-Sim implements
("[the Voigt lineshape's Gaussian component] is *also* meant to represent
inhomogeneous broadening from intra-voxel field variation -- the same
physical effect `B0_inhomogeneities()` explicitly simulates spatially,
just modeled as a simple Gaussian statistical assumption instead" --
Milestone 13's writeup, never acted on until now). Unlike the already-
fixed `b0`/`V1_0` T2* double-counting (which raises a clear `ValueError`
when both are active), **there is no equivalent guard for `b0` vs.
`g`** -- confirmed by reading `forward()`'s only related check (`b0 and
not self.V1_0`, nothing referencing `g`/`broadening`). Confirmed live
(not just theoretical) by grepping every shipped config for
`"b0": true` + a nonzero `"_g"` range with `broadening: true`:
`src/config/templates/B0_samples_15.json`, `src/config/templates/
B0_samples.json`, `src/config/predefined/B0_samples.json`, and
`src/config/predefined/clean_PRESS_144_GE.json` all hit this combination
today; `cows.json`/`kelley.json` set `b0: false` and are unaffected only
incidentally (not because anything prevents it), and `forward()`'s own
default is `b0=True`. **Not fixed in this milestone** -- this changes
behavior for shipped configs (same significance as the `b0`/`V1_0` fix,
which was confirmed with the repo owner directly before implementing);
raised as an explicit question rather than assumed. See progress_log's
open-items note below.

**Finding 3 -- TE/TR/phase/frequency-shift: no double-application found,
but for reasons that were implicit rather than recorded**: basis FIDs
start at t=0=echo with nothing before it modeled (no loader applies TE-
decay or TR/T1 saturation), matching what the already-existing `V1_0=
False` T2* term and dormant `t1_cfg` feature already assume -- now made
explicit via Finding 1's new flags instead of left as an unstated
convention. No phase-correction or frequency-realignment code exists
anywhere in the basis-set conversion pipeline (checked directly), so
MRS-Sim's own `phi0`/`phi1`/frequency-shift sampling is additive-by-
design against a presumed-canonical basis, not a double-application --
but there is no metadata field confirming any given basis set actually
*is* phase-canonical, which is a real (if currently unexercised) gap for
unusual-provenance basis sets (e.g. LCModel `.basis`/`.raw` imports, which
already have documented orientation quirks elsewhere in this file).

**Housekeeping, not fixed**: `src/aux/io_writeospreyBASIS.py` and
`io_writelcmBASIS.py` (and `convert_MRSS_to_MRS-Sim.py`, which imports the
former) have unparseable syntax and are already commented out of
`src/aux/__init__.py` -- confirmed dead code, not reachable, their
hardcoded `linewidth=1` placeholders don't affect anything live.

**Tests**: 5 new (`tests/test_process_basis_functions.py`) covering the
three new `build_header_fields()` fields (present/absent-`lw` cases, the
always-`False` decay flags) and a regression check that `t` still starts
at 0. Full suite: 130/130 passing.

**Resolved in Milestone 20 below**: the `g`/`b0` guard question (Finding
2) and the SNR-decibel question raised while reviewing this milestone's
own documentation.

## Milestone 20 — SNR is unitless (not decibels): real bug fixed; 'g'/b0 guard implemented (commit TBD)

**Handover sections addressed**: 7 (SNR) and 11 (double-application),
follow-up to Milestones 18-19, both resolved directly by the repo owner.

**Correction and a real bug it uncovered**: while documenting Milestone
18's SNR audit, I incorrectly described `target_snr` as being "in
decibels" -- the repo owner corrected this directly: SNR in MRS is a
unitless ratio, never decibels, per expert consensus. That correction led
to checking whether `generate_noise()`'s own `lin_snr = 10**(param / 10)
# convert from decibels to linear scale` was still live code or a dead
leftover (asked directly by the repo owner) -- confirmed live: this
function is shared by both `compile_outputs()` (every existing caller)
and `_compile_result()`, executed on every simulation with `noise=True`;
grepped the whole codebase and found no other dB<->linear conversion for
SNR anywhere. Per the repo owner's explicit instruction ("It should not
be active in the legacy code or this updated code"), **removed** -- `lin_
snr = param` now, using the sampled `'snr'` column directly as the
unitless ratio. This is a real behavior change to every noisy simulation
(not just a documentation fix): config ranges like `cows.json`'s `"snr":
[10, 20]` and `artifacts.mat`'s default `[0, 100]` now mean what they
look like they mean (a plain ratio of 10-20, or up to 100) rather than
being additionally exponentiated (`10**(20/10)=100`, `10**(100/10)=10
billion` under the old, incorrect formula). Updated the stale historical
docstring in `generate_noise()` (which quoted "8.5278dB"-style figures
from before this fix) to note those numbers no longer reflect current
behavior. Updated `SimulationResult`'s field comments and `provenance.
py`'s `snr_definitions` to say "unitless ratio" throughout, not decibels.

**Verification caveat -- root-caused and fixed in Milestone 22 below**:
an end-to-end check against `cows.json`'s real basis set (target SNR =
15, batch of 200) found `realized_snr['spectral']` at the `snr_metab`
line averaging ~680 -- roughly 45x the target, not a close match. My
first-pass guess (this section, as originally written) was that this was
the same pre-existing "I still don't know why the overall SNRs vary so
much" gap the original author documented in `generate_noise()`'s own
docstring, and left it unexplained. The repo owner pushed back directly
("that code was in fact working before... you need to figure that out"),
which was the right call: it was a distinct, precisely-diagnosable bug
(a factor of exactly `sqrt(N)` from FFT-domain noise normalization -- see
Milestone 22), not the author's older variance concern. Fixed there.

**'g'/`b0` double-counting guard (Finding 2 from Milestone 19)**: repo
owner's decision -- raise a clear error like the `b0`/`V1_0` guard, with
a caveat: "It can be applied to the macromolecule and lipid signals even
when B0 is used, but only one of them should be applied to metabolites."
Investigating how to implement the metabolite-vs-MM/lipid distinction
surfaced a related, previously-unnoticed fact about the parameter
registry: `'dmm'`/`'gmm'` (which the `names`/`mult` list in `initialize()`
appears to define as separate MM-only linewidth parameters) are **not
actually part of `self.index` at all** -- confirmed directly against a
real model (`cows.json`'s basis, `pm.MM=8`): `'dmm'`/`'gmm'` raise
`KeyError` on `self.index`, while `self.index['g']`/`self.index['d']`
each have all 28 entries (20 real metabolites + 8 MM/lipid lines
together, in `self._metab`'s order -- `order_metab()`, `src/aux/aux.py`).
So metabolite and MM/lipid lines already share one combined, per-line
`'g'` array; there was never a separate `'gmm'` mechanism actually wired
up despite the `names`/`mult` list implying one. This didn't block
implementing the requested guard -- `self._metab`'s known ordering
(metabolites first, then MM/lipid) plus `self.MM` (the MM/lipid count)
is enough to slice the metabolite-only leading columns out of the
combined `'g'` index without needing a separate `'gmm'` key -- but it's
worth knowing that `'dmm'`/`'gmm'` occupy columns in the sampled
parameter tensor and get a range computed for them, but nothing reads a
value out of them by name yet. **Correction, repo owner directly**: this
is not dead code or a mistake -- it's the start of an implementation the
repo owner began and never finished, because their own actual parameter-
sampling workflows are written directly in Python (e.g. `sim_COWS.py`),
not through these config keys. `'gmm'` is finished/wired up in Milestone
21 below (as a config convenience, on top of -- not replacing -- direct
Python-based sampling); `'dmm'` is deliberately left as unfinished
groundwork for now (see Milestone 21's scoping note).

Implemented `PhysicsModel._check_g_b0_double_counting(g_cols, n_mm_lines,
max_ranges)` (static, extracted for direct testability without a real
basis set, mirroring `_resolve_t1_config`/`_apply_t1_scaling`'s pattern):
raises `ValueError` when `b0=True` and the *configured range* (not a
specific sampled batch's values -- deterministic given config, matching
the `b0`/`V1_0` guard's style) for any metabolite-only `'g'` column is
nonzero; MM/lipid columns (the trailing `self.MM` entries of `self.index
['g']`) are exempt. Called from `forward()` right after the existing
`b0`/`V1_0` check. Verified end to end:
- `src/config/predefined/B0_samples.json` (one of the 4 previously
  confirmed-affected configs, patched only to add a missing unrelated
  `snr_metab` key so `prepare()` would run) now correctly raises.
- `cows.json` (`b0=False`) is unaffected, as expected.
- A direct test against `cows.json`'s real basis set with metabolite `'g'`
  range manually zeroed and MM `'g'` range left nonzero, `b0=True`: no
  raise -- confirms the MM/lipid exemption actually works, not just that
  the guard fires at all.

**Follow-up, same day, Milestone 21**: the 3 other affected shipped
configs were fixed -- see below.

**Tests**: 4 new in `tests/test_physics_model_bugfixes.py` covering
`_check_g_b0_double_counting`'s raise/no-raise/MM-exemption/no-MM-lines
cases. No new committed test for the SNR formula fix itself (it's a
one-line, directly-inspectable change; the existing SNR-shape tests in
the same file already exercise `generate_noise()`'s surrounding code).
Full suite: 134/134 passing.

## Milestone 21 — finish 'g'/'gmm' config sampling; fix the 3 remaining b0/g configs (commit TBD)

**Handover section addressed**: 11, follow-up to Milestone 20's `'g'`/`b0`
guard, which unexpectedly started rejecting 4 shipped configs whose
metabolite `'g'` range was nonzero alongside `b0=True`.

**Correction, repo owner directly (important -- do not describe as a bug
below)**: every shipped config's `"parameters"` block writes `"_g"`,
`"_d"`, `"_dmm"`, `"_gmm"` with a leading underscore, which never matches
`self.index`'s bare key names (`'g'`, `'d'`) in
`set_parameter_constraints()`. Investigating the 4 failing configs
surfaced that this is true of *every* config in the repo, not just those
4 -- e.g. `cows.json`'s stated `"_g": [5, 20]` has never actually applied;
the model has always sampled `'g'` from `artifacts.mat`'s default
`[0, 70.71]` instead. **This is not a bug or dead code.** Per the repo
owner directly: their actual parameter-sampling workflows are written
directly in Python (e.g. `sim_COWS.py`), not through these config keys --
`_g`/`_d`/`_dmm`/`_gmm` in the JSON configs were the start of a
config-driven sampling convenience the repo owner began and never
finished. `'d'`'s case is more benign in practice: it's overridden by a
separate, correct per-metabolite T2-derived computation in
`define_parameter_ranges()` before any config value would even apply, so
the unfinished key has had no numeric consequence for `'d'` specifically
-- but `'g'` has no such override, so every dataset ever generated by
this pipeline has sampled Gaussian broadening from the wide `artifacts.
mat` default rather than whatever narrower range a config's `"_g"`
appeared to request.

**Repo owner's decision**: finish it now for `'g'`/`'gmm'`, across every
config, keeping the same numeric values already written (just under
their correct, matching key names) -- `'d'`/`'dmm'` deliberately left
as unfinished groundwork, out of scope here (would override the
scientifically-grounded per-metabolite T2-derived default with one
config-wide range, a bigger and separate change).

**Implementation, per the repo owner's explicit instruction to keep this
as one grouped tensor array** (not a second index or a second
`lineshape_correction()` call for MM/lipid): `self.index['g']` remains a
single combined array spanning every line (metabolites first, then MM/
lipid -- confirmed via `order_metab()`'s ordering). Added
`PhysicsModel._split_metab_mm_columns(cols, n_mm_lines)` (static, shared
by `set_parameter_constraints()` and Milestone 20's
`_check_g_b0_double_counting`, so both agree on the same split) and
extended `set_parameter_constraints()`: a config `"g"` entry writes into
the metabolite-only leading columns of `self.index['g']`, a `"gmm"` entry
writes into the MM/lipid-only trailing columns -- both slices of the
exact same `self.min_ranges`/`self.max_ranges` tensor, one `'g'` index,
one lineshape path.

**Config files fixed** (renamed `"_g"`->`"g"`, `"_gmm"`->`"gmm"`, same
values, in every config that had them): `cows.json`, `kelley.json`,
`testing.json` (all `b0=False`, so kept their original `[5, 20]`/`[5, 20]`
values for both, now actually applied for the first time) and the 4
`b0=True` configs from Milestone 20 (`templates/B0_samples.json`,
`templates/B0_samples_15.json`, `predefined/B0_samples.json`,
`predefined/clean_PRESS_144_GE.json`) -- these got `"g": [0, 0]`
(metabolite broadening now supplied solely by `b0`'s spatial simulation,
satisfying Milestone 20's guard) while keeping their original `"gmm"`
value (MM/lipid broadening is exempt from the guard and stays sampled
normally). `clean_PRESS_144_GE.json` also had an unrelated, pre-existing
JSON syntax error (a trailing comma) fixed while in there.

**Verified end to end**: all 4 previously-guard-rejected configs now pass
through `pm.forward()` without raising (patched only with a missing,
unrelated `snr_metab` key that `mainFcns.prepare()` requires with no
default -- itself a separate, pre-existing issue confirmed present since
this file's earliest commits, not introduced this session, not fixed
here). `clean_PRESS_144_GE.json` additionally references a basis-set file
(`GE_PRESS_144_test.mat`) that isn't present in the repo -- pre-existing
and unrelated to this fix, not resolved (guessing a replacement filename
would be fabricating the repo owner's intent). Directly confirmed
`cows.json`'s `'g'`/`'gmm'` now apply with the correct per-group values
(previously both silently defaulted to `artifacts.mat`'s `[0, 70.71]`).

**Tests**: 5 new in `tests/test_physics_model_bugfixes.py`
(`_split_metab_mm_columns`'s three split cases, plus two
`set_parameter_constraints()` tests using a minimal `__init__`-free
`PhysicsModel` instance -- no real basis set needed). Full suite:
139/139 passing.

**Not yet started**: `'d'`/`'dmm'` config-driven sampling (deliberately
left unfinished, as above); the pre-existing `snr_metab`-required-with-
no-default issue in `mainFcns.prepare()` (blocks every config except
`cows.json` from even loading, unrelated to this session's work).

## Milestone 22 — CRITICAL: generate_noise() was off by sqrt(N) for every noisy simulation; default SNR=15 (commit TBD)

**Handover section addressed**: 7 (SNR), follow-up. Repo owner pushed
back on Milestone 20's "unexplained ~45x gap, not root-caused" writeup --
correctly: this was a real, precisely-diagnosable bug, not the older
"varies a lot" mystery.

**The repo owner's own description of the intended formula, confirmed
correct and exactly what the code should do**: "SNR = max(Real(spectrum))
/ (1 std of noise). Therefore noise_std = max(real(spectrum)) /
target_SNR. The noise distribution is therefore Normal(mean=0, sigma=
noise_std)." This is exactly `std_dev = max_val / lin_snr` (now `lin_snr
= param` post-Milestone-20) -- confirmed correct, not the bug.

**Root cause, isolated from all basis-set/physics content and confirmed
with an exact, reproducible calculation**: `generate_noise()` draws white
noise, Fourier-transforms it, normalizes THAT (the frequency-domain
representation) to have mean 0 and std `std_dev`, then inverse-Fourier-
transforms it back to time domain -- and returns *that* as the noise
actually added to the spectrum. `Fourier_Transform`/`inv_Fourier_Transform`
(`src/aux/aux.py`) use `torch.fft`'s default ("backward") normalization:
the forward `fft` is unscaled, the inverse `ifft` is scaled by `1/N`. That
convention means normalizing a signal to std `X` in the frequency domain
and then inverse-transforming it does NOT give a time-domain signal with
std `X` -- it gives one with std `X / sqrt(N)` (N = number of spectral
points). Verified directly, isolated from any PhysicsModel/basis-set
content, for N=1024/2048/8192: requesting `std_dev=8.5` produced actual
time-domain stds of 0.265/0.184/0.094 respectively -- ratios of
32.05/46.2/90.4, matching `sqrt(1024)=32.0`, `sqrt(2048)=45.25`,
`sqrt(8192)=90.5` to within statistical noise from a single random draw.
This is why the `cows.json` end-to-end check (N=2048 for that basis)
showed ~45x: `sqrt(2048)=45.25`, matching essentially exactly.

**Consequence**: every simulated dataset with `noise=True` (i.e. every
real dataset -- `noise` defaults to being requested in every shipped
config) has had actual noise added at a magnitude far *smaller* than the
sampled target SNR implied -- realized SNR far *higher* than requested,
by a factor of `sqrt(N)` where N is that basis set's spectral length.
This is a real, live, previously-uncaught bug, not something introduced
this session (the FFT-domain-normalization approach predates this
refactor).

**Fix**: pre-scale the frequency-domain target by `sqrt(N)` before
normalizing, so the inverse FFT's `1/N` scaling brings the time-domain
result back to exactly `std_dev`
(`std_dev.unsqueeze(-1) * (fid.shape[-1] ** 0.5)`). Verified precisely:
the isolated calculation above, redone with this fix, gives actual stds
of 8.484/8.346/8.527 against a target of 8.5 for N=1024/2048/8192 --
all within ~2% (single-draw statistical noise, not a residual bug).
**Verified end to end against `cows.json`'s real basis set**: target SNR
15, batch of 200 -> realized `sSNR` at the `snr_metab` line now averages
15.02 (std 0.16, range 14.66-15.52) -- matching the target almost
exactly, compared to ~680 (45x off) before this fix. Also verified the
multi-transient scaling relationship (Milestone 18) still holds correctly
post-fix: with `num_coils=3` and every `coil_snr` weight fixed at 1.0,
per-transient realized SNR averaged 8.666, matching the predicted
`15/sqrt(3)=8.660` closely.

**Default SNR range changed to [5, 30]** (repo owner's explicit request:
"if you are defining min and max, then it should be 5 and 30"; a fixed 15
was only the fallback if the mechanism couldn't hold a range, which it
can): `src/basis_sets/artifacts.mat`'s `'snr'` entry changed from
`{'min': 0, 'max': 100}` to `{'min': 5, 'max': 30}` -- this is the
fallback range used whenever nothing else overrides the `'snr'` column
(confirmed: the top-level config `"snr": [min, max]` key, e.g.
`cows.json`'s `[10, 20]`, is *not* read by `mainFcns.prepare()`/
`PhysicsModel` at all -- it's only consumed by the separate driver
template scripts, e.g. `mrs-sim_template.py:121`'s
`params[:,ind['snr']].uniform_(config.snr[0], config.snr[1])`, matching
the repo owner's "I always use py files to define how to sample the
parameters" -- so this default only matters for callers that don't
already override SNR sampling themselves). `'snr'` remains a genuine,
per-sample sampled column of `params` (`self.index['snr']`, read directly
by both `generate_noise()` and `SimulationResult.target_snr` -- confirmed
unchanged) -- only the *default range* changed, nothing hardcodes SNR to
a constant. Edited the binary `.mat` file directly via `scipy.io.loadmat`/
`savemat`, verified the round trip preserves every other artifact default
byte-for-byte (`d`, `g`, `phi0`, `coil_snr`, etc. all unchanged) and loads
correctly through the real `PhysicsModel.__init__()`/`convertdict()` path
before overwriting the shared file.

**Additional multicoil verification, requested directly before
committing**: repeated the multicoil check with `coil_snr` weights *and*
`coil_sens` (some coils zeroed) both randomly sampled per sample per
transient, rather than fixed at 1.0 -- 300 samples x 4 transients = 1200
data points, `realized / predicted` (predicted =
`target_snr / sqrt(n_effective_coils) * coil_snr_weight`) had mean 1.0001,
std 0.011, range [0.965, 1.039].

**Tests**: 3 new (parametrized over N=512/2048/8192) in
`tests/test_physics_model_bugfixes.py`, calling `generate_noise()`
directly (it doesn't reference `self`, so no basis set is needed) and
asserting the realized noise std matches the target within 5%. Full
suite: 142/142 passing.

## Milestone 23 — per-component compute skipping (baseline spline); provenance actually wired into SimulationResult (commit TBD)

**Handover sections addressed**: 3 (per-component skip-to-save-memory/
compute, deferred since Milestone 5) and 9 (provenance, follow-up to the
repo owner's "what do you mean by 'not wired in'?" question earlier this
session).

**Per-component compute skipping, clarified by the repo owner directly**:
"I use the flags in the forward pass to only do what is necessary for
that pass. That means the CRLBs don't need to be calculated for every
pass and the splines don't need to fit every baseline unless explicitly
flagged." Checked both:
- `compute_crlb` was already correctly gated (`compute_crlb: bool=False`
  default, only computed in `_compile_result()` when explicitly `True`) --
  no change needed, confirmed by reading the existing code.
- Baseline spline fitting was **not** gated -- `fit_baseline_spline()`
  ran unconditionally whenever a baseline existed (`if baseline is not
  None:`), with no way to get the raw generated baseline back without
  also paying for the spline fit. Fixed: `forward()` gained
  `fit_baseline_spline: bool=False`, threaded through to
  `_compile_result()`; the fit now only runs when a baseline exists
  *and* this flag is `True`. `compute_crlb` already tolerated
  `spline_coefficients=None` (falls back to a zero-baseline nuisance
  term), so this doesn't break CRLB when both are used together --
  verified directly. Verified end to end against `cows.json`'s real
  basis set: default call generates a baseline but leaves
  `baseline_fit`/`spline_coefficients` as `None`; `fit_baseline_spline=
  True` populates both; `compute_crlb=True` with the default (spline
  fit skipped) still returns a valid `crlb`.

**Provenance wiring**: `collect_provenance()` (Milestone 9) was always a
correct, comprehensive standalone utility, but nothing called it
automatically -- `SimulationResult.provenance` stayed a two-key
placeholder dict (`{'noise_enabled', 'offsets_enabled'}`) regardless.
Added `forward(..., collect_provenance: bool=False)`: when `True`,
`_compile_result()` calls `collect_provenance()` (basis-set name/content
hash, git commit, acquisition metadata, SNR definitions, the enabled-
components dict, dtype/device) and attaches its `.to_dict()` as
`SimulationResult.provenance`; the old placeholder dict is kept as the
default (unchanged behavior when not requested). Gated behind a flag
rather than made automatic, matching the same "flags control compute"
principle above -- `collect_provenance()` shells out to `git` and hashes
the loaded basis set on every call, a real (if small) per-call cost that
shouldn't happen by default in a tight sampling/training loop. Verified
end to end against `cows.json`'s real basis set: default provenance
unchanged; `collect_provenance=True` returns all 15
`Provenance` fields populated (confirmed `basis_set_name`,
`basis_set_hash`, `git_commit`, and `enabled_components` all correct).

**Tests**: verified manually against a real basis set (both features
need a real `PhysicsModel`/`_compile_result()`, which committed tests
avoid depending on -- see `test_parameters.py`'s module docstring),
following the same pattern already used for `compute_crlb()` itself.
Full suite: 142/142 passing (no regression).

## Milestone 24 — overlapping parameter-range mechanisms: make the conflict visible, not silent (commit TBD)

**Handover section addressed**: 2 (composable sampler), follow-up to the
repo owner's much-earlier open question about the three overlapping
range-definition mechanisms ("probably excessive and very certainly
redundant") -- asked directly to "make sure that the overlapping
parameter range mechanisms don't conflict with each other."

**The three mechanisms, and their actual precedence (confirmed by
reading, not assumed)**:
1. Basis-set/metabolite-database defaults (`define_parameter_ranges()`,
   lowest precedence).
2. Config `"parameters"` block (`set_parameter_constraints()`) --
   overrides 1, writes into `self.min_ranges`/`max_ranges`.
3. `CopulaInVivoSampler` (`src/sampling.py`) -- for its covered columns,
   writes absolute, already-real-world values **directly** into the
   sampled tensor (`tensor[:, target] = x`), completely bypassing
   `min_ranges`/`max_ranges`/`quantify_params()`. For columns it doesn't
   cover, it falls back to `UniformRangeSampler`, which *does* respect
   1/2 normally.

**The actual conflict**: mechanisms 2 and 3 can both target the same
column (e.g. a config sets `"g": [5, 20]` while a copula's distributions
JSON also has a `gaussLB`-mapped entry) -- mechanism 3 always wins
*silently*, with no error or warning, since it never consults
`min_ranges`/`max_ranges` at all for that column. A user relying on their
config override would see it silently have no effect. This precedence
(copula's fitted in-vivo distribution outranks a min/max range) is a
reasonable, intentional design -- the problem was that it was invisible,
not that it was wrong.

**Fix, scoped to making this visible rather than restructuring the three
mechanisms** (a full redesign wasn't asked for, and the repo owner's
"redundant" comment was about the mechanisms' existence, not necessarily
a request to collapse them): `PhysicsModel.define_parameter_ranges()`
now initializes `self.explicitly_configured_columns = set()`;
`set_parameter_constraints()` records every column it actually writes
into. `CopulaInVivoSampler.__init__()` checks its own covered columns
against `self.pm.explicitly_configured_columns` and raises a `UserWarning`
naming exactly which parameters collide and that the copula will win, if
any overlap -- rather than a user discovering it by noticing their
config override had no effect. Uses `getattr(self.pm,
'explicitly_configured_columns', set())` so a `PhysicsModel`-like object
without this tracking (e.g. the fake used in `tests/test_sampling.py`)
degrades to "no warning" rather than crashing.

**Verified end to end**: `mainFcns.prepare()` against `cows.json`'s real
config correctly populates `pm.explicitly_configured_columns` (confirmed
non-empty, includes the expected column indices for every `"parameters"`
block key that matches a real `self.index` entry).

**Tests**: 2 new in `tests/test_sampling.py` (warns on overlap; does not
warn -- confirmed via `warnings.simplefilter("error")` -- when there is
none). Full suite: 144/144 passing.

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
