# MRS-Sim v1 Architecture Audit (pre-refactor baseline)

Status: reference document for the V2.0 refactor. Written before any behavior-changing
code was touched, per the handover doc's requirement to inspect and document the
existing architecture first. All line numbers refer to the codebase as of the `V2.0`
branch point (parent of the first v2 commit).

## 1. Parameter representation

- All simulation parameters live in one flat `torch.float32` tensor `params`, shape
  `[batch_size, n_columns]`. Values are sampled in `[0, 1]` and later de-normalized by
  `PhysicsModel.quantify_params()` (physics_model.py:1256-1264) using per-column
  `min_ranges`/`max_ranges` tensors (`[1, n_columns]`, physics_model.py:385-386).
- `PhysicsModel._index` / `.index` (built in `initialize()`, physics_model.py:247-378)
  is already a name -> column-index(es) registry, e.g. `ind['naa']` (int),
  `ind['d']` (tuple covering all metabolite+MM Lorentzian-linewidth columns),
  `ind['snr']`, `ind['coil_sens']` (tuple, one per coil), etc. This is effectively the
  "parameter registry" the handover doc asks for in section 1 — it needs
  formalizing/wrapping (nested access, dtype/shape docs), not rebuilding.
- **Coverage gaps in `_index`**: baseline and residual-water generation parameters are
  *not* represented in `params`/`_index` at all — they live in separate
  `baseline_cfg`/`resWater_cfg` dicts consumed directly by
  `sample_baselines`/`sample_resWater` (src/aux/aux.py). There is also no T1 or TR
  column anywhere.
- `mainFcns.prepare()` (mainFcns.py:60-77) casts `_index` entries to plain
  ints/lists and adds two derived keys, `ind['mac']`/`ind['lip']`, via substring
  matching (`'mm' in k`, `'lip' in k`) on existing key names — fragile, but low risk
  today since key names are controlled internally.

## 2. Forward simulation pipeline order (`PhysicsModel.forward()`, physics_model.py:1421-1813)

Order is intentional and must be preserved (README.md:31-34 explicitly states the
pipeline runs in "the opposite order of spectral fitting and processing").

1. B0 field-map computation (`B0_inhomogeneities`, 1460-1461 / 632-739) — computed, not yet applied.
2. Baseline / residual-water pre-generation (`simulate_offsets`, 1474 / 1370-1395) — generated early, added later.
3. Metabolite basis combination (`modulate`, 1500-1501 / 1119-1129).
4. Lineshape/broadening — Voigt/Lorentzian/Gaussian (`lineshape_correction`, 1520-1521 / 1028-1092).
5. B0 inhomogeneity application (`add_inhomogeneities`, 1528 / 449-531).
6. Metabolite-level power/spectral SNR reference capture (`pSNR`/`sSNR`, 1535-1540), from the still artifact-free per-line FID.
7. Per-line frequency shift (`frequency_shift`, 1549-1550 / 860-899).
8. Sum basis lines -> `fidSum` (final signal) and `spectral_fit` (clone; becomes the noise-free fit track) (`line_summing`, 1557-1559 / 1003-1025).
9. Noise vector computed from target SNR (`generate_noise`, 1566-1571 / 902-983) — not yet added.
10. Baseline + residual water added (`add_offsets`, 1601-1604 / 536-577).
11. Multi-coil transient replication (`multicoil`, 1609-1612 / 1132-1148).
12. Noise added, `[noisy, clean]` stacked along a new axis `d` (1615-1636) — **see bug #2 below**.
13. Coil sensitivity scaling (`coil_sensitivity`, 1643-1647 / 782-801).
14. Zero-order phase (`zero_order_phase`, 1653-1654 / 1413-1418).
15. First-order phase (`first_order_phase`, 1659-1660 / 839-857).
16. Global frequency shift (`frequency_shift`, 1665-1666).
17. Coil frequency/phase drift (`coil_freq_drift`/`coil_phi0_drift`, 1675-1685 / 742-779).
18. Eddy currents (`first_order_eddy_currents`, 1700-1701 / 812-836).
19. Apodization (optional, 1707).
20. Zero-filling (optional, 1712).
21. FFT to spectrum, crop/resample (`Fourier_Transform`, `resample_`, 1717-1724).
22. Magnitude channel appended (`magnitude`, 1732-1733 / 1095-1116).
23. Normalization (`normalize`, 1738-1741 / 1151-1186).
24. Difference-editing subtraction (optional, largely unimplemented, 1764-1771).
25. Quantification against `wrt_metab` (`quantify_metab`, 1777 / 1218-1253).
26. Output packaging (`compile_outputs`, 1782-1813).

## 3. Noisy/clean coupling via an extra tensor axis (confirmed)

The handover doc's claim that "the current implementation uses an extra tensor
dimension to couple signal and nuisance components" is confirmed. In
`generate_noise()` (physics_model.py:932-933), a stacking axis `d` (`-3` or `-4`
depending on whether multi-coil transients are active) is computed and returned.
At the noise-addition step it is used to `torch.stack` a `[noisy, clean]` pair for
both `fidSum` and `spectral_fit` (1624, 1628-1629). Every later operation (coil
sensitivity, phase, coil drift, eddy currents, FFT, normalize, ...) broadcasts across
this axis, applying the same physical corrections to both variants identically. A
refactor for nuisance-component removal (handover section 4) should formalize this
axis's semantics rather than replace the mechanism.

## 4. SNR: target vs. realized (already implemented)

`params[:, ind['snr']]` stores a **target SNR in dB**. `generate_noise()`
(902-983) converts it to linear scale, references it against the peak amplitude of
the metabolite(s) named in `snr_metab` (default: `wrt_metab`), and derives the noise
std dev. Per-transient SNR variability is supported via `coil_snr`
(mis-named — it is really a per-transient SNR weight vector, not a coil property).
**Realized/measured SNR** (`pSNR`, `sSNR`) is captured from the pre-noise signal and
then divided by the actual drawn noise's std dev (1585-1593) — i.e., computed from
the realized noise, not the target. Both are saved (`compile_outputs`;
`mainFcns._save` writes `SNR` and `params` separately). This mostly needs
formalizing/documenting per handover section 7, not building from scratch. Note: a
long inline comment (914-930) documents the original author's own uncertainty about
why realized SNRs vary as much as they do — a known, self-acknowledged soft spot to
revisit during the SNR audit.

## 5. Baseline generation — NOT spline-based, and not intended to become spline-based

`src/baselines.py` (`bounded_random_walk`) generates the baseline as a bounded random
walk: cumulative sum of scaled uniform steps, detrended, iteratively reflected back
inside `[lower, upper]` bounds, then smoothed (`batch_smooth`, src/aux/aux.py:52-85)
and resampled onto the acquired ppm grid. **Confirmed with the repo owner: this
generator is intentional and will remain a random-walk generator in v2 — it is not
being replaced by a spline model.** Per handover section 5, splines are only ever a
*post-hoc fit* of the already-generated baseline (fit immediately after generation,
before noise/phase/other contamination), used for (a) a differentiable local
parameterization needed for CRLB calculations, and (b) as a `baseline_fit`
output. The spline coefficients are not differentiated through the fitting procedure
itself, and the spline fit never replaces the baseline actually used in `add_offsets`.

## 6. Relaxation / TE / TR — mostly absent, currently double-applies broadening

- `TE` is accepted by `PhysicsModel.__init__`/`initialize()` but is not used anywhere
  else in the forward physics (grep-confirmed). No TR handling exists anywhere.
- No first-principles T1/T2 exponential relaxation model exists (no `exp(-TE/T2)` or
  `1-exp(-TR/T1)` term). What exists instead:
  - The sampled Lorentzian linewidth (`d`) range is bounded using the metabolite
    database's `T2.metab` field (physics_model.py:406) — T2 only bounds a sampled
    *add-on* broadening parameter, it is not applied as an actual relaxation equation.
  - **Confirmed bug**: `initialize()` (208-211) hardcodes `lw = 1` (the intended
    `lw = 1 - self.linewidth` is dead code — `self.linewidth` is never defined
    anywhere in the file) and unconditionally multiplies every basis FID by
    `exp(-1*t)` at load time, *in addition to* whatever T2 decay the basis-set
    software already baked in, and *in addition to* the per-sample `d`/`g` broadening
    applied later in `lineshape_correction()`. This is the double-application
    failure mode handover section 11 explicitly warns about — and today it is
    unconditional and unconfigurable.
  - **Resolution (agreed with repo owner)**: this will be controlled by a new
    `V1_0` flag, default `True`, which preserves today's unconditional broadening
    exactly for backward compatibility with existing v1-generated datasets/trained
    models. Setting `V1_0=False` will apply the corrected behavior once relaxation
    handling is implemented (handover section 10). This will be implemented and
    documented as part of the relaxation/TE/TR work item, not before — flagging it
    here now so it isn't lost.

## 7. CRLB / Fisher information — none found

No CRLB, Fisher-information-matrix, or Jacobian-based code exists anywhere in the
repository (grep-confirmed). This is greenfield work (handover section 6).

## 8. Basis-set metadata — no "already applied" flags

Loaded `.mat` basis sets carry a `header` block (spectral width, carrier frequency in
MHz, `Ns`, `t`, `centerFreq`, `B0`, `TE`, pulse sequence, vendor, simulation software,
`ppm`) registered as buffers on `PhysicsModel`. **There is no field indicating
whether linewidth/T2/TE/TR effects are already baked into a given basis set's FIDs**,
and nothing checks such a flag before applying additional broadening — this is the
structural gap behind bug #6 above and needs a metadata field per handover section 11.

Two basis-set loader implementations currently exist with an overlapping/duplicated
function set (`src/aux/io_read_basis_sets.py` vs `src/aux/process_basis_functions.py`),
and two independent basis-export implementations exist for LCModel/Osprey formats
(`io_writelcmBASIS.py`/`io_writeospreyBASIS.py` vs. `export_*` functions inside
`process_basis_functions.py`). These should be consolidated during the basis-set
metadata audit rather than left to drift independently.

## 9. NIfTI-MRS export — functional but under-provenanced

`src/NIfTIMRS/mat2niftimrs.py` writes acquisition metadata (`SpectrometerFrequency`,
`EchoTime`, dwell time via `pixdim[4]`) but: `RepetitionTime` is hardcoded to the
**string** `'NA'` rather than numeric/omitted (likely violates the NIfTI-MRS JSON
schema's expectation of a numeric field for consumers that parse it strictly), several
fields are unconditionally hardcoded placeholders (`Manufacturer='MRS-Sim'`,
`SoftwareVersions='NA'`, all `Patient*` fields), no processing-history list is written
despite the NIfTI-MRS convention supporting one, and the format-version tag
`intent_name = b'MRS-Sim_v0_0'` is a hardcoded literal, not tied to any actual
installed package version. To be addressed per handover section 12.

## 10. Provenance — none

No git commit, package version, RNG seed, or config snapshot is saved alongside
simulated data anywhere in `mainFcns._save()` or `sim_COWS.py`. No global seed is set
in the main simulation path (`sim_COWS.py`, `mainFcns.py`) — seeding utilities exist
only in the unrelated GIRF synthetic-data modules. Reproducing an exact v1 dataset
today depends on independently archiving the config JSON and being lucky about RNG
state. Greenfield work per handover section 9.

## 11. Config schema (from `src/config/*.json`)

Representative fields across `cows.json`/`kelley.json`/`config_template.json`:
`totalEntries`, `PM_basis_set`, `metabolites`, `wrt_metab`, `snr_metab`,
`param_distributions` / `corr_matrix` (COWS-only — see section 12 below), `B0`, `TE`,
`vendor`, `cropRange`, `lineshape`, `spectralwidth`, `spectrum_length`,
`basis_fcn_length`, boolean toggles mirroring `forward()`'s flags (`b0`, `fids`,
`eddy`, `fshift_g`, `fshift_i`, `phi0`, `phi1`, `noise`, `apodize`, `num_coils`,
`coil_phi0`, `coil_sens`, `broadening`, `magnitude`, `coil_fshift`, `zero_fill`,
`resample`, `drop_prob`), a `parameters` block consumed by
`set_parameter_constraints()` for per-parameter range overrides, a `baseline_cfg`
block, and a `resWater_cfg` block.

**Known schema inconsistency**: `baseline_cfg` uses key `ppm_range` in
`cows.json`/`kelley.json` but `cropRange` in `config_template.json`;
`aux.prepareConfig()` unconditionally reads `cfg['ppm_range']` — a config authored
against the template's naming would raise `KeyError`. To be resolved (pick one name,
fix the template) during the config-schema pass.

## 12. In-vivo-fitting-based parameter sampler (`sim_COWS.py` + `sample_from_fitted_dist.py`)

This is a **Gaussian-copula sampler**, separate from and complementary to the
straightforward uniform/range sampling used elsewhere: per-parameter marginal
distributions are fit to real in-vivo spectral-fitting results (via
`findParamDist.py`, currently only supporting one fitting software's export format,
to be expanded later), and combined with a correlation matrix (computed externally,
e.g. via a `findCorr`-style function, from normal-score-transformed fit data) so that
sampled synthetic parameters preserve realistic inter-parameter correlations rather
than being drawn independently. `sample_from_copula()`
(src/aux/sample_from_fitted_dist.py:61-356) draws `Z ~ MVN(0, R)`, maps through the
standard normal CDF, then through each parameter's fitted marginal PPF.

This is a legitimate, reusable sampling *strategy* and is intended to become one
pluggable backend of the v2 composable sampler (handover section 2), alongside a
plain-range/uniform sampler. Known fragility to resolve during generalization:

- `sim_COWS.py:46-62` hardcodes dataset-specific name-remapping dicts
  (`name_map`, `suffix_to_ind`, `global_param_map`, e.g. `"Cr_SNR": "snr"`,
  `"PCh": "cho"`) tying the sampler to the COWS basis-set naming convention.
- `sample_from_fitted_dist.py:157-170` contains an unexplained, magic-number
  reordering of the correlation matrix's rows/columns (deletes row/col `-2`, moves
  indices `26`/`27` to the end, moves the item at `-3` to position `52`). This must
  be reverse-engineered against `findParamDist.py`'s / the correlation-matrix
  builder's actual variable ordering before it can be generalized — tracked as a
  follow-up investigation, not yet resolved as of this document.
- Per repo owner: v2 should add the ability to select/exclude specific fitting
  parameters from the copula for a given simulation run (e.g. to simulate only a
  subset of the physical effects the in-vivo fits captured).

## 13. Tests

No `tests/` directory exists. The only executable self-checks anywhere in the repo
are inline `assert np.allclose(...)` calls inside
`src/NIfTIMRS/mat2niftimrs.py:test_output()`, which check NIfTI round-trip
consistency only. No unit/integration/regression coverage exists for the physics
model, baseline/residual-water generation, parameter sampling, or basis-set I/O.
Greenfield per handover section 13.

## 14. Other confirmed/likely bugs found during this audit

Two bugs above (double-broadening, §6; noisy/clean coupling axis, §3) are the most
consequential for the refactor design and are called out separately. Additional
findings, roughly ranked by confidence and impact:

1. **RETRACTED — was reading uncommitted debug code, not the committed
   codebase.** This originally claimed `simulate_offsets()`/`add_offsets()`
   crash when exactly one of baseline/residual-water is configured, due to
   an unconditional debug `print(...baselines.shape...)` dereferencing
   `.shape` on `None`. That print statement turned out to be part of the
   repo owner's own *uncommitted* working-tree edits (present when this
   audit was first written, since discarded at their request once
   identified) -- it is not in the committed `physics_model.py` and never
   was. The actual committed `simulate_offsets()`/`add_offsets()` guard
   `None` correctly and do not crash for single-component configs. See
   `docs/v2/progress_log.md`, Milestone 4, for how this was found.
2. **Fixed (see progress_log.md, Milestone 4) — noisy/clean coupling bug**:
   at physics_model.py's noise-addition step, noise was added to *both*
   branches of the `[noisy, clean]` stack for the main signal `fidSum`
   (`fidSum[...,0,:,:] + noise_vec` AND `fidSum + noise_vec`), while the
   parallel `spectral_fit` tensor correctly left its "clean" branch
   unmodified. Fixed by extracting the pattern into a single shared
   `PhysicsModel._stack_noisy_clean()` used by both, with a unit-level
   regression test (`tests/test_physics_model_bugfixes.py`).
3. **Fixed (see progress_log.md, Milestone 4) — dead `'temperature'`
   column broke `quantify_params()` unconditionally**: `initialize()`
   appended a `'temperature'` entry to the `header` list used to size
   `min_ranges`/`max_ranges`, with no corresponding entry in `ind`/`dct`,
   making `min_ranges`/`max_ranges` exactly one column wider than the
   params tensor. `quantify_params()` raised a shape-mismatch
   `RuntimeError` the moment it was called with any config/basis set.
   `sim_COWS.py` never hit this because it comments out its own
   `quantify_params()` call; `mrs-sim_template.py`/
   `deep_learning_dataset_template.py` call it directly and would have hit
   this. Found while validating `UniformRangeSampler` (v2.0 section 2)
   against a real basis set.
3. `header = self._metab` aliasing bug (physics_model.py:251): `header` is bound to
   the same list object as `self._metab`, and the subsequent loop
   (`for n, m in zip(names, mult): header.append(n)`, 297-298) appends dozens of
   parameter-category name strings directly onto `self._metab`. Any code that later
   trusts `pm.metab`/`pm._metab` to be "just the metabolite names" would see it
   polluted. Currently appears numerically harmless (nothing downstream seems to
   depend on `len(self._metab)` post-`initialize()`), but should be fixed (use
   `list(self._metab)` at line 251) since it is a real correctness landmine for any
   future code (e.g. plotting/labeling) that reads `pm.metab`.
4. `lineshape_lorentzian()` (physics_model.py, in `lineshape_correction`'s
   Lorentzian-only branch) references an undefined variable `g` — selecting
   `lineshape: "lorentzian"` in a config would raise `NameError` the first time it's
   exercised. Not yet hit by any shipped config (all current configs use `"voigt"`).
5. `src/config/predefined/clean_PRESS_144_GE.json` contains invalid JSON (illegal
   trailing comma) and cannot currently be loaded.
6. `PhysicsModel.basis_metab` property (physics_model.py:82-88) always returns an
   empty list — it iterates `self._basis_metab`, which is initialized to `[]` and
   never populated anywhere in the file.
7. `params[:, ind['coil_sens']]` is overloaded: used both as an actual per-coil
   sensitivity-scaling factor and, separately, reinterpreted inside
   `generate_noise()` as a "number of zeroed-out coils" indicator
   (physics_model.py:1569, 932). Worth disambiguating into two named quantities
   during the registry formalization.
8. The individual-spin (as opposed to summed-spin) basis-function code path in
   `initialize()`'s `except:` branch (172-200) is explicitly marked
   `# TODO: Not finished!!!` and contains what looks like a copy-paste shape-mutation
   bug (192-194). Relevant if v2 basis sets ever use individual-spin representations.

## 15. Summary: what's new work vs. what's formalization

| Handover section | Status |
|---|---|
| 1. Parameter registry | Mostly exists (`PhysicsModel.index`); needs wrapping/extension (baseline, residual-water, T1, TR not yet covered) |
| 2. Composable sampler | Partial — copula-based in-vivo sampler exists but is bespoke; needs generalizing into a pluggable backend |
| 3. Forward sim / component outputs | Partial — components are computed internally but not all individually returned/toggleable; noisy/clean axis exists but has bug #2 above |
| 4. Nuisance-component removal | Not implemented as a standalone operation; the coupling mechanism it would build on exists |
| 5. Baseline + spline fit | Baseline generator exists (random walk, staying as-is); spline *fitting* of it is entirely new |
| 6. CRLB/FIM | Entirely new |
| 7. Target vs. realized SNR | Mostly exists; needs formalizing/documenting |
| 8. Parameter replay | Not implemented as a first-class feature (no seed/param persistence for exact replay across basis sets) |
| 9. Provenance | Entirely new |
| 10. Relaxation/TE/TR | Mostly absent; interacts with the double-broadening bug (`V1_0` flag plan above) |
| 11. Basis-set metadata / double-application audit | Metadata exists but lacks "already applied" flags; the double-broadening bug is the concrete instance to fix |
| 12. NIfTI-MRS export | Exists but under-provenanced; several concrete issues listed above |
| 13. Tests | Entirely new |
