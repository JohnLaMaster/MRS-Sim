"""
Nuisance-component removal (v2.0 handover, section 4).

Two removal paths, per the handover doc:

1. **During simulation**: already available for free. `PhysicsModel`
   never adds baseline/residual water to `spectral_fit` -- only to
   `fidSum`/`specSummed` (via `add_offsets()`). `SimulationResult.
   nuisance_free` (see `simulation_result.py`) already *is* the correctly
   physically-transformed, nuisance-free signal; no subtraction is
   involved and no numerical approximation is made. Prefer this path
   whenever you're generating data fresh.

2. **Path-based, from a saved simulation**: `remove_nuisance_from_saved()`
   below, for when only the saved `.mat` output (`mainFcns._save()`
   format) is available and the original `SimulationResult` is not.

IMPORTANT CAVEAT for path-based removal: it reconstructs an
*approximation* of the nuisance-free signal by subtracting the saved
baseline/residual-water arrays from the saved spectrum, and this does
**not** currently reproduce the true nuisance-free signal exactly.
Measured directly against a real basis set (cows.json), comparing this
function's reconstruction to the true nuisance-free signal from a live
`SimulationResult` computed with the identical seed/parameters (see
docs/v2/progress_log.md, Milestone 5, for the exact script): max absolute
error 0.64 against a signal whose own max absolute value is 0.99 -- a ~65%
relative error, clearly not numerical noise. The most likely cause:
`baseline_cfg`/`resWater_cfg` each define their own internal ppm range
(e.g. `cows.json`'s baseline_cfg uses `[-1.6, 6]` ppm and resWater_cfg uses
`[4.4, 4.85]`/`[4.0, 6.4]` ppm) which do not match the main spectrum's
`cropRange` (`[0.2, 4.2]` ppm) -- the baseline/residual-water generators
resample onto their own grid, and it was not confirmed that this lines up
exactly with the main spectrum's final grid at the point they're saved.
This has not been fully root-caused -- flagged here rather than silently
shipping an "exact" removal that isn't. Use `SimulationResult.
nuisance_free` (from `forward(..., return_components=True)`) directly
when exactness matters; it is not an approximation.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

__all__ = ['remove_nuisance', 'remove_nuisance_from_saved']

ArrayLike = Union[torch.Tensor, np.ndarray]


def remove_nuisance(
    noise_free_total: ArrayLike,
    baseline: Optional[ArrayLike] = None,
    residual_water: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Subtract designated nuisance components from a noise-free signal.

    `noise_free_total` should already have noise removed/never added (see
    `SimulationResult.noise_free_total`); this function only ever removes
    baseline/residual water, never noise -- consistent with the handover
    doc's terminology that noise is not itself a "nuisance component".

    Returns the same type (`torch.Tensor` or `np.ndarray`) as
    `noise_free_total`.
    """
    out = noise_free_total
    if baseline is not None:
        out = out - baseline
    if residual_water is not None:
        out = out - residual_water
    return out


def remove_nuisance_from_saved(mat_path: str, branch: str = 'clean') -> np.ndarray:
    """
    Path-based nuisance removal from a dataset saved by
    `mainFcns._save()`.

    Parameters
    ----------
    mat_path :
        Path to a `.mat` file saved by `mainFcns._save()` (has `spectra`,
        `baselines`, `residual_water` keys).
    branch :
        `'clean'` (default) selects the noise-free half of the saved
        spectrum's noisy/clean axis (matching handover terminology's
        `noise_free_total`); `'noisy'` selects the noisy half instead, so
        the returned signal still contains noise with only baseline/
        residual water removed. `spectra`'s noisy/clean axis is axis 1
        (index 0 = noisy, index 1 = clean), matching
        `PhysicsModel._stack_noisy_clean`'s convention.

    Raises
    ------
    ValueError
        If the saved file has no baseline or residual-water component to
        remove (i.e. offsets were not enabled during the original
        simulation) -- required per the handover doc ("path-based removal
        is only possible when the required component realizations were
        saved"), rather than silently returning the untouched spectrum.

    See the module docstring for why this is an *approximation*, not an
    exact reconstruction, in the current pipeline.
    """
    from scipy.io import loadmat

    data = loadmat(mat_path, simplify_cells=True)
    spectra = np.asarray(data['spectra'])

    baselines = data.get('baselines')
    residual_water = data.get('residual_water')
    # mainFcns._save() writes `[]` (not None) when a component was disabled
    # -- scipy round-trips that as an empty array.
    if baselines is not None and np.asarray(baselines).size == 0:
        baselines = None
    if residual_water is not None and np.asarray(residual_water).size == 0:
        residual_water = None

    if baselines is None and residual_water is None:
        raise ValueError(
            f"'{mat_path}' has no saved baseline or residual-water component "
            f"to remove. Path-based nuisance removal is only possible when "
            f"offsets were enabled (and saved) during the original "
            f"simulation -- see the handover doc's note on this."
        )

    branch_index = {'clean': 1, 'noisy': 0}.get(branch)
    if branch_index is None:
        raise ValueError(f"branch must be 'clean' or 'noisy', got {branch!r}.")

    signal = spectra[:, branch_index, ...]

    # PhysicsModel.compile_outputs() saves baselines/residual_water with a
    # size-1 axis at position 1 (shape (batch, 1, channels, length), matching
    # `signal`'s shape once that axis is squeezed out). scipy.io.loadmat
    # squeezes size-1 axes on its own during the save/load round trip
    # (verified directly: a (4,1,2,16) array round-trips to (4,2,16)), so the
    # loaded arrays already match `signal`'s shape with no further indexing.
    # Handle both cases defensively in case a caller passes in an array that
    # was never round-tripped through .mat (e.g. loaded some other way).
    def _squeeze_to_match(arr):
        arr = np.asarray(arr)
        if arr.ndim == signal.ndim + 1:
            arr = arr[:, 0, ...]
        return arr

    baseline = _squeeze_to_match(baselines) if baselines is not None else None
    res_water = _squeeze_to_match(residual_water) if residual_water is not None else None

    return remove_nuisance(signal, baseline, res_water)
