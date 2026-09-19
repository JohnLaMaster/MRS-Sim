"""
MRSsynMRS minimum-reporting-standards table export (v2.0 handover,
section 9).

The handover doc points at a specific Google Sheet
(docs.google.com/spreadsheets/d/1NpAGbK50KFFrc_NTz4LdeTN1xhA4pQ5g) as the
MRSsynMRS reporting standard and says: inspect it if accessible, and if
not, do not invent its contents. It *was* accessible from this
environment; the field structure below (section names, field names, and
nesting) is transcribed directly from it, not invented or recalled from
memory.

Per "Synthetic Data in MR Spectroscopy: Current Practices, Applications,
and Considerations" (the paper the handover doc cites elsewhere for the
relaxation equations), this table is meant to be tailored per dataset, not
treated as one fixed schema -- e.g. edited spectra need extra rows (edited
ppm, editing targets) that unedited spectra don't. `extra_sections` is
exactly for that: pass a dict to merge into the 'Pulse Sequence' section
(e.g. `{'Editing included': {...}}`) for whatever this specific dataset
needs beyond the base table. Nothing here infers that automatically --
MRS-Sim's difference-editing support is itself flagged as largely
unimplemented in docs/v2/architecture_v1_audit.md, so there is currently
nothing in `provenance`/`pm` reliable enough to detect editing from.

Many sheet fields describe real scanner acquisition/sequence detail
(voxel size, water suppression, shimming, RF pulse shapes, gradients,
patient population) that MRS-Sim does not model at all -- these are left
`None` explicitly, not guessed, so a filled-in table makes clear what
MRS-Sim can attest to versus what still needs to be supplied by whoever
is documenting the specific dataset (e.g. because it describes the
in-vivo acquisitions a copula sampler's parameters were fit to, which is
external information MRS-Sim has no access to).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .provenance import Provenance

__all__ = ['export_mrssynmrs_table']

_SAMPLER_DISTRIBUTION_DESCRIPTIONS = {
    'UniformRangeSampler': 'Uniform over configured min/max ranges',
    'CopulaInVivoSampler': (
        'Gaussian copula fit to in-vivo spectral-fitting results '
        '(preserves inter-parameter correlation structure)'
    ),
}


def _concentration_ranges(pm) -> Dict[str, List[Optional[float]]]:
    """
    Per-metabolite [min, max] concentration ranges, read from the
    *effective* ranges actually in force (`pm.min_ranges`/`max_ranges`,
    which reflect any `set_parameter_constraints()` override from a
    config's `"parameters"` block) rather than re-reading the raw
    `metabolites_database.json` values, which a config may have overridden.
    """
    names, _ = pm.metab
    out = {}
    for name in names:
        key = name.lower()
        if key not in pm.index:
            continue
        col = pm.index[key]
        if isinstance(col, tuple):
            continue
        lo = pm.min_ranges[0, col].item()
        hi = pm.max_ranges[0, col].item()
        out[name] = [lo, hi]
    return out


def _t2_ranges(pm) -> Dict[str, List[Optional[float]]]:
    """
    Per-metabolite T2 [min, max] ranges (ms), read from
    `metabolites_database.json` via `pm.ranges` -- there is no per-sample
    T2 column in the parameter tensor to check for an override (T2 is not
    sampled directly; it only bounds the sampled Lorentzian linewidth
    range at initialization -- see docs/v2/architecture_v1_audit.md
    section 6).
    """
    names, _ = pm.metab
    out = {}
    for name in names:
        key = name.lower()
        entry = pm.ranges.get(key) if isinstance(getattr(pm, 'ranges', None), dict) else None
        if not entry:
            continue
        t2 = entry.get('T2', {}).get('metab')
        if not t2:
            continue
        lo = t2.get('min')
        hi = t2.get('max')
        out[name] = [
            lo[0] if isinstance(lo, list) else lo,
            hi[0] if isinstance(hi, list) else hi,
        ]
    return out


def export_mrssynmrs_table(
    pm,
    provenance: Provenance,
    config=None,
    sampler=None,
    extra_sections: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a dict matching the MRSsynMRS reporting table's structure for
    one simulated dataset.

    Parameters
    ----------
    pm :
        The PhysicsModel used.
    provenance :
        A `Provenance` record (see `collect_provenance()`).
    config :
        The config `SimpleNamespace` from `mainFcns.prepare()`, if
        available (used for `drop_prob`).
    sampler :
        The `ParameterSampler` used, if any -- determines the
        'Distribution' field and whether an 'Additional Code' entry is
        added for a non-trivial (e.g. copula) sampling method.
    extra_sections :
        Merged into the 'Pulse Sequence' section -- see the module
        docstring for why this table is meant to be tailored per dataset
        (e.g. spectral editing needs extra rows this base table doesn't
        have).
    """
    acq = provenance.acquisition

    sampler_name = type(sampler).__name__ if sampler is not None else None
    distribution = _SAMPLER_DISTRIBUTION_DESCRIPTIONS.get(sampler_name, sampler_name)

    additional_code = []
    if sampler_name == 'CopulaInVivoSampler':
        additional_code.append({
            'Name': 'CopulaInVivoSampler',
            'Repository': 'https://github.com/JohnLaMaster/MRS-Sim (src/sampling.py)',
            'Purpose': distribution,
        })

    drop_prob = getattr(config, 'drop_prob', None) if config is not None else None

    table: Dict[str, Any] = {
        'Experiment': {
            'ID': None,  # assign a dataset-specific identifier when generating a dataset
            'Nucleus': acq.get('nucleus'),
        },
        'Software': {
            'Basis Set Simulation': {
                'Name': acq.get('basis_set_software'),
                'Repository': None,  # not tracked by MRS-Sim
            },
            'Signal Model': {
                'Name': 'MRS-Sim',
                'Repository': 'https://github.com/JohnLaMaster/MRS-Sim',
            },
            'Additional Code': additional_code,
        },
        'Pulse Sequence': {
            'Field Strength [T]': acq.get('field_strength_T'),
            'Localization': acq.get('pulse_sequence'),  # see provenance.py: not available (stripped on load)
            'Spectral Width [Hz]': acq.get('spectral_width_Hz'),
            'Voxel Size (X, Y, Z) [mm]': None,  # not modeled by MRS-Sim
            'Sequence Timing(s)': {
                'Repetition Times [ms]': acq.get('repetition_time_ms'),
                'Echo Time [ms]': acq.get('echo_time_ms'),
                'Mixing Time [ms]': None,   # not modeled by MRS-Sim
                'Inversion Time [ms]': None,  # not modeled by MRS-Sim
            },
            'Water Suppression': {'Method': None, 'Bandwidth [Hz]': None},  # not modeled
            'Shimming Method': None,  # not modeled
            'Radio Frequency Pulse(s)': {'Shape': None, 'Duration [ms]': None, 'Delay [ms]': None},
            'Slice-Select Gradients': {'Shape': None, 'Duration [ms]': None, 'Delay [ms]': None},
            'Crusher/Rephasing Gradients': {'Shape': None, 'Duration [ms]': None, 'Delay [ms]': None},
        },
        'Metabolites': {
            'Population': None,  # not modeled/tracked by MRS-Sim; supply externally if a copula
                                  # sampler's fitting data came from a specific population
            'Region of interest': None,  # not modeled by MRS-Sim
            'Distribution': distribution,
            'Component dropout probability': drop_prob,
            'Units': (
                f'ratio to {pm.wrt_metab}' if getattr(pm, 'wrt_metab', None) else None
            ),
            'Sources of Parameter Values': {
                'Concentration ranges': 'src/basis_sets/metabolites_database.json (Conc field), possibly overridden per-config',
                'T2 relaxation ranges': 'src/basis_sets/metabolites_database.json (T2.metab field)',
                'T1 relaxation ranges': (
                    'not tracked -- T1 relaxation is not implemented in this '
                    'codebase (see docs/v2/architecture_v1_audit.md section 6)'
                ),
            },
            'Concentration Ranges': _concentration_ranges(pm),
            'T2 Relaxation Ranges': _t2_ranges(pm),
        },
    }

    if extra_sections:
        table['Pulse Sequence'].update(extra_sections)

    return table
