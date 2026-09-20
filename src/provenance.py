"""
Reproducibility and provenance (v2.0 handover, section 9).

`collect_provenance()` gathers machine-readable metadata about one
simulation call into a `Provenance` record, covering what the handover doc
asks for "as applicable" to this codebase today. Fields that nothing in
MRS-Sim currently tracks (TR, vendor/pulse-sequence once loaded through
`PhysicsModel` -- see the note in `_acquisition_metadata`) are left `None`
rather than guessed, per "do not silently guess missing acquisition or
basis-set metadata".

This is not itself the MRSsynMRS reporting-table export -- see
`src/mrssynmrs.py` for that; `export_mrssynmrs_table()` there consumes a
`Provenance` record as one of its inputs.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

__all__ = ['Provenance', 'collect_provenance']

# Bumped whenever SimulationParameters/ParameterRegistry's column layout
# semantics change in a way that would break replaying params sampled
# under an older schema against a newer PhysicsModel.
PARAMETER_SCHEMA_VERSION = 'v2.0-params-1'


@dataclass
class Provenance:
    mrs_sim_version: Optional[str] = None
    git_commit: Optional[str] = None
    parameter_schema_version: str = PARAMETER_SCHEMA_VERSION

    basis_set_name: Optional[str] = None
    basis_set_hash: Optional[str] = None

    acquisition: Dict[str, Any] = field(default_factory=dict)
    simulation_config: Dict[str, Any] = field(default_factory=dict)
    enabled_components: Dict[str, bool] = field(default_factory=dict)

    rng_seed: Optional[int] = None
    dtype: Optional[str] = None
    device: Optional[str] = None
    package_versions: Dict[str, str] = field(default_factory=dict)

    processing_history: List[str] = field(default_factory=list)
    snr_definitions: Dict[str, str] = field(default_factory=dict)
    data_state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _git_commit(repo_root: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_root,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _package_versions() -> Dict[str, str]:
    versions = {}
    for name in ('torch', 'numpy', 'scipy'):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            pass
    import sys
    versions['python'] = sys.version.split()[0]
    return versions


def _basis_set_hash(pm) -> Optional[str]:
    """
    A content hash identifying the specific basis functions used, not just
    the filename (two files with the same name can differ; this catches
    that). Hashes `pm.syn_basis_fids` -- the actual data the forward model
    consumes -- not the raw file on disk.
    """
    try:
        import hashlib
        data = pm.syn_basis_fids.detach().cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()[:16]
    except Exception:
        return None


def _acquisition_metadata(pm, config=None) -> Dict[str, Any]:
    """
    Pulls whatever acquisition metadata is actually available.

    `vendor` is read from `config` (the JSON config, e.g. cows.json's
    `"vendor"` field) since `pm.header` never has it.

    `pulse_sequence`/`localization` (e.g. "PRESS", "sLASER") is left `None`
    even though `process_basis_functions.py`'s basis-set compiler *does*
    write a `pulse_sequence` field into a compiled basis set's `.mat`
    header (`build_header_fields()`, e.g.
    `--pulse_sequence 'COWS7_sLASER'`) -- but `aux.convertdict()`, which
    runs when `PhysicsModel` loads that `.mat` file, unconditionally
    deletes `'seq'`/`'vendor'`/`'pulse_sequence'`/`'pulseSequence'` keys
    (verified directly: `pm.header` only ever has `spectralwidth`,
    `carrier_frequency`, `Ns`, `t`, `centerFreq`, `B0`, `TE`,
    `basis_set_software`, `ppm`). This is a real basis-set-metadata gap
    (relevant to handover section 11's basis-set metadata audit, not just
    provenance) -- the pulse-sequence name is written when the basis set
    is compiled and then silently discarded before `PhysicsModel` ever
    sees it. Not guessed here; flagged instead.

    TR is not tracked anywhere in this codebase (see
    docs/v2/architecture_v1_audit.md section 6) -- also left `None`.
    """
    header = getattr(pm, 'header', {}) or {}

    def _scalar(key):
        v = header.get(key)
        if v is None:
            return None
        return v.item() if torch.is_tensor(v) else v

    return {
        'field_strength_T': _scalar('B0'),
        'echo_time_ms': _scalar('TE'),
        'repetition_time_ms': None,  # not tracked anywhere in this codebase
        'spectral_width_Hz': _scalar('spectralwidth'),
        'carrier_frequency_MHz': _scalar('carrier_frequency'),
        'n_points': _scalar('Ns'),
        'nucleus': '1H',  # the only nucleus this codebase models (see NIfTI export)
        'basis_set_software': header.get('basis_set_software'),
        'vendor': getattr(config, 'vendor', None) if config is not None else None,
        'pulse_sequence': None,  # see docstring: stripped by convertdict() before PhysicsModel sees it
    }


def collect_provenance(
    pm,
    config=None,
    enabled_components: Optional[Dict[str, bool]] = None,
    sampler=None,
    processing_history: Optional[List[str]] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Provenance:
    """
    Build a `Provenance` record for one simulation.

    Parameters
    ----------
    pm :
        The `PhysicsModel` instance the simulation used.
    config :
        The (SimpleNamespace) config object from `mainFcns.prepare()`, if
        available -- used for fields not tracked on `pm` itself (vendor).
    enabled_components :
        Dict of which optional forward() components were enabled for this
        call (e.g. `{'b0': False, 'eddy': False, 'noise': True, ...}`).
    sampler :
        The `ParameterSampler` instance used, if any -- records its class
        name and seed.
    processing_history :
        Free-text list describing processing steps applied (e.g.
        `['simulated', 'cropped']`).
    """
    return Provenance(
        mrs_sim_version=None,  # this repo has no package __version__ yet
        git_commit=_git_commit(),
        basis_set_name=getattr(pm, 'PM_basis_set', None),
        basis_set_hash=_basis_set_hash(pm),
        acquisition=_acquisition_metadata(pm, config),
        simulation_config={
            'wrt_metab': getattr(pm, 'wrt_metab', None),
            'snr_metab': getattr(pm, 'snr_metab', None),
            'lineshape_type': getattr(pm, 'lineshape_type', None),
            'crop_range': getattr(pm, 'cropRange', None),
        },
        enabled_components=dict(enabled_components or {}),
        rng_seed=getattr(sampler, 'seed', None) if sampler is not None else None,
        dtype=str(dtype) if dtype is not None else None,
        device=str(device) if device is not None else None,
        package_versions=_package_versions(),
        processing_history=list(processing_history or []),
        snr_definitions={
            'target_snr': (
                'Sampled per-sample target SNR (unitless ratio, per MRS '
                'expert-consensus convention), stored in '
                'params[:, index["snr"]] -- referenced against the peak '
                'amplitude of pm.snr_metab (defaults to wrt_metab). '
                'generate_noise() previously applied an incorrect '
                'decibel-style log conversion to this value before using '
                'it; removed as of the handover section 7 audit (see '
                'docs/v2/progress_log.md) -- the stored value is used '
                'directly as the linear ratio.'
            ),
            'realized_snr': (
                'Computed post-noise-generation from the actual drawn noise '
                'realization\'s measured std, not the target: power/spectral '
                'SNR (unitless ratio) referenced against pm.snr_metab '
                '(defaults to wrt_metab). Computed from each basis-function '
                'line\'s own pre-baseline/pre-multicoil/pre-phase clean '
                'signal, not from the final noisy/spectrum output. See '
                'src/simulation_result.py\'s field comments and '
                'docs/v2/progress_log.md (handover section 7 audit) for the '
                'full definition, including why "power" (pSNR, a time-domain '
                't=0 amplitude) and "spectral" (sSNR, a frequency-domain peak '
                'height) are not the same kind of quantity despite sharing a '
                'denominator.'
            ),
        },
        data_state={
            'acquired': True,
            'cropped': getattr(pm, 'cropRange', None) is not None,
            'resampled': bool(enabled_components.get('resample')) if enabled_components else None,
            'zero_filled': bool(enabled_components.get('zero_fill')) if enabled_components else None,
            'filtered': False,
        },
    )
