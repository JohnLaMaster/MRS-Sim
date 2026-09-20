"""
Unit tests for src.aux.process_basis_functions.build_header_fields
(v2.0 handover section 11: basis-set metadata / double-application audit).
"""
import numpy as np

from src.aux.process_basis_functions import build_header_fields


def _base_config():
    return {
        'centerFreq': 4.65, 'B0': 3.0, 'TE': 30.0,
        'pulse_sequence': 'PRESS', 'vendor': 'GE',
    }


def test_dwelltime_is_stored_as_its_own_field():
    """Regression: dt was already computed to build `t`, but never saved
    as its own header field -- callers had to re-derive it elsewhere
    (e.g. mat2niftimrs.py's dwelltime-missing bugfix)."""
    header_info = {'sw': 2000.0, 'sf': 127.7e6, 'ns': 8192}
    header = build_header_fields({}, header_info, _base_config())
    assert header['dwelltime'] == 1.0 / 2000.0


def test_pre_existing_linewidth_captures_loader_reported_value():
    """Regression: load_marss_mat/load_fsl_mrs_basis_dir already compute
    header_info['lw'] (the source simulator's own applied broadening),
    but build_header_fields used to silently drop it."""
    header_info = {'sw': 2000.0, 'sf': 127.7e6, 'ns': 8192, 'lw': 1.0}
    header = build_header_fields({}, header_info, _base_config())
    assert header['pre_existing_linewidth_hz'] == 1.0


def test_pre_existing_linewidth_defaults_to_zero_when_not_reported():
    """0.0 here means 'not reported by this loader', not 'confirmed zero
    broadening' -- documented in build_header_fields' own comment."""
    header_info = {'sw': 2000.0, 'sf': 127.7e6, 'ns': 8192}
    header = build_header_fields({}, header_info, _base_config())
    assert header['pre_existing_linewidth_hz'] == 0.0


def test_te_and_tr_decay_flags_are_always_false():
    """Every loader in this file starts `t` at 0 with no pre-echo offset
    and never references TR/T1 -- so a basis FID never has excitation-to
    -echo decay or TR-dependent T1 saturation already applied."""
    header_info = {'sw': 2000.0, 'sf': 127.7e6, 'ns': 8192}
    header = build_header_fields({}, header_info, _base_config())
    assert header['te_decay_applied'] is False
    assert header['tr_relaxation_applied'] is False


def test_time_axis_still_starts_at_zero():
    header_info = {'sw': 2000.0, 'sf': 127.7e6, 'ns': 100}
    header = build_header_fields({}, header_info, _base_config())
    assert header['t'][0] == 0.0
    assert np.isclose(header['t'][1] - header['t'][0], 1.0 / 2000.0)
