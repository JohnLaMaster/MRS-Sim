"""
Unit tests for src.metabolite_database (prep for v2.0 handover section 18).
"""
import json

import pytest

from src.metabolite_database import (
    MoietyRangeError,
    apply_range_overrides,
    get_concentration_range,
    get_omega,
    get_t1_range,
    get_t2_range,
    validate_database,
)

GABA_LIKE = {
    'gaba': {
        'omega': [2.284, 2.284, 1.889, 1.889, 3.0128, 3.0128],
        'T2': {
            'spins': {'min': [25, 25, 25, 25, 25, 25], 'max': [229, 229, 229, 229, 229, 229]},
            'metab': {'min': [77.37], 'max': [161.77]},
        },
        # Matches the real database's current state: T1 is a schema
        # placeholder (null/null) until real values are supplied.
        'T1': {
            'spins': {'min': None, 'max': None},
            'metab': {'min': None, 'max': None},
        },
        'Conc': {'min': [0.033], 'max': [10.720]},
    },
    'broken': {
        'omega': [1.0, 2.0, 3.0],
        'T2': {
            'spins': {'min': [10, 10, 10], 'max': [20, 20]},  # mismatched lengths
            'metab': {'min': [50], 'max': [100]},
        },
        'T1': {
            'spins': {'min': [1000, 1000, 1000], 'max': [1400, 1400]},  # mismatched lengths
            'metab': {'min': [1200], 'max': [1400]},
        },
        'Conc': {'min': [0.1], 'max': [1.0]},
    },
    'has_real_t1': {
        'omega': [2.0],
        'T2': {'spins': {'min': [200], 'max': [300]}, 'metab': {'min': [200], 'max': [300]}},
        'T1': {'spins': {'min': [1100], 'max': [1500]}, 'metab': {'min': [1100], 'max': [1500]}},
        'Conc': {'min': [0.1], 'max': [1.0]},
    },
}


def test_get_concentration_range():
    lo, hi = get_concentration_range(GABA_LIKE, 'gaba')
    assert (lo, hi) == (0.033, 10.720)


def test_get_concentration_range_case_insensitive():
    lo, hi = get_concentration_range(GABA_LIKE, 'GABA')
    assert (lo, hi) == (0.033, 10.720)


def test_get_concentration_range_unknown_metabolite_raises():
    with pytest.raises(MoietyRangeError):
        get_concentration_range(GABA_LIKE, 'not_a_real_metabolite')


def test_get_t2_range_metab_level():
    mins, maxs = get_t2_range(GABA_LIKE, 'gaba', level='metab')
    assert mins == [77.37]
    assert maxs == [161.77]


def test_get_t2_range_spins_level_matches_omega_order():
    omega = get_omega(GABA_LIKE, 'gaba')
    mins, maxs = get_t2_range(GABA_LIKE, 'gaba', level='spins')
    assert len(omega) == len(mins) == len(maxs) == 6


def test_get_t2_range_invalid_level_raises():
    with pytest.raises(ValueError):
        get_t2_range(GABA_LIKE, 'gaba', level='bogus')


def test_get_t2_range_spins_length_mismatch_raises_not_guesses():
    with pytest.raises(MoietyRangeError):
        get_t2_range(GABA_LIKE, 'broken', level='spins')


def test_get_t1_range_raises_for_unpopulated_placeholder():
    """The real database's current state: T1 exists as a schema
    placeholder (min/max: null) for every metabolite. Must raise rather
    than return/fabricate a range until real values are supplied."""
    with pytest.raises(MoietyRangeError):
        get_t1_range(GABA_LIKE, 'gaba', level='metab')
    with pytest.raises(MoietyRangeError):
        get_t1_range(GABA_LIKE, 'gaba', level='spins')


def test_get_t1_range_metab_level_once_populated():
    mins, maxs = get_t1_range(GABA_LIKE, 'has_real_t1', level='metab')
    assert mins == [1100]
    assert maxs == [1500]


def test_get_t1_range_invalid_level_raises():
    with pytest.raises(ValueError):
        get_t1_range(GABA_LIKE, 'has_real_t1', level='bogus')


def test_get_t1_range_spins_length_mismatch_raises_not_guesses():
    with pytest.raises(MoietyRangeError):
        get_t1_range(GABA_LIKE, 'broken', level='spins')


def test_get_t1_range_missing_t1_block_entirely_raises():
    """A metabolite with no 'T1' key at all (e.g. an override dict that
    only ever touched Conc/T2) must raise the same way as an unpopulated
    placeholder, not KeyError."""
    db = {'x': {'T2': {'metab': {'min': [1], 'max': [2]}}, 'Conc': {'min': [0], 'max': [1]}}}
    with pytest.raises(MoietyRangeError):
        get_t1_range(db, 'x', level='metab')


def test_get_omega_missing_raises():
    db = {'x': {'T2': {'metab': {'min': [1], 'max': [2]}}, 'Conc': {'min': [0], 'max': [1]}}}
    with pytest.raises(MoietyRangeError):
        get_omega(db, 'x')


def test_validate_database_flags_length_mismatch():
    problems = validate_database(GABA_LIKE)
    assert len(problems) == 1
    assert 'broken' in problems[0]


def test_validate_database_clean_db_has_no_problems():
    clean = {'gaba': GABA_LIKE['gaba']}
    assert validate_database(clean) == []


def test_validate_database_against_real_database_finds_known_issues():
    """
    Regression test: as of this writing, the real database has 3 known
    moiety-array length mismatches (glc, naa, try) -- see
    docs/v2/progress_log.md. This test pins that known state so a future
    fix to the database (or a newly introduced one) is visible here rather
    than silently passing or failing.
    """
    with open('src/basis_sets/metabolites_database.json') as f:
        db = json.load(f)
    problems = validate_database(db)
    flagged = {p.split("'")[1] for p in problems}
    assert flagged == {'glc', 'naa', 'try'}, (
        f"Expected exactly the known mismatches {{'glc', 'naa', 'try'}}, got {flagged}. "
        f"If you just fixed the database, update this test to reflect the new state."
    )


def test_real_database_has_t1_schema_placeholder_for_every_metabolite():
    """
    Regression test for the T1 schema-space addition (repo owner request,
    docs/v2/progress_log.md Milestone 15): every metabolite must have a
    'T1' block shaped like 'T2' (spins/metab, each with min/max), with
    null values until real data is supplied -- and get_t1_range() must
    raise (not silently return something) for every one of them right now.
    """
    with open('src/basis_sets/metabolites_database.json') as f:
        db = json.load(f)
    for name, entry in db.items():
        assert 'T1' in entry, f"'{name}' has no T1 entry"
        assert entry['T1'] == {
            'spins': {'min': None, 'max': None},
            'metab': {'min': None, 'max': None},
        }, f"'{name}': unexpected T1 placeholder shape {entry['T1']!r}"
        with pytest.raises(MoietyRangeError):
            get_t1_range(db, name, level='metab')


def test_apply_range_overrides_overrides_only_specified_fields():
    overridden = apply_range_overrides(GABA_LIKE, {'gaba': {'Conc': {'min': [0.5], 'max': [5.0]}}})
    lo, hi = get_concentration_range(overridden, 'gaba')
    assert (lo, hi) == (0.5, 5.0)
    # T2/omega untouched
    assert get_omega(overridden, 'gaba') == get_omega(GABA_LIKE, 'gaba')


def test_apply_range_overrides_does_not_mutate_original():
    original_conc = get_concentration_range(GABA_LIKE, 'gaba')
    apply_range_overrides(GABA_LIKE, {'gaba': {'Conc': {'min': [9.0], 'max': [9.9]}}})
    assert get_concentration_range(GABA_LIKE, 'gaba') == original_conc


def test_apply_range_overrides_can_add_a_new_metabolite():
    overridden = apply_range_overrides(GABA_LIKE, {'newmet': {'Conc': {'min': [1.0], 'max': [2.0]}}})
    assert get_concentration_range(overridden, 'newmet') == (1.0, 2.0)
