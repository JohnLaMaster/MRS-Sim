"""
Metabolite/moiety physiological range access
(v2.0 handover, prep for section 18).

Wires `src/basis_sets/metabolites_database.json`'s per-metabolite ranges
(``Conc``, ``T2.metab``, ``T2.spins``, ``omega``) up as accessible,
config-overridable default sampling ranges -- at both the metabolite
level (``Conc``/``T2.metab``, already consumed by
``PhysicsModel.define_parameter_ranges()`` for the amplitude and
Lorentzian-linewidth ``'d'``/``'dmm'`` columns) and the moiety/spin level
(``T2.spins``/``omega``).

The moiety-level values are **not yet consumed anywhere in the forward
simulation** -- individual-spin ("separated moieties") basis support is
explicitly unfinished (see docs/v2/architecture_v1_audit.md section 6 and
the handover doc's new section 18, which the repo owner has deferred
finishing for now). This module makes that data loadable, validated, and
overridable ahead of that pipeline work, per the repo owner's explicit
request, without wiring up individual-spin simulation itself.

Also validates the database's internal consistency -- per handover
section 18.7 ("[moiety] ordering is part of the database contract") and
18.11 ("moiety ordering is unchanged" as an explicit test requirement).
Running ``validate_database()`` against the current database surfaced 3
pre-existing moiety-array length mismatches (``glc``, ``naa``, ``try`` --
see docs/v2/progress_log.md for the exact values). These are **not**
silently corrected here: guessing which array element is wrong or extra
would be fabricating data the repo owner needs to resolve themselves.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

__all__ = [
    'MoietyRangeError',
    'validate_database',
    'get_concentration_range',
    'get_t2_range',
    'get_omega',
    'apply_range_overrides',
]


class MoietyRangeError(ValueError):
    """Raised for a missing or internally-inconsistent database entry."""


def _entry(db: dict, metabolite: str) -> dict:
    key = metabolite.lower()
    if key not in db:
        raise MoietyRangeError(f"'{metabolite}' is not in the metabolite database.")
    return db[key]


def validate_database(db: dict) -> List[str]:
    """
    Check every metabolite entry for internal consistency.

    Returns a list of human-readable problem descriptions (empty if none
    found) rather than raising, so a caller can choose to warn, log, or
    fail based on the result.
    """
    problems = []
    for name, entry in db.items():
        omega = entry.get('omega')
        t2 = entry.get('T2', {}) if isinstance(entry, dict) else {}
        spins = t2.get('spins') if isinstance(t2, dict) else None
        if omega is not None and spins is not None:
            lengths = {'omega': len(omega)}
            for bound in ('min', 'max'):
                if bound in spins:
                    lengths[f'T2.spins.{bound}'] = len(spins[bound])
            if len(set(lengths.values())) > 1:
                problems.append(f"'{name}': moiety array length mismatch -- {lengths}")
    return problems


def get_concentration_range(db: dict, metabolite: str) -> Tuple[float, float]:
    """Metabolite-level ``(min, max)`` concentration range."""
    entry = _entry(db, metabolite)
    conc = entry.get('Conc')
    if not conc:
        raise MoietyRangeError(f"'{metabolite}' has no 'Conc' entry in the database.")
    return float(conc['min'][0]), float(conc['max'][0])


def get_t2_range(db: dict, metabolite: str, level: str = 'metab') -> Tuple[List[float], List[float]]:
    """
    T2 range(s) for ``metabolite``.

    ``level='metab'`` (default): the single combined metabolite-level
    range -- matches what ``PhysicsModel.define_parameter_ranges()``
    already uses for the sampled Lorentzian linewidth ``'d'``/``'dmm'``
    columns.

    ``level='spins'``: one ``[min, max]`` pair per moiety/spin, in the
    same order as ``get_omega()`` (the database's ordering contract).
    Raises ``MoietyRangeError`` if the entry has no per-moiety T2
    breakdown, or if ``T2.spins.min``/``T2.spins.max`` have inconsistent
    lengths (per handover section 18.7: do not fabricate independent
    moiety-specific distributions where the source does not provide
    them -- this surfaces the inconsistency as an error instead of
    silently guessing which value to drop).
    """
    if level not in ('metab', 'spins'):
        raise ValueError(f"level must be 'metab' or 'spins', got {level!r}.")
    entry = _entry(db, metabolite)
    t2 = entry.get('T2', {})
    block = t2.get(level)
    if not block:
        raise MoietyRangeError(f"'{metabolite}' has no T2.{level} entry in the database.")
    mins, maxs = list(block['min']), list(block['max'])
    if level == 'spins' and len(mins) != len(maxs):
        raise MoietyRangeError(
            f"'{metabolite}': T2.spins.min has {len(mins)} entries but "
            f"T2.spins.max has {len(maxs)} -- inconsistent database entry, "
            f"refusing to guess which is correct. See "
            f"docs/v2/progress_log.md for known cases."
        )
    return mins, maxs


def get_omega(db: dict, metabolite: str) -> List[float]:
    """Per-moiety chemical shift (ppm), in the database's fixed moiety order."""
    entry = _entry(db, metabolite)
    omega = entry.get('omega')
    if omega is None:
        raise MoietyRangeError(f"'{metabolite}' has no 'omega' entry in the database.")
    return list(omega)


def apply_range_overrides(db: dict, overrides: dict) -> dict:
    """
    Return a new database dict with ``overrides`` deep-merged on top of
    ``db``, without mutating either input.

    ``overrides`` uses the same nested shape as the database itself, e.g.::

        {"naa": {"Conc": {"min": [0.05], "max": [2.0]}}}

    overrides just NAA's concentration range while leaving everything
    else (including NAA's own T2/omega) untouched. Mirrors
    ``PhysicsModel.set_parameter_constraints()``'s existing "config
    overrides database defaults" pattern for ``Conc``/``T2.metab``,
    extended to also cover ``T2.spins``/``omega`` for when moiety-level
    sampling is wired up.
    """
    merged = copy.deepcopy(db)

    def _merge(base: dict, override: dict):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                _merge(base[key], value)
            else:
                base[key] = value

    for metabolite, override_entry in overrides.items():
        key = metabolite.lower()
        merged.setdefault(key, {})
        _merge(merged[key], override_entry)

    return merged
