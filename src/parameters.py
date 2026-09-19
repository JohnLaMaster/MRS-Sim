"""
Human-readable parameter representation for MRS-Sim's simulation parameters.

This module formalizes access to the flat ``[batch, n_columns]`` parameter
tensor that ``PhysicsModel`` already uses, and to its existing
``PhysicsModel.index`` mapping (semantic name -> column index/indices), into a
lightweight registry and a ``SimulationParameters`` container. Per
docs/v2/architecture_v1_audit.md section 1, this does *not* introduce a new
storage layout: ``SimulationParameters.tensor`` is exactly the tensor
``PhysicsModel.forward()`` already consumes, and ``ParameterRegistry`` wraps
``PhysicsModel.index`` rather than replacing it.

Baseline and residual-water generation parameters (from
``aux.sample_baselines``/``aux.sample_resWater``) are not columns of this
tensor -- they are produced by an independent stochastic process with a
different per-sample convention (``[N, 1, 1]`` dicts) and are not yet
represented in ``PhysicsModel.index``. ``SimulationParameters`` carries them
alongside the tensor so that a full simulation's parameters can be inspected,
saved, and replayed as one object, without forcing them into the physical
model's column layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch

__all__ = ['ParameterRegistry', 'MetaboliteParameterView', 'SimulationParameters']

IndexValue = Union[int, Tuple[int, ...]]

# Per-line parameter families: one column per basis-function line (metabolite
# or MM/Lip), in the same order as PhysicsModel._metab / ParameterRegistry
# .metabolite_names. Maps the human-readable accessor name to the ind[] key.
# Extend this mapping as new per-line families are added (e.g. T1/T2 once
# relaxation is implemented -- see docs/v2/architecture_v1_audit.md section 6).
_PER_LINE_FAMILIES = {
    'lorentzian': 'd',
    'gaussian': 'g',
    'frequency_shift': 'f_shifts',
}

# Attributes the v2.0 handover's SimulationParameters sketch names that are
# not yet represented anywhere in the parameter tensor. Looked up explicitly
# so callers get a clear, actionable error instead of a wrong/missing value.
_NOT_YET_MODELED = {
    't1': (
        "T1 relaxation is not yet modeled in MRS-Sim "
        "(see docs/v2/architecture_v1_audit.md, section 6/10)."
    ),
    't2': (
        "T2 is currently only used to bound the sampled 'lorentzian' linewidth "
        "range at initialization; it is not applied as first-principles "
        "relaxation and has no per-sample column "
        "(see docs/v2/architecture_v1_audit.md, section 6)."
    ),
}

# Keys in PhysicsModel.index that are aggregates over other columns, not
# individual parameters -- excluded from per-column labeling.
_AGGREGATE_KEYS = {'metabolites', 'parameters', 'overall'}


class MetaboliteParameterView:
    """
    Read-only, per-metabolite (or per-MM/Lip-line) view into a parameter
    tensor. Supports both attribute and item access::

        view.concentration       == view['concentration']
        view.lorentzian          == view['lorentzian']
        view.frequency_shift     == view['frequency_shift']

    Every accessor indexes into the shared tensor; nothing is copied.
    """

    __slots__ = ('_tensor', '_index', '_name', '_pos')

    def __init__(self, tensor: torch.Tensor, index: Dict[str, IndexValue], name: str, pos: int):
        self._tensor = tensor
        self._index = index
        self._name = name  # lowercase name, also the ind[] key for concentration
        self._pos = pos    # this line's position among all basis-function lines

    def _line_column(self, family: str) -> int:
        col_group = self._index[family]
        if not isinstance(col_group, tuple):
            raise KeyError(
                f"ind['{family}'] is not a per-line tuple; cannot index it by "
                f"metabolite/line position."
            )
        return col_group[self._pos]

    @property
    def concentration(self) -> torch.Tensor:
        return self._tensor[:, self._index[self._name]]

    @property
    def lorentzian(self) -> torch.Tensor:
        return self._tensor[:, self._line_column(_PER_LINE_FAMILIES['lorentzian'])]

    @property
    def gaussian(self) -> torch.Tensor:
        return self._tensor[:, self._line_column(_PER_LINE_FAMILIES['gaussian'])]

    @property
    def frequency_shift(self) -> torch.Tensor:
        return self._tensor[:, self._line_column(_PER_LINE_FAMILIES['frequency_shift'])]

    def __getitem__(self, key: str) -> torch.Tensor:
        key_l = key.lower()
        if key_l == 'concentration':
            return self.concentration
        if key_l in _PER_LINE_FAMILIES:
            return self._tensor[:, self._line_column(_PER_LINE_FAMILIES[key_l])]
        if key_l in _NOT_YET_MODELED:
            raise NotImplementedError(_NOT_YET_MODELED[key_l])
        raise KeyError(
            f"'{key}' is not a recognized per-metabolite parameter for "
            f"'{self._name}'. Known: concentration, {', '.join(_PER_LINE_FAMILIES)}."
        )

    def __repr__(self) -> str:
        return f"MetaboliteParameterView({self._name!r}, pos={self._pos})"


class ParameterRegistry:
    """
    Semantic-name -> tensor-column registry for a ``PhysicsModel``'s
    parameter tensor.

    Holds no tensor data itself -- only the index mapping and the ordered
    list of basis-function-line names -- so the same registry can label
    CRLB/FIM axes for a computation that has no single "current" sampled
    batch. See ``SimulationParameters`` for the tensor-holding counterpart
    that supports ``params["NAA"]["concentration"]``-style access.
    """

    def __init__(self, index: Dict[str, IndexValue], metabolite_names: List[str]):
        self.index: Dict[str, IndexValue] = dict(index)
        self.metabolite_names: List[str] = list(metabolite_names)
        self._metabolite_pos = {m.lower(): i for i, m in enumerate(self.metabolite_names)}

    @classmethod
    def from_physics_model(cls, pm) -> 'ParameterRegistry':
        """Build a registry from an initialized ``PhysicsModel`` instance."""
        metab_names, _ = pm.metab
        return cls(index=pm.index, metabolite_names=metab_names)

    def is_metabolite(self, name: str) -> bool:
        return name.lower() in self._metabolite_pos

    def position(self, metabolite: str) -> int:
        """Position of ``metabolite`` among all basis-function lines (metabolites + MM/Lip)."""
        return self._metabolite_pos[metabolite.lower()]

    def n_columns(self) -> int:
        overall = self.index['overall']
        return len(overall) if isinstance(overall, tuple) else 1

    def labels(self) -> List[str]:
        """
        One semantic label per tensor column, ordered to match column index
        (e.g. ``['naa.concentration', 'naa.d', ..., 'snr', 'phi0', ...]``).
        Used to make CRLB/FIM dimensions interpretable.
        """
        n = self.n_columns()
        out: List[Optional[str]] = [None] * n

        per_line_ind_keys = set(_PER_LINE_FAMILIES.values())

        for name, cols in self.index.items():
            if name in _AGGREGATE_KEYS:
                continue
            if not isinstance(cols, tuple) and name in self._metabolite_pos:
                out[cols] = f'{name}.concentration'
                continue
            if isinstance(cols, tuple):
                if name in per_line_ind_keys and len(cols) == len(self.metabolite_names):
                    for pos, col in enumerate(cols):
                        out[col] = f'{self.metabolite_names[pos]}.{name}'
                else:
                    for k, col in enumerate(cols):
                        out[col] = f'{name}[{k}]'
            else:
                out[cols] = name

        for i, label in enumerate(out):
            if label is None:
                out[i] = f'column[{i}]'
        return out

    def __contains__(self, key: str) -> bool:
        return key.lower() in self.index or self.is_metabolite(key)

    def __repr__(self) -> str:
        return f"ParameterRegistry({len(self.index)} names, {len(self.metabolite_names)} lines)"


@dataclass
class SimulationParameters:
    """
    Human-readable, replayable representation of one sampled batch of
    simulation parameters.

    ``tensor`` is exactly the flat ``[batch, n_columns]`` tensor
    ``PhysicsModel.forward()`` already consumes -- no new storage layout is
    introduced. ``baseline``/``residual_water``, when present, are the dicts
    produced by ``aux.sample_baselines``/``aux.sample_resWater``; see the
    module docstring for why they are kept alongside rather than merged in.
    """

    tensor: torch.Tensor
    registry: ParameterRegistry
    baseline: Optional[Dict[str, torch.Tensor]] = None
    residual_water: Optional[Dict[str, torch.Tensor]] = None
    metadata: dict = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, key: str):
        """
        ``params["NAA"]["concentration"]``, ``params["NAA"]["lorentzian"]``,
        etc. for basis-function lines; ``params["snr"]`` for flat,
        non-per-line families.
        """
        key_l = key.lower()
        if self.registry.is_metabolite(key_l):
            return MetaboliteParameterView(
                self.tensor, self.registry.index, key_l, self.registry.position(key_l)
            )
        if key_l not in self.registry.index:
            raise KeyError(f"'{key}' is not present in this SimulationParameters' registry.")
        return self.tensor[:, self.registry.index[key_l]]

    def __contains__(self, key: str) -> bool:
        return key in self.registry

    def labels(self) -> List[str]:
        return self.registry.labels()

    def clone(self) -> 'SimulationParameters':
        """An independent copy safe to mutate without affecting the original (e.g. for parameter replay)."""
        def _clone_dict(d):
            if d is None:
                return None
            return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()}

        return SimulationParameters(
            tensor=self.tensor.clone(),
            registry=self.registry,
            baseline=_clone_dict(self.baseline),
            residual_water=_clone_dict(self.residual_water),
            metadata=dict(self.metadata),
        )
