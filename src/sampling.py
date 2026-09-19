"""
Composable parameter samplers for MRS-Sim (v2.0 handover, section 2).

A sampler owns all stochastic parameter generation for a simulation and
returns a fully-populated ``SimulationParameters`` object; ``PhysicsModel``
only ever *consumes* already-sampled parameters (see
docs/v2/architecture_v1_audit.md, section 1/12 for how this replaces the
per-script sampling code in sim_COWS.py / mrs-sim_template.py).

Two backends are provided:

- ``UniformRangeSampler``: draws every column uniformly in [0, 1] and
  de-normalizes with ``PhysicsModel.quantify_params`` (the pattern used by
  mrs-sim_template.py / deep_learning_dataset_template.py).
- ``CopulaInVivoSampler``: draws (a subset of) columns from a Gaussian
  copula fit to real in-vivo spectral-fitting results, preserving their
  inter-parameter correlations (the pattern used by sim_COWS.py /
  src/aux/sample_from_fitted_dist.py), and falls back to
  ``UniformRangeSampler``-style sampling for every column the copula does
  not cover.

Neither sampler applies dataset-specific overrides (e.g. "Cr concentration
is always 1.0", "zero out B0 columns because b0 is disabled in this
config") -- those remain explicit, visible steps in the calling script, as
they are today, because they are genuinely per-dataset choices (see the
"This next section of code will need to be customized for your own
implementations" comments in mrs-sim_template.py / sim_COWS.py) rather than
something a generic sampler can decide.

Neither sampler relies on global RNG state (``torch.manual_seed`` /
``numpy.random.seed``): each instance owns an explicit ``torch.Generator``
and ``numpy.random.Generator`` seeded independently, so enabling/sampling
one component never perturbs another's random stream.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import stats
from scipy.io import loadmat
from scipy.linalg import eigh

from .parameters import ParameterRegistry, SimulationParameters

__all__ = [
    'ParameterSampler',
    'UniformRangeSampler',
    'CopulaInVivoSampler',
]


def _seed_pair(seed: Optional[int]) -> Tuple[torch.Generator, np.random.Generator]:
    """
    Derive an independent torch.Generator and numpy Generator from one
    integer seed (or from fresh entropy if seed is None), so a sampler's
    random stream never depends on / mutates global RNG state.
    """
    ss = np.random.SeedSequence(seed)
    np_seed, torch_seed = ss.generate_state(2)
    g = torch.Generator()
    g.manual_seed(int(torch_seed))
    return g, np.random.default_rng(np_seed)


class ParameterSampler(ABC):
    """
    Base class for a composable, config-driven parameter sampler.

    Subclasses must implement :meth:`sample`. ``self.registry`` is built
    once from the supplied ``PhysicsModel`` and reused for every call to
    :meth:`sample`, so repeated sampling does not repeatedly rebuild it.
    """

    def __init__(self, pm, seed: Optional[int] = None):
        self.pm = pm
        self.registry = ParameterRegistry.from_physics_model(pm)
        self.seed = seed
        self._torch_gen, self._np_rng = _seed_pair(seed)

    @abstractmethod
    def sample(self, batch_size: int) -> SimulationParameters:
        ...

    def _empty_tensor(self, batch_size: int) -> torch.Tensor:
        n_columns = self.registry.n_columns()
        return torch.empty((batch_size, n_columns), dtype=torch.float32)


class UniformRangeSampler(ParameterSampler):
    """
    Draws every parameter column uniformly at random within the ranges
    configured on the ``PhysicsModel`` (``pm.min_ranges``/``pm.max_ranges``,
    set via ``initialize()``/``set_parameter_constraints()``), i.e. the
    ``torch.rand -> PhysicsModel.quantify_params`` pattern used today in
    mrs-sim_template.py / deep_learning_dataset_template.py.
    """

    def sample(self, batch_size: int) -> SimulationParameters:
        n_columns = self.registry.n_columns()
        raw = torch.rand((batch_size, n_columns), generator=self._torch_gen, dtype=torch.float32)
        tensor = self.pm.quantify_params(raw)
        return SimulationParameters(
            tensor=tensor,
            registry=self.registry,
            metadata={'sampler': 'UniformRangeSampler', 'seed': self.seed},
        )


def _load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def _nearest_psd(R: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = eigh(R)
    eigvals = np.clip(eigvals, 1e-8, None)
    R_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(R_psd))
    return R_psd / np.outer(d, d)


def _load_correlation_matrix(path: str) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Load a correlation matrix saved by ``findParamDist.py --findCorr``.

    Returns ``(corr, names)``. ``names`` is the ordered list of variable
    names the matrix's rows/columns correspond to, if the file provides one
    (current ``findParamDist.py`` saves this -- see the module docstring's
    note on why positional alignment is fragile). Older correlation-matrix
    files saved before this field existed will return ``names=None``. In
    that case, alignment against ``dist_json_path`` falls back to trusting
    that the JSON's key order already matches the matrix's row/column
    order (the assumption ``src/aux/sample_from_fitted_dist.py`` makes) --
    verify this assumption for any specific legacy file before relying on
    it, since it can silently drift when the JSON was accumulated across
    multiple ``findParamDist.py`` runs (each run's ``dict.update()`` keeps a
    key's original position, while a re-run of ``--findCorr`` recomputes
    the matrix fresh in the *current* iteration order).
    """
    mat = loadmat(path, simplify_cells=True)
    corr = np.asarray(mat['corr'], dtype=np.float64)
    names = None
    if 'names' in mat:
        raw_names = mat['names']
        raw_names = raw_names if isinstance(raw_names, (list, tuple, np.ndarray)) else [raw_names]
        # scipy.io.savemat right-pads a list of strings with spaces to the
        # longest entry's width when storing them as a fixed-width char
        # array; strip that back off on load.
        names = [str(n).strip() for n in raw_names]
    return corr, names


class CopulaInVivoSampler(ParameterSampler):
    """
    Samples parameters from a Gaussian copula fit to real in-vivo spectral-
    fitting results (per-parameter marginals + a correlation matrix of the
    normal-score-transformed data), generalizing
    ``src/aux/sample_from_fitted_dist.sample_from_copula`` /
    ``sim_COWS.py`` into a reusable, config-driven sampler.

    Differences from the ``sim_COWS.py`` implementation this generalizes:

    - Alignment between the distributions JSON and the correlation matrix
      is done **by name** when the correlation-matrix file provides a
      ``names`` array (see ``_load_correlation_matrix``), not by trusting
      positional order or by a hand-maintained index permutation. This
      removes the need for the unexplained magic-number row/column
      reordering that ``sample_from_fitted_dist.sample_from_copula``
      hard-codes for one specific legacy file pair.
    - ``include_params``/``exclude_params`` let a caller select or exclude
      specific fitting parameters (by exact JSON key, e.g. ``"NAA_ampl"``,
      or by family suffix, e.g. ``"lorentzLB"``) from the copula for a
      given simulation run.
    - Any parameter column *not* covered by the (possibly filtered) copula
      is filled by ``UniformRangeSampler`` rather than left as a raw,
      never-quantified ``[0, 1)`` value (which is what happens today in
      sim_COWS.py for any column the copula's JSON does not mention and
      that isn't explicitly overridden afterward -- e.g. phi0/phi1/eddy
      current columns). This is a deliberate behavior improvement over the
      current script, not merely a generalization: those leftover columns
      are inert whenever the corresponding component is disabled by a
      config flag, and correctly quantified (rather than raw noise)
      whenever it is not.
    """

    def __init__(
        self,
        pm,
        dist_json_path: str,
        corr_matrix_path: str,
        name_map: Optional[Mapping[str, str]] = None,
        global_param_map: Optional[Mapping[str, str]] = None,
        suffix_to_ind: Optional[Mapping[str, str]] = None,
        include_params: Optional[Iterable[str]] = None,
        exclude_params: Optional[Iterable[str]] = None,
        legacy_reorder: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(pm, seed=seed)

        self.name_map = {k.lower(): v.lower() for k, v in (name_map or {}).items()}
        self.global_param_map = dict(global_param_map or {})
        self._global_json_to_ind = {v.lower(): k.lower() for k, v in self.global_param_map.items()}
        self.suffix_to_ind = {k.lower(): v.lower() for k, v in (suffix_to_ind or {}).items()}

        dist_db = _load_json(dist_json_path)
        corr, names = _load_correlation_matrix(corr_matrix_path)

        if names is None:
            # Legacy file with no embedded name order. Trust the JSON's key
            # order (this codebase's long-standing assumption -- see
            # sample_from_fitted_dist.py's docstring), or apply an explicit
            # caller-supplied permutation if the matrix is known to have
            # drifted out of sync with the JSON (replaces the old hardcoded
            # magic-number reordering with a documented, per-call
            # parameter instead of a silent, unexplained transformation).
            names = list(dist_db.keys())
            if legacy_reorder is not None:
                corr = corr[np.ix_(legacy_reorder, legacy_reorder)]
            elif corr.shape[0] != len(names):
                raise ValueError(
                    f"corr_matrix has shape {corr.shape} but the JSON at "
                    f"'{dist_json_path}' has {len(names)} entries, and the "
                    f"correlation-matrix file does not embed a 'names' array "
                    f"to align them by name. Pass legacy_reorder=... to "
                    f"specify how to permute/trim the matrix, or regenerate "
                    f"the correlation matrix with the current "
                    f"findParamDist.py (which saves 'names' automatically)."
                )

        if corr.shape[0] != corr.shape[1] or corr.shape[0] != len(names):
            raise ValueError(
                f"corr_matrix has shape {corr.shape}, inconsistent with "
                f"{len(names)} named variables."
            )

        by_name = {n: dist_db[n] for n in names if n in dist_db}
        missing = [n for n in names if n not in dist_db]
        if missing:
            raise KeyError(
                f"Correlation matrix names not found in the distributions "
                f"JSON: {missing[:5]}{'...' if len(missing) > 5 else ''}."
            )

        keep = self._resolve_selection(names, include_params, exclude_params)
        keep_idx = [i for i, n in enumerate(names) if n in keep]
        if not keep_idx:
            raise ValueError("include_params/exclude_params left no parameters to sample.")

        self.names = [names[i] for i in keep_idx]
        self.dist_db = {n: by_name[n] for n in self.names}
        self.corr = _nearest_psd(corr[np.ix_(keep_idx, keep_idx)])

        self._marginals = []
        for n in self.names:
            spec = self.dist_db[n]
            dist_name, kwargs = next(iter(spec.items()))
            if not hasattr(stats, dist_name):
                raise ValueError(f"Unknown scipy distribution '{dist_name}' for '{n}'.")
            self._marginals.append(getattr(stats, dist_name)(**kwargs))

        self._col_targets = [self._resolve_column_target(n) for n in self.names]

    @staticmethod
    def _resolve_selection(
        names: List[str],
        include_params: Optional[Iterable[str]],
        exclude_params: Optional[Iterable[str]],
    ) -> set:
        keep = set(names)
        if include_params is not None:
            include_params = set(include_params)
            keep = {
                n for n in keep
                if n in include_params or n.rsplit('_', 1)[-1] in include_params
            }
        if exclude_params is not None:
            exclude_params = set(exclude_params)
            keep = {
                n for n in keep
                if n not in exclude_params and n.rsplit('_', 1)[-1] not in exclude_params
            }
        return keep

    def _resolve_column_target(self, json_key: str):
        """Resolve one JSON key to a column index (or list of indices for a broadcast group)."""
        index = self.registry.index
        json_key_l = json_key.lower()

        if json_key_l in self._global_json_to_ind:
            ind_key = self._global_json_to_ind[json_key_l]
            col_group = index[ind_key]
            return list(col_group) if isinstance(col_group, tuple) else [col_group]

        if '_' in json_key:
            prefix, suffix = json_key.rsplit('_', 1)
            met_name = self.name_map.get(prefix.lower(), prefix.lower())
            canonical_suffix = self.suffix_to_ind.get(suffix.lower(), suffix.lower())

            if canonical_suffix in index and isinstance(index[canonical_suffix], tuple) \
                    and self.registry.is_metabolite(met_name):
                pos = self.registry.position(met_name)
                return index[canonical_suffix][pos]

            if met_name in index:
                col_idx = index[met_name]
                return col_idx[0] if isinstance(col_idx, tuple) and len(col_idx) == 1 else col_idx

            raise KeyError(
                f"Could not resolve JSON key '{json_key}' to a column index. "
                f"'{prefix}' is not in the registry and is not covered by name_map."
            )

        canonical_key = self.suffix_to_ind.get(json_key_l, json_key_l)
        if canonical_key not in index:
            raise KeyError(
                f"Could not resolve JSON key '{json_key}' to a registry key. "
                f"Add it to suffix_to_ind."
            )
        col_idx = index[canonical_key]
        if isinstance(col_idx, tuple):
            return col_idx[0] if len(col_idx) == 1 else list(col_idx)
        return col_idx

    def sample(self, batch_size: int) -> SimulationParameters:
        # Fill every column with a properly quantified fallback value first,
        # then overwrite whatever the (filtered) copula covers.
        fallback_seed = None if self.seed is None else self.seed + 1
        base = UniformRangeSampler(self.pm, seed=fallback_seed).sample(batch_size)
        tensor = base.tensor

        z = self._np_rng.multivariate_normal(np.zeros(len(self.names)), self.corr, size=batch_size)
        u = stats.norm.cdf(z)
        samples = np.column_stack([m.ppf(u[:, i]) for i, m in enumerate(self._marginals)])

        for i, target in enumerate(self._col_targets):
            x = torch.as_tensor(samples[:, i].astype(np.float32), dtype=tensor.dtype)
            if isinstance(target, list):
                for col in target:
                    tensor[:, col] = x
            else:
                tensor[:, target] = x

        return SimulationParameters(
            tensor=tensor,
            registry=self.registry,
            metadata={
                'sampler': 'CopulaInVivoSampler',
                'seed': self.seed,
                'copula_params': list(self.names),
            },
        )
