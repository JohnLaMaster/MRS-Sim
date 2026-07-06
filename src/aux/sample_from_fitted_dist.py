import json
from typing import Optional, Mapping

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.linalg import eigh
from scipy.io import loadmat

from .aux import order_metab, gaussian_copula_sample


__all__ = ['populate_params_from_distributions', 'sample_from_copula']


def _sample_scipy_spec(spec, n, rng):
    """
    spec looks like:
        {"gamma": {"a": ..., "loc": ..., "scale": ...}}
    """
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ValueError(f"Expected one scipy distribution spec, got: {spec}")

    dist_name, kwargs = next(iter(spec.items()))
    if not hasattr(stats, dist_name):
        raise ValueError(f"Unknown scipy distribution: {dist_name}")

    dist = getattr(stats, dist_name)(**kwargs)
    x = dist.rvs(size=n, random_state=rng)
    return np.asarray(x, dtype=np.float32).reshape(n)


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _infer_metabolite_keys(ind):
    """
    Returns metabolite names in the order they appear in ind.
    Example:
        ['asc', 'asp', 'ch', 'cr', ...]
    """
    metabolite_indices = set(ind.get("metabolites", ()))
    return [
        k for k, v in ind.items()
        if isinstance(v, int) and v in metabolite_indices
    ]


def nearest_psd(R):
    eigvals, eigvecs = eigh(R)
    eigvals = np.clip(eigvals, 1e-8, None)  # zero out negatives
    R_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # re-normalize to correlation matrix
    d = np.sqrt(np.diag(R_psd))
    return R_psd / np.outer(d, d)


def sample_from_copula(
    ind,
    params,
    dist_json_path: str,
    corr_matrix: str,
    name_map: Optional[Mapping[str, str]] = None,
    global_param_map: Optional[Mapping[str, str]] = None,
    suffix_to_ind: Optional[Mapping[str, str]] = None,
    seed: Optional[int] = None,
):
    """
    Sample all parameters jointly from a Gaussian copula.

    Unlike ``populate_params_from_distributions``, which draws each parameter
    independently from its fitted marginal, this function preserves the
    inter-parameter correlation structure by embedding those marginals inside a
    Gaussian copula whose Gaussian correlation matrix ``R`` is derived from the
    Spearman rank correlation matrix ``corr_matrix`` via the standard relation

        R_ij = 2 sin(pi * rho_s_ij / 6)

    This relation is exact for bivariate-normal margins and is a good
    approximation for general continuous marginals.

    The JSON at ``dist_json_path`` must list variables in the same order as the
    rows and columns of ``corr_matrix``.  Each entry maps a canonical label to a
    single scipy distribution spec, e.g.::

        {
            "Asc_ampl":   {"lognorm":  {"s": ..., "loc": ..., "scale": ...}},
            "Asc_lorentzLB": {"gamma": {"a": ..., "loc": ..., "scale": ...}},
            ...
            "gaussLB":    {"norm":     {"loc": ..., "scale": ...}},
            "ph0":        {"vonmises": {"kappa": ..., "loc": ...}},
        }

    Global parameters (those that appear once in the JSON but govern every
    metabolite column, e.g. ``gaussLB``) are sampled once per draw and then
    broadcast to all corresponding columns in ``params``, consistent with the
    behaviour of ``populate_params_from_distributions``.

    The correlation matrix is computed once externally — typically by
    ``findCorr`` — and passed in here.  ``sample_from_copula`` does not build
    it from data; doing so inside a sampling hot-path would be both slow and
    conceptually wrong.

    Parameters
    ----------
    ind :
        OrderedDict mapping internal parameter names to column indices
        (``int``) or tuples of column indices (``tuple[int, ...]``).
    params :
        Preallocated parameter matrix of shape ``[batch_size, n_params]``.
        Accepts both NumPy arrays and PyTorch tensors; the return type matches
        the input type.
    dist_json_path :
        Path to the distributions JSON.  Keys must be in the same order as the
        rows/columns of ``corr_matrix``; the two must have been produced from
        the same variable set.
    corr_matrix :
        Pre-computed Spearman rank correlation matrix, shape ``[d, d]``, where
        ``d`` is the number of entries in the JSON.  Must be positive
        semi-definite; ``nearest_psd`` is applied as a safety projection before
        use.
    name_map :
        Optional prefix remapping from JSON metabolite prefix to the
        corresponding internal ``ind`` key.  Matching is case-insensitive.
        Example: ``{"crch2": "ch"}``
    global_param_map :
        Optional mapping from internal ``ind`` key to the JSON key for
        parameters that appear once in the JSON but are broadcast to all
        metabolite columns in the group.
        Example: ``{"g": "gaussLB"}``
    suffix_to_ind :
        Optional mapping from the JSON parameter suffix to the internal ``ind``
        key that holds the column indices for that parameter family.
        Example: ``{"lorentzLB": "d", "freqShift": "f_shifts", "ph0": "phi0"}``
    seed :
        Integer seed passed to ``numpy.random.default_rng`` for reproducibility.

    Returns
    -------
    params :
        Same object (and type) as the input, with all sampled values written
        in-place into the appropriate columns.

    Raises
    ------
    ValueError
        If ``corr_matrix`` is not square or its dimension does not match the
        number of JSON entries.
    KeyError
        If a JSON key cannot be resolved to a column index through the supplied
        maps.
    """
    dist_db = _load_json(dist_json_path)
    if len(dist_db) == 0:
        return params
    
    corr_matrix = loadmat(corr_matrix)
    corr_matrix = corr_matrix["corr"]
    corr_matrix = np.delete(corr_matrix, -2, 0)
    corr_matrix = np.delete(corr_matrix, -2, 1)
    n = corr_matrix.shape[0] 
    order = list(range(n)) 
    # Move 26 and 27 to the end 
    for idx in [27, 26]: 
        order.append(order.pop(idx)) 
    # Move the element currently at -3 to before position 52 
    item = order.pop(-3) 
    order.insert(52, item) 
    # Reorder rows and columns together 
    corr_matrix = corr_matrix[np.ix_(order, order)]
    

    # ------------------------------------------------------------------
    # Validate that the correlation matrix is consistent with the JSON.
    # Both must have been produced from the same ordered variable list.
    # ------------------------------------------------------------------
    corr_matrix = np.asarray(corr_matrix, dtype=np.float64)
    d = len(dist_db)
    if corr_matrix.ndim != 2 or corr_matrix.shape != (d, d):
        raise ValueError(
            f"corr_matrix has shape {corr_matrix.shape} but the JSON contains "
            f"{d} entries.  Both must be prepared from the same ordered variable "
            f"list (e.g. via findCorr)."
        )

    # Normalize all user-supplied maps to lowercase for case-insensitive lookup.
    name_map_l        = {k.lower(): v.lower() for k, v in (name_map or {}).items()}
    # global_param_map  : ind_key  -> json_key  (e.g. "g"  -> "gaussLB")
    # global_json_to_ind: json_key -> ind_key   (inverse of the above)
    global_json_to_ind = {
        v.lower(): k.lower() for k, v in (global_param_map or {}).items()
    }
    suffix_to_ind_l = {k.lower(): v.lower() for k, v in (suffix_to_ind or {}).items()}

    is_torch = torch.is_tensor(params)
    out = params.clone() if is_torch else np.array(params, copy=True)
    if out.ndim != 2:
        raise ValueError("params must be 2-D with shape [batch_size, n_params].")
    batch_size = out.shape[0]
    rng = np.random.default_rng(seed)

    metabolite_keys = _infer_metabolite_keys(ind)
    # Map each metabolite name to its positional index within the metabolite list.
    # This index is used to index into tuple-valued families such as ind["d"].
    metabolite_pos = {k.lower(): i for i, k in enumerate(metabolite_keys)}

    # ------------------------------------------------------------------
    # Build two parallel ordered lists that will be consumed after sampling:
    #   marginals   - frozen scipy distributions, one per JSON entry
    #   col_targets - int  for a single column
    #               - list[int] for a broadcast group (global parameters)
    #
    # We iterate dist_db in insertion order, which is guaranteed to match the
    # row/column ordering of corr_matrix.
    # ------------------------------------------------------------------
    marginals:   list = []
    col_targets: list = []

    for json_key, spec in dist_db.items():
        # Construct a frozen scipy distribution from the JSON spec.
        dist_name, dist_kwargs = next(iter(spec.items()))
        if not hasattr(stats, dist_name):
            raise ValueError(f"Unknown scipy distribution '{dist_name}' in JSON key '{json_key}'.")
        marginals.append(getattr(stats, dist_name)(**dist_kwargs))

        json_key_l = json_key.lower()

        # ----------------------------------------------------------
        # Priority 1: global broadcast parameter.
        # The JSON entry is keyed by the raw parameter name (e.g.
        # "gaussLB") and global_json_to_ind maps it to the ind key
        # whose column group should all receive the same draw.
        # ----------------------------------------------------------
        if json_key_l in global_json_to_ind:
            ind_key   = global_json_to_ind[json_key_l]
            col_group = ind[ind_key]
            col_targets.append(
                list(col_group) if isinstance(col_group, tuple) else [col_group]
            )
            continue

        # ----------------------------------------------------------
        # Priority 2: prefixed key  "{metabolite}_{parameter}".
        # Covers amplitudes, Lorentzian linewidths, frequency shifts,
        # and any other per-metabolite per-parameter combination.
        # ----------------------------------------------------------
        if "_" in json_key:
            prefix, suffix = json_key.rsplit("_", 1)
            met_name         = name_map_l.get(prefix.lower(), prefix.lower())
            canonical_suffix = suffix_to_ind_l.get(suffix.lower(), suffix.lower())

            # Tuple-indexed family (e.g. "d", "f_shifts"):
            # ind[canonical_suffix] is a tuple aligned with metabolite_keys,
            # so we index it by the metabolite's positional rank.
            if canonical_suffix in ind and isinstance(ind[canonical_suffix], tuple):
                pos = metabolite_pos.get(met_name)
                if pos is None:
                    raise KeyError(
                        f"Metabolite '{met_name}' (from JSON key '{json_key}') "
                        f"was not found in ind.  Add it to name_map if the prefix "
                        f"does not match the ind key directly."
                    )
                col_targets.append(ind[canonical_suffix][pos])
                continue

            # Per-metabolite scalar (e.g. amplitude):
            # ind[met_name] is an int giving the column for that metabolite.
            if met_name in ind:
                col_idx = ind[met_name]
                # Guard against accidentally tuple-valued entries of length 1.
                if isinstance(col_idx, tuple) and len(col_idx) == 1:
                    col_idx = col_idx[0]
                col_targets.append(col_idx)
                continue

            raise KeyError(
                f"Could not resolve JSON key '{json_key}' to a column index. "
                f"The prefix '{prefix}' does not appear in ind and is not covered "
                f"by name_map.  Add the appropriate mapping."
            )

        # ----------------------------------------------------------
        # Priority 3: plain scalar key without a metabolite prefix.
        # Examples: "SNR", "ph0", "ph1".
        # suffix_to_ind remaps e.g. "ph0" -> "phi0" before lookup.
        # ----------------------------------------------------------
        canonical_key = suffix_to_ind_l.get(json_key_l, json_key_l)
        if canonical_key not in ind:
            raise KeyError(
                f"Could not resolve JSON key '{json_key}' to an ind key. "
                f"Add the mapping to suffix_to_ind (e.g. {{'{json_key}': '<ind_key>'}})."
            )
        col_idx = ind[canonical_key]
        if isinstance(col_idx, tuple):
            # A length-1 tuple is treated as a scalar; longer tuples become a
            # broadcast group (same draw into every column).
            col_targets.append(col_idx[0] if len(col_idx) == 1 else list(col_idx))
        else:
            col_targets.append(col_idx)

    # ------------------------------------------------------------------
    # Convert the Spearman rank correlation matrix to the Gaussian copula
    # correlation matrix R using the van der Waerden approximation:
    #
    #     R_ij = 2 sin(pi * rho_s_ij / 6)
    #
    # This is exact for bivariate-normal marginals and is a well-known
    # first-order approximation for general continuous marginals.
    # np.fill_diagonal enforces exact ones on the diagonal after the
    # element-wise transform, which may perturb them slightly.
    # nearest_psd then projects R onto the cone of positive semi-definite
    # matrices by clipping negative eigenvalues to a small epsilon, guarding
    # against numerical issues introduced by the transform or data noise.
    # ------------------------------------------------------------------
    R = 2.0 * np.sin(np.pi * corr_matrix / 6.0)
    np.fill_diagonal(R, 1.0)
    R = nearest_psd(R)

    # ------------------------------------------------------------------
    # Draw batch_size joint samples from the Gaussian copula.
    # gaussian_copula_sample is expected to:
    #   1. Draw Z ~ MVN(0, R),  shape [batch_size, d]
    #   2. Apply the standard-normal CDF element-wise: U = Phi(Z)
    #   3. Apply the marginal PPF element-wise: X_i = F_i^{-1}(U_i)
    # and return X of shape [batch_size, d].
    # ------------------------------------------------------------------
    samples = gaussian_copula_sample(batch_size, marginals, R)  # [batch_size, d]

    if samples.shape != (batch_size, d):
        raise RuntimeError(
            f"gaussian_copula_sample returned shape {samples.shape}; "
            f"expected ({batch_size}, {d})."
        )

    # ------------------------------------------------------------------
    # Write the sampled columns back into the output parameter matrix.
    # For broadcast groups (global parameters) the single draw is written
    # to every column in the group, matching the behaviour in
    # populate_params_from_distributions.
    # ------------------------------------------------------------------
    for i, target in enumerate(col_targets):
        x = samples[:, i].astype(np.float32)

        if is_torch:
            x_t = torch.as_tensor(x, dtype=out.dtype, device=out.device)
            if isinstance(target, list):
                for col in target:
                    out[:, col] = x_t
            else:
                out[:, target] = x_t
        else:
            x = x.astype(out.dtype, copy=False)
            if isinstance(target, list):
                for col in target:
                    out[:, col] = x
            else:
                out[:, target] = x

    return out
 


def populate_params_from_distributions(
    ind,
    params,
    dist_json_path: str,
    metabolites: list = None,
    target_key: Optional[str] = None,
    name_map: Optional[Mapping[str, str]] = None,
    global_param_map: Optional[Mapping[str, str]] = None,
    suffix_to_ind: Optional[Mapping[str, str]] = None,
    seed: Optional[int] = None,
):
    """
    Fill params from fitted distributions stored in dist_json_path.

    Parameters
    ----------
    ind
        OrderedDict mapping internal parameter names to column indices.
    params
        Preallocated matrix, shape [batch_size, n_params]. Can be numpy or torch.
    dist_json_path
        Path to parameter_distributions_best_fit.json.
    target_key
        Parameter family to populate. This is resolved through suffix_to_ind
        before matching against ind.
        Examples:
            "ampl"       -> metabolite-specific scalar family, keys like Asc_ampl
            "d"          -> tuple-valued metabolite-aligned family
            "g"          -> tuple-valued metabolite-aligned family unless global_param_map maps it to a global JSON key
            "f_shifts"   -> tuple-valued metabolite-aligned family
            "ph0"        -> resolved through suffix_to_ind to ind['phi0'] if provided
            "ph1"        -> resolved through suffix_to_ind to ind['phi1'] if provided
    name_map
        Optional manual prefix map for JSON prefixes that do not match ind metabolite names.
        Example:
            {"crch2": "ch"}
    global_param_map
        Optional mapping from internal target_key to a JSON key that should be sampled once
        and broadcast across all columns in the corresponding ind tuple.
        Example:
            {"g": "gaussLB"}
    suffix_to_ind
        Optional mapping from JSON suffixes to internal ind keys.
        Example:
            {"lorentzLB": "d", "freqShift": "f_shifts", "ph0": "phi0", "ph1": "phi1"}
    seed
        RNG seed.

    Returns
    -------
    params
        Same object type as input, filled with sampled values.
    """
    dist_db = _load_json(dist_json_path)
    if len(dist_db) == 0:
        return params
    metabolites = set(m.lower() for m in metabolites)

    dist_db = {
        k: v
        for k, v in dist_db.items()
        if (
            not k.endswith(("_ampl", "_freqShift", "_lorentzLB"))
            or k.rsplit("_", 1)[0].lower() in metabolites
        )
    }
        

    json_lookup = {k.lower(): k for k in dist_db}
    name_map = {k.lower(): v.lower() for k, v in (name_map or {}).items()}
    global_param_map = {k.lower(): v for k, v in (global_param_map or {}).items()}
    global_json_to_ind = {v.lower(): k for k, v in global_param_map.items()}
    suffix_to_ind = {k.lower(): v.lower() for k, v in (suffix_to_ind or {}).items()}

    is_torch = torch.is_tensor(params)
    out = params.clone() if is_torch else np.array(params, copy=True)

    if out.ndim != 2:
        raise ValueError("params must be 2D with shape [batch_size, n_params]")

    batch_size = out.shape[0]
    rng = np.random.default_rng(seed)

    metabolite_keys = _infer_metabolite_keys(ind)
    metabolite_pos = {k.lower(): i for i, k in enumerate(metabolite_keys)}

    def resolve_prefix(prefix):
        p = prefix.lower()
        if p.endswith(f"_{target_key}"):
            p = p[:-(len(target_key) + 1)]
        return name_map.get(p, p)

    def write_column(col_idx, spec):
        x = _sample_scipy_spec(spec, batch_size, rng)
        # Functionality was moved to the findParamDist code so that the distributions
        # are fitted to be nonnegative
        # if (x<0).any(): x = np.abs(x)
        if is_torch:
            out[:, col_idx] = torch.as_tensor(x, dtype=out.dtype, device=out.device)
        else:
            out[:, col_idx] = x.astype(out.dtype, copy=False)

    # Infer family from the first JSON key if not provided
    if target_key is None:
        first_json_key = next(iter(dist_db.keys()))
        if "_" in first_json_key:
            target_key = first_json_key.rsplit("_", 1)[1]
        else:
            target_key = first_json_key

    target_key = target_key.lower()
    canonical_target_key = suffix_to_ind.get(target_key, target_key)

    # ------------------------------------------------------------------
    # Populate metabolite-specific parameters together.
    #
    # Calling with target_key="ampl" now also populates the corresponding
    # Lorentzian linewidths ("d") and metabolite frequency shifts
    # ("f_shifts") using the same metabolite index.
    # ------------------------------------------------------------------
    if target_key == "ampl":

        if "d" not in ind or not isinstance(ind["d"], tuple):
            raise KeyError("ind['d'] must exist and be a tuple.")

        if "f_shifts" not in ind or not isinstance(ind["f_shifts"], tuple):
            raise KeyError("ind['f_shifts'] must exist and be a tuple.")

        d_cols = ind["d"]
        f_cols = ind["f_shifts"]

        if len(d_cols) != len(metabolite_keys):
            raise ValueError(
                f"ind['d'] has {len(d_cols)} entries but "
                f"{len(metabolite_keys)} metabolites were found."
            )

        if len(f_cols) != len(metabolite_keys):
            raise ValueError(
                f"ind['f_shifts'] has {len(f_cols)} entries but "
                f"{len(metabolite_keys)} metabolites were found."
            )

        # Build a lookup:
        # family_lookup["asc"]["ampl"] = ...
        # family_lookup["asc"]["d"] = ...
        # family_lookup["asc"]["f_shifts"] = ...
        family_lookup = {}

        for json_key, spec in dist_db.items():

            if "_" not in json_key:
                continue

            prefix, suffix = json_key.rsplit("_", 1)

            met_name = resolve_prefix(prefix)
            if met_name not in metabolite_pos:
                continue

            family = suffix_to_ind.get(suffix.lower(), suffix.lower())

            if family not in {"ampl", "d", "f_shifts"}:
                continue

            family_lookup.setdefault(met_name, {})[family] = spec

        # Populate one metabolite at a time
        for met_name in metabolite_keys:

            met_name = met_name.lower()

            if met_name not in family_lookup:
                continue

            pos = metabolite_pos[met_name]

            families = family_lookup[met_name]

            # amplitude
            if "ampl" in families:
                write_column(ind[met_name], families["ampl"])

            # Lorentzian linewidth
            if "d" in families:
                write_column(d_cols[pos], families["d"])

            # metabolite frequency shift
            if "f_shifts" in families:
                write_column(f_cols[pos], families["f_shifts"])

        return out

    # Handle global parameters
    # print(f'target_key: {target_key}')
    if target_key in global_param_map:
        json_name = global_param_map[target_key]
        json_key = json_lookup.get(json_name.lower())
        if json_key is None:
            return out

        col_group = ind[target_key]
        if isinstance(col_group, int):
            col_group = (col_group,)

        x = _sample_scipy_spec(dist_db[json_key], batch_size, rng)

        if is_torch:
            x = torch.as_tensor(x, dtype=out.dtype, device=out.device)

        for col in col_group:
            out[:, col] = x

        return out

    # Otherwise, assume a scalar family like ampl, ph0, ph1, snr, etc.
    for json_key, spec in dist_db.items():
        json_suffix = json_key.rsplit("_", 1)[1] if "_" in json_key else json_key
        resolved_suffix = suffix_to_ind.get(json_suffix.lower(), json_suffix.lower())
        # print(f'json_suffix {json_suffix}; resolved_suffix {resolved_suffix}')
        if resolved_suffix != canonical_target_key:
            continue            
        
        json_key_l = json_key.lower()

        # Exact global-key handling
        if json_key_l in global_json_to_ind:
            ind_key = global_json_to_ind[json_key_l]
            col_group = ind[ind_key]

            if isinstance(col_group, int):
                col_group = (col_group,)

            x = _sample_scipy_spec(spec, batch_size, rng)
            if is_torch:
                x = torch.as_tensor(x, dtype=out.dtype, device=out.device)

            for col in col_group:
                out[:, col] = x
            continue

        if "_" in json_key:
            prefix, _ = json_key.rsplit("_", 1)
            met_name = resolve_prefix(prefix)
            # print(f'prefix {prefix} and met_name {met_name}')
            
            if met_name in ind:
                col_idx = ind[met_name]
            elif met_name.lower() in ind:
                col_idx = ind[met_name.lower()]
            else:
                # if canonical_target_key not in ind:
                raise KeyError(
                    f"Could not resolve JSON key '{json_key}' to an internal ind key. "
                    f"Add the mapping to name_map if needed."
                )
                # col_idx = ind[canonical_target_key]
        else:
            if canonical_target_key not in ind:
                raise KeyError(
                    f"Could not resolve JSON key '{json_key}' to an internal ind key. "
                    f"Add the mapping to suffix_to_ind if needed."
                )
            col_idx = ind[canonical_target_key]

        if isinstance(col_idx, tuple):
            if len(col_idx) != 1:
                raise ValueError(
                    f"ind['{canonical_target_key}'] is a tuple with length {len(col_idx)}, "
                    f"but a scalar column was expected for '{json_key}'."
                )
            col_idx = col_idx[0]

        write_column(col_idx, spec)

    return out