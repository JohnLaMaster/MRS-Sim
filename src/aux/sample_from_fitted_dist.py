import json
from typing import Optional, Mapping

import numpy as np
import torch
from scipy import stats


__all__ = ["populate_params_from_distributions"]


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