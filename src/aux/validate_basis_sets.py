"""
validate_basis_sets.py
======================

Standalone validation tool for exported MRS basis set files.

Usage
-----
Run after a complete export to validate all files in an output directory::

    python validate_basis_sets.py --export_dir ./basis_sets/exported/my_basis \
                                  --formats lcmodel_raw lcmodel_basis fsl_mrs_json \
                                            osprey_mat marss_native nifti_mrs

Or call programmatically from process_basis_functions.py for early per-file
validation::

    from validate_basis_sets import validate_single_file
    ok, message = validate_single_file('/path/to/file.raw', 'lcmodel_raw')

Design
------
Each format has a dedicated validator that returns (passed: bool, report: str).
Validators are deliberately conservative: they check *structural* and
*convention* compliance, not physical plausibility of the spectral data.

The entry point ``validate_export_dir`` discovers files produced by
process_basis_functions.py (mirroring the layout written by export_basis_set),
validates each one, and prints a summary table.  It returns a non-zero exit
code if any file fails, making it suitable for use in CI pipelines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import numpy as np
import scipy.io as sio


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _pass(msg: str = '') -> Tuple[bool, str]:
    return True, msg or 'OK'


def _fail(reason: str) -> Tuple[bool, str]:
    return False, reason


def _fmt_header(fmt: str, path: str) -> str:
    return f"[{fmt}] {os.path.basename(path)}"


# ─────────────────────────────────────────────────────────────────────────────
# Per-format validators
# ─────────────────────────────────────────────────────────────────────────────

def validate_lcmodel_raw(path: str) -> Tuple[bool, str]:
    """
    Validate a single LCModel .raw file.

    Required structure:
      - File must start with a $NMID ... $END block.
      - Block must contain ID and FMTDAT keys.
      - FID data section (after the last $END) must contain an even number of
        whitespace-delimited floating-point tokens (interleaved real/imag).
      - At least one complex point must be present.
    """
    try:
        with open(path, 'r') as f:
            content = f.read()
    except OSError as e:
        return _fail(f"Cannot open file: {e}")

    # $NMID block
    import re
    m = re.search(r'\$NMID(.*?)\$END', content, re.DOTALL | re.IGNORECASE)
    if not m:
        return _fail("Missing $NMID ... $END block.")

    block = m.group(1)
    if 'ID' not in block.upper():
        return _fail("$NMID block does not contain an ID field.")
    if 'FMTDAT' not in block.upper():
        return _fail("$NMID block does not contain a FMTDAT field.")

    # FID data section
    data_section = re.split(r'\$END', content, flags=re.IGNORECASE)[-1]
    tokens = []
    for line in data_section.splitlines():
        for tok in line.split():
            try:
                tokens.append(float(tok))
            except ValueError:
                pass

    if len(tokens) == 0:
        return _fail("No numeric data found after $END block.")
    if len(tokens) % 2 != 0:
        return _fail(
            f"Odd number of data tokens ({len(tokens)}); interleaved re/im "
            "pairs require an even count."
        )
    return _pass(f"{len(tokens)//2} complex point(s)")


def validate_lcmodel_basis(path: str) -> Tuple[bool, str]:
    """
    Validate an LCModel .basis file.

    Required structure:
      - A $SEQPAR block (may be empty but must be present).
      - A $BASIS1 block containing HZPPPM, BADELT, and NDATAB.
      - At least one $BASIS (not $BASIS1) block containing ID or METABO and
        followed by a non-empty FID data section.
    """
    try:
        with open(path, 'r') as f:
            content = f.read()
    except OSError as e:
        return _fail(f"Cannot open file: {e}")

    import re

    # SEQPAR
    if not re.search(r'\$SEQPAR', content, re.IGNORECASE):
        return _fail("Missing $SEQPAR block.")

    # BASIS1
    m1 = re.search(r'\$BASIS1(.*?)\$END', content, re.DOTALL | re.IGNORECASE)
    if not m1:
        return _fail("Missing $BASIS1 block.")
    b1 = m1.group(1).upper()
    for key in ('HZPPPM', 'BADELT', 'NDATAB'):
        if key not in b1:
            return _fail(f"$BASIS1 block is missing required key: {key}.")

    # At least one $BASIS (not $BASIS1) block
    basis_pattern = re.compile(
        r'\$(BASIS)(?!1)\b(.*?)\$END(.*?)(?=\$BASIS(?!1)\b|\Z)',
        re.DOTALL | re.IGNORECASE,
    )
    blocks = list(basis_pattern.finditer(content))
    if not blocks:
        return _fail("No $BASIS metabolite blocks found.")

    issues = []
    for i, block in enumerate(blocks):
        header_text = block.group(2).upper()
        data_text   = block.group(3)
        if 'ID' not in header_text and 'METABO' not in header_text:
            issues.append(f"Block {i}: missing ID and METABO fields.")
        tokens = []
        for line in data_text.splitlines():
            for tok in line.split():
                try:
                    tokens.append(float(tok))
                except ValueError:
                    pass
        if len(tokens) == 0:
            issues.append(f"Block {i}: no FID data.")
        elif len(tokens) % 2 != 0:
            issues.append(
                f"Block {i}: odd token count ({len(tokens)}); expected even."
            )

    if issues:
        return _fail('; '.join(issues))
    return _pass(f"{len(blocks)} metabolite block(s)")


def validate_fsl_mrs_json(path: str) -> Tuple[bool, str]:
    """
    Validate a single FSL-MRS .json basis file.

    Required fields:
      - basis_name (str)
      - basis (dict) with keys: basis_re, basis_im, basis_dwell, basis_centre
      - basis_re and basis_im must be equal-length non-empty lists of numbers.
      - basis_dwell must be a positive float.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return _fail(f"Cannot parse JSON: {e}")

    if 'basis_name' not in data:
        return _fail("Missing field: 'basis_name'.")
    if 'basis' not in data:
        return _fail("Missing top-level 'basis' dict.")

    b = data['basis']
    for key in ('basis_re', 'basis_im', 'basis_dwell', 'basis_centre'):
        if key not in b:
            return _fail(f"Missing field in 'basis': '{key}'.")

    re_arr = b['basis_re']
    im_arr = b['basis_im']
    if not isinstance(re_arr, list) or len(re_arr) == 0:
        return _fail("'basis_re' must be a non-empty list.")
    if not isinstance(im_arr, list) or len(im_arr) == 0:
        return _fail("'basis_im' must be a non-empty list.")
    if len(re_arr) != len(im_arr):
        return _fail(
            f"'basis_re' length ({len(re_arr)}) != 'basis_im' length ({len(im_arr)})."
        )

    dt = b['basis_dwell']
    if not isinstance(dt, (int, float)) or dt <= 0:
        return _fail(f"'basis_dwell' must be a positive number; got {dt!r}.")

    return _pass(f"{len(re_arr)} complex point(s), dwell={dt:.4g}s")


def validate_osprey_mat(path: str) -> Tuple[bool, str]:
    """
    Validate an Osprey-format .mat file.

    Required structure:
      - Top-level variable 'BASIS'.
      - BASIS must contain: fids, name, nMets, nMM, spectralwidth, dwelltime,
        n, Bo, te, centerFreq, scale.
      - fids must be 2-D (ns × n_mets); n_mets must equal nMets + nMM.
      - name cell array length must equal fids column count.
      - spectralwidth and dwelltime must be positive; 1/dwelltime ≈ spectralwidth
        to within 1 %.
    """
    try:
        raw = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        return _fail(f"Cannot load .mat file: {e}")

    if 'BASIS' not in raw:
        return _fail("No 'BASIS' variable found.")
    B = raw['BASIS']

    required_fields = [
        'fids', 'name', 'nMets', 'nMM', 'spectralwidth',
        'dwelltime', 'n', 'Bo', 'te', 'centerFreq', 'scale',
    ]
    for field in required_fields:
        if not hasattr(B, field):
            return _fail(f"BASIS struct missing field: '{field}'.")

    fids = np.atleast_2d(np.squeeze(B.fids))
    if fids.ndim != 2:
        return _fail(
            f"BASIS.fids must be 2-D (ns × n_mets); got shape {fids.shape}."
        )
    ns_fids, n_cols = fids.shape

    try:
        n_mets = int(np.squeeze(B.nMets))
        n_mm   = int(np.squeeze(B.nMM))
    except Exception as e:
        return _fail(f"Cannot read nMets / nMM: {e}")

    if n_mets + n_mm != n_cols:
        return _fail(
            f"nMets ({n_mets}) + nMM ({n_mm}) = {n_mets+n_mm} "
            f"but fids has {n_cols} column(s)."
        )

    # Name count
    name_arr = np.atleast_1d(np.squeeze(B.name))
    if len(name_arr) != n_cols:
        return _fail(
            f"BASIS.name length ({len(name_arr)}) != fids column count ({n_cols})."
        )

    try:
        sw = float(np.squeeze(B.spectralwidth))
        dt = float(np.squeeze(B.dwelltime))
    except Exception as e:
        return _fail(f"Cannot read spectralwidth / dwelltime: {e}")

    if sw <= 0:
        return _fail(f"spectralwidth must be positive; got {sw}.")
    if dt <= 0:
        return _fail(f"dwelltime must be positive; got {dt}.")

    sw_from_dt = 1.0 / dt
    if abs(sw - sw_from_dt) / sw_from_dt > 0.01:
        return _fail(
            f"spectralwidth ({sw:.4f} Hz) and 1/dwelltime ({sw_from_dt:.4f} Hz) "
            "differ by more than 1 %."
        )

    return _pass(
        f"{n_mets} met(s) + {n_mm} MM/Lip, ns={ns_fids}, sw={sw:.1f} Hz"
    )


def validate_marss_native(path: str) -> Tuple[bool, str]:
    """
    Validate a MARSS-native per-metabolite .mat file.

    Required structure:
      - Top-level variable 'exptDat'.
      - exptDat must contain: sw_h, sf, nspecC, fid.
      - sw_h and sf must be positive.
      - fid length must equal nspecC.
    """
    try:
        mat = sio.loadmat(path)
    except Exception as e:
        return _fail(f"Cannot load .mat file: {e}")

    if 'exptDat' not in mat:
        return _fail("No 'exptDat' variable found.")

    exptDat = mat['exptDat']

    for field in ('sw_h', 'sf', 'nspecC', 'fid'):
        try:
            _ = exptDat[field]
        except (KeyError, ValueError):
            return _fail(f"exptDat missing field: '{field}'.")

    try:
        sw  = float(np.squeeze(exptDat['sw_h']))
        sf  = float(np.squeeze(exptDat['sf']))
        ns  = int(np.squeeze(exptDat['nspecC']))
        fid = np.squeeze(exptDat['fid'])
    except Exception as e:
        return _fail(f"Cannot extract fields: {e}")

    if sw <= 0:
        return _fail(f"sw_h must be positive; got {sw}.")
    if sf <= 0:
        return _fail(f"sf must be positive; got {sf}.")

    if fid.size != ns:
        return _fail(
            f"fid length ({fid.size}) != nspecC ({ns})."
        )

    return _pass(f"ns={ns}, sw={sw:.1f} Hz, sf={sf:.4f} MHz")


def validate_nifti_mrs(path: str) -> Tuple[bool, str]:
    """
    Validate a NIfTI-MRS .nii / .nii.gz file.

    Required conventions (Clarke et al., MRM 2022):
      - Data must be complex (complex64 or complex128).
      - Shape must be (x, y, z, ns [, n_mets]); x=y=z=1 for SVS.
      - pixdim[4] (dwell time) must be positive.
      - JSON extension with ecode=44 must be present and valid JSON.
      - SpectrometerFrequency must be a non-empty list.
      - If dim_5_header is present it must contain a 'Basis' list whose length
        matches dim 5 of the data array.
    """
    try:
        import nibabel as nib
        img = nib.load(path)
    except Exception as e:
        return _fail(f"Cannot load NIfTI file: {e}")

    data = np.asarray(img.dataobj)

    # Complex dtype
    if not np.iscomplexobj(data):
        return _fail(
            f"Data dtype is {data.dtype}; NIfTI-MRS requires complex data."
        )

    # Minimum 4 dimensions
    if data.ndim < 4:
        return _fail(
            f"Data has {data.ndim} dimension(s); at least 4 required."
        )

    # pixdim[4] — dwell time
    dt_raw = float(img.header['pixdim'][4])
    if dt_raw <= 0:
        return _fail(f"pixdim[4] (dwell time) must be positive; got {dt_raw}.")

    # JSON extension with ecode=44
    hdr_ext = None
    for ext in img.header.extensions:
        if ext.get_code() == 44:
            try:
                hdr_ext = json.loads(ext.get_content().decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                return _fail(f"ecode-44 extension is not valid JSON: {e}")
            break

    if hdr_ext is None:
        return _fail("No ecode-44 JSON extension found (required by NIfTI-MRS spec).")

    sf_field = hdr_ext.get('SpectrometerFrequency')
    if not isinstance(sf_field, list) or len(sf_field) == 0:
        return _fail(
            "'SpectrometerFrequency' must be a non-empty JSON array "
            f"(got {sf_field!r})."
        )

    # dim_5_header consistency check
    dim5_hdr = hdr_ext.get('dim_5_header')
    if dim5_hdr is not None:
        basis_names = dim5_hdr.get('Basis', [])
        if data.ndim >= 5:
            n_mets_data = data.shape[4]
            if len(basis_names) != n_mets_data:
                return _fail(
                    f"dim_5_header 'Basis' list length ({len(basis_names)}) "
                    f"!= data shape[4] ({n_mets_data})."
                )

    ns  = data.shape[3]
    n_m = data.shape[4] if data.ndim >= 5 else 1
    return _pass(
        f"ns={ns}, n_mets={n_m}, dt={dt_raw:.4g}s, "
        f"sf={sf_field[0]:.4f} MHz"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch table
# ─────────────────────────────────────────────────────────────────────────────

#: Maps format identifier → (file extension(s), validator function).
#  Extensions are used by validate_export_dir to discover files automatically.
FORMAT_VALIDATORS = {
    'lcmodel_raw':   (['.raw'],        validate_lcmodel_raw),
    'lcmodel_basis': (['.basis'],      validate_lcmodel_basis),
    'fsl_mrs_json':  (['.json'],       validate_fsl_mrs_json),
    'osprey_mat':    (['.mat'],        validate_osprey_mat),
    'marss_native':  (['.mat'],        validate_marss_native),
    'nifti_mrs':     (['.nii', '.gz'], validate_nifti_mrs),
}


def validate_single_file(path: str, fmt: str) -> Tuple[bool, str]:
    """
    Validate one file against the given format's conventions.

    Parameters
    ----------
    path : str   Path to the file to validate.
    fmt  : str   Format identifier (key in FORMAT_VALIDATORS).

    Returns
    -------
    (passed, message) where passed is True if all checks pass.
    """
    if fmt not in FORMAT_VALIDATORS:
        return _fail(
            f"Unknown format '{fmt}'. Valid: {', '.join(FORMAT_VALIDATORS)}"
        )
    _, validator = FORMAT_VALIDATORS[fmt]
    return validator(path)


# ─────────────────────────────────────────────────────────────────────────────
# Directory-level validation
# ─────────────────────────────────────────────────────────────────────────────

def _discover_files(export_dir: str, fmt: str) -> list:
    """
    Return a sorted list of file paths to validate for ``fmt`` inside
    ``export_dir``, following the directory layout written by
    export_basis_set() in process_basis_functions.py.

    Multi-file formats are written to <export_dir>/<fmt>/.
    Single-file formats are written directly to <export_dir>/.
    """
    MULTI_FILE_FORMATS = {'lcmodel_raw', 'fsl_mrs_json', 'marss_native'}
    exts, _ = FORMAT_VALIDATORS[fmt]

    if fmt in MULTI_FILE_FORMATS:
        search_dir = os.path.join(export_dir, fmt)
    else:
        search_dir = export_dir

    if not os.path.isdir(search_dir):
        return []

    paths = []
    for name in sorted(os.listdir(search_dir)):
        _, ext = os.path.splitext(name)
        # .nii.gz: check the combined suffix
        if name.endswith('.nii.gz') and fmt == 'nifti_mrs':
            paths.append(os.path.join(search_dir, name))
        elif ext.lower() in exts and not name.endswith('.nii.gz'):
            paths.append(os.path.join(search_dir, name))
    return paths


def validate_export_dir(export_dir: str,
                        formats: list,
                        early_abort: bool = True) -> dict:
    """
    Validate all exported basis set files found under ``export_dir``.

    The function validates each format in the order given.  For multi-file
    formats it validates the first file immediately; if it fails and
    ``early_abort`` is True the remaining files for that format are skipped
    (matching the early-abort behaviour of the exporter).

    Parameters
    ----------
    export_dir   : Root directory of the exported basis set (the value passed
                   to export_basis_set as ``output_dir``).
    formats      : List of format identifiers to validate.
    early_abort  : If True, skip remaining files in a format after the first
                   failure.  If False, validate every file regardless.

    Returns
    -------
    dict mapping format → {'passed': int, 'failed': int, 'files': [
        {'path': str, 'ok': bool, 'msg': str}, ...
    ]}
    """
    results = {}

    for fmt in formats:
        if fmt not in FORMAT_VALIDATORS:
            print(f"\n[SKIP] Unknown format '{fmt}'.")
            continue

        files = _discover_files(export_dir, fmt)
        if not files:
            print(f"\n[{fmt}] No files found under '{export_dir}'.")
            results[fmt] = {'passed': 0, 'failed': 0, 'files': []}
            continue

        print(f"\n── Validating {len(files)} file(s) for format '{fmt}' ──")
        fmt_result = {'passed': 0, 'failed': 0, 'files': []}

        aborted = False
        for i, path in enumerate(files):
            ok, msg = validate_single_file(path, fmt)
            label = '  ✓' if ok else '  ✗'
            short = os.path.relpath(path, export_dir)
            print(f"{label}  {short}  —  {msg}")
            fmt_result['files'].append({'path': path, 'ok': ok, 'msg': msg})
            if ok:
                fmt_result['passed'] += 1
            else:
                fmt_result['failed'] += 1
                if early_abort and i == 0 and len(files) > 1:
                    print(
                        f"  [early-abort] First file failed; skipping "
                        f"{len(files)-1} remaining file(s) for '{fmt}'."
                    )
                    aborted = True
                    break

        if aborted:
            fmt_result['aborted'] = True

        results[fmt] = fmt_result

    return results


def _print_summary(results: dict):
    """Print a concise per-format pass/fail summary table."""
    print("\n" + "=" * 60)
    print(f"{'FORMAT':<20}  {'PASSED':>6}  {'FAILED':>6}  STATUS")
    print("-" * 60)
    all_passed = True
    for fmt, r in results.items():
        p, f = r.get('passed', 0), r.get('failed', 0)
        aborted = r.get('aborted', False)
        status = 'PASS' if f == 0 and p > 0 else ('ABORTED' if aborted else 'FAIL')
        if status != 'PASS':
            all_passed = False
        print(f"{fmt:<20}  {p:>6}  {f:>6}  {status}")
    print("=" * 60)
    print("Overall:", "PASS" if all_passed else "FAIL")
    print()
    return all_passed


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Validate exported MRS basis set files produced by '
            'process_basis_functions.py.'
        ),
    )
    parser.add_argument(
        '--export_dir',
        type=str,
        required=True,
        help=(
            'Root export directory passed to export_basis_set() as output_dir, '
            'e.g. ./basis_sets/exported/PRESS_30ms_Siemens_2000.'
        ),
    )
    parser.add_argument(
        '--formats',
        type=str,
        nargs='+',
        default=list(FORMAT_VALIDATORS.keys()),
        help=(
            'One or more format identifiers to validate. '
            f'Choices: {", ".join(FORMAT_VALIDATORS)}. '
            'Defaults to all known formats.'
        ),
    )
    parser.add_argument(
        '--no_early_abort',
        action='store_true',
        default=False,
        help=(
            'By default validation aborts a multi-file format after the first '
            'failure. Pass this flag to validate every file regardless.'
        ),
    )

    args = parser.parse_args()

    if not os.path.isdir(args.export_dir):
        print(f"[ERROR] export_dir not found: {args.export_dir}", file=sys.stderr)
        sys.exit(2)

    results = validate_export_dir(
        export_dir=args.export_dir,
        formats=args.formats,
        early_abort=not args.no_early_abort,
    )

    all_passed = _print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()