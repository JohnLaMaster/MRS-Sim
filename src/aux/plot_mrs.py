#!/usr/bin/env python3
"""
plot_mrs.py
-----------
Visualize MR spectroscopy fitting results.

Layout
------
  Top panel  : real spectrum (black) + baseline (blue) + fit (red, optional)
  Middle rows: per-metabolite basis functions – broadened & amplitude-scaled
  Bottom row : residual water (no broadening)

Expected data files
-------------------
MAT_FILE  (simulation outputs)
    ppm              [N_pts]
    spectra          [batch, 2, N_pts]  or  [batch, N_pts] complex
    baselines        same shape as spectra
    residual_water   same shape as spectra
    params           struct with fields:
                       .asc, .asp, .ch, …, .lip20   each [1, batch] (amplitudes)
                       .d   [batch, N_met]  Lorentzian decay (Hz)
                       .g   [batch, N_met]  Gaussian decay   (Hz²)
                       (amplitude fields are all params fields that come before .d)

BASIS_FILE  (basis functions + acquisition header)
    metabolites/<key>/fid   [2, N_pts]   row-0 real, row-1 imag
    header/t                [N_pts]      time vector (seconds)

Requirements
------------
    pip install numpy scipy matplotlib h5py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import scipy.io
import h5py

from collections import OrderedDict

# ══════════════════════════════════════════════════════════════
#  USER SETTINGS
# ══════════════════════════════════════════════════════════════
MAT_FILE     = "/home/john/Documents/Repositories/MRS-Sim/dataset/VERI_sample/dataset_spectra_0.mat"
# BASIS_FILE   = "/home/john/Documents/Repositories/MRS-Sim/src/basis_sets/PRESS_30_GE_2000.mat"
BASIS_FILE   = "/home/john/Documents/Repositories/MRS-Sim/src/basis_sets/VERI_PRESS_30_GE_30_2000.mat"
SPECTRUM_IDX = 0                # batch index (0-based)
# PPM_RANGE    = (4.2, 0.2)       # (high_ppm, low_ppm)
# PPM_RANGE    = (9.0, -1.0)       # (high_ppm, low_ppm)
PPM_RANGE    = (11.0, -3.0)       # (high_ppm, low_ppm)
PPM_RANGE    = (5.0,  0.5)       # (high_ppm, low_ppm)
SHOW_FIT     = True
FIG_WIDTH    = 8.5              # inches
TOP_RATIO    = 10                # height of spectrum panel vs. one metabolite row
SAVE_PATH    = "/home/john/Documents/Repositories/MRS-Sim/dataset/VERI_sample/mrs_plot_wMM.png" # set None to skip saving

LABEL_FS   = 11   # x-axis label
TICK_FS    = 10   # tick labels
ANNOT_FS   = 9    # metabolite labels
LEGEND_FS  = 9    # legend
# ══════════════════════════════════════════════════════════════

# Metabolite order must match the column order of params.d and params.g,
# and is the order in which params.<key> amplitude fields are stacked.
MET_LABELS = [
    "Asc",  "Asp",  "Ch",   "Cr",   "GABA", "Gln",  "Glu",  "GPC",
    "GSH",  "Lac",  "MI",   "NAA",  "NAAG", "PCh",  "PCr",  "PE",
    "SI",   "Tau",
    "MM092","MM121", "MM139", "MM167",  "MM204", "MM226", "MM270",
    "MM299", "MM321", "MM375",
    # "MM09", "MM12", "MM14", "MM17",  "MM20",
    # "Lip09","Lip13","Lip20",
]
MET_KEYS = [l.lower() for l in MET_LABELS]
N_MET = len(MET_KEYS)


# ──────────────────────────────────────────────────────────────
#  I/O helpers
# ──────────────────────────────────────────────────────────────

def load_mat(filepath):
    """
    Try scipy.io first (MATLAB ≤ v7); fall back to h5py (MATLAB v7.3 / HDF5).
    Returns (data_object, 'scipy' | 'h5py').
    """
    try:
        d = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        return d, "scipy"
    except Exception:
        return h5py.File(filepath, "r"), "h5py"


def to_complex(arr):
    """
    Ensure the array is complex.
    Accepts a complex array directly, or a real array whose second-to-last
    dimension (dim -2) is [real, imag].
    """
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr
    return arr[..., 0, :] + 1j * arr[..., 1, :]


def load_basis(basis_file):
    """
    Load basis FIDs and the time vector from the basis file.
    NOTE: header (including t) lives in the BASIS file, not the simulation file.

    Returns
    -------
    fids : dict  {met_key: complex_1d_array  shape [N_pts]}
    t    : 1-D float array  (time vector in seconds, from header.t)
    """
    # print(basis_file)
    # ── h5py (HDF5 / MATLAB v7.3) ─────────────────────────────
    try:
        f = h5py.File(basis_file, "r")
        t = np.asarray(f["header"]["t"]).squeeze()
        fids = OrderedDict()#{}
        for key in MET_KEYS:
            raw = np.asarray(f["metabolites"][key]["fid"])   # [2, N]
            fids[key] = raw[0] + 1j * raw[1]
        return fids, t
    except (OSError, KeyError):
        pass

    # ── scipy fallback (MATLAB ≤ v7) ──────────────────────────
    d = scipy.io.loadmat(basis_file, squeeze_me=True, struct_as_record=False)
    t = np.asarray(d["header"].t).squeeze()
    fids = OrderedDict()
    for key in MET_KEYS:
        try:
            raw = np.asarray(d["metabolites"].__dict__[key].fid)  # [2, N]
            fids[key] = raw[0] + 1j * raw[1]
        except KeyError:
            print("KeyError when loading the basis set.")
            pass
    return fids, t


# ──────────────────────────────────────────────────────────────
#  Signal processing
# ──────────────────────────────────────────────────────────────

def apply_broadening(fid, d, g, t):
    """
    Lorentzian + Gaussian apodization in the time domain.

        fid_out = fid · exp(−d·t − g·t²)

    Parameters
    ----------
    fid : complex 1-D ndarray
    d   : Lorentzian decay rate  (from params.d for this metabolite)
    g   : Gaussian decay rate    (from params.g for this metabolite)
    t   : time vector in seconds (from basis header.t)
    """
    return fid * np.exp(-d * t - g * t ** 2)


def fid_to_spectrum(fid):
    """FFT + fftshift; return real part only."""
    return np.fft.fftshift(np.fft.fft(fid)).real


# ──────────────────────────────────────────────────────────────
#  Figure helpers
# ──────────────────────────────────────────────────────────────

def _clean_ax(ax):
    """Strip spines, y-ticks, and x-tick labels."""
    ax.set_yticks([])
    ax.tick_params(bottom=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _pad_ylim(ax, frac=0.15):
    """Add vertical breathing room above and below the plotted data."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax.set_ylim(lo - frac * span, hi + frac * span)


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main():
    idx = SPECTRUM_IDX

    # ── 1. Load simulation data ───────────────────────────────
    mat, loader = load_mat(MAT_FILE)

    ppm       = np.asarray(mat["ppm"]).squeeze()
    spec      = to_complex(np.asarray(mat["spectra"]))[idx,:,:]
    bl_real   = to_complex(np.asarray(mat["baselines"]))[idx,:].real
    rw_real   = to_complex(np.asarray(mat["residual_water"]))[idx,:].real
    
    spec_real = np.fft.fftshift(np.fft.fft(spec),axes=-1).real[0,:]
    print('spec.shape (FID): ',spec.shape)
    print('spec.amax (FID): ',np.amax(spec))
    print('spec_real.amax: ',np.amax(spec_real))
    denom = np.amax(spec_real)
    spec_real = spec_real / denom
    print('spec_real.amax: ',np.amax(spec_real))
    bl_real   = bl_real / denom
    rw_real   = rw_real / denom

    # Amplitudes are stored per-metabolite as params.<key>, each shape [1, batch].
    # Stack them in MET_KEYS order to get a [N_met] vector for this index.
    if loader == "scipy":
        p = mat["params"]
        # try:
        a_vec = np.array([np.atleast_1d(getattr(p, key)).squeeze()[idx]
                        for key in MET_KEYS])
        d_vec = np.asarray(p.d)[idx]
        g_vec = np.asarray(p.g)[idx]
        # except AttributeError:
        #     print("AttributeError with scipy")
        #     pass
    else:
        p = mat["params"]
        try:
            a_vec = np.array([np.asarray(p[key]).squeeze()[idx]
                            for key in MET_KEYS])
            d_vec = np.asarray(p["d"])[idx]
            g_vec = np.asarray(p["g"])[idx]
        except AttributeError:
            print("AttributeError with alternate loader")
            pass

    # ── 2. Load basis FIDs + time vector ─────────────────────
    #    header.t is stored in the BASIS file, not the simulation file
    basis_fids, t = load_basis(BASIS_FILE)

    # ── 3. Build per-metabolite spectra ──────────────────────
    met_spectra = []
    for i, key in enumerate(MET_KEYS):
        # if not ("mm" in key or "lip" in key):
        # if not ("lip" in key):
        fid_b = apply_broadening(basis_fids[key], d_vec[i], g_vec[i], t)
        fid_b = fid_to_spectrum(fid_b)
        # if "mm" in key or "lip" in key: fid_b = np.zeros_like(fid_b)#np.flip(fid_b)
        met_spectra.append(fid_b * a_vec[i])
            # try:
            #     fid_b = fid_to_spectrum(basis_fids[key])#*1000
            #     met_spectra.append(fid_b)
            # except KeyError:
            #     pass
    met_spectra = np.fliplr(np.array(met_spectra))   # [N_met, N_pts]
    
    met_spectra = met_spectra / np.amax(met_spectra) * np.amax(spec_real)

    # ── 4. Reconstructed fit ──────────────────────────────────
    fit_real = (met_spectra.sum(axis=0) + bl_real) 
    fit_real = fit_real / np.amax(fit_real) * np.amax(spec_real)

    # ── 5. Figure layout ──────────────────────────────────────
    ppm_hi, ppm_lo = PPM_RANGE
    n_rows         = 1 + N_MET #+ 1
    height_ratios  = [TOP_RATIO] + [1] * N_MET #+ [1]
    total_h        = FIG_WIDTH * (sum(height_ratios) / (TOP_RATIO + 1)) * 0.55#+ 2)) * 0.55

    # fig = plt.figure(figsize=(FIG_WIDTH, total_h))
    # gs  = gridspec.GridSpec(
    #     n_rows, 1, figure=fig,
    #     height_ratios=height_ratios,
    #     hspace=0.0,
    #     left=0.03, right=0.86,
    #     top=0.98,  bottom=0.05,
    # )
    # axes = [fig.add_subplot(gs[i]) for i in range(n_rows)]
    # for ax in axes[1:]:
    #     ax.sharex(axes[0])

    # # ── 6. Top panel ──────────────────────────────────────────
    # ax0 = axes[0]
    # ax0.axvline(x=0, color="silver", lw=0.8, zorder=0)
    # ax0.plot(ppm, spec_real, color="black",     lw=0.9, label="data",     zorder=2)
    # if SHOW_FIT:
    #     ax0.plot(ppm, fit_real,  color="red",   lw=0.9, label="fit",      zorder=3)
    # ax0.plot(ppm, rw_real+bl_real, color="green", lw=0.9, label="residual water", zorder=3)
    # ax0.plot(ppm, bl_real,       color="royalblue", lw=0.9, label="baseline", zorder=3)
    # ax0.legend(loc="upper right", fontsize=7, frameon=False, handlelength=1.2)
    # # xlim inverted (high → low); the renderer clips everything outside automatically
    # ax0.set_xlim(ppm_hi, ppm_lo)
    # _clean_ax(ax0)

    # # ── 7. Metabolite panels ──────────────────────────────────
    # for i, (lbl, sp) in enumerate(zip(MET_LABELS, met_spectra)):
    #     ax = axes[i + 1]
    #     ax.axvline(x=0, color="silver", lw=0.4, zorder=0)
    #     ax.plot(ppm, sp, color="black", lw=0.6)
    #     _clean_ax(ax)
    #     _pad_ylim(ax)
    #     ax.annotate(lbl, xy=(1.01, 0.5), xycoords="axes fraction",
    #                 fontsize=6.5, va="center", ha="left")

    # # ── 8. Residual water panel ───────────────────────────────
    # ax_w = axes[-1]
    # residual = spec_real - bl_real - fit_real
    # ax_w.axvline(x=0, color="silver", lw=0.4, zorder=0)
    # # ax_w.plot(ppm, rw_real, color="black", lw=0.6)
    # ax_w.plot(ppm, residual, color="black", lw=0.6)
    # _clean_ax(ax_w)
    # # _pad_ylim(ax_w)
    # ax_w.annotate("H\u2082O", xy=(1.01, 0.5), xycoords="axes fraction",
    #               fontsize=6.5, va="center", ha="left")

    # # x-axis ticks on the bottom panel only
    # ax_w.tick_params(bottom=True, labelbottom=True, labelsize=8)
    # ax_w.xaxis.set_major_locator(MultipleLocator(0.5))
    # ax_w.xaxis.set_minor_locator(MultipleLocator(0.1))
    # ax_w.tick_params(which="major", length=4)
    # ax_w.tick_params(which="minor", length=2)
    # ax_w.spines["bottom"].set_visible(True)
    # ax_w.set_xlabel("Chemical Shift (ppm)", fontsize=9)
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH*1.2, FIG_WIDTH))

    # ── Define vertical offset ───────────────────────────────────
    spec_range = np.max(spec_real) - np.min(spec_real)
    OFFSET_STEP = 1.0 * spec_range   # 10% of spec_real height

    # Keep track of current vertical level (top → bottom)
    offset = 0.0
    rx = 7.5*1.25
    print('spec_range: ',spec_range, np.amax(spec_real*rx+offset))

    # ── 6. Top spectrum (data + model components) ────────────────
    # ax.axvline(x=0, color="silver", lw=2.8, zorder=0)

    ax.plot(ppm, spec_real*rx + offset, color="black", lw=0.9, label="data", zorder=2)

    if SHOW_FIT:
        ax.plot(ppm, fit_real*rx + offset, color="red", lw=0.9, label="fit", zorder=3)

    # ax.plot(ppm, (rw_real + bl_real)*rx + offset, color="green", lw=0.9,
    #         label="residual water", zorder=3)

    ax.plot(ppm, bl_real*rx + offset, color="royalblue", lw=0.9,
            label="baseline", zorder=3)

    # Legend only once
    ax.legend(loc="upper right", fontsize=LEGEND_FS, frameon=False, handlelength=1.2)

    # Move down for next traces
    offset -= OFFSET_STEP
    rx /= 1.25
    rx1 = rx
    mm = 0

    # ── 7. Metabolite spectra (stacked with offsets) ─────────────
    for lbl, sp in zip(MET_LABELS, met_spectra):
        # if "mm" in lbl and rx1==rx:
        #     rx1 = 2*rx
        # if "mm" in lbl:
        #     sp = np.flip(sp)
        # ax.plot(ppm, sp*.025 + offset, color="black", lw=0.6)
        ax.plot(ppm, sp*rx1 + offset, color="black", lw=0.6)

        # Label on the right
        ax.annotate(lbl,
                    xy=(1.01, offset),
                    xycoords=("axes fraction", "data"),
                    fontsize=ANNOT_FS, va="center", ha="left")

        offset -= OFFSET_STEP
        if lbl.lower().startswith("mm"): mm += sp*rx1
        
    ax.plot(ppm, mm + offset, color='black', lw=0.6)
    ax.annotate("MM Sum", xy=(1.01, offset), xycoords=("axes fraction", "data"),
                  fontsize=ANNOT_FS, va="center", ha="left")
    offset -= OFFSET_STEP
    
    # ── 8. Residual water panel ───────────────────────────────
    # residual = spec_real - bl_real - fit_real
    # ax_w.axvline(x=0, color="silver", lw=0.4, zorder=0)
    ax.plot(ppm, rw_real*rx + offset, color="black", lw=0.6)
    # ax_w.plot(ppm, residual, color="black", lw=0.6)
    # _clean_ax(ax_w)
    # _pad_ylim(ax_w)
    ax.annotate("Res H\u2082O", xy=(1.01, offset), xycoords=("axes fraction", "data"),
                  fontsize=ANNOT_FS, va="center", ha="left")
    offset -= OFFSET_STEP

    # ── 8. Axis formatting ───────────────────────────────────────
    ax.set_xlim(ppm_lo, ppm_hi)
    ax.invert_xaxis()

    # X-axis ticks only (bottom, as requested)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="major", length=4, labelsize=TICK_FS)
    ax.tick_params(which="minor", length=2)

    ax.set_xlabel("Chemical Shift (ppm)", fontsize=LABEL_FS)

    # Clean up y-axis (since offsets make absolute values less meaningful)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Optional: tighten vertical limits to content
    all_data = [spec_real] + list(met_spectra)
    y_min = min(np.min(d*.05) for d in all_data) - OFFSET_STEP * (len(all_data)+1)
    y_min = min(np.min(d) for d in all_data) - OFFSET_STEP * (len(all_data)+2)#1)
    y_max = np.max(spec_real)*rx*1.25 + OFFSET_STEP
    ax.set_ylim(y_min, y_max)
    # ax.set_ylim(-20,15)
    print("ymin: {}, ymax: {}".format(y_min,y_max))
    

    # ── 9. Save / show ────────────────────────────────────────
    if SAVE_PATH:
        fig.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
        print(f"Saved → {SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
