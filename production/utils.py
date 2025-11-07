import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib import RcParams

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import os

import pyabf
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences, filtfilt, savgol_filter, butter
plt.style.use("seaborn-v0_8")

plt.rcParams.update({
    "axes.labelsize": 19,          # axis labels (x/y)
    "axes.titlesize": 21,          # individual subplot titles
    "xtick.labelsize": 16,         # tick labels
    "ytick.labelsize": 16,
    "legend.fontsize": 15,         # legend entries
    "legend.title_fontsize": 17,   # legend title
    "figure.titlesize": 23,        # global suptitle
})
## Helper functions
def to_samples(ms, fs) -> int:
    return int(round(ms * fs / 1000))

def to_ms(samples, fs):
    return samples / fs * 1000

def derivative(y: np.ndarray, fs: float):
    return np.gradient(y) * fs

def logistic(x, A, x0, k):
    """Standard 3-parameter logistic (sigmoidal) function."""
    return A / (1 + np.exp(-(x - x0) / k))

def exp_saturation(x, A, k):
    return A * (1 - np.exp(-k * x))

def extend_palette(base_colors, n):
    base = np.array(base_colors)
    idx = np.linspace(0, len(base)-1, n)
    return ListedColormap(np.array([base[int(round(i)) % len(base)] for i in idx]))

def smooth(x, window_samples=5): # zero phase smoothing
    if window_samples <= 1:
        return np.asarray(x, dtype=float)
    b = np.ones(window_samples) / window_samples
    a = [1]
    padlen = min(3 * (window_samples - 1), len(x) - 1)
    return filtfilt(b, a, x, padlen=padlen)

def savgol(x, window_samples=11, poly=3):
    if window_samples <= 1:
        return np.asarray(x, dtype=float)
    return savgol_filter(x, window_length=window_samples, polyorder=poly)

def butterw(x, cutoff_hz, fs, order=4):
    nyq = 0.5 * fs
    Wn = cutoff_hz / nyq
    b, a = butter(order, Wn, btype='low')
    return filtfilt(b, a, x)

def read_abf(abf_path: str):
    return pyabf.ABF(abf_path)

def switch_channels(abf, channel: int = 0):
    if not isinstance(abf, pyabf.abf.ABF):
        return TypeError("Not an pyabf.abf.ABF object!")
    return abf.setSweep(channel=channel)

def switch_sweep(abf, sweep: int = 0):
    if not isinstance(abf, pyabf.abf.ABF):
        return TypeError("Not an pyabf.abf.ABF object!")
    return abf.setSweep(sweep=sweep)

def set_xaxis_ms(ax, xlim=None):
    """
    Format the x-axis to display milliseconds instead of seconds.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to modify.
    xlim : tuple or None
        Optional (xmin, xmax) in seconds. If provided, will set limits scaled to ms.
    """
    # Convert existing or given limits to ms
    if xlim is None:
        xmin, xmax = ax.get_xlim()
    else:
        xmin, xmax = xlim

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Time (ms)")

    # Format tick labels: seconds → ms
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 1000:.0f}"))

def tidy_results(abf_df: pd.DataFrame,
                 fv_df: Optional[pd.DataFrame],
                 epsp_df: Optional[pd.DataFrame],
                 ps_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Join FV, fEPSP, and PS metrics on 'stim_intensity' into one tidy DataFrame.
    Drops fit_t and fit_v columns from fEPSP results before joining.
    """
    out = abf_df[["stim_intensity"]].drop_duplicates().sort_values("stim_intensity")

    if fv_df is not None and not fv_df.empty:
        out = pd.merge(out, fv_df, on="stim_intensity", how="left")

    if epsp_df is not None and not epsp_df.empty:
        epsp_clean = epsp_df.drop(columns=["fit_t", "fit_v"], errors="ignore")
        out = pd.merge(out, epsp_clean, on="stim_intensity", how="left")

    if ps_df is not None and not ps_df.empty:
        out = pd.merge(out, ps_df, on="stim_intensity", how="left")

    out.attrs.update(abf_df.attrs)
    return out

## Convenience running functions
def run_all(paths,
            stim_intensities,
            stim_window=[0.0,1.0],
            fv_window=[0.0, 1.5],
            epsp_window=[1.5, 5.25],
            ps_lag=3.0,
            repnum=3,
            summarize=False):
    """
    Run the full electrophysiology analysis pipeline on one or many ABF files.

    Parameters
    ----------
    paths : str | list[str]
        Either a folder containing .abf files or a list of file paths.
    stim_intensities : list[int]
        Stimulus intensities (µA) corresponding to sweeps.
    repnum : int, default=3
        Number of repetitions per stimulus.
    summarize : bool, default=False
        If True, combines all results into one summary DataFrame (results_summary.tsv). If False, returns a concatenated per-slice DataFrame.
    """

    if isinstance(paths, (str, Path)):
        p = Path(paths)
        abf_files = sorted(p.glob("*.abf")) if p.is_dir() else [p]
    else:
        abf_files = [Path(f) for f in paths]

    if not abf_files:
        print("No .abf files found.")
        return None

    summary_dfs = []

    for abf_path in abf_files:
        fname = abf_path.stem
        print(f"Processing {fname} ...")

        base_dir = abf_path.parent / fname
        deriv_dir = base_dir / "plots" / "derivative"
        io_dir = base_dir / "plots" / "io"
        other = base_dir / "plots" / "other"
        results_dir = base_dir / "results"
        for d in [deriv_dir, io_dir, results_dir]:
            d.mkdir(parents=True, exist_ok=True)

        tidy = tidy_abf(str(abf_path), stim_intensities=stim_intensities, repnum=repnum)
        norm = normalize_abf(tidy)
        cropped = remove_stim_artifact(norm, stim_window=stim_window)
        avg = average(cropped)  # adds per-stim zero-phase `smooth` column

        fv = calculate_fv(avg, fv_window=fv_window)
        epsp = calculate_epsp(avg, fv, epsp_window=epsp_window)
        ps = calculate_ps(avg, epsp, ps_lag=ps_lag)

        # Derivative plots per stimulus
        for stim, g in avg.groupby("stim_intensity"):
            x = g["time"].to_numpy()
            v = g["smooth"].to_numpy()     # smoothed mean trace
            dv = np.gradient(v, x) / 1000.0        # perfectly aligned derivative (mV/ms)

            fv_pts, epsp_pts, ps_pts = [], [], []

            if fv is not None and (fv.stim_intensity == stim).any():
                row = fv.loc[fv.stim_intensity == stim].iloc[0]
                if np.isfinite(row["fv_min_s"]) and np.isfinite(row["fv_min_v"]):
                    fv_pts.append((row["fv_min_s"], row["fv_min_v"]))
                if np.isfinite(row["fv_max_s"]) and np.isfinite(row["fv_max_v"]):
                    fv_pts.append((row["fv_max_s"], row["fv_max_v"]))

            if epsp is not None and (epsp.stim_intensity == stim).any():
                row = epsp.loc[epsp.stim_intensity == stim].iloc[0]
                if np.isfinite(row["slope_mid_s"]) and np.isfinite(row["slope_mid_v"]):
                    epsp_pts.append((row["slope_mid_s"], row["slope_mid_v"]))
                if np.isfinite(row["epsp_min_s"]) and np.isfinite(row["epsp_min_v"]):
                    epsp_pts.append((row["epsp_min_s"], row["epsp_min_v"]))

            if ps is not None and (ps.stim_intensity == stim).any():
                row = ps.loc[ps.stim_intensity == stim].iloc[0]
                if np.isfinite(row["ps_max_s"]) and np.isfinite(row["ps_max_v"]):
                    ps_pts.append((row["ps_max_s"], row["ps_max_v"]))

            savepath = deriv_dir / f"stim_{stim}.png"
            fig = plot_derivative(
                x, v, dv,
                fv_points=fv_pts if fv_pts else None,
                epsp_points=epsp_pts if epsp_pts else None,
                ps_points=ps_pts if ps_pts else None,
                fv_window=fv_window,
                epsp_window=epsp_window,
                label=f"{stim} µA",
                savepath=str(savepath),
                show=False
            )
            plt.close(fig)

        # IO curve
        fig1 = plot_fp(avg, fv_df=fv, epsp_df=epsp, ps_df=ps,
                            savepath=io_dir / "evoked_fp.png", show=False)
        plt.close(fig1)

        fig2, _ = plot_excitability(fv_df=fv, epsp_df=epsp, savepath=other / "excitability.png", show=False)
        plt.close(fig2)

        fig3, _ = plot_presynaptic(fv_df=fv, savepath=other/ "presynaptic.png", show=False)
        plt.close(fig3)

        fig4, _ = plot_io_curve(epsp_df=epsp, savepath=io_dir / "io_curve.png", show=False)
        plt.close(fig4)

        fig5 = combo_plots(fv, epsp, ps, savepath=other/ "combo.png", show=False)
        plt.close(fig5)

        first_ps_stim = ps_onset(ps)
        avg.attrs["first_ps_stimulus"] = first_ps_stim if first_ps_stim is not None else "None detected"

        results = tidy_results(avg, fv, epsp, ps)
        results_path = results_dir / f"{fname}.csv"

        attr_lines = [f"# {k}: {v}" for k, v in results.attrs.items()]
        with open(results_path, "w", encoding="utf-8") as f:
            if attr_lines:
                f.write("\n".join(attr_lines) + "\n")
            results.to_csv(f, sep=",", index=False)

        print(f"→ Saved results to {results_path}")

        if summarize:
            results.insert(0, "filename", fname)
            summary_dfs.append(results)

    if summarize and summary_dfs:
        summary_df = pd.concat(summary_dfs, ignore_index=True)
        summary_path = Path.cwd() / "results_summary.csv"
        summary_df.to_csv(summary_path, sep=",", index=False)
        print(f"\nSummary written to: {summary_path}")
        return summary_df
    
    return None

ran = run_all(
    paths="C:/Users/bbyer/OneDrive/Documents/UniversityofKentucky/BachstetterLab/ephys_toolbox/ephys_toolbox/data/2025_03_06_0000.abf",
    stim_intensities=[25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600],
    repnum=3,
    summarize=False
)

#TODO: Add ES plot and other plots for results on the poster. You MUST do some concordance and MAE calculations for results too
# EPSP amplitude from zero not from FV end
# Work on smoothing and say validation study to come
# Stim response curve not unique to ephys. In conclusion focus on broader potential impact EEG
# Asking questions of the ephys people at the poster presentation
# fix epsp 25 microamps slope must be before most negative