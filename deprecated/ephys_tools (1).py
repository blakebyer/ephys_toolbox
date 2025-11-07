
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import os

try:
    import pyabf  # to load .abf files
except Exception as e:
    pyabf = None

from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

###############################################################################
# Core helpers
###############################################################################

def _to_samples(ms: float, sampling_hz: float) -> int:
    return int(round((ms/1000.0) * sampling_hz))

def _smooth_once(x: np.ndarray, window_samples: int = 5) -> np.ndarray:
    w = max(3, int(window_samples))
    return uniform_filter1d(x.astype(float), size=w, mode='nearest')

def _first_derivative(x: np.ndarray) -> np.ndarray:
    return np.diff(x, prepend=x[0])

def _scale_derivative_to_mV_per_ms(dv_per_sample: float, sampling_hz: float) -> float:
    return dv_per_sample * (sampling_hz / 1000.0)

###############################################################################
# ABF loading and sweep organization
###############################################################################

def load_abf_into_sweeps(
    abf_path: str,
    stim_artifact_ms: float = 1.0,
    pre_artifact_baseline_ms: float = 0.3,
    repnum: int = 3,
    current_sweep_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if pyabf is None:
        raise RuntimeError("pyabf not available in this environment. Install pyabf to use this loader.")
    abf = pyabf.ABF(abf_path)
    sampling_hz = float(abf.dataRate)

    start_idx = _to_samples(stim_artifact_ms, sampling_hz)
    baseline_idx = _to_samples(pre_artifact_baseline_ms, sampling_hz)

    if current_sweep_list is None:
        current_sweep_list = [f"amp{x}" for x in (25,50,75,100,150,200,250,300,400,500,600)]

    abf.setSweep(sweepNumber=0, channel=0)
    n_points = abf.sweepPointCount
    n_sweeps = abf.sweepCount

    n_currents = len(current_sweep_list)
    if n_currents * repnum != n_sweeps:
        raise ValueError(f"repnum * len(current_sweep_list) != sweepCount ({repnum*n_currents} != {n_sweeps}). Adjust inputs.")

    raw_groups: Dict[str, np.ndarray] = {}
    avg_groups: Dict[str, np.ndarray] = {}
    baseline_by_current: Dict[str, float] = {}

    sweep_idx = 0
    for label in current_sweep_list:
        reps = []
        bases = []
        for _ in range(repnum):
            abf.setSweep(sweep_idx, channel=0)
            trace = abf.sweepY.copy()
            base_mean = float(np.mean(trace[:baseline_idx])) if baseline_idx > 0 else 0.0
            bases.append(base_mean)
            trace = trace[start_idx:]
            reps.append(trace)
            sweep_idx += 1
        group = np.column_stack(reps)  # [T x repnum]
        raw_groups[label] = group
        avg_groups[label] = np.mean(group, axis=1)
        baseline_by_current[label] = float(np.mean(bases))

    T = len(next(iter(avg_groups.values())))
    time_axis_ms = np.arange(T) * (1000.0 / sampling_hz)

    return dict(
        sampling_hz=sampling_hz,
        raw_groups=raw_groups,
        avg_groups=avg_groups,
        baseline_by_current=baseline_by_current,
        time_axis_ms=time_axis_ms,
        current_sweep_list=current_sweep_list,
        repnum=repnum,
    )

###############################################################################
# Feature extraction
###############################################################################

def find_fv_indices_from_derivative(
    d1_smooth: np.ndarray,
    sampling_hz: float,
    FV_min_window_ms: float = 0.1,
    FV_max_window_ms: float = 2.5,
    FV_start_max_ms: float = 1.5,
    FV_end_min_ms: float = 1.2,
    last_good: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    w_min = _to_samples(FV_min_window_ms, sampling_hz)
    w_max = _to_samples(FV_max_window_ms, sampling_hz)
    start_max = _to_samples(FV_start_max_ms, sampling_hz)
    end_min = _to_samples(FV_end_min_ms, sampling_hz)

    peaks, props = find_peaks(d1_smooth)
    if peaks.size >= 2:
        mask = (peaks >= w_min) & (peaks <= w_max)
        win_peaks = peaks[mask]
        if win_peaks.size >= 2:
            prominences = props.get("prominences", None)
            if prominences is None:
                prominences = np.ones_like(peaks, dtype=float)
            win_prom = prominences[mask] if len(prominences)==len(peaks) else np.ones_like(win_peaks, dtype=float)
            order = np.argsort(win_prom)[::-1]
            p1, p2 = np.sort(win_peaks[order[:2]])
            start = p1 if p1 <= start_max else (last_good[0] if last_good else w_min)
            end   = p2 if p2 >= end_min   else (last_good[1] if last_good else w_max)
            return int(start), int(end)

    if last_good is not None:
        return int(last_good[0]), int(last_good[1])
    return int(w_min), int(w_max)

def epsp_minimum(trace_smooth: np.ndarray, d1_smooth: np.ndarray) -> Tuple[float, int]:
    max_df_idx = int(np.argmax(d1_smooth))
    epsp_idx = int(np.argmin(trace_smooth[: max_df_idx+1]))
    return float(trace_smooth[epsp_idx]), epsp_idx

def epsp_slope_mV_per_ms(
    trace_smooth: np.ndarray,
    d1_smooth: np.ndarray,
    sampling_hz: float,
    fv_end_idx: int,
    epsp_min_idx: int,
    window_half: int = 4,
) -> Tuple[float, int]:
    lo = int(max(0, fv_end_idx + 5))
    hi = int(max(lo+1, epsp_min_idx))
    if hi - lo < 3:
        target_idx = int(np.argmin(d1_smooth))
    else:
        local = d1_smooth[lo:hi]
        target_local_idx = int(np.argmin(local))
        target_idx = lo + target_local_idx

    i0 = max(0, target_idx - window_half)
    i1 = min(len(d1_smooth), target_idx + window_half + 1)
    local_derivs = d1_smooth[i0:i1]
    slope_per_sample = float(np.median(local_derivs))
    slope_mV_ms = _scale_derivative_to_mV_per_ms(slope_per_sample, sampling_hz)
    return slope_mV_ms, target_idx

def detect_population_spike_threshold(
    slopes_by_current: List[float],
    method: str = "zscore",
    z: float = 2.5,
) -> Optional[int]:
    arr = np.asarray(slopes_by_current, dtype=float)
    diffs = np.diff(np.abs(arr))
    if method == "zscore" and len(diffs) >= 3:
        mu = float(np.mean(diffs[:max(3, len(diffs)//3)]))
        sd = float(np.std(diffs[:max(3, len(diffs)//3)]) + 1e-9)
        hits = np.where(diffs > mu + z*sd)[0]
        if hits.size > 0:
            return int(hits[0] + 1)
    return None

###############################################################################
# Full per-file analysis
###############################################################################

def analyze_abf_file(
    abf_path: str,
    repnum: int = 3,
    stim_artifact_ms: float = 1.0,
    pre_artifact_baseline_ms: float = 0.3,
    FV_min_window_ms: float = 0.1,
    FV_max_window_ms: float = 2.5,
    FV_start_max_ms: float = 1.5,
    FV_end_min_ms: float = 1.2,
    smooth_window_samples: int = 5,
    current_sweep_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    info = load_abf_into_sweeps(
        abf_path=abf_path,
        stim_artifact_ms=stim_artifact_ms,
        pre_artifact_baseline_ms=pre_artifact_baseline_ms,
        repnum=repnum,
        current_sweep_list=current_sweep_list,
    )
    sampling_hz = info['sampling_hz']
    avg_groups = info['avg_groups']
    baseline_by_current = info['baseline_by_current']
    current_sweep_list = info['current_sweep_list']

    last_good = None
    rows = []
    epsp_vals = []
    slopes_vals = []
    fv_vals = []

    for cur in current_sweep_list:
        avg = avg_groups[cur]
        avg_s = _smooth_once(avg, smooth_window_samples)
        d1 = _first_derivative(avg_s)
        fv_start, fv_end = find_fv_indices_from_derivative(
            d1_smooth=d1, sampling_hz=sampling_hz,
            FV_min_window_ms=FV_min_window_ms, FV_max_window_ms=FV_max_window_ms,
            FV_start_max_ms=FV_start_max_ms, FV_end_min_ms=FV_end_min_ms,
            last_good=last_good,
        )
        last_good = (fv_start, fv_end)

        epsp_min, epsp_idx = epsp_minimum(avg_s, d1)
        epsp_min_corr = float(epsp_min - baseline_by_current[cur])

        slope_mV_ms, slope_idx = epsp_slope_mV_per_ms(
            avg_s, d1, sampling_hz, fv_end_idx=fv_end, epsp_min_idx=epsp_idx
        )

        fv_amp = float(avg_s[fv_start] - avg_s[fv_end])

        rows.append(dict(
            stim_current=cur,
            fv_start_idx=fv_start,
            fv_end_idx=fv_end,
            fv_amplitude=fv_amp,
            epsp_min_mV=epsp_min_corr,
            epsp_min_idx=epsp_idx,
            epsp_slope_mV_per_ms=slope_mV_ms,
            slope_idx=slope_idx,
        ))
        epsp_vals.append(epsp_min_corr)
        slopes_vals.append(slope_mV_ms)
        fv_vals.append(fv_amp)

    per_current = pd.DataFrame(rows)

    pop_idx = detect_population_spike_threshold(slopes_vals, z=2.5)
    pop_threshold_slope = slopes_vals[pop_idx] if pop_idx is not None else np.nan
    pop_threshold_current = current_sweep_list[pop_idx] if pop_idx is not None else None

    last3 = per_current.tail(3)
    last3_ratio = (last3['epsp_min_mV'] / last3['fv_amplitude']).values

    summary = dict(
        max_epsp_amplitude_mV=float(np.min(per_current['epsp_min_mV'])),
        epsp_slope_at_max_epsp=float(per_current.loc[per_current['epsp_min_mV'].idxmin(), 'epsp_slope_mV_per_ms']),
        fv_amplitude_by_current=fv_vals,
        epsp_slope_by_current=slopes_vals,
        pop_spike_threshold_slope=float(pop_threshold_slope) if not np.isnan(pop_threshold_slope) else None,
        pop_spike_threshold_current=pop_threshold_current,
        epsp_fv_ratio_last3=last3_ratio.tolist(),
    )

    return dict(
        sampling_hz=sampling_hz,
        per_current=per_current,
        summary=summary,
        current_sweep_list=current_sweep_list,
        avg_groups=avg_groups,
        time_axis_ms=info['time_axis_ms'],
    )

def run_all(
    abf_path: str,
    repnum: int = 3,
    current_sweep_list: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    return analyze_abf_file(
        abf_path,
        repnum=repnum,
        current_sweep_list=current_sweep_list,
        stim_artifact_ms=kwargs.get('stim_artifact_ms', 1.0),
        pre_artifact_baseline_ms=kwargs.get('pre_artifact_baseline_ms', 0.3),
        FV_min_window_ms=kwargs.get('FV_min_window_ms', 0.1),
        FV_max_window_ms=kwargs.get('FV_max_window_ms', 2.5),
        FV_start_max_ms=kwargs.get('FV_start_max_ms', 1.5),
        FV_end_min_ms=kwargs.get('FV_end_min_ms', 1.2),
        smooth_window_samples=kwargs.get('smooth_window_samples', 5),
    )

###############################################################################
# Plotting (return figures; optional saving)
###############################################################################

def plot_traces_with_features(
    time_ms: np.ndarray,
    avg_groups: Dict[str, np.ndarray],
    per_current: pd.DataFrame,
    show: bool = True,
    savepath: Optional[str] = None,
):
    fig = plt.figure(figsize=(9, 6))
    for _, row in per_current.iterrows():
        cur = row['stim_current']
        avg = avg_groups[cur]
        plt.plot(time_ms[:len(avg)], avg, linewidth=1, label=cur)
        plt.plot(time_ms[int(row['fv_start_idx'])], avg[int(row['fv_start_idx'])], 'o')
        plt.plot(time_ms[int(row['fv_end_idx'])],   avg[int(row['fv_end_idx'])],   's')
        plt.plot(time_ms[int(row['epsp_min_idx'])], avg[int(row['epsp_min_idx'])], 'v')
        plt.plot(time_ms[int(row['slope_idx'])],    avg[int(row['slope_idx'])],    '*')
    plt.xlabel('Time (ms)')
    plt.ylabel('mV')
    plt.title('Averaged traces with FV/EPSP markers')
    plt.legend(loc='best')
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig

def plot_io_curves(
    per_current: pd.DataFrame,
    current_sweep_list: List[str],
    show: bool = True,
    savepath_epsp: Optional[str] = None,
    savepath_fv: Optional[str] = None,
):
    x = np.arange(len(current_sweep_list))

    fig1 = plt.figure()
    plt.plot(x, per_current['epsp_min_mV'].values, marker='o')
    plt.xticks(x, current_sweep_list, rotation=45)
    plt.xlabel('Stimulus (label)')
    plt.ylabel('EPSP amplitude (mV)')
    plt.title('IO curve (EPSP amplitude)')
    plt.tight_layout()
    if savepath_epsp:
        fig1.savefig(savepath_epsp, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    fig2 = plt.figure()
    plt.plot(x, per_current['fv_amplitude'].values, marker='s')
    plt.xticks(x, current_sweep_list, rotation=45)
    plt.xlabel('Stimulus (label)')
    plt.ylabel('FV amplitude (mV)')
    plt.title('FV amplitude vs stimulus')
    plt.tight_layout()
    if savepath_fv:
        fig2.savefig(savepath_fv, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    return fig1, fig2

def plot_epsp_slope_curve(
    per_current: pd.DataFrame,
    current_sweep_list: List[str],
    show: bool = True,
    savepath: Optional[str] = None,
):
    x = np.arange(len(current_sweep_list))
    fig = plt.figure()
    plt.plot(x, per_current['epsp_slope_mV_per_ms'].values, marker='^')
    plt.xticks(x, current_sweep_list, rotation=45)
    plt.xlabel('Stimulus (label)')
    plt.ylabel('EPSP slope (mV/ms)')
    plt.title('EPSP slope vs stimulus')
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig

def plot_all_figures(
    res: Dict[str, Any],
    outprefix: Optional[str] = None,
    show: bool = True,
) -> List[str]:
    """Generate and optionally save all figures for a single analyzed ABF result.
    Returns list of saved file paths (if outprefix is provided).
    """
    saved = []
    save_traces = f"{outprefix}_traces.png" if outprefix else None
    save_io_epsp = f"{outprefix}_io_epsp.png" if outprefix else None
    save_io_fv   = f"{outprefix}_io_fv.png" if outprefix else None
    save_slope   = f"{outprefix}_slope.png" if outprefix else None

    plot_traces_with_features(res['time_axis_ms'], res['avg_groups'], res['per_current'],
                              show=show, savepath=save_traces)
    plot_io_curves(res['per_current'], res['current_sweep_list'],
                   show=show, savepath_epsp=save_io_epsp, savepath_fv=save_io_fv)
    plot_epsp_slope_curve(res['per_current'], res['current_sweep_list'],
                          show=show, savepath=save_slope)
    for p in [save_traces, save_io_epsp, save_io_fv, save_slope]:
        if p:
            saved.append(p)
    return saved

def plot_all_for_abf(
    abf_path: str,
    repnum: int = 3,
    current_sweep_list: Optional[List[str]] = None,
    outdir: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> Dict[str, Any]:
    res = run_all(abf_path, repnum=repnum, current_sweep_list=current_sweep_list, **kwargs)
    outprefix = None
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        outprefix = str(Path(outdir) / Path(abf_path).stem)
    plot_all_figures(res, outprefix=outprefix, show=show)
    return res

def batch_plot_in_dir(
    abf_dir: str,
    pattern: str = "*.abf",
    outdir: Optional[str] = None,
    show: bool = False,
    **kwargs
) -> List[Dict[str, Any]]:
    """Process every ABF in a directory and (optionally) save all figures.
    Returns list of results dicts (one per file).
    """
    abf_paths = sorted([str(p) for p in Path(abf_dir).glob(pattern)])
    results = []
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    for p in abf_paths:
        res = plot_all_for_abf(p, outdir=outdir, show=show, **kwargs)
        results.append(res)
    return results

res = plot_all_for_abf(
    abf_path="C:/Users/bbyer/OneDrive/Documents/UniversityofKentucky/BachstetterLab/ephys_toolbox/ephys_toolbox/data/2025_03_06_0000.abf",
    repnum=3,
    current_sweep_list=["amp25","amp50","amp75","amp100","amp150","amp200","amp250","amp300","amp400","amp500","amp600"],
    # optional knobs:
    stim_artifact_ms=1.0, pre_artifact_baseline_ms=0.3,
    FV_min_window_ms=0.1, FV_max_window_ms=2.5,
    FV_start_max_ms=1.5,  FV_end_min_ms=1.2,
    smooth_window_samples=5,
    outdir="plots",    # omit to only show without saving
    show=True          # set False in big batches
)