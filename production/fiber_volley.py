"""
TODO: Fiber volley methods:
- derivative based approach top two peaks based on prominence
- template correlation. Averaged FV across multiple of the same stimulus
- cursor method 
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences


def calculate_fv(abf_df, fv_window: list[float] = [0.0, 1.5]):
    """
    Detect Fiber Volley (FV) amplitude from averaged traces.
    Prefers a true y-minimum (dy crosses 0 from -→+) inside the FV window for fv_min.
    Falls back to derivative-peak concavity method if no true minimum is found.

    Assumes df columns: ['stim_intensity', 'time', 'mean', 'smooth'] and abf_df.attrs['sampling_rate'].
    """
    results = []

    for stim, g in abf_df.groupby("stim_intensity"):
        x = g["time"].to_numpy()
        y = g["smooth"].to_numpy()

        # Window indices based on actual time
        t0, t1 = [v / 1000.0 for v in fv_window]  # convert ms to s
        start_idx = np.searchsorted(x, t0)
        stop_idx  = np.searchsorted(x, t1)

        dy = np.gradient(y, x)

        y_w, dy_w = y[start_idx:stop_idx], dy[start_idx:stop_idx]

        # Prefer a true minimum
        fv_min_idx = None
        neg_peaks, _ = find_peaks(-y_w)
        if len(neg_peaks):
            fv_min_rel = neg_peaks[np.argmin(y_w[neg_peaks])]
            fv_min_idx = start_idx + fv_min_rel

        # Derivative peaks for fv_max
        fv_max_idx = None
        dy_peaks, _ = find_peaks(dy_w)
        if len(dy_peaks) >= 2:
            prom = peak_prominences(dy_w, dy_peaks)[0]
            top2_rel = dy_peaks[np.argsort(prom)[-2:]]
            abs2 = start_idx + top2_rel
            fv_max_idx = abs2[np.argmax(y[abs2])]
            if fv_min_idx is None:
                fv_min_idx = abs2[np.argmin(y[abs2])]

        fv_amp = fv_min_s = fv_min_v = fv_max_s = fv_max_v = np.nan
        if fv_min_idx is not None:
            fv_min_s, fv_min_v = x[fv_min_idx], y[fv_min_idx]
        if fv_max_idx is not None:
            fv_max_s, fv_max_v = x[fv_max_idx], y[fv_max_idx]
        if np.isfinite(fv_max_v) and np.isfinite(fv_min_v):
            fv_amp = abs(fv_max_v - fv_min_v)

        results.append({
            "stim_intensity": stim,
            "fv_amp": fv_amp,
            "fv_max_s": fv_max_s,
            "fv_max_v": fv_max_v,
            "fv_min_s": fv_min_s,
            "fv_min_v": fv_min_v
        })

    return pd.DataFrame(results)