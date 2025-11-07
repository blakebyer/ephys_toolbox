"""
TODO: Slope methods:
- cursor method (deltaV/deltaT between two points)
- 20/80 + linear fit slope
- maximum magnitude slope in descending/ascending arm
- from end of fv to epsp amplitude deltaV/deltaT
"""
import numpy as np
import pandas as pd
from typing import Optional

def calculate_epsp(
    abf_df: pd.DataFrame,
    fv_df: Optional[pd.DataFrame],
    epsp_window: list[float] = [1.5, 5.25],
    fit_distance: int = 4
) -> pd.DataFrame:
    """
    Calculates fEPSP slope given information about the fiber volley.
    Uses filtfilt-based smoothing to prevent time shift so that the
    most-negative dV/dt corresponds exactly to the steepest fEPSP segment.
    """
    results = []

    for stim, g in abf_df.groupby("stim_intensity"):
        x = g["time"].to_numpy()
        y = g["smooth"].to_numpy()

        # Window indices from actual time (post-crop safe)
        t0, t1 = [v / 1000.0 for v in epsp_window] # ms to s
        start_idx = np.searchsorted(x, t0)
        stop_idx  = np.searchsorted(x, t1)

        # Derivative in mV/s, perfectly aligned
        dy_full = np.gradient(y, x)

        epsp_segment = slice(start_idx, stop_idx)
        slope_center_idx = start_idx + int(np.argmin(dy_full[epsp_segment]))
        epsp_min_idx     = start_idx + int(np.argmin(y[epsp_segment]))

        # Baseline from FV
        if fv_df is not None and (fv_df.stim_intensity == stim).any():
            fv_row = fv_df.loc[fv_df.stim_intensity == stim].iloc[0]
            fv_min_v = fv_row["fv_min_v"]
            fv_amp   = fv_row["fv_amp"]
        else:
            fv_min_v = np.median(y[:start_idx])
            fv_amp = np.nan

        epsp_min_s, epsp_min_v = x[epsp_min_idx], y[epsp_min_idx]
        epsp_amp = abs(epsp_min_v)

        # Linear fit ±fit_distance samples around slope minimum
        i0 = max(0, slope_center_idx - fit_distance)
        i1 = min(len(y) - 1, slope_center_idx + fit_distance)
        t_win = x[i0:i1 + 1] - x[slope_center_idx]
        v_win = y[i0:i1 + 1]
        m, b = np.polyfit(t_win, v_win, 1)
        v_fit = m * t_win + b
        r2 = np.corrcoef(t_win, v_win)[0, 1] ** 2 if np.std(v_win) else np.nan

        results.append({
            "stim_intensity": stim,
            "epsp_min_s": float(epsp_min_s),
            "epsp_min_v": float(epsp_min_v),
            "slope_mid_s": float(x[slope_center_idx]),
            "slope_mid_v": float(y[slope_center_idx]),
            "epsp_amp": float(epsp_amp),
            "epsp_to_fv": float(abs(m) / fv_amp) if fv_amp and m else np.nan,
            "epsp_slope": float(abs(m)), # canonical mV/s
            "epsp_slope_ms": float(abs(m) / 1000.0),     # mV/ms (for plotting)
            "epsp_r2": float(r2),
            "fit_t": list(x[i0:i1 + 1]),
            "fit_v": list(v_fit),
        })

    return pd.DataFrame(results)