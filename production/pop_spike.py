"""
TODO: 
- derivative based approach: peak (dy=0) in specified range
- template match ps: normalized cross-correlation with a PS waveform template; accept if NCC ≥ τ, then measure height on raw trace. 
- ratio qc: normal derivative based approach but require PS height to exceed a fraction of EPSP amplitude (or mV)
- cursor method
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def calculate_ps(abf_df, epsp_df, ps_lag=3.0, height=0.2):
    """
    Detect Population Spike (PS) deflection after the fEPSP trough.
    Uses simple peak-based logic:
        - Find peaks in the voltage trace after the fEPSP minimum.
        - Keep only those ≥ height mV above the fEPSP minimum.
    """
    results = []
    for stim, g in abf_df.groupby("stim_intensity"):
        x = g["time"].to_numpy()
        y = g["smooth"].to_numpy()

        epsp_row = epsp_df.loc[epsp_df.stim_intensity == stim].iloc[0]
        epsp_min_s = epsp_row["epsp_min_s"]
        epsp_min_v = epsp_row["epsp_min_v"]

        # Time window after fEPSP trough
        start_idx = np.argmin(np.abs(x - epsp_min_s))
        stop_idx = np.searchsorted(x, epsp_min_s + ps_lag / 1000.0)

        yw = y[start_idx:stop_idx]
        peaks, _ = find_peaks(yw)

        ps_max_s = ps_max_v = ps_amp = np.nan
        ps_present = False

        if peaks.size:
            rel = yw[peaks] - epsp_min_v
            keep = rel >= height
            if np.any(keep):
                pk_rel = peaks[np.argmax(rel[keep])]
                ps_idx = start_idx + pk_rel
                ps_max_s, ps_max_v = x[ps_idx], y[ps_idx]
                ps_amp = ps_max_v - epsp_min_v
                ps_present = True

        results.append({
            "stim_intensity": stim,
            "ps_present": ps_present,
            "ps_min_s": epsp_min_s,
            "ps_min_v": epsp_min_v,
            "ps_max_s": ps_max_s,
            "ps_max_v": ps_max_v,
            "ps_amp": ps_amp
        })

    return pd.DataFrame(results)

def ps_onset(ps_df: pd.DataFrame):
    if ps_df is None or ps_df.empty:
        return None
    hits = ps_df.loc[ps_df["ps_present"], "stim_intensity"]
    return None if hits.empty else hits.min()
