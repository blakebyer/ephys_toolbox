import pyabf 
import pandas as pd
from ephys_toolbox.production.utils import smooth
from typing import List, Optional
import numpy as np
from pathlib import Path
import os

def read_abf(abf_path: str):
    return pyabf.ABF(abf_path)

## Load ABF
def tidy_abf(abf_path: str, stim_intensities: List[int], repnum: int = 3) -> pd.DataFrame:
    abf = pyabf.ABF(abf_path)
    fs = abf.sampleRate
    sweeps = abf.sweepList
    sweep_groups = [sweeps[i:i + repnum] for i in range(0, len(sweeps), repnum)]

    records = []
    for stim, sweep_idxs in zip(stim_intensities, sweep_groups):
        for rep, sweep_i in enumerate(sweep_idxs, start=1):
            abf.setSweep(sweep_i)
            x = abf.sweepX.astype("float32")
            y = abf.sweepY.astype("float32")
            records.extend(zip([stim]*len(x), [rep]*len(x), x, y))

    df = pd.DataFrame(records, columns=["stim_intensity", "sweep", "time", "value"])
    df.attrs["abf_path"] = abf_path
    df.attrs["stim_intensities"] = stim_intensities
    df.attrs["sampling_rate"] = fs

    return df

## Normalize abf so that f(0) = 0
def normalize_abf(abf_df: pd.DataFrame) -> pd.DataFrame:
    df = abf_df.copy()
    df["baseline"] = df.groupby(["stim_intensity", "sweep"])["value"].transform("first")
    df["time0"] = df.groupby(["stim_intensity", "sweep"])["time"].transform("first")
    df["value"] = df["value"] - df["baseline"]
    df["time"] = df["time"] - df["time0"]
    return df.drop(columns=["baseline", "time0"])

def average(abf_df: pd.DataFrame) -> pd.DataFrame:
    avg = (abf_df.groupby(["stim_intensity", "time"])
        .agg(
            mean=("value", "mean"),
            sem=("value", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        )
        .reset_index()
        .sort_values(["stim_intensity", "time"])
    )
    avg.attrs.update(abf_df.attrs)

    # per-stim zero-phase smoothing (no cross-stim bleed)
    avg["smooth"] = np.nan
    for stim, g in avg.groupby("stim_intensity"):
        avg.loc[g.index, "smooth"] = smooth(g["mean"].to_numpy())

    return avg

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