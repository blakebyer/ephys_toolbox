"""
Population spike analyzer implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ephys_toolbox.src.analyzers import Analyzer, AnalyzerConfig
from ephys_toolbox.src.results import RecordingContext


class PopSpikeAnalyzer(Analyzer):
    """Detects population spikes following the EPSP trough."""

    def __init__(self, config: AnalyzerConfig | None = None):
        super().__init__("pop_spike", config)
        params = self.config.params
        self.ps_lag = params.get("lag_ms", 3.0) if params else 3.0
        self.height = params.get("height", 0.2) if params else 0.2

    def run(self, context: RecordingContext) -> RecordingContext:
        epsp_df = context.get_feature("epsp")
        if epsp_df is None or epsp_df.empty:
            raise ValueError("EPSP results required before running PopSpikeAnalyzer.")
        ps_df = self._calculate(context.averaged, epsp_df)
        context.register_feature(self.name, ps_df)
        return context

    def _calculate(self, abf_df: pd.DataFrame, epsp_df: pd.DataFrame) -> pd.DataFrame:
        results = []

        for stim, g in abf_df.groupby("stim_intensity"):
            x = g["time"].to_numpy()
            y = g["smooth"].to_numpy()

            epsp_row = epsp_df.loc[epsp_df.stim_intensity == stim].iloc[0]
            epsp_min_s = epsp_row["epsp_min_s"]
            epsp_min_v = epsp_row["epsp_min_v"]

            start_idx = np.argmin(np.abs(x - epsp_min_s))
            stop_idx = np.searchsorted(x, epsp_min_s + self.ps_lag / 1000.0)

            yw = y[start_idx:stop_idx]
            peaks, _ = find_peaks(yw)

            ps_max_s = ps_max_v = ps_amp = np.nan
            ps_present = False

            if peaks.size:
                rel = yw[peaks] - epsp_min_v
                keep = rel >= self.height
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
                "ps_amp": ps_amp,
            })

        return pd.DataFrame(results)


def ps_onset(ps_df: pd.DataFrame):
    if ps_df is None or ps_df.empty:
        return None
    hits = ps_df.loc[ps_df["ps_present"], "stim_intensity"]
    return None if hits.empty else hits.min()
