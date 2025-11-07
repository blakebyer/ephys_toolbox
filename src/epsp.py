"""
fEPSP slope analyzer implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from ephys_toolbox.src.analyzers import Analyzer, AnalyzerConfig
from ephys_toolbox.src.results import RecordingContext


class EpspAnalyzer(Analyzer):
    """Computes EPSP minima and slopes given a fiber volley baseline."""

    def __init__(self, config: AnalyzerConfig | None = None):
        super().__init__("epsp", config)
        params = self.config.params
        self.window = params.get("window", [1.5, 5.25]) if params else [1.5, 5.25]
        self.fit_distance = params.get("fit_distance", 4) if params else 4

    def run(self, context: RecordingContext) -> RecordingContext:
        fv_df = context.get_feature("fiber_volley")
        epsp_df = self._calculate(context.averaged, fv_df)
        context.register_feature(self.name, epsp_df)
        return context

    def _calculate(
        self,
        abf_df: pd.DataFrame,
        fv_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        t0, t1 = [v / 1000.0 for v in self.window]
        results = []

        for stim, g in abf_df.groupby("stim_intensity"):
            x = g["time"].to_numpy()
            y = g["smooth"].to_numpy()

            start_idx = np.searchsorted(x, t0)
            stop_idx = np.searchsorted(x, t1)
            dy_full = np.gradient(y, x)

            epsp_segment = slice(start_idx, stop_idx)
            slope_center_idx = start_idx + int(np.argmin(dy_full[epsp_segment]))
            epsp_min_idx = start_idx + int(np.argmin(y[epsp_segment]))

            if fv_df is not None and (fv_df.stim_intensity == stim).any():
                fv_row = fv_df.loc[fv_df.stim_intensity == stim].iloc[0]
                fv_amp = fv_row["fv_amp"]
            else:
                fv_amp = np.nan

            epsp_min_s, epsp_min_v = x[epsp_min_idx], y[epsp_min_idx]
            epsp_amp = abs(epsp_min_v)

            i0 = max(0, slope_center_idx - self.fit_distance)
            i1 = min(len(y) - 1, slope_center_idx + self.fit_distance)
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
                "epsp_slope": float(abs(m)),
                "epsp_slope_ms": float(abs(m) / 1000.0),
                "epsp_r2": float(r2),
                "fit_t": list(x[i0:i1 + 1]),
                "fit_v": list(v_fit),
            })

        return pd.DataFrame(results)
