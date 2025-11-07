"""
Fiber volley feature detection implemented as an Analyzer subclass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

from ephys_toolbox.src.analyzers import Analyzer, AnalyzerConfig
from ephys_toolbox.src.results import RecordingContext


class FiberVolleyAnalyzer(Analyzer):
    """Detect fiber volley extrema and amplitude for each stimulus intensity."""

    def __init__(self, config: AnalyzerConfig | None = None):
        super().__init__("fiber_volley", config)
        params = self.config.params
        self.window = params.get("window", [0.0, 1.5]) if params else [0.0, 1.5]

    def run(self, context: RecordingContext) -> RecordingContext:
        fv_df = self._calculate(context.averaged)
        context.register_feature(self.name, fv_df)
        return context

    def _calculate(self, abf_df: pd.DataFrame) -> pd.DataFrame:
        t0, t1 = [v / 1000.0 for v in self.window]
        results = []

        for stim, g in abf_df.groupby("stim_intensity"):
            x = g["time"].to_numpy()
            y = g["smooth"].to_numpy()

            start_idx = np.searchsorted(x, t0)
            stop_idx = np.searchsorted(x, t1)
            dy = np.gradient(y, x)

            y_w, dy_w = y[start_idx:stop_idx], dy[start_idx:stop_idx]

            fv_min_idx = None
            neg_peaks, _ = find_peaks(-y_w)
            if neg_peaks.size:
                fv_min_rel = neg_peaks[np.argmin(y_w[neg_peaks])]
                fv_min_idx = start_idx + fv_min_rel

            fv_max_idx = None
            dy_peaks, _ = find_peaks(dy_w)
            if dy_peaks.size >= 2:
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
                "fv_min_v": fv_min_v,
            })

        return pd.DataFrame(results)
