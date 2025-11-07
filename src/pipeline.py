from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ephys_toolbox.src.analyzers import Analyzer
from ephys_toolbox.src.plots import (
    plot_derivative,
    plot_excitability,
    plot_fp,
    plot_io_curve,
    plot_es_curve,
    plot_presynaptic,
)
from ephys_toolbox.src.pop_spike import ps_onset
from ephys_toolbox.src.read_write import average, normalize_abf, tidy_abf, tidy_results
from ephys_toolbox.src.results import RecordingContext
from ephys_toolbox.src.stim_artifact import remove_stim_artifact


@dataclass
class PipelineConfig:
    paths: Sequence[str]
    stim_intensities: Sequence[int]
    repnum: int = 3
    stim_window: List[float] = field(default_factory=lambda: [0.0, 1.0])
    fv_window: List[float] = field(default_factory=lambda: [0.0, 1.5])
    epsp_window: List[float] = field(default_factory=lambda: [1.5, 5.25])
    ps_lag: float = 3.0
    summarize: bool = False
    output_dir: str | None = None


class Pipeline:
    """High-level orchestrator tying loaders, analyzers, plots, and exports together."""

    def __init__(self, config: PipelineConfig, analyzers: Iterable[Analyzer]):
        self.config = config
        self.analyzers = [an for an in analyzers if an.config.enabled]

    def run(self):
        summary_tables: List[pd.DataFrame] = []

        for abf_path in self._resolve_paths(self.config.paths):
            base_dir = (
                Path(self.config.output_dir) / abf_path.stem
                if self.config.output_dir
                else abf_path.parent / abf_path.stem
            )
            deriv_dir = base_dir / "plots" / "derivative"
            io_dir = base_dir / "plots" / "io"
            other_dir = base_dir / "plots" / "other"
            results_dir = base_dir / "results"
            for folder in (deriv_dir, io_dir, other_dir, results_dir):
                folder.mkdir(parents=True, exist_ok=True)

            context = self._prepare_context(abf_path)
            context.metadata["directories"] = {
                "derivative": deriv_dir,
                "io": io_dir,
                "other": other_dir,
                "results": results_dir,
            }

            for analyzer in self.analyzers:
                context = analyzer.run(context)

            fv = context.get_feature("fiber_volley")
            epsp = context.get_feature("epsp")
            ps = context.get_feature("pop_spike")

            self._make_derivative_plots(context, fv, epsp, ps, deriv_dir)
            self._make_summary_plots(context, fv, epsp, ps, io_dir, other_dir)

            first_ps_stim = ps_onset(ps) if ps is not None else None
            context.averaged.attrs["first_ps_stimulus"] = (
                first_ps_stim if first_ps_stim is not None else "None detected"
            )

            results = tidy_results(context.averaged, fv, epsp, ps)
            results_path = results_dir / f"{abf_path.stem}.csv"
            attr_lines = [f"# {k}: {v}" for k, v in results.attrs.items()]
            with open(results_path, "w", encoding="utf-8") as f:
                if attr_lines:
                    f.write("\n".join(attr_lines) + "\n")
                results.to_csv(f, index=False)

            if self.config.summarize:
                results.insert(0, "filename", abf_path.stem)
                summary_tables.append(results)

        if self.config.summarize and summary_tables:
            summary_df = pd.concat(summary_tables, ignore_index=True)
            summary_path = Path.cwd() / "results_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            return summary_df

        return None

    def _prepare_context(self, abf_path: Path) -> RecordingContext:
        tidy = tidy_abf(str(abf_path), stim_intensities=self.config.stim_intensities, repnum=self.config.repnum)
        norm = normalize_abf(tidy)
        cropped = remove_stim_artifact(norm, stim_window=self.config.stim_window)
        avg = average(cropped)
        metadata = {
            "stim_intensities": self.config.stim_intensities,
            "sampling_rate": avg.attrs.get("sampling_rate"),
        }
        return RecordingContext(abf_path=abf_path, tidy=tidy, averaged=avg, metadata=metadata)

    def _make_derivative_plots(self, context, fv, epsp, ps, deriv_dir: Path):
        avg = context.averaged
        for stim, g in avg.groupby("stim_intensity"):
            x = g["time"].to_numpy()
            v = g["smooth"].to_numpy()
            dv = np.gradient(v, x) / 1000.0

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
                x,
                v,
                dv,
                fv_points=fv_pts if fv_pts else None,
                epsp_points=epsp_pts if epsp_pts else None,
                ps_points=ps_pts if ps_pts else None,
                fv_window=self.config.fv_window,
                epsp_window=self.config.epsp_window,
                label=f"{stim} uA",
                savepath=str(savepath),
                show=False,
            )
            context.register_plot("derivative", savepath)
            plt.close(fig)

    def _make_summary_plots(self, context, fv, epsp, ps, io_dir: Path, other_dir: Path):
        fp_path = io_dir / "evoked_fp.png"
        excitability_path = other_dir / "excitability.png"
        presynaptic_path = other_dir / "presynaptic.png"
        io_curve_path = io_dir / "io_curve.png"
        es_path = other_dir / "es_curve.png"

        fig_fp = plot_fp(context.averaged, fv_df=fv, epsp_df=epsp, ps_df=ps, savepath=fp_path, show=False)
        fig_exc, _ = plot_excitability(fv_df=fv, epsp_df=epsp, savepath=excitability_path, show=False)
        fig_pre, _ = plot_presynaptic(fv_df=fv, savepath=presynaptic_path, show=False)
        fig_io, _ = plot_io_curve(epsp_df=epsp, savepath=io_curve_path, show=False)
        fig_es = None
        if ps is not None and not ps.empty and ps.get("ps_present", pd.Series(dtype=bool)).any():
            fig_es, _ = plot_es_curve(epsp_df=epsp, ps_df=ps, savepath=es_path, show=False)

        plot_refs = []
        if fig_fp is not None:
            plot_refs.append(("fp", fig_fp, fp_path))
        if fig_exc is not None:
            plot_refs.append(("excitability", fig_exc, excitability_path))
        if fig_pre is not None:
            plot_refs.append(("presynaptic", fig_pre, presynaptic_path))
        if fig_io is not None:
            plot_refs.append(("io_curve", fig_io, io_curve_path))
        if fig_es:
            plot_refs.append(("es_curve", fig_es, es_path))

        for name, fig, path in plot_refs:
            context.register_plot(name, path)
            plt.close(fig)

    @staticmethod
    def _resolve_paths(paths: Iterable[str]) -> List[Path]:
        resolved: List[Path] = []
        for entry in paths:
            p = Path(entry)
            if p.is_dir():
                resolved.extend(sorted(p.glob("*.abf")))
            else:
                resolved.append(p)
        return resolved
