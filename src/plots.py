from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import RcParams
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit

from ephys_toolbox.src.utils import exp_saturation, logistic, set_xaxis_ms

def extend_palette(base_colors, n):
    base = np.array(base_colors)
    idx = np.linspace(0, len(base)-1, n)
    return ListedColormap(np.array([base[int(round(i)) % len(base)] for i in idx]))

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


@dataclass
class PlotBase:
    figsize: tuple = (14, 7)
    savepath: Optional[str] = None
    show: bool = False

    def _finalize(self, fig):
        if self.savepath:
            Path(self.savepath).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.savepath, dpi=300, bbox_inches="tight")
        if self.show:
            plt.show()
        return fig


@dataclass
class DerivativePlotter(PlotBase):
    color: str = "black"
    label: Optional[str] = None
    fv_window: Optional[list[float]] = None
    epsp_window: Optional[list[float]] = None

    def render(
        self,
        time,
        voltage,
        derivative,
        fv_points=None,
        epsp_points=None,
        ps_points=None,
    ):
        fv_color = to_rgba("darkviolet", alpha=0.7)
        epsp_slope_color = to_rgba("firebrick", alpha=0.8)
        epsp_min_color = to_rgba("royalblue", alpha=0.8)
        ps_color = to_rgba("darkorange", alpha=0.8)
        derivative_color = "#008b8b"

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=self.figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )

        ax1.plot(time, voltage, color=self.color, lw=1.2, label="Smoothed Mean")

        fv_window = self.fv_window
        epsp_window = self.epsp_window
        fv_window_s = [v / 1000 for v in fv_window] if fv_window else None
        epsp_window_s = [v / 1000 for v in epsp_window] if epsp_window else None

        span_alpha = 0.20
        if fv_window_s:
            ax1.axvspan(*fv_window_s, color="gray", label="FV Window", alpha=span_alpha)
            ax2.axvspan(*fv_window_s, color="gray", alpha=span_alpha)
        if epsp_window_s:
            ax1.axvspan(
                *epsp_window_s, color="green", label="fEPSP Window", alpha=span_alpha
            )
            ax2.axvspan(*epsp_window_s, color="green", alpha=span_alpha)

        if fv_points and len(fv_points) >= 2:
            fv_start_t, fv_start_v = fv_points[0]
            fv_end_t, fv_end_v = fv_points[1]
            ax1.scatter(
                [fv_start_t, fv_end_t],
                [fv_start_v, fv_end_v],
                s=60,
                marker="^",
                facecolors=fv_color,
                edgecolors="black",
                linewidths=1.0,
                zorder=5,
                label="Fiber Volley",
            )
            ax2.scatter(
                [fv_start_t, fv_end_t],
                np.interp([fv_start_t, fv_end_t], time, derivative),
                s=50,
                marker="^",
                facecolors=fv_color,
                edgecolors="black",
                linewidths=1.0,
                zorder=5,
            )

        if epsp_points and len(epsp_points) >= 2:
            epsp_slope_t, epsp_slope_v = epsp_points[0]
            epsp_min_t, epsp_min_v = epsp_points[1]

            ax1.scatter(
                epsp_slope_t,
                epsp_slope_v,
                s=70,
                marker="s",
                facecolors=epsp_slope_color,
                edgecolors="black",
                linewidths=1.0,
                zorder=6,
                label="fEPSP Slope Min",
            )
            ax2.scatter(
                epsp_slope_t,
                np.interp(epsp_slope_t, time, derivative),
                s=60,
                marker="s",
                facecolors=epsp_slope_color,
                edgecolors="black",
                linewidths=1.0,
                zorder=6,
            )

            ax1.scatter(
                epsp_min_t,
                epsp_min_v,
                s=70,
                marker="o",
                facecolors=epsp_min_color,
                edgecolors="black",
                linewidths=1.0,
                zorder=6,
                label="fEPSP Min",
            )
            ax2.scatter(
                epsp_min_t,
                np.interp(epsp_min_t, time, derivative),
                s=60,
                marker="o",
                facecolors=epsp_min_color,
                edgecolors="black",
                linewidths=1.0,
                zorder=6,
            )

        if ps_points and len(ps_points) >= 1:
            ps_t, ps_v = zip(*ps_points)
            ax1.scatter(
                ps_t,
                ps_v,
                s=70,
                marker="d",
                facecolors=ps_color,
                edgecolors="black",
                linewidths=1.0,
                zorder=7,
                label="PS Max",
            )
            ax2.scatter(
                ps_t,
                np.interp(ps_t, time, derivative),
                s=60,
                marker="d",
                facecolors=ps_color,
                edgecolors="black",
                linewidths=1.0,
                zorder=7,
            )

        ax1.set_ylabel("Response (mV)")
        if self.label:
            ax1.set_title(self.label)
        ax1.legend(loc="best", frameon=False)

        ax2.plot(time, derivative, color=derivative_color, lw=1.2, label="dV/dt (mV/ms)")
        ax2.axhline(0, color="k", lw=0.5, ls="--")
        ax2.set_ylabel("Derivative (mV/ms)")
        ax2.legend(loc="best", frameon=False)

        set_xaxis_ms(ax2)
        plt.tight_layout()
        return self._finalize(fig)


def plot_derivative(
    time,
    voltage,
    derivative,
    fv_points=None,
    epsp_points=None,
    ps_points=None,
    fv_window: list[float] | None = None,
    epsp_window: list[float] | None = None,
    label: str | None = None,
    color: str = "black",
    figsize=(14, 7),
    show: bool = False,
    savepath: Optional[str] = None,
):
    """Convenience wrapper to render the derivative plot using the plotter class."""
    plotter = DerivativePlotter(
        figsize=figsize,
        savepath=savepath,
        show=show,
        label=label,
        color=color,
        fv_window=fv_window,
        epsp_window=epsp_window,
    )
    return plotter.render(
        time,
        voltage,
        derivative,
        fv_points=fv_points,
        epsp_points=epsp_points,
        ps_points=ps_points,
    )


@dataclass
class FieldPotentialPlotter(PlotBase):
    stimuli: Optional[list[int]] = None

    def render(self, abf_df, fv_df, epsp_df, ps_df):
        stimuli = self.stimuli or sorted(abf_df["stim_intensity"].unique())

        violet_rgba = to_rgba("violet", alpha=0.6)
        purple_rgba = to_rgba("purple", alpha=0.6)
        gray_rgba = to_rgba("gray", alpha=0.6)
        lightblue_rgba = to_rgba("lightblue", alpha=0.6)

        cmap = plt.get_cmap("viridis", len(stimuli))
        fig, ax = plt.subplots(figsize=self.figsize)

        for i, stim in enumerate(stimuli):
            g = abf_df.query("stim_intensity == @stim").sort_values("time")
            x = g["time"].to_numpy()
            y = g["smooth"].to_numpy()
            color = cmap(i)
            ax.plot(x, y, lw=1.2, label=stim, color=color)

            if fv_df is not None and (fv_df.stim_intensity == stim).any():
                r = fv_df.loc[fv_df.stim_intensity == stim].iloc[0]
                if np.isfinite(r.get("fv_min_s", np.nan)):
                    ax.scatter(
                        r.fv_min_s,
                        r.fv_min_v,
                        s=80,
                        marker="v",
                        facecolors=violet_rgba,
                        edgecolors=purple_rgba,
                        linewidths=1.0,
                        zorder=5,
                    )
                if np.isfinite(r.get("fv_max_s", np.nan)):
                    ax.scatter(
                        r.fv_max_s,
                        r.fv_max_v,
                        s=80,
                        marker="^",
                        facecolors=violet_rgba,
                        edgecolors=purple_rgba,
                        linewidths=1.0,
                        zorder=5,
                    )

            if epsp_df is not None and (epsp_df.stim_intensity == stim).any():
                r = epsp_df.loc[epsp_df.stim_intensity == stim].iloc[0]
                if np.isfinite(r.get("epsp_min_s", np.nan)):
                    ax.scatter(
                        r.epsp_min_s,
                        r.epsp_min_v,
                        s=80,
                        marker="o",
                        facecolors=to_rgba(color, alpha=0.6),
                        edgecolors="black",
                        linewidths=1.0,
                        zorder=5,
                    )
                if "fit_t" in r and isinstance(r.fit_t, list):
                    ax.plot(r.fit_t, r.fit_v, lw=2.5, color=color)

            if ps_df is not None and (ps_df.stim_intensity == stim).any():
                r = ps_df.loc[ps_df.stim_intensity == stim].iloc[0]
                if np.isfinite(r.get("ps_max_s", np.nan)):
                    ax.scatter(
                        r.ps_max_s,
                        r.ps_max_v,
                        s=80,
                        marker="d",
                        facecolors=lightblue_rgba,
                        edgecolors="black",
                        linewidths=1.0,
                        zorder=5,
                    )

        trace_leg = ax.legend(
            title="Stimulus Intensity (µA)",
            loc="lower right",
            bbox_to_anchor=(1.0, 0.0),
            frameon=False,
        )

        feature_handles = [
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="none",
                markerfacecolor=violet_rgba,
                markeredgecolor=purple_rgba,
                markeredgewidth=1.2,
                markersize=8,
                fillstyle="full",
                label="FV Start",
            ),
            Line2D(
                [0],
                [0],
                marker="v",
                linestyle="none",
                markerfacecolor=violet_rgba,
                markeredgecolor=purple_rgba,
                markeredgewidth=1.2,
                markersize=8,
                fillstyle="full",
                label="FV End",
            ),
            Line2D([0], [0], color=gray_rgba, linewidth=2.5, label="fEPSP Slope"),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.2,
                markersize=8,
                label="fEPSP Min",
            ),
            Line2D(
                [0],
                [0],
                marker="d",
                linestyle="none",
                markerfacecolor=lightblue_rgba,
                markeredgecolor="black",
                markeredgewidth=1.2,
                markersize=8,
                label="PS Max",
            ),
        ]

        feature_leg = ax.legend(handles=feature_handles, loc="upper right", frameon=False)
        ax.add_artist(trace_leg)
        ax.add_artist(feature_leg)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (mV)")
        ax.set_title("Evoked Field Potential with Feature Annotations")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._finalize(fig)


def _fit_curve(x: np.ndarray, y: np.ndarray, fit_type: str | None):
    if fit_type is None:
        return None
    if len(x) < 4:
        return None

    fit_type = fit_type.lower()
    if fit_type in {"logistic", "log"}:
        func = logistic
        p0 = [np.max(y), np.median(x), 0.1]
        param_names = ["A", "x0", "k"]
        label = "logistic"
    elif fit_type in {"exp", "exponential"}:
        func = exp_saturation
        p0 = [np.max(y), 0.01]
        param_names = ["A", "k"]
        label = "exponential"
    else:
        return None

    try:
        popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
    except RuntimeError:
        print(f"{label.title()} fit failed.")
        return None

    y_pred = func(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = func(x_fit, *popt)

    return {
        "fit_type": label,
        "params": dict(zip(param_names, popt)),
        "r2": r2,
        "n_points": len(x),
        "x_fit": x_fit,
        "y_fit": y_fit,
    }


def plot_scatter_fit(
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    fit_type: str | None = "logistic",
    hue_col: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    color: str = "steelblue",
    point_color: str | None = None,
    figsize=(12, 8),
    savepath: str | None = None,
    show: bool = False,
):
    required_cols = [x_col, y_col]
    if hue_col:
        required_cols.append(hue_col)
    df = data[required_cols].dropna()
    if df.empty:
        print("No data available for scatter plot.")
        return None, pd.DataFrame()

    fig, ax = plt.subplots(figsize=figsize)
    point_color = point_color or color

    if hue_col:
        unique_vals = df[hue_col].unique()
        cmap = extend_palette(plt.get_cmap("Set3").colors, len(unique_vals))
        for i, val in enumerate(unique_vals):
            subset = df[df[hue_col] == val]
            ax.scatter(
                subset[x_col],
                subset[y_col],
                s=80,
                color=cmap(i / len(unique_vals)),
                edgecolor="black",
                linewidth=0.8,
                label=str(val),
            )
        ax.legend(title=hue_col.replace("_", " ").title(), frameon=False)
    else:
        ax.scatter(
            df[x_col],
            df[y_col],
            s=80,
            color=point_color,
            edgecolor="black",
            linewidth=0.8,
        )

    fit_result = _fit_curve(df[x_col].to_numpy(), df[y_col].to_numpy(), fit_type)
    fit_stats = []
    if fit_result:
        fit_color = color
        ax.plot(fit_result["x_fit"], fit_result["y_fit"], "--", color=fit_color, lw=2.5)
        ax.text(
            0.05,
            0.90,
            f"$R^2$ = {fit_result['r2']:.3f}",
            transform=ax.transAxes,
            fontsize=15,
            color=fit_color,
            ha="left",
            va="top",
            weight="bold",
        )
        fit_stats.append(
            {
                "fit_type": fit_result["fit_type"],
                **fit_result["params"],
                "r2": fit_result["r2"],
                "n_points": fit_result["n_points"],
            }
        )

    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    if title:
        ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, pd.DataFrame(fit_stats)


def plot_fp(
    abf_df,
    fv_df,
    epsp_df,
    ps_df,
    stimuli=None,
    figsize=(14, 9),
    savepath: str | None = None,
    show: bool = False,
):
    plotter = FieldPotentialPlotter(
        stimuli=list(stimuli) if stimuli is not None else None,
        figsize=figsize,
        savepath=savepath,
        show=show,
    )
    return plotter.render(abf_df, fv_df, epsp_df, ps_df)


def plot_presynaptic(
    fv_df: pd.DataFrame,
    figsize=(12, 8),
    savepath: str | None = None,
    show: bool = False,
):
    return plot_scatter_fit(
        fv_df,
        x_col="stim_intensity",
        y_col="fv_amp",
        fit_type="exp",
        xlabel="Stimulus Intensity (µA)",
        ylabel="FV Amplitude (mV)",
        title="Presynaptic Excitability Curve",
        color="darkorange",
        point_color="lightgray",
        figsize=figsize,
        savepath=savepath,
        show=show,
    )


def plot_io_curve(
    epsp_df,
    figsize=(12, 8),
    savepath: str | None = None,
    show: bool = False,
):
    return plot_scatter_fit(
        epsp_df,
        x_col="stim_intensity",
        y_col="epsp_slope_ms",
        fit_type="exp",
        xlabel="Stimulus Intensity (µA)",
        ylabel="fEPSP Slope (mV/ms)",
        title="Input-Output Curve",
        color="seagreen",
        point_color="lightgray",
        figsize=figsize,
        savepath=savepath,
        show=show,
    )


def plot_excitability(
    fv_df,
    epsp_df,
    figsize=(12, 8),
    savepath: str | None = None,
    show: bool = False,
):
    merged = pd.merge(
        fv_df[["stim_intensity", "fv_amp"]],
        epsp_df[["stim_intensity", "epsp_slope_ms"]],
        on="stim_intensity",
        how="inner",
    ).dropna()
    if merged.empty:
        print("No overlapping stimuli between fv_df and epsp_df.")
        return None, pd.DataFrame()

    return plot_scatter_fit(
        merged,
        x_col="epsp_slope_ms",
        y_col="fv_amp",
        hue_col="stim_intensity",
        fit_type="logistic",
        xlabel="fEPSP Slope (mV/ms)",
        ylabel="FV Amplitude (mV)",
        title="Excitability Curve",
        color="steelblue",
        figsize=figsize,
        savepath=savepath,
        show=show,
    )


def plot_es_curve(
    epsp_df,
    ps_df,
    figsize=(8, 6),
    savepath=None,
    show=False,
):
    merged = pd.merge(
        epsp_df[["stim_intensity", "epsp_slope_ms"]],
        ps_df[["stim_intensity", "ps_amp"]],
        on="stim_intensity",
        how="inner",
    ).dropna()
    if merged.empty:
        print("No overlapping stimuli between fEPSP and PS data.")
        return None, pd.DataFrame()

    return plot_scatter_fit(
        merged,
        x_col="epsp_slope_ms",
        y_col="ps_amp",
        fit_type="logistic",
        xlabel="fEPSP Slope (mV/ms)",
        ylabel="Population spike amplitude (mV)",
        title="E-S Curve (Postsynaptic Excitability)",
        color="firebrick",
        point_color="teal",
        figsize=figsize,
        savepath=savepath,
        show=show,
    )
