import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import numpy as np
from typing import Optional
import pandas as pd

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

## Plotting functions
def plot_derivative(
    time, voltage, derivative,
    fv_points=None, epsp_points=None, ps_points=None,
    fv_window: list[float] = None, epsp_window: list[float]=None,
    label=None, color="black", figsize=(14,7),
    show=False, savepath: Optional[str]=None
):
    """
    Plot smoothed voltage and its derivative with markers for detected features.
    Uses zero-phase smoothed signals to ensure time alignment.
    Labels fEPSP slope maximum and minimum separately.
    """
    # --- Color definitions ---
    fv_color = to_rgba('darkviolet', alpha=0.7)
    epsp_slope_color = to_rgba('firebrick', alpha=0.8)
    epsp_min_color = to_rgba('royalblue', alpha=0.8)
    ps_color = to_rgba('darkorange', alpha=0.8)
    derivative_color = "#008b8b"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    # --- Main smoothed voltage trace ---
    ax1.plot(time, voltage, color=color, lw=1.2, label="Smoothed Mean")

    if fv_window:
        fv_window_s = [v / 1000 for v in fv_window]
    else:
        fv_window_s = None
    if epsp_window:
        epsp_window_s = [v / 1000 for v in epsp_window]
    else:
        epsp_window_s = None

    # --- Shaded search windows (same alpha both panels) ---
    span_alpha = 0.20
    if fv_window:
        ax1.axvspan(*fv_window_s, color="gray", label="FV Window", alpha=span_alpha)
        ax2.axvspan(*fv_window_s, color="gray", alpha=span_alpha)
    if epsp_window:
        ax1.axvspan(*epsp_window_s, color="green", label="fEPSP Window", alpha=span_alpha)
        ax2.axvspan(*epsp_window_s, color="green", alpha=span_alpha)

    # --- FV markers ---
    if fv_points and len(fv_points) >= 2:
        fv_start_t, fv_start_v = fv_points[0]
        fv_end_t, fv_end_v = fv_points[1]
        ax1.scatter([fv_start_t, fv_end_t], [fv_start_v, fv_end_v],
                    s=60, marker='^', facecolors=fv_color, edgecolors='black',
                    linewidths=1.0, zorder=5, label="Fiber Volley")
        ax2.scatter([fv_start_t, fv_end_t],
                    np.interp([fv_start_t, fv_end_t], time, derivative),
                    s=50, marker='^', facecolors=fv_color, edgecolors='black',
                    linewidths=1.0, zorder=5)

    # --- fEPSP markers ---
    if epsp_points and len(epsp_points) >= 2:
        epsp_slope_t, epsp_slope_v = epsp_points[0]
        epsp_min_t, epsp_min_v = epsp_points[1]

        # fEPSP Slope Max
        ax1.scatter(epsp_slope_t, epsp_slope_v,
                    s=70, marker='s', facecolors=epsp_slope_color, edgecolors='black',
                    linewidths=1.0, zorder=6, label="fEPSP Slope Min")
        ax2.scatter(epsp_slope_t, np.interp(epsp_slope_t, time, derivative),
                    s=60, marker='s', facecolors=epsp_slope_color, edgecolors='black',
                    linewidths=1.0, zorder=6)

        # fEPSP Min
        ax1.scatter(epsp_min_t, epsp_min_v,
                    s=70, marker='o', facecolors=epsp_min_color, edgecolors='black',
                    linewidths=1.0, zorder=6, label="fEPSP Min")
        ax2.scatter(epsp_min_t, np.interp(epsp_min_t, time, derivative),
                    s=60, marker='o', facecolors=epsp_min_color, edgecolors='black',
                    linewidths=1.0, zorder=6)

    # --- PS markers ---
    if ps_points and len(ps_points) >= 1:
        ps_t, ps_v = zip(*ps_points)
        ax1.scatter(ps_t, ps_v,
                    s=70, marker='d', facecolors=ps_color, edgecolors='black',
                    linewidths=1.0, zorder=7, label="PS Max")
        ax2.scatter(ps_t, np.interp(ps_t, time, derivative),
                    s=60, marker='d', facecolors=ps_color, edgecolors='black',
                    linewidths=1.0, zorder=7)

    # --- Formatting ---
    ax1.set_ylabel("Response (mV)")
    if label:
        ax1.set_title(label)
    ax1.legend(loc="best", frameon=False)

    ax2.plot(time, derivative, color=derivative_color, lw=1.2, label="dV/dt (mV/ms)")
    ax2.axhline(0, color="k", lw=0.5, ls="--")
    ax2.set_ylabel("Derivative (mV/ms)")
    ax2.legend(loc="best", frameon=False)
    
    set_xaxis_ms(ax2)

    plt.tight_layout()

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig

def plot_fp(abf_df, 
                  fv_df, 
                  epsp_df, 
                  ps_df, 
                  stimuli=None, 
                  figsize=(14,9),
                  savepath: str | None = None, 
                  show: bool = False):
    """
    Plot I/O curves for each stimulus with FV, fEPSP, PS, and slope fit overlays.
    Returns matplotlib Figure object.
    """
    if stimuli is None:
        stimuli = sorted(abf_df["stim_intensity"].unique())

    # Common RGBA colors
    violet_rgba = to_rgba('violet', alpha=0.6)
    purple_rgba = to_rgba('purple', alpha=0.6)
    gray_rgba = to_rgba('gray', alpha=0.6)
    lightblue_rgba = to_rgba('lightblue', alpha=0.6)

    cmap = plt.get_cmap("viridis", len(stimuli))
    fig, ax = plt.subplots(figsize=figsize)

    for i, stim in enumerate(stimuli):
        g = abf_df.query("stim_intensity == @stim").sort_values("time")
        x = g["time"].to_numpy() #* 1000 # s to ms
        y = g["smooth"].to_numpy()  # use already smoothed trace
        color = cmap(i)
        ax.plot(x, y, lw=1.2, label=stim, color=color)

        # FV markers
        if fv_df is not None and (fv_df.stim_intensity == stim).any():
            r = fv_df.loc[fv_df.stim_intensity == stim].iloc[0]
            if np.isfinite(r.get("fv_min_s", np.nan)):
                ax.scatter(r.fv_min_s, r.fv_min_v,
                           s=80, marker='v',
                           facecolors=violet_rgba, edgecolors=purple_rgba,
                           linewidths=1.0, zorder=5)
            if np.isfinite(r.get("fv_max_s", np.nan)):
                ax.scatter(r.fv_max_s, r.fv_max_v,
                           s=80, marker='^',
                           facecolors=violet_rgba, edgecolors=purple_rgba,
                           linewidths=1.0, zorder=5)

        # fEPSP features and slope fit
        if epsp_df is not None and (epsp_df.stim_intensity == stim).any():
            r = epsp_df.loc[epsp_df.stim_intensity == stim].iloc[0]
            if np.isfinite(r.get("epsp_min_s", np.nan)):
                ax.scatter(r.epsp_min_s, r.epsp_min_v,
                           s=80, marker='o',
                           facecolors=to_rgba(color, alpha=0.6), edgecolors='black',
                           linewidths=1.0, zorder=5)
            if "fit_t" in r and isinstance(r.fit_t, list):
                ax.plot(r.fit_t, r.fit_v, lw=2.5, color=color)

        # PS markers
        if ps_df is not None and (ps_df.stim_intensity == stim).any():
            r = ps_df.loc[ps_df.stim_intensity == stim].iloc[0]
            if np.isfinite(r.get("ps_max_s", np.nan)):
                ax.scatter(r.ps_max_s, r.ps_max_v,
                           s=80, marker='d',
                           facecolors=lightblue_rgba, edgecolors='black',
                           linewidths=1.0, zorder=5)

    # Legends
    trace_leg = ax.legend(
        title="Stimulus Intensity (µA)",
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        frameon=False
    )

    feature_handles = [
        Line2D([0], [0], marker='^', linestyle='none',
               markerfacecolor=violet_rgba, markeredgecolor=purple_rgba,
               markeredgewidth=1.2, markersize=8,
               fillstyle='full', label='FV Start'),
        Line2D([0], [0], marker='v', linestyle='none',
               markerfacecolor=violet_rgba, markeredgecolor=purple_rgba,
               markeredgewidth=1.2, markersize=8,
               fillstyle='full', label='FV End'),
        Line2D([0], [0], color=gray_rgba, linewidth=2.5, label='fEPSP Slope'),
        Line2D([0], [0], marker='o', linestyle='none',
               markerfacecolor=gray_rgba, markeredgecolor='black',
               markeredgewidth=1.0, markersize=8,
               fillstyle='full', label='fEPSP Min'),
        Line2D([0], [0], marker='d', linestyle='none',
               markerfacecolor=lightblue_rgba, markeredgecolor='black',
               markeredgewidth=1.0, markersize=8,
               fillstyle='full', label='PS Max'),
    ]

    ax.add_artist(trace_leg)
    ax.legend(handles=feature_handles, title="Detected Features",
              loc="lower right", bbox_to_anchor=(1.0, 0.5), frameon=False)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Response (mV)")
    ax.set_title("Evoked Field Potentials")

    set_xaxis_ms(ax)
    plt.tight_layout()

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig

def plot_presynaptic(fv_df: pd.DataFrame, figsize=(12,8), savepath: str | None = None, show: bool = False):
    x = fv_df["stim_intensity"].to_numpy()
    y = fv_df["fv_amp"].to_numpy()

    # Sort for plotting
    order = np.argsort(x)
    y, x = y[order], x[order]

    fig, ax = plt.subplots(figsize=figsize)

    base_colors = plt.get_cmap("Set3").colors

    unique_stims = np.unique(x)
    cmap = extend_palette(base_colors, len(unique_stims))
    colors = [cmap(i / len(unique_stims)) for i in range(len(unique_stims))]
    
    ax.scatter(x, y, s=80, color="lightgray", edgecolor="black", linewidth=0.8)

    fit_stats = []

    if len(x) >= 4:
        try:
            p0 = [np.max(y), 0.01]
            popt, _ = curve_fit(exp_saturation, x, y, p0=p0, maxfev=10000)
            A, k = popt
            y_pred = exp_saturation(x, *popt)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res/ss_tot

            x_fit = np.linspace(x.min(), x.max(), 200)
            y_fit = exp_saturation(x_fit, *popt)
            ax.plot(x_fit, y_fit, "--", color="darkorange", lw=2.5)
            ax.text(0.05, 0.90, f"$R^2$ = {r2:.3f}",
                    transform=ax.transAxes,
                    fontsize=15, color="darkorange",
                    ha="left", va="top", weight="bold")

            fit_stats.append({
                "fit_type": "exponential",
                "A": A,
                "k": k,
                "r2": r2,
                "n_points": len(x)
            })

        except RuntimeError:
            # fallback if desired
            print("Exponential fit failed.")

    # ax.legend(
    #     frameon=False,
    #     title="Stimulus Intensity (µA)",
    #     loc="best"
    # )
    ax.set_xlabel("Stimulus Intensity (µA)")
    ax.set_ylabel("FV Amplitude (mV)")
    ax.set_title("Presynaptic Excitability Curve")
    
    ax.grid(True)
    fig.tight_layout()

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    
    fit_df = pd.DataFrame(fit_stats)

    return fig, fit_df

def plot_io_curve(epsp_df, figsize=(12,8), 
            savepath: str | None = None, show: bool = False):
    """
    Plot IO scatter: fEPSP slope vs stimulus intensity.
    Fits a logistic curve across stimuli (each point = one fEPSP slope). 
    Returns (fig, fit_df)
    """
    x = epsp_df["stim_intensity"].to_numpy()
    y = epsp_df["epsp_slope_ms"].to_numpy()

    # Sort for plotting
    order = np.argsort(x)
    y, x = y[order], x[order]

    fig, ax = plt.subplots(figsize=figsize)

    base_colors = plt.get_cmap("Set3").colors

    unique_stims = np.unique(x)
    cmap = extend_palette(base_colors, len(unique_stims))
    colors = [cmap(i / len(unique_stims)) for i in range(len(unique_stims))]
    
    ax.scatter(x, y, s=80, color="lightgray", edgecolor="black", linewidth=0.8)

    fit_stats = []

    if len(x) >= 4:
        try:
            p0 = [np.max(y), 0.01]
            popt, _ = curve_fit(exp_saturation, x, y, p0=p0, maxfev=10000)
            A, k = popt
            y_pred = exp_saturation(x, *popt)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res/ss_tot

            x_fit = np.linspace(x.min(), x.max(), 200)
            y_fit = exp_saturation(x_fit, *popt)
            ax.plot(x_fit, y_fit, "--", color="seagreen", lw=2.5)
            ax.text(0.05, 0.90, f"$R^2$ = {r2:.3f}",
                    transform=ax.transAxes,
                    fontsize=15, color="seagreen",
                    ha="left", va="top", weight="bold")

            fit_stats.append({
                "fit_type": "exponential",
                "A": A,
                "k": k,
                "r2": r2,
                "n_points": len(x)
            })

        except RuntimeError:
            # fallback if desired
            print("Exponential fit failed.")
    
    #ax.legend(
      #  frameon=False,
      #  title="Stimulus Intensity (µA)",
     #   loc="best"
    #)
    ax.set_xlabel("Stimulus Intensity (µA)")
    ax.set_ylabel("fEPSP Slope (mV/ms)")
    ax.set_title("Input-Output Curve")
    ax.grid(True)
    fig.tight_layout()

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    
    fit_df = pd.DataFrame(fit_stats)

    return fig, fit_df


def plot_excitability(fv_df, epsp_df, figsize=(12,8),
                      savepath: str | None = None, show: bool = False):
    """
    Plot synaptic excitability: Fiber volley amplitude vs fEPSP slope.
    Fits a logistic curve across all stimuli (each point = one stimulus).
    Returns (fig, fit_df)
    """

    merged = pd.merge(
        fv_df[["stim_intensity", "fv_amp"]],
        epsp_df[["stim_intensity", "epsp_slope_ms"]],
        on="stim_intensity",
        how="inner"
    ).dropna()

    if merged.empty:
        print("No overlapping stimuli between fv_df and epsp_df.")
        return None, pd.DataFrame()

    x = merged["epsp_slope_ms"].to_numpy()
    y = merged["fv_amp"].to_numpy()
    stimuli = merged["stim_intensity"].to_numpy()

    # Sort for plotting
    order = np.argsort(x)
    x, y, stimuli = x[order], y[order], stimuli[order]

    fig, ax = plt.subplots(figsize=figsize)

    # --- Build an extendable discrete palette ---
    base_cmap = plt.get_cmap("Set3")  # change here: e.g., "tab10", "Pastel1", "viridis"
    base_colors = np.array(base_cmap.colors)
    unique_stims = np.unique(stimuli)
    cmap = extend_palette(base_colors, len(unique_stims))

    # --- Plot each stimulus separately for clean legend + colors ---
    for i, stim in enumerate(unique_stims):
        mask = stimuli == stim
        ax.scatter(
            x[mask], y[mask],
            s=80, color=cmap(i / len(unique_stims)),
            edgecolor="black", linewidth=0.8,
            label=f"{stim}"
        )

    fit_stats = []

    # --- Fit logistic curve if enough points ---
    if len(x) >= 4:
        try:
            p0 = [np.max(y), np.median(x), 0.1]
            popt, _ = curve_fit(logistic, x, y, p0=p0, maxfev=10000)
            A, x0, k = popt

            y_pred = logistic(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot

            x_fit = np.linspace(x.min(), x.max(), 200)
            y_fit = logistic(x_fit, *popt)
            ax.plot(x_fit, y_fit, "--", color="steelblue", lw=2.5)
            ax.text(0.05, 0.90, f"$R^2$ = {r2:.3f}",
                    transform=ax.transAxes,
                    fontsize=15, color="steelblue",
                    ha="left", va="top", weight="bold")

            fit_stats.append({
                "fit_type": "logistic",
                "A": A,
                "x0": x0,
                "k": k,
                "r2": r2,
                "n_points": len(x)
            })

        except RuntimeError:
           print("Logistic fit failed.")
    # --- Legend *after* fit, so all entries appear ---
    #ax.legend(frameon=False, title="Stimulus Intensity (µA)", loc="best")

    ax.set_xlabel("fEPSP Slope (mV/ms)")
    ax.set_ylabel("FV Amplitude (mV)")
    ax.set_title("Excitability Curve")
    ax.grid(True)
    fig.tight_layout()

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    
    fit_df = pd.DataFrame(fit_stats)
    return fig, fit_df

def plot_es_curve(epsp_df, ps_df, figsize=(8,6), savepath=None, show=False):
    """
    Plot E–S curve: population spike amplitude vs fEPSP slope.
    Fits a logistic function to quantify postsynaptic excitability.
    """
    merged = pd.merge(
        epsp_df[["stim_intensity", "epsp_slope_ms"]],
        ps_df[["stim_intensity", "ps_amp"]],
        on="stim_intensity",
        how="inner"
    ).dropna()

    if merged.empty:
        print("No overlapping stimuli between fEPSP and PS data.")
        return None, pd.DataFrame()

    x = merged["epsp_slope_ms"].to_numpy()
    y = merged["ps_amp"].to_numpy()

    order = np.argsort(x)
    x, y = x[order], y[order]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, color="teal", edgecolor="black", s=80)

    fit_stats = []
    if len(x) >= 4:
        try:
            p0 = [np.max(y), np.median(x), 0.5]
            popt, _ = curve_fit(logistic, x, y, p0=p0, maxfev=10000)
            A, x0, k = popt

            y_pred = logistic(x, *popt)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res/ss_tot

            x_fit = np.linspace(x.min(), x.max(), 200)
            y_fit = logistic(x_fit, *popt)
            ax.plot(x_fit, y_fit, "--", color="firebrick",
                    label=f"Logistic fit ($R^2$={r2:.3f})")

            fit_stats.append({
                "fit_type": "logistic",
                "A": A,
                "x0": x0,
                "k": k,
                "r2": r2,
                "n_points": len(x)
            })
        except RuntimeError:
            print("Logistic fit failed.")

    ax.set_xlabel("fEPSP slope (mV/ms)")
    ax.set_ylabel("Population spike amplitude (mV)")
    ax.set_title("E–S Curve (Postsynaptic Excitability)")
    #ax.legend(frameon=False)
    ax.grid(True)
    fig.tight_layout()

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    fit_df = pd.DataFrame(fit_stats)
    return fig, fit_df

def combo_plots(fv_df, epsp_df, ps_df,
                figsize=(16, 12),
                savepath: str | None = None,
                show: bool = False):
    """
    Combine individual plots into a 2×2 layout:
      A: Presynaptic Excitability
      B: Input–Output Curve
      C: Synaptic Excitability
      D: Common Legend
    """

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axA, axB, axC, axD = axes.flat

    def embed(fig_src, ax_target):
        """Render a standalone figure and embed it into an existing axis."""
        fig_src.canvas.draw()
        img = np.asarray(fig_src.canvas.buffer_rgba())
        ax_target.imshow(img)
        plt.close(fig_src)
        ax_target.set_xticks([])
        ax_target.set_yticks([])
        ax_target.axis("off")

    # --- A) Presynaptic Excitability ---
    figA, _ = plot_presynaptic(fv_df, figsize=None, show=False)
    handlesA, labelsA = figA.axes[0].get_legend_handles_labels()
    embed(figA, axA)

    # --- B) Input–Output Curve ---
    figB, _ = plot_io_curve(epsp_df, figsize=None, show=False)
    handlesB, labelsB = figB.axes[0].get_legend_handles_labels()
    embed(figB, axB)

    # --- C) Synaptic Excitability ---
    figC, _ = plot_excitability(fv_df, epsp_df, figsize=None, show=False)
    handlesC, labelsC = figC.axes[0].get_legend_handles_labels()
    embed(figC, axC)

    panel_label_style = dict(
        ha="left", va="top",
        weight="bold",
        fontsize=22
    )
    axA.text(0.01, 0.97, "A)", transform=axA.transAxes, **panel_label_style)
    axB.text(0.01, 0.97, "B)", transform=axB.transAxes, **panel_label_style)
    axC.text(0.01, 0.97, "C)", transform=axC.transAxes, **panel_label_style)


    # --- D) Shared Legend Only ---
    axD.axis("off")

    # merge all legend handles (unique only)
    handles, labels = [], []
    for hlist, llist in [(handlesA, labelsA), (handlesB, labelsB), (handlesC, labelsC)]:
        for h, l in zip(hlist, llist):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    # draw one clean legend in panel D
    axD.legend(handles, labels, loc="center left", frameon=False,
               title="Stimulus Intensity (µA)")

    # --- Final layout ---
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig