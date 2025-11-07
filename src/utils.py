import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib import RcParams

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import os

import pyabf
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences, filtfilt, savgol_filter, butter

## Helper functions
def to_samples(ms, fs) -> int:
    return int(round(ms * fs / 1000))

def to_ms(samples, fs):
    return samples / fs * 1000

def derivative(y: np.ndarray, fs: float):
    return np.gradient(y) * fs

def logistic(x, A, x0, k):
    """Standard 3-parameter logistic (sigmoidal) function."""
    return A / (1 + np.exp(-(x - x0) / k))

def exp_saturation(x, A, k):
    return A * (1 - np.exp(-k * x))

def smooth(x, window_samples=5): # zero phase smoothing
    if window_samples <= 1:
        return np.asarray(x, dtype=float)
    b = np.ones(window_samples) / window_samples
    a = [1]
    padlen = min(3 * (window_samples - 1), len(x) - 1)
    return filtfilt(b, a, x, padlen=padlen)

def savgol(x, window_samples=11, poly=3):
    if window_samples <= 1:
        return np.asarray(x, dtype=float)
    return savgol_filter(x, window_length=window_samples, polyorder=poly)

def butterw(x, cutoff_hz, fs, order=4):
    nyq = 0.5 * fs
    Wn = cutoff_hz / nyq
    b, a = butter(order, Wn, btype='low')
    return filtfilt(b, a, x)

def switch_channels(abf, channel: int = 0):
    if not isinstance(abf, pyabf.abf.ABF):
        return TypeError("Not an pyabf.abf.ABF object!")
    return abf.setSweep(channel=channel)

def switch_sweep(abf, sweep: int = 0):
    if not isinstance(abf, pyabf.abf.ABF):
        return TypeError("Not an pyabf.abf.ABF object!")
    return abf.setSweep(sweep=sweep)

def set_xaxis_ms(ax, xlim=None):
    """
    Format the x-axis to display milliseconds instead of seconds.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to modify.
    xlim : tuple or None
        Optional (xmin, xmax) in seconds. If provided, will set limits scaled to ms.
    """
    # Convert existing or given limits to ms
    if xlim is None:
        xmin, xmax = ax.get_xlim()
    else:
        xmin, xmax = xlim

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Time (ms)")

    # Format tick labels: seconds → ms
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 1000:.0f}"))

