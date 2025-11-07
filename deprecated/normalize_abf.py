import pyabf
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
from scipy.interpolate import CubicSpline, interp1d

abf = pyabf.ABF("ephys_toolbox/data/2025_03_06_0000.abf")

sweep_list = abf.sweepList

sweep_stims = ["amp25","amp50","amp75","amp100","amp150","amp200","amp250","amp300","amp400","amp500","amp600"]

sweep_count = abf.sweepCount

## TODO: custom function where user supplies number of sweeps per stim intensity
if sweep_count == 33:
    num_traces = 3
elif sweep_count == 55:
    num_traces = 5
else:
    ValueError("Improper number of traces, inspect .abf file")

sweep_indices = [sweep_list[i:i + num_traces] for i in range(0, len(sweep_list), num_traces)]

def build_sweeps(abf, sweep_stims, sweep_indices):
    sweep_dict = {}
    for stim, idxs in zip(sweep_stims, sweep_indices):
        traces = []
        for i in idxs:
            abf.setSweep(i)
            x = abf.sweepX.copy()
            y = abf.sweepY.copy()
            traces.append(np.column_stack((x, y)))  # shape (N, 2), list of 3 np.arrays
        sweep_dict[stim] = traces # shape (num_traces * N, 2) e.g. 3 traces * 500 x-y pairs is shape (1500,2)
    return sweep_dict

sweep_dict = build_sweeps(abf, sweep_indices=sweep_indices, sweep_stims=sweep_stims)

def normalize_abf(sweep_dict):
    normalized_dict = {}
    for amp, traces in sweep_dict.items():
        norm_traces = []
        for trace in traces:
            trace = np.asarray(trace)
            x = trace[:,0]
            y = trace[:,1]
            t0 = np.isclose(x, 0.0)
            if not np.any(t0):
                raise ValueError(f"No t=0 found for stim stim:{amp}, trace:{trace}")
            offset = np.abs(y[t0][0]) # if multiple y's at x = 0 select the first
            norm_traces.append(np.column_stack((x, y - offset)))
            normalized_dict[amp] = norm_traces
    return normalized_dict

normalized = normalize_abf(sweep_dict=sweep_dict)

def interpolate_artifact(x, y, sec_window=2e-3, sec_baseline=0.2e-3, k=2, percent_max = 0.05):
    """
    Interpolates only within the first sec_window (default = 2 ms).
    Finds artifact boundaries by slope threshold, then interpolates
    between flat regions on either side.
    """
    dt = x[1] - x[0]
    dydt = np.gradient(y, dt)

    # baseline slope from first 0.2 ms
    # baseline_pts = int(sec_baseline / dt)
    # sigma = np.std(dydt[:baseline_pts]) # standard deviation of slopes in first X seconds88
    # if sigma == 0:
    #     sigma = 100 # small slope threshold
    # slope_thresh = k * sigma

    # restrict to first 2 ms
    window_pts = int(sec_window / dt)
    abs_slope = np.abs(dydt[:window_pts])
    slope_thresh = max(abs_slope) * percent_max

    # find where slope jumps above threshold (artifact start)
    above = np.where(abs_slope > slope_thresh)[0]
    if above.size == 0:
        return y.copy(), None, None  # no artifact detected
    start_idx = above[0]

    # find where slope falls back below threshold (artifact end)
    below_after = np.where(abs_slope[start_idx:] < slope_thresh)[0]
    if below_after.size == 0:
        return y.copy(), None, None
    end_idx = start_idx + below_after[0]

    # anchor points: last flat point before artifact, first flat after artifact
    x_good = [x[start_idx-1], x[end_idx]]
    y_good = [y[start_idx-1], y[end_idx]]

    cs = CubicSpline(x_good, y_good, bc_type="natural")

    # interpolate only inside artifact region
    y_fixed = y.copy()
    mask_bad = (x >= x[start_idx]) & (x <= x[end_idx])
    y_fixed[mask_bad] = cs(x[mask_bad])

    return y_fixed, start_idx, end_idx

# pick one trace (say first trace from "amp200")
amp = "amp25"
trace = normalized[amp][2]   # first trace
x = trace[:, 0]              # time (s)
y = trace[:, 1]              # amplitude

y_fixed, i, j = interpolate_artifact(x, y, percent_max=0.04)

plt.plot(x, y, "-", label = "original", color = "red")
plt.plot(x, y_fixed, "-", label = "interpolated", color = "blue")
plt.legend()
plt.show()


# def plot_traces_by_amp(normalized_dict):
#     for amp, traces in normalized_dict.items():
#         plt.figure(figsize=(6, 4))
#         for i, trace in enumerate(traces):
#             x = trace[:, 0]
#             y = trace[:, 1]
# #             plt.plot(x, y, alpha=0.7, label=f"trace {i+1}")
# #             plt.title(f"{amp} (normalized)")
# #             plt.xlabel("Time")
# #             plt.ylabel("Amplitude")
# #             plt.legend()
# #             plt.tight_layout()
# #         plt.show()

# # test = plot_traces_by_amp(normalized)

# def average_traces(normalized_dict):
#     averaged = {}
#     for amp, traces in normalized_dict.items():
#         if not traces:
#             continue
#         ys_fixed = []
#         for trace in traces:
#             # stack all three ys into stacked array
#             x = trace[:,0]
#             y_raw = trace[:,1]
#             y_fixed, i, j = interpolate_artifact(x, y_raw)
#             ys_fixed.append(y_fixed)
#         ys = np.vstack(ys_fixed)
#         y_mean = np.mean(ys, axis=0) # shape (N,)
#         se = sp.sem(ys, axis=0)
#         averaged[amp] = np.column_stack((x, y_mean, se)) # shape (N,2)
#     return averaged

# averaged = average_traces(normalized)

# def plot_by_amp(averaged_dict):
#     for amp, xy in averaged_dict.items():
#         plt.figure(figsize= (6,4))
#         plt.plot(xy[:,0], xy[:,1])
#         plt.fill_between(xy[:,0], xy[:,1] - xy[:,2], xy[:,1] + xy[:,2], color='lightblue')
#         plt.title(f"{amp} mean")
#         plt.tight_layout()
#     plt.show()

# # # ## IO curve (input output curve), plot with no background
# test = plot_by_amp(averaged)

# # def normalize_abf(stim):
# #     offset = sweep_dict[stim]["y"][0]
# #     print(offset)

# # test = normalize_abf("amp50")
# # print(test)

# # # abf.SampleRate is in Hz
# # sample_rate = abf.sampleRate
# # ms = 1.2 # time in msec
# # idx = ms*1000/sample_rate # (msec to seconds / Hz) = index time in seconds

# # def average_sweeps(stim):
# #     y_avg = np.array(sweep_dict[stim]["y"]).mean(axis=0)
# #     x = sweep_dict[stim]["x"][0]
# #     plt.plot(x, y_avg)
# #     plt.show()

# # test = average_sweeps("amp600")



