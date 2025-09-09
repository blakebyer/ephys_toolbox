import pyabf
import matplotlib.pyplot as plt
import numpy as np

abf = pyabf.ABF("ephys_toolbox\data\\2025_03_06_0000.abf")

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

sweep_dict = {}
for stim, idxs in zip(sweep_stims, sweep_indices):
    xs, ys = [], []
    for i in idxs:
        abf.setSweep(i)
        xs.append(abf.sweepX.copy())
        ys.append(abf.sweepY.copy())
    sweep_dict[stim] = {
        "x": np.array(xs),   # shape (num_traces, N)
        "y": np.array(ys)    # shape (num_traces, N)
    }

# abf.SampleRate is in Hz
sample_rate = abf.sampleRate
ms = 1.2 # time in msec
idx = ms*1000/sample_rate # msec to seconds / Hz = index time in seconds

def average_sweeps(stim):
    y_avg = np.array(sweep_dict[stim]["y"]).mean(axis=0)
    x = sweep_dict[stim]["x"][0]
    plt.plot(x, y_avg)
    plt.show()


test = average_sweeps("amp25")



