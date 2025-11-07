"""
Module for removing stimulation artifacts from electrophysiological recordings.
"""
from ephys_toolbox.src.utils import to_samples
import pandas as pd

def remove_stim_artifact(abf_df, stim_window: list[float]=[0.0,1.0]):
    fs = abf_df.attrs.get("sampling_rate")
    if fs is None:
        raise ValueError("Sampling Rate Missing from DataFrame")

    start_idx = to_samples(stim_window[0], fs)
    stop_idx  = to_samples(stim_window[1], fs)

    def crop(g):
        # keep pre-artifact and post-artifact sections
        parts = [g.iloc[:start_idx], g.iloc[stop_idx:]]
        cropped = pd.concat(parts, ignore_index=True)
        # re-zero time so all traces align perfectly
        cropped["time"] = cropped["time"] - cropped["time"].iloc[0]
        return cropped

    return (
        abf_df.groupby(["stim_intensity", "sweep"], group_keys=False)[
            ["stim_intensity", "sweep", "time", "value"]
        ].apply(crop)
    )
