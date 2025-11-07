# ephys_toolbox

Utilities for running fiber-volley / fEPSP / population-spike analyses on ABF recordings and producing tidy outputs plus publication-ready plots.

## Usage (Python API)

Create a script (e.g., `test.py`) that constructs the pipeline configuration and analyzers, then run it as a module:

```python
from ephys_toolbox.src import (
    AnalyzerConfig,
    Pipeline,
    PipelineConfig,
    FiberVolleyAnalyzer,
    EpspAnalyzer,
    PopSpikeAnalyzer,
)

cfg = PipelineConfig(
    paths=[
        r"C:\path\to\your\data\2025_03_06_0000.abf",
        # you can list multiple ABFs here
    ],
    stim_intensities=[25, 50, 75, 100],
    repnum=3,
    stim_window=[0.0, 1.0],
    fv_window=[0.0, 1.5],
    epsp_window=[1.5, 5.25],
    ps_lag=3.0,
    summarize=True,
    output_dir=r"C:\path\to\output\directory",
)

analyzers = [
    FiberVolleyAnalyzer(AnalyzerConfig(params={"window": cfg.fv_window})),
    EpspAnalyzer(AnalyzerConfig(params={"window": cfg.epsp_window, "fit_distance": 4})),
    PopSpikeAnalyzer(AnalyzerConfig(params={"lag_ms": cfg.ps_lag, "height": 0.2})),
]

Pipeline(cfg, analyzers).run()
```

Run it from the repo root so the package resolves correctly:

```
python -m ephys_toolbox.src.test
```

This creates per-recording folders under `output_dir/ABF_STEM/` with individual feature CSVs, plots, and a tidy results CSV. When `summarize=True`, a timestamped `run_summary_<ISO>.csv` is written alongside the per-recording directories.

## Usage (CLI)

Install dependencies (or `pip install -e .`), then run the CLI module:

```
python -m ephys_toolbox.src.cli ^
  -p C:\path\to\your\data\2025_03_06_0000.abf ^
  -s 25,50,75,100 ^
  --repnum 3 ^
  --stim-window 0.0 1.0 ^
  --fv-window 0.0 1.5 ^
  --epsp-window 1.5 5.25 ^
  --ps-lag 3.0 ^
  --ps-height 0.2 ^
  --fit-distance 4 ^
  --summarize ^
  -o C:\path\to\output\directory
```

Flags mirror the Python configuration. Multiple `-p` arguments are allowed (files or directories). The CLI writes the same outputs as the Python API, including the summary CSV in `output_dir`.