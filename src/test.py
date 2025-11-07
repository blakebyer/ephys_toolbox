from ephys_toolbox.src.analyzers import AnalyzerConfig
from ephys_toolbox.src.fiber_volley import FiberVolleyAnalyzer
from ephys_toolbox.src.epsp import EpspAnalyzer
from ephys_toolbox.src.pipeline import Pipeline, PipelineConfig
from ephys_toolbox.src.pop_spike import PopSpikeAnalyzer

cfg = PipelineConfig(
    paths=[
        r"C:\Users\bbyer\OneDrive\Documents\UniversityofKentucky\BachstetterLab\ephys_toolbox\ephys_toolbox\data\2025_03_06_0000.abf"
    ],
    stim_intensities=[75, 100, 400, 500, 600],
    repnum=3,
    stim_window=[0.0, 1.0],
    fv_window=[0.0, 1.5],
    epsp_window=[1.5, 5.25],
    ps_lag=3.0,
    summarize=True,
    output_dir=r"C:\Users\bbyer\OneDrive\Documents\UniversityofKentucky\BachstetterLab\ephys_toolbox\ephys_toolbox\data2",
)

analyzers = [
    FiberVolleyAnalyzer(AnalyzerConfig(params={"window": cfg.fv_window})),
    EpspAnalyzer(AnalyzerConfig(params={"window": cfg.epsp_window, "fit_distance": 4})),
    PopSpikeAnalyzer(AnalyzerConfig(params={"lag_ms": cfg.ps_lag, "height": 0.2})),
]

Pipeline(cfg, analyzers).run()
