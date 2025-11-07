"""
Command line entry point for running the electrophysiology pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import click

from ephys_toolbox.src.analyzers import AnalyzerConfig
from ephys_toolbox.src.epsp import EpspAnalyzer
from ephys_toolbox.src.fiber_volley import FiberVolleyAnalyzer
from ephys_toolbox.src.pipeline import Pipeline, PipelineConfig
from ephys_toolbox.src.pop_spike import PopSpikeAnalyzer


def _parse_csv_ints(_: click.Context, __: click.Option, value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise click.BadParameter("Provide comma-separated integers, e.g. '25,50,75'.") from exc
    if not values:
        raise click.BadParameter("At least one stimulus intensity is required.")
    return values


@dataclass
class RunnerOptions:
    paths: Sequence[Path]
    stimuli: Sequence[int]
    repnum: int
    stim_window: tuple[float, float]
    fv_window: tuple[float, float]
    epsp_window: tuple[float, float]
    ps_lag: float
    ps_height: float
    fit_distance: int
    summarize: bool
    output_dir: Path | None

    def build_pipeline(self) -> Pipeline:
        config = PipelineConfig(
            paths=[str(p) for p in self.paths],
            stim_intensities=self.stimuli,
            repnum=self.repnum,
            stim_window=list(self.stim_window),
            fv_window=list(self.fv_window),
            epsp_window=list(self.epsp_window),
            ps_lag=self.ps_lag,
            summarize=self.summarize,
            output_dir=str(self.output_dir) if self.output_dir else None,
        )

        analyzers = [
            FiberVolleyAnalyzer(AnalyzerConfig(params={"window": list(self.fv_window)})),
            EpspAnalyzer(
                AnalyzerConfig(
                    params={
                        "window": list(self.epsp_window),
                        "fit_distance": self.fit_distance,
                    }
                )
            ),
            PopSpikeAnalyzer(
                AnalyzerConfig(
                    params={
                        "lag_ms": self.ps_lag,
                        "height": self.ps_height,
                    }
                )
            ),
        ]

        return Pipeline(config, analyzers)


@click.command(context_settings={"show_default": True})
@click.option(
    "--paths",
    "-p",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="ABF files or directories containing ABF files.",
)
@click.option(
    "--stimuli",
    "-s",
    required=True,
    callback=_parse_csv_ints,
    help="Comma-separated stimulus intensities (µA), e.g. '25,50,75'.",
)
@click.option("--repnum", default=3, type=int, help="Number of repetitions per stimulus.")
@click.option(
    "--stim-window",
    nargs=2,
    type=float,
    default=(0.0, 1.0),
    help="Stimulus artifact window in ms to blank/crop.",
)
@click.option(
    "--fv-window",
    nargs=2,
    type=float,
    default=(0.0, 1.5),
    help="Fiber volley search window in ms.",
)
@click.option(
    "--epsp-window",
    nargs=2,
    type=float,
    default=(1.5, 5.25),
    help="EPSP slope/peak search window in ms.",
)
@click.option("--ps-lag", default=3.0, type=float, help="Population spike search window (ms) after EPSP trough.")
@click.option("--ps-height", default=0.2, type=float, help="Minimum PS height above EPSP trough (mV).")
@click.option("--fit-distance", default=4, type=int, help="Samples in each direction for EPSP slope fitting.")
@click.option("--summarize/--no-summarize", default=False, help="Emit a combined results_summary.csv.")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Optional directory to store per-recording outputs (defaults to alongside each ABF).",
)
def main(
    paths,
    stimuli,
    repnum,
    stim_window,
    fv_window,
    epsp_window,
    ps_lag,
    ps_height,
    fit_distance,
    summarize,
    output_dir,
):
    """Run the electrophysiology pipeline from the command line."""
    options = RunnerOptions(
        paths=paths,
        stimuli=stimuli,
        repnum=repnum,
        stim_window=stim_window,
        fv_window=fv_window,
        epsp_window=epsp_window,
        ps_lag=ps_lag,
        ps_height=ps_height,
        fit_distance=fit_distance,
        summarize=summarize,
        output_dir=output_dir,
    )
    pipeline = options.build_pipeline()
    pipeline.run()


if __name__ == "__main__":
    main()

