from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import pandas as pd


@dataclass
class RecordingContext:
    """Shared state object passed between analyzers while processing one ABF file."""

    abf_path: Path
    tidy: pd.DataFrame
    averaged: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, pd.DataFrame] = field(default_factory=dict)
    plots: Dict[str, list[Path]] = field(default_factory=dict)

    def register_feature(self, name: str, table: pd.DataFrame) -> None:
        self.features[name] = table

    def get_feature(self, name: str) -> pd.DataFrame | None:
        return self.features.get(name)

    def register_plot(self, name: str, path: Path) -> None:
        self.plots.setdefault(name, []).append(path)
