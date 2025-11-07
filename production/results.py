from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class RecordingContext:
    abf_path: Path
    averaged: pd.DataFrame
    metadata: Dict[str, Any]
    features: Dict[str, pd.DataFrame] = field(default_factory=dict)
    plots: Dict[str, list[Path]] = field(default_factory=dict)

    def register_feature(self, key: str, table: pd.DataFrame) -> None:
        self.features[key] = table

    def register_plot(self, key: str, path: Path) -> None:
        self.plots.setdefault(key, []).append(path)


@dataclass
class FeatureResult:
    name: str
    table: pd.DataFrame
    cursor_overrides: Optional[pd.DataFrame] = None