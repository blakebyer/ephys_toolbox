from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from ephys_toolbox.src.results import RecordingContext
from ephys_toolbox.src.utils import build_smoother


@dataclass
class AnalyzerConfig:
    """Simple configuration container passed to every analyzer instance."""

    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


class Analyzer(ABC):
    """Base type for every analysis component in the production pipeline."""

    def __init__(self, name: str, config: AnalyzerConfig | None = None):
        self.name = name
        self.config = config or AnalyzerConfig()
        smoothing_cfg = self.config.params.get("smoothing") if self.config.params else None
        self._smoother = build_smoother(smoothing_cfg)

    @abstractmethod
    def run(self, context: RecordingContext) -> RecordingContext:
        """Apply analysis logic to the shared context and return it."""

    def apply_smoothing(self, data, fs=None):
        if self._smoother is None:
            return data
        return self._smoother(data, fs=fs)
