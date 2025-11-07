from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from ephys_toolbox.src.results import RecordingContext


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

    @abstractmethod
    def run(self, context: RecordingContext) -> RecordingContext:
        """Apply analysis logic to the shared context and return it."""
