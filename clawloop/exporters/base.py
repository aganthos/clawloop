"""TraceExporter — abstract interface for episode serialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from clawloop.core.episode import Episode


class TraceExporter(ABC):
    """Base class for exporting episodes to external trace formats."""

    @abstractmethod
    def export(self, episodes: list[Episode]) -> Any:
        """Export a batch of episodes to the target format."""
        ...

    @abstractmethod
    def export_one(self, episode: Episode) -> Any:
        """Export a single episode."""
        ...
