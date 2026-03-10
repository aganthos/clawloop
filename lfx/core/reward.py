"""Composable reward signals for the LfX learning system.

Signals live in [-1.0, 1.0] with an associated confidence in [0.0, 1.0].
Convention: -1 = definitively bad, 0 = neutral/unknown, +1 = definitively good.

``RewardExtractor`` is a protocol that pulls a signal from a completed
``Episode``.  ``RewardPipeline`` chains extractors, populating
``episode.summary.signals``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lfx.core.episode import Episode


@dataclass(frozen=True)
class RewardSignal:
    """A single reward signal with value and confidence.

    Values are clamped to [-1.0, 1.0], confidence to [0.0, 1.0].
    """

    name: str
    value: float
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", max(-1.0, min(1.0, self.value)))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, self.confidence)))


@runtime_checkable
class RewardExtractor(Protocol):
    """Protocol for extracting a reward signal from an episode."""

    name: str

    def extract(self, episode: Episode) -> RewardSignal | None: ...


class RewardPipeline:
    """Run extractors in order, populating episode.summary.signals.

    Judge extractors (name="judge") are automatically skipped when
the episode already has a high-confidence signal.
    """

    def __init__(self, extractors: list[RewardExtractor]) -> None:
        self.extractors = extractors

    def enrich(self, episode: Episode) -> None:
        for ext in self.extractors:
            if ext.name == "judge" and not episode.summary.needs_judge():
                continue
            sig = ext.extract(episode)
            if sig is not None:
                episode.summary.signals[sig.name] = sig
