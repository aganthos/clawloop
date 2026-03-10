"""Composable reward signals for the LfX learning system.

Signals live in [-1.0, 1.0] with an associated confidence in [0.0, 1.0].
Convention: -1 = definitively bad, 0 = neutral/unknown, +1 = definitively good.
"""

from __future__ import annotations

from dataclasses import dataclass


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
