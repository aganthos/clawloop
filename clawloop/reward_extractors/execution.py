"""Execution-based reward extractor.

Scans tool-role messages in an episode for evidence of success or failure
and produces a confidence-weighted :class:`RewardSignal`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from clawloop.core.reward import RewardSignal

if TYPE_CHECKING:
    from clawloop.core.episode import Episode

_ERROR_KEYWORDS = re.compile(
    r"error|exception|traceback|failed|failure|timeout|refused|denied",
    re.IGNORECASE,
)
_HTTP_ERROR_CODES = re.compile(r"\b[45]\d{2}\b")


class ExecutionExtractor:
    """Extract a reward signal from tool-message execution outcomes."""

    name: str = "execution"

    def extract(self, episode: Episode) -> RewardSignal | None:
        """Analyse tool messages and return an aggregated reward signal.

        Returns ``None`` when no tool messages are present.
        """
        tool_messages = [m for m in episode.messages if m.role == "tool" and m.content is not None]
        if not tool_messages:
            return None

        weighted_sum = 0.0
        confidence_sum = 0.0

        for msg in tool_messages:
            value, confidence = self._score_message(msg.content)
            weighted_sum += value * confidence
            confidence_sum += confidence

        if confidence_sum == 0.0:
            return RewardSignal(name="execution", value=0.0, confidence=0.0)

        agg_value = weighted_sum / confidence_sum
        agg_confidence = confidence_sum / len(tool_messages)

        agg_value = max(-1.0, min(1.0, agg_value))

        return RewardSignal(name="execution", value=agg_value, confidence=agg_confidence)

    @staticmethod
    def _score_message(content: str) -> tuple[float, float]:
        """Return ``(value, confidence)`` for a single tool message."""
        if _ERROR_KEYWORDS.search(content):
            return (-1.0, 0.9)

        if _HTTP_ERROR_CODES.search(content):
            return (-1.0, 0.85)

        if len(content) == 0:
            return (-0.5, 0.5)

        if len(content) <= 50:
            return (0.0, 0.3)

        return (0.5, 0.6)
