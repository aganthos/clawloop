"""UserFeedbackExtractor — reads pre-populated user signal from episode."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfx.core.reward import RewardSignal

if TYPE_CHECKING:
    from lfx.core.episode import Episode


class UserFeedbackExtractor:
    """Pass through the ``"user"`` signal already set on the episode summary.

    This extractor does not generate new signals — it simply reads what was
    previously set by ``EpisodeCollector.submit_feedback()``.
    """

    name: str = "user"

    def extract(self, episode: Episode) -> RewardSignal | None:
        return episode.summary.signals.get("user")
