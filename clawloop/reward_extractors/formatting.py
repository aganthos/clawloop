"""FormattingFilter — hard filter for episode quality."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clawloop.core.episode import Episode


class FormattingFilter:
    """Hard filter that excludes episodes with malformed assistant responses.

    Episodes that fail this filter are excluded from training entirely.
    This is *not* a reward signal — it is a binary pass/fail gate.
    """

    def __init__(self, min_response_length: int = 10) -> None:
        self.min_response_length = min_response_length

    def passes(self, episode: Episode) -> bool:
        response: str | None = None
        for msg in reversed(episode.messages):
            if msg.role == "assistant":
                response = msg.content
                break

        if response is None:
            return False

        if not response or len(response) < self.min_response_length:
            return False

        return True
