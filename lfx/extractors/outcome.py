"""OutcomeExtractor — wraps TaskEnvironment.evaluate() as a reward signal."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfx.core.env import Sample
from lfx.core.reward import RewardSignal

if TYPE_CHECKING:
    from lfx.core.env import TaskEnvironment
    from lfx.core.episode import Episode


class OutcomeExtractor:
    """Extract a reward signal from the task environment's evaluation.

    Maps the environment's [0, 1] score to [-1, 1] via ``value = score * 2 - 1``.
    Confidence is always 1.0.  Returns ``None`` if no environment was provided.
    """

    name: str = "outcome"

    def __init__(self, env: TaskEnvironment | None = None) -> None:
        self._env = env

    def extract(self, episode: Episode) -> RewardSignal | None:
        if self._env is None:
            return None

        question: str | None = None
        for msg in episode.messages:
            if msg.role == "user":
                question = msg.content
                break

        if question is None:
            return None

        response: str | None = None
        for msg in reversed(episode.messages):
            if msg.role == "assistant":
                response = msg.content
                break

        if response is None:
            return None

        sample = Sample(question=question)

        try:
            tasks = self._env.get_tasks()
            for task in tasks:
                if task.question == question:
                    sample = task
                    break
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to get tasks from environment: %s", exc,
            )

        result = self._env.evaluate(sample, response)
        value = result.score * 2 - 1
        return RewardSignal(name=self.name, value=value, confidence=1.0)
