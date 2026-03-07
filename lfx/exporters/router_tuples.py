"""Router training data exporter — (task_embedding, model_id, cost, reward) tuples.

Aligned with the RouteLLM/NotDiamond approach: the exported tuples feed a
classifier that maps (query -> best model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lfx.core.episode import Episode
from lfx.exporters.base import TraceExporter


@dataclass
class RouterTuple:
    """One routing training sample."""

    task_embedding: list[float]
    model_id: str
    cost: float
    reward: float


class RouterTupleExporter(TraceExporter):
    """Extract routing training tuples from episodes.

    Parameters
    ----------
    embed_fn:
        Callable that maps a task description (str) to an embedding vector.
        If ``None``, a placeholder zero-vector is used.
    """

    def __init__(self, embed_fn: Any = None, embed_dim: int = 768) -> None:
        self._embed_fn = embed_fn
        self._embed_dim = embed_dim

    def export(self, episodes: list[Episode]) -> list[RouterTuple]:
        return [self.export_one(ep) for ep in episodes]

    def export_one(self, episode: Episode) -> RouterTuple:
        # Extract task text from the first user message
        task_text = ""
        for msg in episode.messages:
            if msg.role == "user":
                task_text = msg.content
                break

        # Compute embedding
        if self._embed_fn is not None:
            embedding = self._embed_fn(task_text)
        else:
            embedding = [0.0] * self._embed_dim

        # Determine model (use the model from the first assistant message)
        model_id = ""
        for msg in episode.messages:
            if msg.role == "assistant" and msg.model:
                model_id = msg.model
                break

        # Estimate cost from token usage
        cost = 0.0
        if episode.summary.token_usage is not None:
            cost = float(episode.summary.token_usage.total_tokens)

        return RouterTuple(
            task_embedding=embedding,
            model_id=model_id,
            cost=cost,
            reward=episode.summary.total_reward,
        )
