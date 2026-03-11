"""EpisodeCollector — constructs episodes from live agent traffic."""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from typing import Callable

from lfx.core.episode import Episode, EpisodeSummary, Message
from lfx.core.reward import RewardPipeline, RewardSignal
from lfx.extractors.formatting import FormattingFilter

log = logging.getLogger(__name__)


class EpisodeCollector:
    """Collects live agent traffic into episodes, enriches with rewards,
    and triggers learning when batch is full.

    Thread-safe: all buffer operations are protected by a lock.
    """

    def __init__(
        self,
        pipeline: RewardPipeline,
        batch_size: int = 16,
        on_batch: Callable[[list[Episode]], None] | None = None,
        formatting_filter: FormattingFilter | None = None,
        max_episode_cache: int = 10_000,
        state_id: str | Callable[[], str] = "live",
    ) -> None:
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.on_batch = on_batch
        self.formatting_filter = formatting_filter or FormattingFilter()
        self._max_cache = max_episode_cache
        self._state_id = state_id

        self._buffer: list[Episode] = []
        self._episode_index: OrderedDict[str, Episode] = OrderedDict()
        self._lock = threading.Lock()

        # Metrics
        self._episodes_collected = 0
        self._episodes_filtered = 0
        self._feedback_received = 0
        self._feedback_missed = 0
        self._eviction_count = 0

    def _resolve_state_id(self) -> str:
        """Return the current state_id, invoking callable if needed."""
        if callable(self._state_id):
            return self._state_id()
        return self._state_id

    def ingest(
        self,
        messages: list[Message],
        *,
        task_id: str = "",
        session_id: str = "",
    ) -> Episode:
        """Convert a completed request/response into an Episode.

        Enriches with reward signals via pipeline. If formatting filter
        fails, marks as filtered and excludes from training buffer.
        """
        episode = Episode(
            id=uuid.uuid4().hex,
            state_id=self._resolve_state_id(),
            task_id=task_id or uuid.uuid4().hex,
            bench="live",
            messages=list(messages),
            step_boundaries=[0] if messages else [],
            steps=[],
            summary=EpisodeSummary(),
            session_id=session_id,
        )

        # Enrich with reward signals
        self.pipeline.enrich(episode)

        batch_to_flush: list[Episode] | None = None

        with self._lock:
            self._episodes_collected += 1

            # Index for feedback lookup
            self._episode_index[episode.id] = episode
            self._maybe_evict()

            # Formatting gate
            if not self.formatting_filter.passes(episode):
                episode.summary.filtered = True
                self._episodes_filtered += 1
                return episode

            # Buffer for training
            self._buffer.append(episode)
            if len(self._buffer) >= self.batch_size:
                batch_to_flush = list(self._buffer)
                self._buffer.clear()

        # Flush outside lock to avoid holding it during callback
        if batch_to_flush and self.on_batch:
            self.on_batch(batch_to_flush)

        return episode

    def submit_feedback(self, episode_id: str, score: float) -> bool:
        """Attach user feedback to an episode. Returns False if not found.

        ``score`` must be in [-1.0, 1.0]: -1 = bad, 0 = neutral, +1 = good.
        Values outside this range are clamped by :class:`RewardSignal`.
        """
        with self._lock:
            ep = self._episode_index.get(episode_id)
            if ep is None:
                self._feedback_missed += 1
                return False
            ep.summary.signals["user"] = RewardSignal("user", score, 1.0)
            self._feedback_received += 1
            # Promote to most-recently-used so feedback isn't evicted early
            self._episode_index.move_to_end(episode_id)
            return True

    @property
    def metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "episodes_collected": self._episodes_collected,
                "episodes_filtered": self._episodes_filtered,
                "feedback_received": self._feedback_received,
                "feedback_missed": self._feedback_missed,
                "evictions": self._eviction_count,
                "buffer_size": len(self._buffer),
                "cache_size": len(self._episode_index),
            }

    def _maybe_evict(self) -> None:
        """Evict oldest episodes from LRU cache when over limit."""
        while len(self._episode_index) > self._max_cache:
            evicted_id, _ = self._episode_index.popitem(last=False)
            self._eviction_count += 1
