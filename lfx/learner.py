"""AsyncLearner — background thread for learning from episode batches."""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from statistics import mean
from typing import Any

from lfx.core.intensity import AdaptiveIntensity
from lfx.core.types import Datum

log = logging.getLogger(__name__)


class AsyncLearner:
    """Run learning in a background worker thread.

    Episodes are submitted via on_batch(). A single worker thread
    processes them sequentially — never blocks the caller.
    """

    def __init__(
        self,
        agent_state: Any,
        active_layers: list[str] | None = None,
        intensity: AdaptiveIntensity | None = None,
        max_queue_size: int = 4,
        overflow: str = "drop_newest",
    ) -> None:
        self.agent_state = agent_state
        self.active_layers = active_layers or ["harness"]
        self.intensity = intensity or AdaptiveIntensity()
        self.overflow = overflow

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._iteration = 0

        self._batches_trained = 0
        self._batches_dropped = 0
        self._batches_failed = 0

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="lfx-learner",
        )
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)

    def on_batch(self, episodes: list) -> None:
        if self.overflow == "block":
            self._queue.put(episodes)
        elif self.overflow == "drop_oldest":
            dropped = 0
            while self._queue.full():
                try:
                    self._queue.get_nowait()
                    dropped += 1
                except queue.Empty:
                    break
            self._batches_dropped += dropped
            try:
                self._queue.put_nowait(episodes)
            except queue.Full:
                self._batches_dropped += 1
        else:  # drop_newest
            try:
                self._queue.put_nowait(episodes)
            except queue.Full:
                self._batches_dropped += 1

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "batches_trained": self._batches_trained,
            "batches_dropped": self._batches_dropped,
            "batches_failed": self._batches_failed,
            "iteration": self._iteration,
            "queue_size": self._queue.qsize(),
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                episodes = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._learn(episodes)

    def _learn(self, episodes: list) -> None:
        batch_id = uuid.uuid4().hex[:8]

        rewards = [ep.summary.normalized_reward() for ep in episodes]
        avg_reward = mean(rewards) if rewards else 0.0
        self.intensity.record_reward(avg_reward)

        log.info(
            "Batch %s: %d episodes, avg_reward=%.3f",
            batch_id, len(episodes), avg_reward,
        )

        datum = Datum(episodes=episodes)

        for name in self.active_layers:
            layer = getattr(self.agent_state, name, None)
            if layer is None:
                continue
            try:
                layer.forward_backward(datum).result()
                layer.optim_step().result()
            except Exception as exc:
                log.error(
                    "Layer %s failed on batch %s: %s", name, batch_id, exc,
                )
                self._batches_failed += 1
                try:
                    layer.clear_pending_state()
                except Exception as clear_exc:
                    log.error(
                        "Failed to clear pending state for %s: %s",
                        name, clear_exc,
                    )
                return

        self._batches_trained += 1
        self._iteration += 1
