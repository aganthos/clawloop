"""AsyncLearner — background thread for learning from episode batches."""

from __future__ import annotations

import copy
import logging
import queue
import threading
import uuid
from statistics import mean
from typing import Any, Callable

from clawloop.core.intensity import AdaptiveIntensity
from clawloop.core.types import Datum, FBResult

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
        on_learn_complete: Callable | None = None,
    ) -> None:
        self.agent_state = agent_state
        self.active_layers = active_layers or ["harness"]
        self.intensity = intensity or AdaptiveIntensity()
        self.overflow = overflow
        self.on_learn_complete = on_learn_complete

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

    def on_batch(self, episodes: list) -> bool:
        """Submit a batch for learning. Returns True if enqueued, False if dropped."""
        if self.overflow == "block":
            self._queue.put(episodes)
            return True
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
                return True
            except queue.Full:
                self._batches_dropped += 1
                return False
        else:  # drop_newest
            try:
                self._queue.put_nowait(episodes)
                return True
            except queue.Full:
                self._batches_dropped += 1
                return False

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
        success = False
        error_msg: str | None = None

        try:
            rewards = [ep.summary.normalized_reward() for ep in episodes]
            avg_reward = mean(rewards) if rewards else 0.0
            self.intensity.record_reward(avg_reward)

            log.info(
                "Batch %s: %d episodes, avg_reward=%.3f",
                batch_id, len(episodes), avg_reward,
            )

            # NOTE: Support-query split disabled — see loop.py.
            layer_datums: dict[str, Datum] = {
                "harness": Datum(episodes=episodes),
                "weights": Datum(episodes=episodes),
                "router": Datum(episodes=episodes),
            }

            # Phase 1: forward_backward all layers, collect results
            fb_results: dict[str, FBResult] = {}
            layers: list[tuple[str, Any]] = []
            for name in self.active_layers:
                layer = getattr(self.agent_state, name, None)
                if layer is None:
                    continue
                # Skip harness when intensity says not to reflect
                if name == "harness" and not self.intensity.should_reflect(self._iteration):
                    log.info("Batch %s: skipping harness fb (adaptive intensity)", batch_id)
                    fb_results[name] = FBResult(status="skipped")
                    continue
                layers.append((name, layer))
                datum = layer_datums.get(name, Datum(episodes=episodes))
                should_clear = False
                try:
                    fb_result = layer.forward_backward(datum).result()
                    fb_results[name] = fb_result
                    if fb_result.status in ("error", "skipped"):
                        should_clear = True
                except Exception as exc:
                    log.error(
                        "forward_backward failed for %s on batch %s: %s",
                        name, batch_id, exc,
                    )
                    fb_results[name] = FBResult(status="error")
                    should_clear = True

                if should_clear:
                    try:
                        layer.clear_pending_state()
                    except Exception:
                        log.exception(
                            "Failed to clear pending state for %s", name,
                        )

            # Phase 2: optim_step with cross-layer rollback
            layers_to_optim = [
                (name, layer) for name, layer in layers
                if fb_results.get(name, FBResult(status="error")).status
                not in ("error", "skipped")
            ]

            if not layers_to_optim:
                log.warning("Batch %s: no layers to optim (all FB error/skipped)", batch_id)
                error_msg = "no layers to optimize"
                return

            # Snapshot for rollback
            snapshots: dict[str, dict[str, Any]] = {}
            try:
                for name, layer in layers_to_optim:
                    snapshots[name] = copy.deepcopy(layer.to_dict())
            except Exception:
                log.exception("Snapshot failed — aborting optim for batch %s", batch_id)
                for name, layer in layers_to_optim:
                    try:
                        layer.clear_pending_state()
                    except Exception:
                        log.exception("Failed to clear pending state for %s", name)
                self._batches_failed += 1
                error_msg = "snapshot failed"
                return

            optim_failed = False
            for name, layer in layers_to_optim:
                try:
                    result = layer.optim_step().result()
                    if result.status == "error":
                        log.error(
                            "optim_step returned error for %s on batch %s",
                            name, batch_id,
                        )
                        optim_failed = True
                        break
                except Exception as exc:
                    log.error(
                        "optim_step failed for %s on batch %s: %s",
                        name, batch_id, exc,
                    )
                    optim_failed = True
                    break

            if optim_failed:
                log.warning(
                    "Rolling back all layers to pre-optim state for batch %s",
                    batch_id,
                )
                for name, layer in layers_to_optim:
                    if name in snapshots:
                        try:
                            lr = layer.load_state(snapshots[name]).result()
                            if lr.status != "ok":
                                log.error(
                                    "Rollback returned %s for %s", lr.status, name,
                                )
                        except Exception:
                            log.exception("Rollback failed for %s", name)
                self._batches_failed += 1
                error_msg = "optim_step failed"
                return

            # Generation flush: clear stale weights buffer on playbook advance
            harness = getattr(self.agent_state, "harness", None)
            weights = getattr(self.agent_state, "weights", None)
            if harness is not None and hasattr(harness, "playbook_generation"):
                current_gen = harness.playbook_generation
                prev_gen = getattr(self, "_prev_playbook_generation", current_gen)
                if current_gen > prev_gen:
                    if weights is not None and hasattr(weights, "pending_advantage_count"):
                        stale = weights.pending_advantage_count()
                        weights.clear_pending_state()
                        log.info(
                            "Generation %d->%d: flushed %d stale episodes from weights buffer",
                            prev_gen, current_gen, stale,
                        )
                self._prev_playbook_generation = current_gen

            self._batches_trained += 1
            self._iteration += 1
            success = True

        except Exception as exc:
            log.exception("Unexpected error in _learn for batch %s", batch_id)
            error_msg = str(exc)

        finally:
            if self.on_learn_complete is not None:
                try:
                    self.on_learn_complete(
                        episodes, success=success, error=error_msg,
                    )
                except Exception:
                    log.exception("on_learn_complete callback failed")
