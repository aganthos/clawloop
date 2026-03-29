"""Async cloud wrapper for SkyDiscover AdaEvolve.

Integrates with SkyDiscover (https://github.com/skydiscover-ai/skydiscover)
by UC Berkeley Sky Computing Lab — Apache 2.0. No SkyDiscover code is copied.

Runs evolution in a background thread and returns immediately with a run_id.
Integrates with ClawLoop's existing async evolver patterns:
- EvolverResult.run_id for tracking
- make_fb_info(status="running") for progress polling
- Harness.cancel() for aborting long-running evolution

Cloud hosting architecture:
    ┌─────────────┐    evolve()     ┌──────────────────┐
    │   Harness    │ ──────────────▶│ CloudAdaEvolve   │
    │ (main loop)  │    run_id      │ (this module)    │
    │              │◀──────────────│                  │
    └──────┬───────┘                └───────┬──────────┘
           │ poll_status(run_id)            │ background thread
           │                               ▼
           │                        ┌──────────────────┐
           │                        │ SkyDiscover       │
           │                        │ run_discovery()   │
           │                        │ (multi-island)    │
           │                        └──────────────────┘
           │ get_result(run_id)
           ▼
    ┌──────────────┐
    │ EvolverResult │
    └──────────────┘

For production deployment, the background thread could be replaced with:
- A Celery/Redis task queue for horizontal scaling
- A Cloud Run job for serverless execution
- A Kubernetes Job for cluster-based execution

The interface stays the same — run_id for tracking, poll for status.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from clawloop.core.episode import Episode
from clawloop.core.evolver import (
    EvolverContext,
    EvolverResult,
    HarnessSnapshot,
    Provenance,
)
from enterprise.evolution.backends.skydiscover_adaevolve import SkyDiscoverAdaEvolve
from enterprise.evolution.backends.skydiscover_evaluator import (
    AdapterLike,
    AgentStateFactory,
)

log = logging.getLogger(__name__)


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EvolutionRun:
    """Tracks state of a single async evolution run."""

    run_id: str
    status: RunStatus = RunStatus.PENDING
    result: EvolverResult | None = None
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    _cancel_event: threading.Event = field(
        default_factory=threading.Event, repr=False,
    )


class CloudAdaEvolve:
    """Async wrapper around SkyDiscoverAdaEvolve for cloud/background execution.

    On evolve():
    - Launches evolution in a background thread
    - Returns immediately with an EvolverResult containing only run_id
    - Use poll_status() to check progress, get_result() to retrieve the final result
    - Use cancel() to abort

    Implements the Evolver protocol so it can be used as a drop-in replacement.
    """

    def __init__(
        self,
        adapter: AdapterLike,
        tasks: list[Any],
        agent_state_factory: AgentStateFactory,
        *,
        iterations: int = 20,
        model: str = "claude-sonnet-4-6",
        num_islands: int = 2,
        population_size: int = 20,
        n_eval_episodes: int = 5,
        max_concurrent: int = 1,
    ) -> None:
        self._backend = SkyDiscoverAdaEvolve(
            adapter=adapter,
            tasks=tasks,
            agent_state_factory=agent_state_factory,
            iterations=iterations,
            model=model,
            num_islands=num_islands,
            population_size=population_size,
            n_eval_episodes=n_eval_episodes,
        )
        self._max_concurrent = max_concurrent
        self._runs: dict[str, EvolutionRun] = {}
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

    def evolve(
        self,
        episodes: list[Episode],
        harness_state: HarnessSnapshot,
        context: EvolverContext,
    ) -> EvolverResult:
        """Launch evolution in background, return immediately with run_id.

        The returned EvolverResult has only run_id set. The Harness should
        check FBResult.info["status"] == "running" and poll via
        evolution_summary(run_id).
        """
        # Check concurrency limit
        with self._lock:
            active = sum(
                1 for r in self._runs.values()
                if r.status in (RunStatus.PENDING, RunStatus.RUNNING)
            )
            if active >= self._max_concurrent:
                return EvolverResult(
                    run_id="",
                    provenance=Provenance(backend="skydiscover_adaevolve_cloud"),
                )

        run_id = f"sky-{uuid.uuid4().hex[:12]}"
        run = EvolutionRun(run_id=run_id)

        with self._lock:
            self._runs[run_id] = run

        thread = threading.Thread(
            target=self._run_evolution,
            args=(run, episodes, harness_state, context),
            daemon=False,
            name=f"skydiscover-{run_id}",
        )
        thread.start()
        with self._lock:
            self._threads.append(thread)

        return EvolverResult(
            run_id=run_id,
            provenance=Provenance(backend="skydiscover_adaevolve_cloud"),
        )

    def poll_status(self, run_id: str) -> dict[str, Any]:
        """Check the status of an evolution run.

        Returns a dict compatible with make_fb_info schema:
        {"status": str, "run_id": str, "elapsed_s": float, ...}
        """
        with self._lock:
            run = self._runs.get(run_id)

        if run is None:
            return {"status": "failed", "run_id": run_id, "error": "Unknown run_id"}

        elapsed = (
            (run.completed_at or time.time()) - run.started_at
            if run.started_at > 0
            else 0.0
        )

        info: dict[str, Any] = {
            "status": run.status.value,
            "run_id": run_id,
            "elapsed_s": round(elapsed, 1),
        }
        if run.error:
            info["error"] = run.error
        return info

    def get_result(self, run_id: str) -> EvolverResult | None:
        """Retrieve the final EvolverResult for a completed run.

        Returns None if the run is still in progress or not found.
        """
        with self._lock:
            run = self._runs.get(run_id)

        if run is None or run.status != RunStatus.COMPLETED:
            return None
        return run.result

    def cancel(self, run_id: str) -> bool:
        """Request cancellation of a running evolution.

        Returns True if cancellation was requested, False if run not found
        or already completed.
        """
        with self._lock:
            run = self._runs.get(run_id)

        if run is None:
            return False
        if run.status not in (RunStatus.PENDING, RunStatus.RUNNING):
            return False

        run._cancel_event.set()
        run.status = RunStatus.CANCELLED
        run.completed_at = time.time()
        log.info("Cancellation requested for run %s", run_id)
        return True

    def active_runs(self) -> list[str]:
        """Return run_ids of currently active evolutions."""
        with self._lock:
            return [
                r.run_id for r in self._runs.values()
                if r.status in (RunStatus.PENDING, RunStatus.RUNNING)
            ]

    def name(self) -> str:
        return "skydiscover_adaevolve_cloud"

    def cleanup(self, timeout: float = 10.0) -> None:
        """Cancel all runs, join threads, and clean up resources.

        Threads are non-daemon so they won't be killed on main exit.
        This method ensures graceful shutdown by signalling cancellation
        and waiting for threads to finish.
        """
        with self._lock:
            for run in self._runs.values():
                if run.status in (RunStatus.PENDING, RunStatus.RUNNING):
                    run._cancel_event.set()
                    run.status = RunStatus.CANCELLED
            threads = list(self._threads)

        for t in threads:
            t.join(timeout=timeout)

        self._backend.cleanup()

    def _run_evolution(
        self,
        run: EvolutionRun,
        episodes: list[Episode],
        harness_state: HarnessSnapshot,
        context: EvolverContext,
    ) -> None:
        """Execute evolution in background thread."""
        run.started_at = time.time()

        # Check cancellation before overwriting status — cancel() may have
        # already set CANCELLED between evolve() and thread start.
        if run._cancel_event.is_set():
            run.status = RunStatus.CANCELLED
            run.completed_at = time.time()
            return

        run.status = RunStatus.RUNNING

        try:
            result = self._backend.evolve(episodes, harness_state, context)

            if run._cancel_event.is_set():
                run.status = RunStatus.CANCELLED
                return

            run.result = result
            run.status = RunStatus.COMPLETED
            log.info(
                "Evolution run %s completed: %d insights, %d candidate benches",
                run.run_id,
                len(result.insights),
                len(result.candidates),
            )

        except Exception as exc:
            run.status = RunStatus.FAILED
            run.error = str(exc)
            log.exception("Evolution run %s failed", run.run_id)

        finally:
            run.completed_at = time.time()
