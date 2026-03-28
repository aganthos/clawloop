"""SkyDiscover AdaEvolve — enterprise Evolver backend for ClawLoop.

Wraps SkyDiscover's multi-island adaptive search (AdaEvolve) as a ClawLoop
Evolver. Serializes the current harness state as a seed program, runs
SkyDiscover's evolutionary search with a ClawLoopEvaluator, and parses the
best result back into an EvolverResult.

Enterprise-only — lives in enterprise/, never synced to public repo.
"""

from __future__ import annotations

import importlib
import logging
import tempfile
from typing import Any, Callable

from clawloop.core.episode import Episode
from clawloop.core.evolver import (
    EvolverContext,
    EvolverResult,
    HarnessSnapshot,
    Provenance,
)
from enterprise.evolution.backends.skydiscover_evaluator import (
    AgentStateFactory,
    ClawLoopEvaluator,
    AdapterLike,
)
from enterprise.evolution.backends.skydiscover_utils import (
    harness_to_program,
    program_to_evolver_result,
)

log = logging.getLogger(__name__)


class SkyDiscoverAdaEvolve:
    """Wraps SkyDiscover's AdaEvolve as a ClawLoop Evolver.

    On evolve():
    1. Serialize current HarnessSnapshot to a seed program file
    2. Create a ClawLoopEvaluator with the provided adapter + tasks
    3. Call skydiscover.run_discovery(evaluator, initial_program=seed, search="adaevolve", ...)
    4. Parse the best result back into an EvolverResult
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
    ) -> None:
        self._adapter = adapter
        self._tasks = tasks
        self._agent_state_factory = agent_state_factory
        self._iterations = iterations
        self._model = model
        self._num_islands = num_islands
        self._population_size = population_size
        self._n_eval_episodes = n_eval_episodes
        self._work_dir = tempfile.mkdtemp(prefix="skydiscover_")

    @staticmethod
    def _get_run_discovery() -> Callable[..., Any]:
        """Lazy-import skydiscover.run_discovery."""
        mod = importlib.import_module("skydiscover")
        return mod.run_discovery  # type: ignore[no-any-return]

    def evolve(
        self,
        episodes: list[Episode],
        harness_state: HarnessSnapshot,
        context: EvolverContext,
    ) -> EvolverResult:
        """Run SkyDiscover AdaEvolve and return an EvolverResult."""
        run_discovery = self._get_run_discovery()

        # 1. Write harness state as seed program
        seed_path = harness_to_program(
            harness_state,
            f"{self._work_dir}/seed.json",
        )

        # 2. Create evaluator
        evaluator = ClawLoopEvaluator(
            adapter=self._adapter,
            tasks=self._tasks,
            agent_state_factory=self._agent_state_factory,
            n_episodes=self._n_eval_episodes,
        )

        # 3. Run SkyDiscover (the expensive part)
        log.info(
            "Starting SkyDiscover AdaEvolve: %d iterations, %d islands, pop=%d",
            self._iterations,
            self._num_islands,
            self._population_size,
        )
        result = run_discovery(
            evaluator=evaluator,
            initial_program=seed_path,
            search="adaevolve",
            model=self._model,
            iterations=self._iterations,
            num_islands=self._num_islands,
            population_size=self._population_size,
        )

        # 4. Parse result back to EvolverResult
        evolver_result = program_to_evolver_result(
            result.best_program, harness_state,
        )

        # Enrich provenance with SkyDiscover metadata
        evolver_result.provenance = Provenance(
            backend="skydiscover_adaevolve",
            version=getattr(result, "version", ""),
            tokens_used=getattr(result, "tokens_used", 0),
        )

        return evolver_result

    def name(self) -> str:
        return "skydiscover_adaevolve"
