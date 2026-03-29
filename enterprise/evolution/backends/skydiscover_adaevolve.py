"""SkyDiscover AdaEvolve — enterprise Evolver backend for ClawLoop.

Wraps SkyDiscover's multi-island adaptive search (AdaEvolve) as a ClawLoop
Evolver. Serializes the current harness state as a seed program, runs
SkyDiscover's evolutionary search with a ClawLoopEvaluator, and parses the
best result back into an EvolverResult.

Enterprise-only — lives in enterprise/, never synced to public repo.

Real SkyDiscover API (skydiscover/api.py):
    run_discovery(
        evaluator, initial_program, model, iterations, search,
        config, agentic, output_dir, system_prompt, api_base, cleanup,
    ) -> DiscoveryResult

    DiscoveryResult:
        best_program: Program  (object, not path)
        best_score: float
        best_solution: str     (actual solution text)
        metrics: dict
        output_dir: str | None
        initial_score: float | None

    AdaEvolve config (num_islands, population_size, decay) is passed via
    config= parameter as a Config object, not as direct kwargs.
"""

from __future__ import annotations

import importlib
import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
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


def _build_config(
    num_islands: int,
    population_size: int,
) -> Any:
    """Build a SkyDiscover Config with AdaEvolve-specific settings.

    The real API accepts config as a Config object or YAML path.
    AdaEvolve params (num_islands, population_size) are config-level
    settings in AdaEvolveDatabaseConfig, not direct run_discovery kwargs.
    """
    try:
        from skydiscover.config import Config
        cfg = Config()
        cfg.search = "adaevolve"
        cfg.num_islands = num_islands
        cfg.population_size = population_size
        return cfg
    except ImportError:
        # Return a dict as fallback for environments without skydiscover
        return {
            "search": "adaevolve",
            "num_islands": num_islands,
            "population_size": population_size,
        }


def _extract_best_program_path(result: Any, fallback_dir: str) -> str:
    """Extract a file path to the best program from a DiscoveryResult.

    DiscoveryResult.best_program is a Program object, not a path string.
    We need to serialize the solution content to a temp file for parsing.
    """
    # Try best_solution (str) first — this is the actual solution text
    solution_text = getattr(result, "best_solution", None)

    # Try best_program.code as fallback
    if solution_text is None:
        best_prog = getattr(result, "best_program", None)
        if best_prog is not None:
            solution_text = getattr(best_prog, "code", None) or str(best_prog)

    if solution_text is None:
        raise ValueError("SkyDiscover result has no best_solution or best_program.code")

    # If the solution is already a file path, return it
    if isinstance(solution_text, str) and Path(solution_text).exists():
        return solution_text

    # Otherwise, parse as JSON program content and write to file
    out_path = Path(fallback_dir) / "best_program.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If it's valid JSON, write as-is; otherwise wrap in program format
    try:
        json.loads(solution_text)
        out_path.write_text(solution_text)
    except (json.JSONDecodeError, TypeError):
        # Treat as system prompt text and wrap
        program = {
            "system_prompt": str(solution_text),
            "playbook": [],
            "model": "",
        }
        out_path.write_text(json.dumps(program, indent=2))

    return str(out_path)


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
        """Run SkyDiscover AdaEvolve and return an EvolverResult.

        Note: ``episodes`` is accepted for Evolver protocol compatibility but
        not used directly — the evaluator generates fresh episodes against each
        candidate config using the adapter and tasks provided at construction.
        """
        run_discovery = self._get_run_discovery()

        # 1. Write harness state as seed program (unique name for concurrency)
        run_id = uuid.uuid4().hex[:8]
        run_dir = f"{self._work_dir}/run_{run_id}"
        seed_path = harness_to_program(
            harness_state,
            f"{run_dir}/seed.json",
        )

        # 2. Create evaluator
        evaluator = ClawLoopEvaluator(
            adapter=self._adapter,
            tasks=self._tasks,
            agent_state_factory=self._agent_state_factory,
            n_episodes=self._n_eval_episodes,
        )

        # 3. Build config for AdaEvolve-specific parameters
        config = _build_config(
            num_islands=self._num_islands,
            population_size=self._population_size,
        )

        # 4. Run SkyDiscover (the expensive part)
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
            config=config,
            output_dir=run_dir,
        )

        # 5. Extract best program from DiscoveryResult
        best_path = _extract_best_program_path(result, run_dir)

        # 6. Parse result back to EvolverResult
        evolver_result = program_to_evolver_result(best_path, harness_state)

        # Enrich provenance with SkyDiscover metadata
        metrics = getattr(result, "metrics", {}) or {}
        evolver_result.provenance = Provenance(
            backend="skydiscover_adaevolve",
            tokens_used=metrics.get("total_tokens", 0),
        )

        # 7. Cleanup run-specific temp files
        shutil.rmtree(run_dir, ignore_errors=True)

        return evolver_result

    def cleanup(self) -> None:
        """Remove the work directory. Safe to call multiple times."""
        shutil.rmtree(self._work_dir, ignore_errors=True)

    def name(self) -> str:
        return "skydiscover_adaevolve"
