# lfx/adapters/car.py
"""CAR-bench adapter — orchestrates agentbeats-run with lfx harness injection.

Uses a custom A2A purple agent server that injects harness system prompt +
playbook into LLM calls. Results parsed from CAR's results.json.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from lfx.adapters._car_rewards import DEFAULT_CAR_WEIGHTS, map_car_scores
from lfx.adapters.base import EnvAdapter
from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta

if TYPE_CHECKING:
    from lfx.adapters._car_purple import CarPurpleAgent
    from lfx.core.loop import AgentState

log = logging.getLogger(__name__)

# All task types in CAR-bench
_ALL_TASK_TYPES = ("base", "hallucination", "disambiguation")

REWARD_METRICS = list(DEFAULT_CAR_WEIGHTS.keys())


class CARAdapter(EnvAdapter):
    """Adapter for CAR-bench. Runs agentbeats-run per learning iteration."""

    CAR_BENCH_TESTED_COMMIT = "TBD"

    def setup(self, config: dict[str, Any]) -> None:
        self._model = config.get("model", "anthropic/claude-haiku-4-5-20251001")
        self._car_bench_path = Path(
            config.get("car_bench_path", "benchmarks/car-bench")
        )
        self._output_dir = Path(
            config.get("output", f"./runs/car/{int(time.time())}")
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._task_type = config.get("task_type", "base")
        self._task_split = config.get("task_split", "test")
        self._agentbeats_cmd = config.get("agentbeats_cmd", "agentbeats-run")
        self._iteration_count = 0
        self._purple: CarPurpleAgent | None = None
        self._purple_port: int = 0
        self._config = config

    def _start_purple(self) -> None:
        """Start the purple agent server (lazy — called on first run_batch)."""
        if self._purple is not None:
            return
        from lfx.adapters._car_purple import CarPurpleAgent, start_purple_server
        from lfx.layers.harness import Harness

        self._purple = CarPurpleAgent(
            model=self._model, harness=Harness(), bench="car"
        )
        _, self._purple_port = start_purple_server(self._purple)
        log.info("Purple agent started on port %d", self._purple_port)

    def run_episode(self, task: Any, agent_state: "AgentState") -> Episode:
        """Run a single task. Delegates to run_batch with one task."""
        episodes = self.run_batch(agent_state, [task])
        return episodes[0] if episodes else self._make_failed_episode(str(task), "empty")

    def run_batch(
        self, agent_state: "AgentState", task_ids: list[Any]
    ) -> list[Episode]:
        """Run a batch of tasks via agentbeats-run."""
        self._start_purple()
        assert self._purple is not None

        # Update harness + clear sessions
        self._purple.update_harness(agent_state.harness)
        self._purple.clear_all_sessions()

        # Generate scenario
        str_ids = [str(tid) for tid in task_ids]
        scenario = self._generate_scenario(str_ids)
        iter_dir = self._output_dir / f"iter_{self._iteration_count}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        scenario_path = iter_dir / "scenario.toml"
        scenario_path.write_text(scenario)
        results_path = iter_dir / "results.json"

        # Run agentbeats-run
        try:
            result = subprocess.run(
                [self._agentbeats_cmd, str(scenario_path), "--show-logs",
                 "--output", str(results_path)],
                cwd=str(self._car_bench_path),
                capture_output=True, text=True, timeout=600,
            )
            (iter_dir / "green_agent.log").write_text(
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
            if result.returncode != 0:
                log.warning(
                    "agentbeats-run exited %d", result.returncode
                )
        except subprocess.TimeoutExpired:
            log.error("agentbeats-run timed out")
            self._iteration_count += 1
            return [self._make_failed_episode(tid, "timeout") for tid in str_ids]

        # Parse results
        episodes = self._parse_results(results_path, str_ids)

        # Save harness state
        harness_path = iter_dir / "harness_state.json"
        harness_path.write_text(json.dumps(agent_state.harness.to_dict(), indent=2))

        self._iteration_count += 1
        return episodes

    def _parse_results(
        self, results_path: Path, expected_task_ids: list[str]
    ) -> list[Episode]:
        """Parse results.json into Episodes."""
        try:
            raw = json.loads(results_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log.error("Failed to parse results: %s", e)
            return [
                self._make_failed_episode(tid, "parse_error")
                for tid in expected_task_ids
            ]

        episodes = []
        detailed = raw.get("detailed_results_by_split", {})
        for task_type_results in detailed.values():
            for task_result in task_type_results:
                episodes.append(self._map_to_episode(task_result))

        # Check for missing tasks
        found_ids = {ep.task_id for ep in episodes}
        for tid in expected_task_ids:
            if f"car:{tid}" not in found_ids:
                episodes.append(self._make_failed_episode(tid, "missing_result"))

        return episodes

    def _generate_scenario(self, task_ids: list[str]) -> str:
        """Generate scenario.toml for this batch."""
        by_type: dict[str, list[str]] = {}
        for tid in task_ids:
            # "base_0" → "base", "hallucination_3" → "hallucination"
            parts = tid.rsplit("_", 1)
            task_type = parts[0] if len(parts) == 2 and parts[1].isdigit() else "base"
            by_type.setdefault(task_type, []).append(tid)

        lines = []
        for tt in _ALL_TASK_TYPES:
            if tt in by_type:
                lines.append(
                    f'tasks_{tt}_task_id_filter = {json.dumps(by_type[tt])}'
                )
            else:
                lines.append(f"tasks_{tt}_num_tasks = 0")

        filter_block = "\n".join(lines)

        return f"""\
[green_agent]
endpoint = "http://127.0.0.1:8081"
cmd = "python src/green_car_bench_agent/server.py --host 127.0.0.1 --port 8081"

[[participants]]
role = "agent"
endpoint = "http://127.0.0.1:{self._purple_port}"

[config]
task_split = "{self._task_split}"
{filter_block}
num_trials = 1
max_steps = 50
"""

    def _map_to_episode(self, task_result: dict) -> Episode:
        """Map a CAR detailed result to an lfx Episode."""
        car_task_id = task_result["task_id"]
        task_id = f"car:{car_task_id}"

        signals, breakdown = map_car_scores(
            task_result.get("reward_info", {}),
            task_reward=task_result.get("reward", 0.0),
        )

        # Convert trajectory to lfx Messages
        messages = []
        for msg in task_result.get("trajectory", []):
            messages.append(
                Message(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                )
            )

        summary = EpisodeSummary(
            signals=signals,
            score_breakdown=breakdown,
        )

        return Episode(
            id=uuid4().hex,
            state_id="",
            task_id=task_id,
            bench="car",
            model=self._model,
            messages=messages,
            step_boundaries=[0] if messages else [],
            steps=[StepMeta(t=0, reward=task_result.get("reward", 0.0),
                            done=True, timing_ms=task_result.get("total_llm_latency_ms", 0.0))],
            summary=summary,
            created_at=time.time(),
            metadata={
                "car_raw_reward": task_result.get("reward"),
                "car_agent_cost": task_result.get("total_agent_cost"),
                "car_llm_latency_ms": task_result.get("total_llm_latency_ms"),
            },
        )

    def _make_failed_episode(self, task_id: str, reason: str) -> Episode:
        """Create a failed episode placeholder."""
        from lfx.core.reward import RewardSignal

        signals = {
            "outcome": RewardSignal(name="outcome", value=-1.0, confidence=0.5)
        }
        return Episode(
            id=uuid4().hex,
            state_id="",
            task_id=f"car:{task_id}",
            bench="car",
            model=self._model,
            messages=[],
            step_boundaries=[],
            steps=[],
            summary=EpisodeSummary(signals=signals),
            metadata={"error": reason},
        )

    def get_traces(self, episode: Episode) -> dict[str, Any]:
        return {"bench": "car", "episode_id": episode.id}

    def list_tasks(self, split: str = "base") -> list[Any]:
        # TODO: parse from CAR-bench task definitions (HuggingFace auto-download)
        # For now, return numbered task IDs
        raise NotImplementedError(
            "list_tasks requires CAR-bench data. Use run_batch with explicit task_ids."
        )
