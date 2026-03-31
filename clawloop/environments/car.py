# clawloop/adapters/car.py
"""CAR-bench adapter — orchestrates agentbeats-run with clawloop harness injection.

Writes harness prompt to a JSON file, generates a scenario.toml that spawns
the purple agent server (which reads the file and injects the prompt into the system
message), then runs agentbeats-run and parses results.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from clawloop.environments._car_rewards import DEFAULT_CAR_WEIGHTS, map_car_scores
from clawloop.environments.base import EnvAdapter
from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta

if TYPE_CHECKING:
    from clawloop.core.loop import AgentState

log = logging.getLogger(__name__)

# All task types in CAR-bench
_ALL_TASK_TYPES = ("base", "hallucination", "disambiguation")

REWARD_METRICS = list(DEFAULT_CAR_WEIGHTS.keys())


class CARAdapter(EnvAdapter):
    """Adapter for CAR-bench. Runs agentbeats-run per learning iteration."""

    def setup(self, config: dict[str, Any]) -> None:
        self._model = config.get("model", "anthropic/claude-haiku-4-5-20251001")
        self._car_bench_path = Path(
            config.get("car_bench_path", "benchmarks/a2a/car-bench")
        )
        self._output_dir = Path(
            config.get("output", f"./runs/car/{int(time.time())}")
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._task_type = config.get("task_type", "base")
        self._task_split = config.get("task_split", "test")
        self._agentbeats_cmd = config.get(
            "agentbeats_cmd",
            str(self._car_bench_path.resolve() / ".venv" / "bin" / "agentbeats-run"),
        )
        self._api_base = config.get("api_base")
        self._api_key = config.get("api_key")
        self._iteration_count = 0
        self._config = config

    def run_episode(self, task: Any, agent_state: "AgentState") -> Episode:
        """Run a single task. Delegates to run_batch with one task."""
        episodes = self.run_batch(agent_state, [task])
        return episodes[0] if episodes else self._make_failed_episode(str(task), "empty")

    def run_batch(
        self, agent_state: "AgentState", task_ids: list[Any]
    ) -> list[Episode]:
        """Run a batch of tasks via agentbeats-run with clawloop harness injection."""
        str_ids = [str(tid) for tid in task_ids]
        self._current_state_id = agent_state.state_id().combined_hash

        iter_dir = (self._output_dir / f"iter_{self._iteration_count}").resolve()
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Write harness prompt to file for the purple agent server to read
        harness_prompt = agent_state.harness.system_prompt("car")
        harness_file = iter_dir / "harness_prompt.json"
        harness_file.write_text(json.dumps({"prompt": harness_prompt}))
        log.info(
            "Harness prompt: %d chars, %d playbook entries",
            len(harness_prompt),
            len(agent_state.harness.playbook.entries),
        )

        # Pick free ports for this iteration to avoid collisions
        green_port, purple_port = self._find_free_ports()

        # Generate scenario pointing to the purple agent server with harness file
        scenario = self._generate_scenario(str_ids, str(harness_file), green_port, purple_port)
        scenario_path = iter_dir / "scenario.toml"
        scenario_path.write_text(scenario)
        results_path = iter_dir / "results.json"

        # Build env with API credentials for purple agent
        env = dict(os.environ)
        if self._api_base:
            env["OPENAI_API_BASE"] = self._api_base
        if self._api_key:
            env["OPENAI_API_KEY"] = self._api_key
        # Remove GOOGLE_API_KEY — litellm prefers it over GEMINI_API_KEY,
        # and it's often a free-tier key from the system environment.
        env.pop("GOOGLE_API_KEY", None)

        # Run agentbeats-run
        try:
            result = subprocess.run(
                [self._agentbeats_cmd, str(scenario_path), "--show-logs",
                 "--output", str(results_path)],
                cwd=str(self._car_bench_path.resolve()),
                capture_output=True, text=True, timeout=600,
                env=env,
            )
            (iter_dir / "green_agent.log").write_text(
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
            if result.returncode != 0:
                log.error(
                    "agentbeats-run exited with code %d. See green_agent.log.",
                    result.returncode,
                )
                self._iteration_count += 1
                return [self._make_failed_episode(tid, "agentbeats_error") for tid in str_ids]
        except subprocess.TimeoutExpired:
            log.error("agentbeats-run timed out")
            self._iteration_count += 1
            return [self._make_failed_episode(tid, "timeout") for tid in str_ids]

        # Parse results
        episodes = self._parse_results(results_path, str_ids)

        # Save harness state for debugging/reproducibility
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

        # agentbeats-run output: {"results": [{"detailed_results_by_split": {...}}]}
        # Unwrap the results array to get detailed results
        detailed = {}
        if "detailed_results_by_split" in raw:
            detailed = raw["detailed_results_by_split"]
        elif "results" in raw and raw["results"]:
            detailed = raw["results"][0].get("detailed_results_by_split", {})

        episodes = []
        for task_type_results in detailed.values():
            for task_result in task_type_results:
                episodes.append(self._map_to_episode(task_result))

        # Check for missing tasks
        found_ids = {ep.task_id for ep in episodes}
        for tid in expected_task_ids:
            if f"car:{tid}" not in found_ids:
                episodes.append(self._make_failed_episode(tid, "missing_result"))

        return episodes

    @staticmethod
    def _find_free_ports() -> tuple[int, int]:
        """Find two free TCP ports for green and purple agents."""
        import socket
        socks = []
        ports = []
        for _ in range(2):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            ports.append(s.getsockname()[1])
            socks.append(s)
        for s in socks:
            s.close()
        return ports[0], ports[1]

    def _generate_scenario(
        self, task_ids: list[str], harness_file: str,
        green_port: int, purple_port: int,
    ) -> str:
        """Generate scenario.toml for this batch."""
        by_type: dict[str, list[str]] = {}
        for tid in task_ids:
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

        gp = green_port
        pp = purple_port
        car_dir = self._car_bench_path.resolve()
        green_server = car_dir / "src" / "green_car_bench_agent" / "server.py"
        # Legacy filename in external car-bench repo (not controlled by us)
        lfx_server = car_dir / "src" / "purple_car_bench_agent" / "lfx_server.py"
        # Derive python from agentbeats_cmd's venv (e.g. .../bin/agentbeats-run → .../bin/python)
        agentbeats_bin = Path(self._agentbeats_cmd).parent
        green_python = agentbeats_bin / "python" if agentbeats_bin.name == "bin" else "python"

        return f"""\
[green_agent]
endpoint = "http://127.0.0.1:{gp}"
cmd = "{green_python} {green_server} --host 127.0.0.1 --port {gp}"

[[participants]]
role = "agent"
endpoint = "http://127.0.0.1:{pp}"
cmd = "{green_python} {lfx_server} --host 127.0.0.1 --port {pp} --agent-llm {self._model} --temperature 0.0 --harness-file {harness_file}"

[config]
task_split = "{self._task_split}"
{filter_block}
num_trials = 1
max_steps = 50
"""

    def _map_to_episode(self, task_result: dict) -> Episode:
        """Map a CAR detailed result to a clawloop Episode."""
        car_task_id = task_result["task_id"]
        task_id = f"car:{car_task_id}"

        signals, breakdown = map_car_scores(
            task_result.get("reward_info", {}),
            task_reward=task_result.get("reward", 0.0),
        )

        # Convert trajectory to clawloop Messages
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
            state_id=getattr(self, "_current_state_id", ""),
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
        from clawloop.core.reward import RewardSignal

        signals = {
            "outcome": RewardSignal(name="outcome", value=-1.0, confidence=0.5)
        }
        return Episode(
            id=uuid4().hex,
            state_id=getattr(self, "_current_state_id", ""),
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
        raise NotImplementedError(
            "list_tasks requires CAR-bench data. Use run_batch with explicit task_ids."
        )
