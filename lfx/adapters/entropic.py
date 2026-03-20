# lfx/adapters/entropic.py
"""Entropic CRMArenaPro adapter — orchestrates the green agent with lfx harness.

Starts the green agent (entropic evaluator) as a subprocess, starts an lfx
purple agent with harness injection, sends an EvalRequest, and parses the
7-dimension scores into lfx Episodes.

Architecture follows the same pattern as the CAR-bench adapter:
  green agent (evaluator, subprocess) ↔ purple agent (lfx, in-process)
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

from lfx.adapters._entropic_rewards import DEFAULT_ENTROPIC_WEIGHTS, map_entropic_scores
from lfx.adapters.base import EnvAdapter
from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta

if TYPE_CHECKING:
    from lfx.core.loop import AgentState

log = logging.getLogger(__name__)

# CRMArenaPro task categories
_ALL_CATEGORIES = (
    "knowledge_qa", "lead_qualification", "monthly_trend_analysis",
    "conversion_rate_comprehension", "handle_time",
    "private_customer_information", "internal_operation_data",
    "confidential_company_knowledge",
)

REWARD_METRICS = list(DEFAULT_ENTROPIC_WEIGHTS.keys())


class EntropicAdapter(EnvAdapter):
    """Adapter for Entropic CRMArenaPro. Runs green + purple agents per iteration."""

    def setup(self, config: dict[str, Any]) -> None:
        self._model = config.get("model", "anthropic/claude-haiku-4-5-20251001")
        self._bench_path = Path(
            config.get("entropic_bench_path", "benchmarks/a2a/entropic-crmarenapro")
        )
        self._output_dir = Path(
            config.get("output", f"./runs/entropic/{int(time.time())}")
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._task_categories = config.get("task_categories")
        self._task_limit = config.get("task_limit")
        self._api_base = config.get("api_base")
        self._api_key = config.get("api_key")
        self._green_timeout = config.get("green_timeout", 600)
        self._iteration_count = 0
        self._config = config

    def run_episode(self, task: Any, agent_state: "AgentState") -> Episode:
        episodes = self.run_batch(agent_state, [task])
        return episodes[0] if episodes else self._make_failed_episode(str(task), "empty")

    def run_batch(
        self, agent_state: "AgentState", task_ids: list[Any]
    ) -> list[Episode]:
        """Run a batch of tasks via the entropic green agent."""
        str_ids = [str(tid) for tid in task_ids]
        self._current_state_id = agent_state.state_id().combined_hash

        iter_dir = (self._output_dir / f"iter_{self._iteration_count}").resolve()
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Write harness prompt for the purple agent
        harness_prompt = agent_state.harness.system_prompt("entropic")
        harness_file = iter_dir / "harness_prompt.json"
        harness_file.write_text(json.dumps({"prompt": harness_prompt}))
        log.info(
            "Harness prompt: %d chars, %d playbook entries",
            len(harness_prompt),
            len(agent_state.harness.playbook.entries),
        )

        # Pick free ports
        green_port, purple_port = self._find_free_ports()

        # Build eval config for the green agent
        eval_config = self._build_eval_config(str_ids, purple_port)
        eval_config_path = iter_dir / "eval_config.json"
        eval_config_path.write_text(json.dumps(eval_config, indent=2))
        results_path = iter_dir / "results.json"

        # Build env with API credentials
        env = dict(os.environ)
        if self._api_base:
            env["OPENAI_API_BASE"] = self._api_base
        if self._api_key:
            env["OPENAI_API_KEY"] = self._api_key
        env.pop("GOOGLE_API_KEY", None)

        # Resolve green agent start command
        bench_dir = self._bench_path.resolve()
        green_python = self._resolve_python(bench_dir)
        green_server = bench_dir / "src" / "server.py"

        # Start green agent as subprocess
        try:
            result = subprocess.run(
                [
                    str(green_python), str(green_server),
                    "--host", "127.0.0.1",
                    "--port", str(green_port),
                    "--eval-config", str(eval_config_path),
                    "--purple-url", f"http://127.0.0.1:{purple_port}",
                    "--harness-file", str(harness_file),
                    "--model", self._model,
                    "--output", str(results_path),
                ],
                cwd=str(bench_dir),
                capture_output=True, text=True,
                timeout=self._green_timeout,
                env=env,
            )
            (iter_dir / "green_agent.log").write_text(
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
            if result.returncode != 0:
                log.error(
                    "Entropic green agent exited with code %d. See green_agent.log.",
                    result.returncode,
                )
                self._iteration_count += 1
                return [self._make_failed_episode(tid, "green_error") for tid in str_ids]
        except subprocess.TimeoutExpired:
            log.error("Entropic green agent timed out")
            self._iteration_count += 1
            return [self._make_failed_episode(tid, "timeout") for tid in str_ids]

        # Parse results
        episodes = self._parse_results(results_path, str_ids)

        # Save harness state
        harness_path = iter_dir / "harness_state.json"
        harness_path.write_text(json.dumps(agent_state.harness.to_dict(), indent=2))

        self._iteration_count += 1
        return episodes

    def _build_eval_config(
        self, task_ids: list[str], purple_port: int
    ) -> dict[str, Any]:
        """Build the eval config dict sent to the green agent."""
        config: dict[str, Any] = {
            "participants": {
                "agent": f"http://127.0.0.1:{purple_port}",
            },
            "task_ids": task_ids,
            "org_type": "b2b",
            "drift_level": "medium",
            "rot_level": "medium",
            "max_steps": 10,
            "timeout": 300,
        }
        if self._task_categories:
            config["task_categories"] = self._task_categories
        if self._task_limit:
            config["task_limit"] = self._task_limit
        return config

    def _parse_results(
        self, results_path: Path, expected_task_ids: list[str]
    ) -> list[Episode]:
        """Parse results JSON into Episodes."""
        try:
            raw = json.loads(results_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log.error("Failed to parse entropic results: %s", e)
            return [
                self._make_failed_episode(tid, "parse_error")
                for tid in expected_task_ids
            ]

        # Handle multiple result formats:
        # 1. {"tasks": [{task_id, scores, ...}]}
        # 2. {"results": [{"tasks": [...]}]}
        task_results = []
        if "tasks" in raw:
            task_results = raw["tasks"]
        elif "results" in raw and raw["results"]:
            first = raw["results"][0] if isinstance(raw["results"], list) else raw["results"]
            task_results = first.get("tasks", [])

        episodes = []
        for task_result in task_results:
            episodes.append(self._map_to_episode(task_result))

        # Fill in missing tasks
        found_ids = {ep.task_id for ep in episodes}
        for tid in expected_task_ids:
            if f"entropic:{tid}" not in found_ids:
                episodes.append(self._make_failed_episode(tid, "missing_result"))

        return episodes

    @staticmethod
    def _find_free_ports() -> tuple[int, int]:
        """Find two free TCP ports."""
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

    @staticmethod
    def _resolve_python(bench_dir: Path) -> Path:
        """Resolve the Python interpreter from the benchmark's venv."""
        venv_python = bench_dir / ".venv" / "bin" / "python"
        if venv_python.exists():
            return venv_python
        return Path("python")

    def _map_to_episode(self, task_result: dict) -> Episode:
        """Map an entropic task result to an lfx Episode."""
        raw_task_id = task_result.get("task_id", "unknown")
        task_id = f"entropic:{raw_task_id}"

        scores = task_result.get("scores", {})
        functional_score = scores.get("functional", 0.0)
        # Functional is 0-100; convert to 0/1 binary for task_reward
        task_reward = 1.0 if functional_score >= 50.0 else 0.0

        signals, breakdown = map_entropic_scores(
            scores, task_reward=task_reward,
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

        total_score = task_result.get("total_score", 0.0)
        summary = EpisodeSummary(
            signals=signals,
            score_breakdown=breakdown,
        )

        return Episode(
            id=uuid4().hex,
            state_id=getattr(self, "_current_state_id", ""),
            task_id=task_id,
            bench="entropic",
            model=self._model,
            messages=messages,
            step_boundaries=[0] if messages else [],
            steps=[StepMeta(t=0, reward=total_score / 100.0,
                            done=True, timing_ms=task_result.get("latency_ms", 0.0))],
            summary=summary,
            created_at=time.time(),
            metadata={
                "entropic_total_score": total_score,
                "entropic_category": task_result.get("category"),
                "entropic_drift_level": task_result.get("drift_level"),
                "entropic_rot_level": task_result.get("rot_level"),
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
            state_id=getattr(self, "_current_state_id", ""),
            task_id=f"entropic:{task_id}",
            bench="entropic",
            model=self._model,
            messages=[],
            step_boundaries=[],
            steps=[],
            summary=EpisodeSummary(signals=signals),
            metadata={"error": reason},
        )

    def get_traces(self, episode: Episode) -> dict[str, Any]:
        return {"bench": "entropic", "episode_id": episode.id}

    def list_tasks(self, split: str = "base") -> list[Any]:
        raise NotImplementedError(
            "list_tasks requires CRMArenaPro data. Use run_batch with explicit task_ids."
        )
