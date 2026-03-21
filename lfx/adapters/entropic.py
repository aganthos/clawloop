# lfx/adapters/entropic.py
"""Entropic CRMArenaPro adapter — orchestrates the green agent with lfx harness.

Starts the entropic green agent as a long-running A2A server, starts an lfx
purple agent with harness injection in a background thread, then uses
``lfx_runner.py`` (inside the benchmark repo) to send the EvalRequest and
collect results.

Architecture follows the same pattern as the CAR-bench adapter:
  green agent (evaluator, subprocess server) ↔ purple agent (lfx, in-process)
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

REWARD_METRICS = list(DEFAULT_ENTROPIC_WEIGHTS.keys())

# Runner script generated into each iteration dir.  Runs with the benchmark's
# venv python (which has a2a-sdk installed) and speaks A2A to the green server.
_RUNNER_SCRIPT = '''\
"""Auto-generated LfX runner — sends EvalRequest to the entropic green agent."""
import asyncio, json, sys, argparse, logging
from pathlib import Path
from uuid import uuid4

# Ensure benchmark src/ is importable
_src = Path(__file__).resolve().parents[0]
for candidate in [_src / "src", _src.parent / "src"]:
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import DataPart, Message, Part, Role, TextPart

logger = logging.getLogger(__name__)

def _create_message(text):
    return Message(kind="message", role=Role.user,
                   parts=[Part(root=TextPart(kind="text", text=text))],
                   message_id=uuid4().hex)

async def send_eval_request(green_url, eval_json, timeout=600):
    async with httpx.AsyncClient(timeout=timeout) as hc:
        resolver = A2ACardResolver(httpx_client=hc, base_url=green_url)
        agent_card = await resolver.get_agent_card()
        client = ClientFactory(ClientConfig(httpx_client=hc, streaming=False)).create(agent_card)
        msg = _create_message(eval_json)
        data_parts, text_parts = [], []
        async for event in client.send_message(msg):
            match event:
                case Message() as m:
                    for p in m.parts:
                        (data_parts if isinstance(p.root, DataPart) else text_parts).append(
                            p.root.data if isinstance(p.root, DataPart) else p.root.text)
                case (t, _):
                    if t.status and t.status.message:
                        for p in t.status.message.parts:
                            if isinstance(p.root, TextPart): text_parts.append(p.root.text)
                    if t.artifacts:
                        for a in t.artifacts:
                            for p in a.parts:
                                (data_parts if isinstance(p.root, DataPart) else text_parts).append(
                                    p.root.data if isinstance(p.root, DataPart) else p.root.text)
    if data_parts: return data_parts[-1]
    for t in reversed(text_parts):
        try: return json.loads(t)
        except Exception: continue
    return {"error": "no results", "text": "\\n".join(text_parts)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--green-url", required=True)
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with open(args.eval_config) as f: eval_config = json.load(f)
    logger.info("Sending EvalRequest to %s", args.green_url)
    results = asyncio.run(send_eval_request(args.green_url, json.dumps(eval_config), args.timeout))
    Path(args.output).write_text(json.dumps(results, indent=2))
    s = results.get("summary", results.get("entropic", {}).get("summary", {}))
    if s: logger.info("Done: %d tasks, pass_rate=%.1f%%", s.get("total_tasks", 0), s.get("pass_rate", 0)*100)

if __name__ == "__main__": main()
'''


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
        self._task_ids = config.get("task_ids")
        self._api_base = config.get("api_base")
        self._api_key = config.get("api_key")
        self._green_timeout = config.get("green_timeout", 600)
        self._iteration_count = 0
        self._config = config
        # Purple agent is started once and reused across iterations.
        # Its harness is updated at the start of each run_batch call.
        self._purple_agent = None
        self._purple_port: int | None = None

    def run_episode(self, task: Any, agent_state: "AgentState") -> Episode:
        episodes = self.run_batch(agent_state, [task])
        return episodes[0] if episodes else self._make_failed_episode(str(task), "empty")

    def run_batch(
        self, agent_state: "AgentState", task_ids: list[Any]
    ) -> list[Episode]:
        """Run a batch of tasks via the entropic green agent.

        1. Start the purple agent in a background thread (harness-injected).
        2. Start the green agent as a subprocess server.
        3. Wait for the green agent to be healthy.
        4. Run lfx_runner.py to send EvalRequest and save results.
        5. Parse results into Episodes.
        6. Terminate the green agent.
        """
        # Prefer explicit task_ids from config over CLI-generated ones
        # (CLI generates "base_0" etc. which are meaningless for CRMArenaPro).
        if self._task_ids:
            str_ids = [str(tid) for tid in self._task_ids]
        else:
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

        # Build eval config (EvalRequest JSON for the green agent)
        eval_config = self._build_eval_config(str_ids, purple_port)
        eval_config_path = iter_dir / "eval_config.json"
        eval_config_path.write_text(json.dumps(eval_config, indent=2))
        results_path = iter_dir / "results.json"

        # Build env with API credentials for both green + purple agents
        env = dict(os.environ)
        if self._api_base:
            env["OPENAI_API_BASE"] = self._api_base
            env["OPENAI_BASE_URL"] = self._api_base
        if self._api_key:
            env["OPENAI_API_KEY"] = self._api_key
        env.pop("GOOGLE_API_KEY", None)

        bench_dir = self._bench_path.resolve()
        green_python = str(self._resolve_python(bench_dir))

        green_proc = None
        try:
            # --- Step 0: Start or reuse purple agent (harness-injected) ---
            from lfx.adapters._entropic_purple import (
                EntropicPurpleAgent,
                start_purple_server,
            )

            if self._purple_agent is None:
                self._purple_agent = EntropicPurpleAgent(
                    model=self._model,
                    harness=agent_state.harness,
                    bench="entropic",
                    api_base=self._api_base,
                    api_key=self._api_key,
                )
                _thread, self._purple_port = start_purple_server(
                    self._purple_agent, host="127.0.0.1", port=purple_port,
                )
                log.info("Purple agent started (port=%d)", self._purple_port)
            else:
                # Reuse existing server — update harness and clear sessions
                self._purple_agent.update_harness(agent_state.harness)
                self._purple_agent.clear_all_sessions()
                log.info("Purple agent reused (port=%d)", self._purple_port)

            purple_port = self._purple_port
            # Rebuild eval config with actual purple port
            eval_config = self._build_eval_config(str_ids, purple_port)
            eval_config_path.write_text(json.dumps(eval_config, indent=2))

            # --- Step 1: Start green agent server ---
            green_proc = subprocess.Popen(
                [
                    green_python, str(bench_dir / "src" / "server.py"),
                    "--host", "127.0.0.1",
                    "--port", str(green_port),
                ],
                cwd=str(bench_dir),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=env,
            )
            log.info("Green agent started (pid=%d, port=%d)", green_proc.pid, green_port)

            # --- Step 2: Wait for green agent health ---
            green_url = f"http://127.0.0.1:{green_port}"
            if not self._wait_for_health(green_url, timeout=30):
                # Terminate first to unblock pipe reads
                green_proc.terminate()
                try:
                    green_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    green_proc.kill()
                stdout = green_proc.stdout.read().decode() if green_proc.stdout else ""
                stderr = green_proc.stderr.read().decode() if green_proc.stderr else ""
                (iter_dir / "green_agent.log").write_text(
                    f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                )
                green_proc = None  # Already cleaned up
                log.error("Green agent failed to start. See green_agent.log.")
                self._iteration_count += 1
                return [self._make_failed_episode(tid, "green_start_failed") for tid in str_ids]

            # --- Step 3: Run lfx_runner.py to send EvalRequest ---
            runner = iter_dir / "lfx_runner.py"
            runner.write_text(_RUNNER_SCRIPT)
            try:
                result = subprocess.run(
                    [
                        green_python, str(runner),
                        "--green-url", green_url,
                        "--eval-config", str(eval_config_path),
                        "--output", str(results_path),
                        "--timeout", str(self._green_timeout),
                    ],
                    cwd=str(bench_dir),
                    capture_output=True, text=True,
                    timeout=self._green_timeout + 30,
                    env=env,
                )
                (iter_dir / "runner.log").write_text(
                    f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                )
                if result.returncode != 0:
                    log.error("lfx_runner.py exited with code %d", result.returncode)
                    self._iteration_count += 1
                    return [self._make_failed_episode(tid, "runner_error") for tid in str_ids]
            except subprocess.TimeoutExpired:
                log.error("lfx_runner.py timed out")
                self._iteration_count += 1
                return [self._make_failed_episode(tid, "timeout") for tid in str_ids]

        finally:
            # --- Cleanup: kill green agent ---
            if green_proc is not None:
                try:
                    green_proc.terminate()
                    green_proc.wait(timeout=5)
                except Exception:
                    green_proc.kill()
                # Save logs
                stdout = green_proc.stdout.read().decode() if green_proc.stdout else ""
                stderr = green_proc.stderr.read().decode() if green_proc.stderr else ""
                log_path = iter_dir / "green_agent.log"
                existing = log_path.read_text() if log_path.exists() else ""
                log_path.write_text(existing + f"\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

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
        """Build the EvalRequest dict for the green agent.

        The CLI generates synthetic task IDs (``base_0``, ``base_1``, …) that
        don't correspond to CRMArenaPro indices.  We use ``task_limit`` to tell
        the green agent how many tasks to sample, unless explicit integer task
        IDs are provided.
        """
        cfg: dict[str, Any] = {
            "skip_original": True,
        }

        # Prefer explicit task_ids from config, then check if CLI-generated
        # IDs are real CRMArenaPro indices (integers).
        if self._task_ids:
            cfg["task_ids"] = [str(tid) for tid in self._task_ids]
        else:
            real_ids = [tid for tid in task_ids if tid.isdigit()]
            if real_ids and len(real_ids) == len(task_ids):
                cfg["task_ids"] = real_ids
            else:
                cfg["task_limit"] = len(task_ids)

        if self._task_categories:
            cfg["task_categories"] = self._task_categories
        if self._task_limit:
            cfg["task_limit"] = self._task_limit

        return {
            "participants": {
                "agent": f"http://127.0.0.1:{purple_port}",
            },
            "config": cfg,
        }

    def _parse_results(
        self, results_path: Path, expected_task_ids: list[str]
    ) -> list[Episode]:
        """Parse results JSON into Episodes.

        The green agent returns aggregated results with per-task entries in
        ``results`` containing ``task_idx``, ``crm_reward``, ``total_score``,
        and ``dimension_scores``.
        """
        try:
            raw = json.loads(results_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log.error("Failed to parse entropic results: %s", e)
            return [
                self._make_failed_episode(tid, "parse_error")
                for tid in expected_task_ids
            ]

        # The artifact data may be:
        #   {"results": [{task_idx, ...}, ...]}
        #   {"entropic": {"results": [...]}, "results": [...]}
        # Prefer the top-level "results" if it's a list, else unwrap "entropic".
        task_results = raw.get("results", [])
        if not isinstance(task_results, list):
            task_results = []
        if not task_results and "entropic" in raw:
            nested = raw["entropic"]
            if isinstance(nested, dict):
                task_results = nested.get("results", [])

        episodes = []
        for task_result in task_results:
            episodes.append(self._map_to_episode(task_result))

        # Fill in missing tasks — but only when expected IDs are real
        # CRMArenaPro indices (digits).  Synthetic CLI IDs (``base_0``, etc.)
        # would always appear missing because the green agent returns numeric
        # task_idx values.
        found_ids = {ep.task_id for ep in episodes}
        for tid in expected_task_ids:
            if not tid.isdigit():
                continue  # skip synthetic IDs
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

    @staticmethod
    def _wait_for_health(url: str, timeout: int = 30) -> bool:
        """Poll the agent card endpoint until healthy or timeout."""
        import httpx
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = httpx.get(f"{url}/.well-known/agent.json", timeout=2)
                if r.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            time.sleep(0.5)
        return False

    def _map_to_episode(self, task_result: dict) -> Episode:
        """Map an entropic task result to an lfx Episode."""
        raw_task_id = str(task_result.get("task_idx", "unknown"))
        task_id = f"entropic:{raw_task_id}"

        # dimension_scores is {dim_name: score_value} on 0-100 scale
        dim_scores = task_result.get("dimension_scores", {})
        # Normalise keys to lowercase for reward mapping
        scores = {k.lower(): v for k, v in dim_scores.items() if isinstance(v, (int, float))}

        crm_reward = task_result.get("crm_reward", 0)
        task_reward = 1.0 if crm_reward > 0 else 0.0

        signals, breakdown = map_entropic_scores(scores, task_reward=task_reward)

        total_score = task_result.get("total_score", 0.0)
        summary = EpisodeSummary(signals=signals, score_breakdown=breakdown)

        # Build a single message from the task query + agent answer
        messages = []
        query = task_result.get("task_query", "")
        answer = task_result.get("agent_answer", "")
        if query:
            messages.append(Message(role="user", content=query))
        if answer:
            messages.append(Message(role="assistant", content=answer))

        timing = task_result.get("timing", {})

        return Episode(
            id=uuid4().hex,
            state_id=getattr(self, "_current_state_id", ""),
            task_id=task_id,
            bench="entropic",
            model=self._model,
            messages=messages,
            step_boundaries=[0] if messages else [],
            steps=[StepMeta(
                t=0, reward=total_score / 100.0, done=True,
                timing_ms=timing.get("total_seconds", 0.0) * 1000,
            )],
            summary=summary,
            created_at=time.time(),
            metadata={
                "entropic_total_score": total_score,
                "entropic_category": task_result.get("task_category"),
                "entropic_crm_reward": crm_reward,
                "entropic_success": task_result.get("success"),
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
