# clawloop/adapters/openclaw.py
"""OpenClaw adapter — runs pi-mono agent tasks via subprocess.

Spawns a runner script (typically Node.js) per episode, feeding the task JSON
on stdin and reading the result JSON from stdout.  Designed for OpenClaw /
pi-mono benchmarks where the agent is an external process.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from clawloop.adapters.base import EnvAdapter
from clawloop.core.episode import Episode, EpisodeSummary, Message

if TYPE_CHECKING:
    from clawloop.core.loop import AgentState

log = logging.getLogger(__name__)


class OpenClawAdapter(EnvAdapter):
    """Adapter for OpenClaw / pi-mono agent tasks via subprocess runner."""

    def __init__(self) -> None:
        self._task_dir: str = ""
        self._runner_script: str = ""
        self._node_bin: str = "node"
        self._timeout_s: int = 120
        self._proxy_port: int = 8080
        self._skip_proxy: bool = False

    def setup(self, config: dict[str, Any]) -> None:
        self._task_dir = config.get("task_dir", self._task_dir)
        self._runner_script = config.get("runner_script", self._runner_script)
        self._node_bin = config.get("node_bin", self._node_bin)
        self._timeout_s = config.get("timeout_s", self._timeout_s)
        self._proxy_port = config.get("proxy_port", self._proxy_port)
        self._skip_proxy = config.get("_skip_proxy", self._skip_proxy)

    def run_episode(self, task: Any, agent_state: AgentState) -> Episode:
        run_id = uuid4().hex
        task_json = json.dumps(task).encode()

        cmd = [self._node_bin, self._runner_script]
        if not self._skip_proxy:
            cmd += ["--base-url", f"http://127.0.0.1:{self._proxy_port}"]
        cmd += ["--run-id", run_id]

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
            stdout, stderr = proc.communicate(input=task_json, timeout=self._timeout_s)
        except subprocess.TimeoutExpired:
            # Kill the entire process group
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            proc.wait()
            log.error("Runner timed out after %ds (run_id=%s)", self._timeout_s, run_id)
            return self._make_failed_episode(task, run_id, "timeout")

        if proc.returncode != 0:
            log.error(
                "Runner exited %d (run_id=%s): %s",
                proc.returncode, run_id, stderr.decode(errors="replace")[:500],
            )
            return self._make_failed_episode(task, run_id, "runner_error")

        try:
            result = json.loads(stdout.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            log.error("Failed to parse runner output (run_id=%s): %s", run_id, e)
            return self._make_failed_episode(task, run_id, "parse_error")

        messages = [
            Message(role="user", content=task.get("instruction", "") if isinstance(task, dict) else ""),
            Message(role="assistant", content=result.get("output", "")),
        ]
        return Episode(
            id=Episode.new_id(),
            state_id="",
            task_id=task.get("task_id", run_id) if isinstance(task, dict) else run_id,
            bench="openclaw",
            messages=messages,
            step_boundaries=[0, len(messages)],
            steps=[],
            summary=EpisodeSummary(),
            session_id=run_id,
            metadata={"runner_status": result.get("status", "unknown")},
        )

    def list_tasks(self, split: str = "base") -> list[Any]:
        task_file = Path(self._task_dir) / f"{split}.jsonl"
        if not task_file.exists():
            return []
        tasks = []
        for line in task_file.read_text().splitlines():
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
        return tasks

    def get_traces(self, episode: Episode) -> dict[str, Any]:
        return {"session_id": episode.session_id}

    def teardown(self) -> None:
        """Placeholder — will be wired to stop proxy later."""

    # -- Internal helpers --------------------------------------------------

    def _make_failed_episode(
        self, task: Any, run_id: str, reason: str
    ) -> Episode:
        """Create a failed episode with error metadata."""
        task_id = task.get("task_id", run_id) if isinstance(task, dict) else run_id
        instruction = task.get("instruction", "") if isinstance(task, dict) else ""
        return Episode(
            id=Episode.new_id(),
            state_id="",
            task_id=task_id,
            bench="openclaw",
            messages=[Message(role="user", content=instruction)],
            step_boundaries=[0],
            steps=[],
            summary=EpisodeSummary(),
            session_id=run_id,
            metadata={"error": reason},
        )
