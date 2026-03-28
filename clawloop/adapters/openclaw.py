# clawloop/adapters/openclaw.py
"""OpenClaw adapter — runs pi-mono agent tasks via the LLM proxy.

The adapter starts a ProxyApp on an ephemeral port, then spawns a Node runner
per episode.  The runner creates a pi-mono Agent pointing at the proxy.
Every LLM call flows through the proxy which:
  1. Injects playbook skills into the system message
  2. Forwards to the real upstream LLM
  3. Captures the full trace into an EpisodeCollector

This means the Episode returned by run_episode() contains the REAL conversation
as captured by the proxy — not just the runner's stdout summary.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import uvicorn
from pydantic import SecretStr

from clawloop.adapters.base import EnvAdapter
from clawloop.collector import EpisodeCollector
from clawloop.core.episode import Episode, EpisodeSummary, Message
from clawloop.core.reward import RewardPipeline
from clawloop.proxy import ProxyApp
from clawloop.proxy_config import ProxyConfig

if TYPE_CHECKING:
    from clawloop.core.loop import AgentState

log = logging.getLogger(__name__)


class OpenClawAdapter(EnvAdapter):
    """Adapter for OpenClaw / pi-mono agent tasks via proxy + subprocess."""

    def __init__(self) -> None:
        self._task_dir: str = ""
        self._runner_script: str = ""
        self._node_bin: str = "node"
        self._timeout_s: int = 120
        self._skip_proxy: bool = False

        # Proxy infrastructure (created in setup)
        self._proxy: ProxyApp | None = None
        self._proxy_port: int | None = None
        self._proxy_server: uvicorn.Server | None = None
        self._proxy_thread: threading.Thread | None = None
        self._collector: EpisodeCollector | None = None
        self._episodes: list[Episode] = []

    def setup(self, config: dict[str, Any]) -> None:
        self._task_dir = config.get("task_dir", self._task_dir)
        self._runner_script = config.get("runner_script", self._runner_script)
        self._node_bin = config.get("node_bin", self._node_bin)
        self._timeout_s = config.get("timeout_s", self._timeout_s)
        self._skip_proxy = config.get("_skip_proxy", self._skip_proxy)

        if self._skip_proxy:
            return

        # Build proxy config from adapter config
        upstream_url = config.get("upstream_url", "")
        upstream_key = config.get("upstream_api_key", "")
        if not upstream_url or not upstream_key:
            log.warning("No upstream_url/upstream_api_key — proxy disabled")
            self._skip_proxy = True
            return

        # Collector captures episodes from proxy traces
        self._collector = EpisodeCollector(
            pipeline=RewardPipeline.with_defaults(),
            on_batch=lambda eps: self._episodes.extend(eps),
            batch_size=1,
        )

        proxy_config = ProxyConfig(
            upstream_url=upstream_url,
            upstream_api_key=SecretStr(upstream_key),
            bench_mode=True,
            bench=config.get("bench", "openclaw"),
        )

        # Get harness from config if provided (for skill injection)
        harness = config.get("harness")

        self._proxy = ProxyApp(
            proxy_config,
            collector=self._collector,
            harness=harness,
        )

        # Start on ephemeral port
        self._proxy_port = self._find_free_port()
        self._proxy_server = uvicorn.Server(uvicorn.Config(
            self._proxy.asgi_app,
            host="127.0.0.1",
            port=self._proxy_port,
            log_level="warning",
        ))
        self._proxy_thread = threading.Thread(
            target=self._proxy_server.run, daemon=True
        )
        self._proxy_thread.start()

        # Wait for proxy to accept connections
        for _ in range(50):
            try:
                import httpx
                httpx.get(f"http://127.0.0.1:{self._proxy_port}/", timeout=1)
                break
            except Exception:
                time.sleep(0.1)

        log.info("Proxy started on port %d → %s", self._proxy_port, upstream_url)

    def run_episode(self, task: Any, agent_state: Any) -> Episode:
        run_id = uuid4().hex
        task_json = json.dumps(task).encode()

        cmd = [self._node_bin, self._runner_script]
        if not self._skip_proxy and self._proxy_port:
            cmd += [
                "--base-url",
                f"http://127.0.0.1:{self._proxy_port}/v1",
                "--run-id", run_id,
            ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
            stdout, stderr = proc.communicate(
                input=task_json, timeout=self._timeout_s
            )
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            proc.wait()
            log.error("Runner timed out after %ds (run_id=%s)", self._timeout_s, run_id)
            return self._make_failed_episode(task, run_id, "timeout")

        if stderr:
            log.debug("Runner stderr: %s", stderr.decode(errors="replace")[:500])

        if proc.returncode != 0:
            log.error(
                "Runner exited %d (run_id=%s): %s",
                proc.returncode, run_id, stderr.decode(errors="replace")[:500],
            )
            return self._make_failed_episode(task, run_id, "runner_error")

        # Parse runner stdout for status
        try:
            result = json.loads(stdout.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            log.error("Failed to parse runner output (run_id=%s): %s", run_id, e)
            return self._make_failed_episode(task, run_id, "parse_error")

        # Wait a beat for proxy post-processing to complete
        time.sleep(0.5)

        # Try to find the proxy-captured episode (richer than stdout)
        proxy_episode = self._pop_episode_by_session(run_id)
        if proxy_episode is not None:
            # Enrich with task metadata
            proxy_episode.task_id = (
                task.get("task_id", run_id) if isinstance(task, dict) else run_id
            )
            proxy_episode.metadata["runner_status"] = result.get("status", "unknown")
            return proxy_episode

        # Fallback: build from runner stdout (proxy didn't capture)
        log.warning("No proxy episode for run_id=%s, using runner stdout", run_id)
        messages = [
            Message(
                role="user",
                content=task.get("instruction", "") if isinstance(task, dict) else "",
            ),
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
        """Stop the proxy server."""
        if self._proxy_server is not None:
            self._proxy_server.should_exit = True
            log.info("Proxy server stopped")

    # -- Internal helpers --------------------------------------------------

    def _pop_episode_by_session(self, session_id: str) -> Episode | None:
        """Find and remove the episode captured by the proxy for this run."""
        for i, ep in enumerate(self._episodes):
            if ep.session_id == session_id:
                return self._episodes.pop(i)
        return None

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _make_failed_episode(
        self, task: Any, run_id: str, reason: str
    ) -> Episode:
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
