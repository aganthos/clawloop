# tests/test_openclaw_adapter.py
"""Tests for OpenClawAdapter — subprocess-based adapter for pi-mono agent tasks."""

import json
import sys
import textwrap
from pathlib import Path

import pytest

from clawloop.adapters.openclaw import OpenClawAdapter
from clawloop.core.loop import AgentState


class TestListTasks:
    """OpenClawAdapter.list_tasks reads JSONL task files."""

    def test_reads_jsonl(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        lines = [
            json.dumps({"task_id": "t1", "instruction": "Do thing one"}),
            json.dumps({"task_id": "t2", "instruction": "Do thing two"}),
        ]
        (tasks_dir / "base.jsonl").write_text("\n".join(lines) + "\n")

        adapter = OpenClawAdapter()
        adapter.setup({"task_dir": str(tasks_dir), "_skip_proxy": True})

        tasks = adapter.list_tasks("base")
        assert len(tasks) == 2
        assert tasks[0]["task_id"] == "t1"
        assert tasks[1]["instruction"] == "Do thing two"

    def test_missing_split_returns_empty(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        adapter = OpenClawAdapter()
        adapter.setup({"task_dir": str(tasks_dir), "_skip_proxy": True})

        tasks = adapter.list_tasks("nonexistent")
        assert tasks == []


class TestRunEpisode:
    """OpenClawAdapter.run_episode spawns a subprocess and returns an Episode."""

    def _make_runner_script(self, tmp_path: Path) -> Path:
        """Create a mock runner script that reads stdin JSON, writes stdout JSON."""
        script = tmp_path / "mock_runner.py"
        script.write_text(textwrap.dedent("""\
            import json, sys
            task = json.loads(sys.stdin.read())
            result = {
                "task_id": task.get("task_id", "unknown"),
                "status": "success",
                "output": f"Completed: {task.get('instruction', '')}",
            }
            print(json.dumps(result))
        """))
        return script

    def _make_timeout_script(self, tmp_path: Path) -> Path:
        """Create a script that sleeps forever (for timeout testing)."""
        script = tmp_path / "slow_runner.py"
        script.write_text(textwrap.dedent("""\
            import time, sys
            sys.stdin.read()
            time.sleep(999)
        """))
        return script

    def test_runs_subprocess_and_returns_episode(self, tmp_path):
        runner = self._make_runner_script(tmp_path)

        adapter = OpenClawAdapter()
        adapter.setup({
            "runner_script": str(runner),
            "node_bin": sys.executable,
            "timeout_s": 10,
            "_skip_proxy": True,
        })

        task = {"task_id": "abc", "instruction": "Say hello"}
        episode = adapter.run_episode(task, AgentState())

        assert episode.bench == "openclaw"
        assert episode.task_id == "abc"
        assert len(episode.messages) == 2
        assert episode.messages[0].role == "user"
        assert episode.messages[0].content == "Say hello"
        assert episode.messages[1].role == "assistant"
        assert "Completed: Say hello" in episode.messages[1].content
        assert episode.metadata["runner_status"] == "success"
        assert episode.session_id  # non-empty run_id

    def test_timeout_kills_subprocess(self, tmp_path):
        runner = self._make_timeout_script(tmp_path)

        adapter = OpenClawAdapter()
        adapter.setup({
            "runner_script": str(runner),
            "node_bin": sys.executable,
            "timeout_s": 1,
            "_skip_proxy": True,
        })

        task = {"task_id": "slow", "instruction": "Wait forever"}
        # Should not hang — timeout kills the process
        episode = adapter.run_episode(task, AgentState())

        assert episode.bench == "openclaw"
        assert episode.metadata.get("error") == "timeout"
