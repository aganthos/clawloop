"""Integration tests — learning_loop with a JsonlArchiveStore captures data."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from clawloop.archive.jsonl_store import JsonlArchiveStore
from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.loop import AgentState, learning_loop
from clawloop.learning_layers.harness import Harness
from clawloop.learning_layers.router import Router
from clawloop.learning_layers.weights import Weights


class FakeAdapter:
    def run_episode(self, task, agent_state):  # type: ignore[no-untyped-def]
        return Episode(
            id=Episode.new_id(),
            state_id="test-state",
            task_id=str(task),
            bench="test",
            messages=[Message(role="user", content="hello")],
            step_boundaries=[0],
            steps=[StepMeta(t=0, reward=0.8, done=True, timing_ms=100.0)],
            summary=EpisodeSummary(total_reward=0.8),
            model="gpt-4",
            created_at=time.time(),
        )


class TestLearningLoopArchive:
    def test_archive_captures_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_dir = Path(tmpdir) / "arch"
            archive = JsonlArchiveStore(archive_dir=archive_dir)
            agent_state = AgentState(
                harness=Harness(system_prompts={"test": "You are helpful."}),
                router=Router(),
                weights=Weights(),
            )
            adapter = FakeAdapter()

            learning_loop(
                adapter,
                agent_state,
                ["task-1", "task-2"],
                n_episodes=2,
                n_iterations=2,
                output_dir=tmpdir,
                archive=archive,
            )

            runs_file = archive_dir / "runs.jsonl"
            assert runs_file.exists()
            run_lines = [json.loads(ln) for ln in runs_file.read_text().splitlines() if ln.strip()]
            starts = [r for r in run_lines if r["record_type"] == "run_start"]
            completes = [r for r in run_lines if r["record_type"] == "run_complete"]
            assert len(starts) == 1
            assert len(completes) == 1

            iterations_file = archive_dir / "iterations.jsonl"
            iter_lines = [json.loads(ln) for ln in iterations_file.read_text().splitlines() if ln.strip()]
            assert len(iter_lines) == 2
            assert {r["iteration_num"] for r in iter_lines} == {0, 1}

            run_id = starts[0]["run_id"]
            episodes_file = archive_dir / run_id / "episodes.jsonl"
            ep_lines = [json.loads(ln) for ln in episodes_file.read_text().splitlines() if ln.strip()]
            assert len(ep_lines) == 4  # 2 iterations * 2 episodes

            got = archive.get_run(run_id)
            assert got is not None
            assert got.completed_at is not None

    def test_archive_none_is_safe(self) -> None:
        agent_state = AgentState(
            harness=Harness(system_prompts={"test": "You are helpful."}),
            router=Router(),
            weights=Weights(),
        )
        learning_loop(
            FakeAdapter(),
            agent_state,
            ["task-1"],
            n_episodes=1,
            n_iterations=1,
            archive=None,
        )
