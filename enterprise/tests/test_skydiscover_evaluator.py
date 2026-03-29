"""Tests for ClawLoopEvaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta, TokenUsage
from clawloop.core.reward import RewardSignal
from enterprise.evolution.backends.skydiscover_evaluator import ClawLoopEvaluator


def _make_episode(reward: float) -> Episode:
    """Create a minimal Episode with a given outcome reward."""
    return Episode(
        id=f"ep-test",
        state_id="s-test",
        task_id="t-1",
        bench="test",
        messages=[Message(role="user", content="hello")],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
        summary=EpisodeSummary(
            signals={"outcome": RewardSignal(name="outcome", value=reward, confidence=1.0)},
        ),
    )


def _make_adapter(rewards: list[float]) -> MagicMock:
    """Create a mock adapter that returns episodes with given rewards."""
    adapter = MagicMock()
    episodes = [_make_episode(r) for r in rewards]
    adapter.run_episode.side_effect = episodes
    return adapter


def _make_factory() -> MagicMock:
    """Create a mock AgentStateFactory."""
    return MagicMock(return_value=MagicMock())


class TestClawLoopEvaluator:
    def test_returns_combined_score(self, tmp_path: Path) -> None:
        adapter = _make_adapter([0.8, 0.6])
        factory = _make_factory()
        evaluator = ClawLoopEvaluator(
            adapter=adapter,
            tasks=["task-a", "task-b"],
            agent_state_factory=factory,
            n_episodes=2,
        )

        program = {"system_prompt": "test", "playbook": []}
        prog_path = tmp_path / "prog.json"
        prog_path.write_text(json.dumps(program))

        result = evaluator(str(prog_path))
        assert result["combined_score"] == pytest.approx(0.7)
        assert result["n_episodes"] == 2

    def test_cycles_through_tasks(self, tmp_path: Path) -> None:
        adapter = _make_adapter([0.5, 0.5, 0.5])
        factory = _make_factory()
        evaluator = ClawLoopEvaluator(
            adapter=adapter,
            tasks=["task-a"],
            agent_state_factory=factory,
            n_episodes=3,
        )

        program = {"system_prompt": "test", "playbook": []}
        prog_path = tmp_path / "prog.json"
        prog_path.write_text(json.dumps(program))

        evaluator(str(prog_path))
        # All 3 calls should use "task-a" (cycling)
        calls = adapter.run_episode.call_args_list
        assert len(calls) == 3
        assert all(c.args[0] == "task-a" for c in calls)

    def test_passes_config_to_factory(self, tmp_path: Path) -> None:
        adapter = _make_adapter([0.5])
        factory = _make_factory()
        evaluator = ClawLoopEvaluator(
            adapter=adapter,
            tasks=["task-a"],
            agent_state_factory=factory,
            n_episodes=1,
        )

        program = {
            "system_prompt": "You are helpful.",
            "playbook": [{"content": "Be safe", "tags": ["safety"]}],
        }
        prog_path = tmp_path / "prog.json"
        prog_path.write_text(json.dumps(program))

        evaluator(str(prog_path))
        factory.assert_called_once_with(
            "You are helpful.",
            [{"content": "Be safe", "tags": ["safety"]}],
        )

    def test_handles_all_failures(self, tmp_path: Path) -> None:
        adapter = MagicMock()
        adapter.run_episode.side_effect = RuntimeError("boom")
        factory = _make_factory()
        evaluator = ClawLoopEvaluator(
            adapter=adapter,
            tasks=["task-a"],
            agent_state_factory=factory,
            n_episodes=2,
        )

        program = {"system_prompt": "test", "playbook": []}
        prog_path = tmp_path / "prog.json"
        prog_path.write_text(json.dumps(program))

        result = evaluator(str(prog_path))
        assert result["combined_score"] == -1.0
        assert result["n_episodes"] == 0

    def test_negative_rewards(self, tmp_path: Path) -> None:
        adapter = _make_adapter([-0.5, -1.0])
        factory = _make_factory()
        evaluator = ClawLoopEvaluator(
            adapter=adapter,
            tasks=["task-a"],
            agent_state_factory=factory,
            n_episodes=2,
        )

        program = {"system_prompt": "test", "playbook": []}
        prog_path = tmp_path / "prog.json"
        prog_path.write_text(json.dumps(program))

        result = evaluator(str(prog_path))
        assert result["combined_score"] == pytest.approx(-0.75)

    def test_empty_tasks_returns_failure(self, tmp_path: Path) -> None:
        """Empty task list should not crash (was ZeroDivisionError)."""
        adapter = MagicMock()
        factory = _make_factory()
        evaluator = ClawLoopEvaluator(
            adapter=adapter,
            tasks=[],
            agent_state_factory=factory,
            n_episodes=3,
        )

        program = {"system_prompt": "test", "playbook": []}
        prog_path = tmp_path / "prog.json"
        prog_path.write_text(json.dumps(program))

        result = evaluator(str(prog_path))
        assert result["combined_score"] == -1.0
        assert result["n_episodes"] == 0
        adapter.run_episode.assert_not_called()

    def test_partial_failures(self, tmp_path: Path) -> None:
        """Some episodes fail, others succeed — score uses only successes."""
        adapter = MagicMock()
        # First call succeeds, second fails, third succeeds
        adapter.run_episode.side_effect = [
            _make_episode(0.6),
            RuntimeError("timeout"),
            _make_episode(0.4),
        ]
        factory = _make_factory()
        evaluator = ClawLoopEvaluator(
            adapter=adapter,
            tasks=["task-a", "task-b", "task-c"],
            agent_state_factory=factory,
            n_episodes=3,
        )

        program = {"system_prompt": "test", "playbook": []}
        prog_path = tmp_path / "prog.json"
        prog_path.write_text(json.dumps(program))

        result = evaluator(str(prog_path))
        assert result["n_episodes"] == 2
        assert result["combined_score"] == pytest.approx(0.5)

    def test_rewards_list_in_result(self, tmp_path: Path) -> None:
        """Result should include individual rewards for analysis."""
        adapter = _make_adapter([0.3, 0.7])
        factory = _make_factory()
        evaluator = ClawLoopEvaluator(
            adapter=adapter,
            tasks=["t1", "t2"],
            agent_state_factory=factory,
            n_episodes=2,
        )

        program = {"system_prompt": "test", "playbook": []}
        prog_path = tmp_path / "prog.json"
        prog_path.write_text(json.dumps(program))

        result = evaluator(str(prog_path))
        assert result["rewards"] == pytest.approx([0.3, 0.7])
