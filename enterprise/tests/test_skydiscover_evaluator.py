"""Tests for ClawLoopEvaluator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from enterprise.evolution.backends.skydiscover_evaluator import ClawLoopEvaluator
from enterprise.tests.conftest import make_adapter, make_episode, make_factory


class TestClawLoopEvaluator:
    def test_returns_combined_score(self, tmp_path: Path) -> None:
        adapter = make_adapter([0.8, 0.6])
        factory = make_factory()
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
        adapter = make_adapter([0.5, 0.5, 0.5])
        factory = make_factory()
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
        adapter = make_adapter([0.5])
        factory = make_factory()
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
        factory = make_factory()
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
        adapter = make_adapter([-0.5, -1.0])
        factory = make_factory()
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
        factory = make_factory()
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
            make_episode(0.6),
            RuntimeError("timeout"),
            make_episode(0.4),
        ]
        factory = make_factory()
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
        adapter = make_adapter([0.3, 0.7])
        factory = make_factory()
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

    def test_non_json_program_treated_as_prompt(self, tmp_path: Path) -> None:
        """SkyDiscover may mutate JSON into non-JSON — evaluator should cope."""
        adapter = make_adapter([0.5])
        factory = make_factory()
        evaluator = ClawLoopEvaluator(
            adapter=adapter,
            tasks=["task-a"],
            agent_state_factory=factory,
            n_episodes=1,
        )

        # Write non-JSON content (simulating a broken LLM mutation)
        prog_path = tmp_path / "mutated.py"
        prog_path.write_text("You are an expert math tutor. Show all work.")

        result = evaluator(str(prog_path))
        assert result["combined_score"] == pytest.approx(0.5)
        assert result["n_episodes"] == 1
        # Factory should have received the raw text as system_prompt
        factory.assert_called_once_with(
            "You are an expert math tutor. Show all work.",
            [],
        )
