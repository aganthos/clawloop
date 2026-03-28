"""Tests for SkyDiscoverAdaEvolve backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.evolver import EvolverContext, HarnessSnapshot
from clawloop.core.reward import RewardSignal
from enterprise.evolution.backends.skydiscover_adaevolve import SkyDiscoverAdaEvolve

_PATCH_TARGET = (
    "enterprise.evolution.backends.skydiscover_adaevolve"
    ".SkyDiscoverAdaEvolve._get_run_discovery"
)


def _make_snapshot() -> HarnessSnapshot:
    return HarnessSnapshot(
        system_prompts={"default": "You are a helpful agent."},
        playbook_entries=[
            {
                "id": "e1",
                "content": "Always verify inputs.",
                "tags": ["safety"],
                "helpful": 5,
                "harmful": 1,
            },
        ],
        pareto_fronts={},
        playbook_generation=2,
        playbook_version=4,
    )


def _make_context() -> EvolverContext:
    return EvolverContext(
        reward_history=[0.3, 0.5, 0.4],
        is_stagnating=False,
        iteration=3,
    )


def _make_episode(reward: float = 0.5) -> Episode:
    return Episode(
        id="ep-test",
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


def _make_discovery_result(evolved_program_path: str) -> MagicMock:
    """Create a mock SkyDiscover result pointing at the evolved program."""
    result = MagicMock()
    result.best_program = evolved_program_path
    result.version = "0.1.0"
    result.tokens_used = 1500
    return result


class TestSkyDiscoverAdaEvolve:
    def test_evolve_returns_evolver_result(self, tmp_path: Path) -> None:
        """End-to-end: mock run_discovery, verify EvolverResult shape."""
        snap = _make_snapshot()
        ctx = _make_context()
        episodes = [_make_episode()]

        # Write an "evolved" program with a new playbook entry + changed prompt
        evolved = {
            "system_prompt": "You are an expert problem-solver.",
            "playbook": [
                {"content": "Always verify inputs.", "tags": ["safety"], "helpful": 5, "harmful": 1},
                {"content": "Think step by step.", "tags": ["reasoning"], "helpful": 0, "harmful": 0},
            ],
            "model": "",
        }
        evolved_path = tmp_path / "evolved.json"
        evolved_path.write_text(json.dumps(evolved))

        mock_result = _make_discovery_result(str(evolved_path))
        mock_run = MagicMock(return_value=mock_result)

        adapter = MagicMock()
        adapter.run_episode.return_value = _make_episode(0.8)
        factory = MagicMock(return_value=MagicMock())

        backend = SkyDiscoverAdaEvolve(
            adapter=adapter,
            tasks=["task-a"],
            agent_state_factory=factory,
            iterations=5,
            num_islands=1,
            population_size=10,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            result = backend.evolve(episodes, snap, ctx)

        # Verify run_discovery was called with correct params
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["search"] == "adaevolve"
        assert call_kwargs.kwargs["iterations"] == 5
        assert call_kwargs.kwargs["num_islands"] == 1
        assert call_kwargs.kwargs["population_size"] == 10

        # Verify result shape
        assert result.provenance.backend == "skydiscover_adaevolve"
        assert result.provenance.version == "0.1.0"
        assert result.provenance.tokens_used == 1500

        # Should have 1 add insight (new entry) + 1 prompt candidate
        add_insights = [i for i in result.insights if i.action == "add"]
        assert len(add_insights) == 1
        assert add_insights[0].content == "Think step by step."

        assert "default" in result.candidates
        assert result.candidates["default"][0].text == "You are an expert problem-solver."

    def test_evolve_no_changes(self, tmp_path: Path) -> None:
        """When evolution produces no changes, result should be empty."""
        snap = _make_snapshot()

        # Evolved program is identical to seed
        evolved = {
            "system_prompt": "You are a helpful agent.",
            "playbook": [
                {"content": "Always verify inputs.", "tags": ["safety"], "helpful": 5, "harmful": 1},
            ],
            "model": "",
        }
        evolved_path = tmp_path / "same.json"
        evolved_path.write_text(json.dumps(evolved))

        mock_result = _make_discovery_result(str(evolved_path))
        mock_run = MagicMock(return_value=mock_result)

        adapter = MagicMock()
        adapter.run_episode.return_value = _make_episode()
        factory = MagicMock(return_value=MagicMock())

        backend = SkyDiscoverAdaEvolve(
            adapter=adapter,
            tasks=["task-a"],
            agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            result = backend.evolve([_make_episode()], snap, _make_context())

        assert result.insights == []
        assert result.candidates == {}

    def test_name(self) -> None:
        adapter = MagicMock()
        factory = MagicMock()
        backend = SkyDiscoverAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )
        assert backend.name() == "skydiscover_adaevolve"

    def test_seed_program_written_to_work_dir(self, tmp_path: Path) -> None:
        """Verify that the seed program is written before calling run_discovery."""
        snap = _make_snapshot()

        evolved_path = tmp_path / "evolved.json"
        evolved_path.write_text(json.dumps({
            "system_prompt": "You are a helpful agent.",
            "playbook": [
                {"content": "Always verify inputs.", "tags": ["safety"], "helpful": 5, "harmful": 1},
            ],
        }))

        mock_result = _make_discovery_result(str(evolved_path))
        seed_paths: list[str] = []

        def capture_seed(**kwargs: Any) -> MagicMock:
            seed_paths.append(kwargs["initial_program"])
            return mock_result

        mock_run = MagicMock(side_effect=capture_seed)

        adapter = MagicMock()
        adapter.run_episode.return_value = _make_episode()
        factory = MagicMock(return_value=MagicMock())

        backend = SkyDiscoverAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            backend.evolve([_make_episode()], snap, _make_context())

        assert len(seed_paths) == 1
        seed_data = json.loads(Path(seed_paths[0]).read_text())
        assert seed_data["system_prompt"] == "You are a helpful agent."
