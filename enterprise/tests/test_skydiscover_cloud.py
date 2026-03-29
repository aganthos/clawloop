"""Tests for CloudAdaEvolve async wrapper."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.evolver import EvolverContext, EvolverResult, HarnessSnapshot, Provenance
from clawloop.core.reward import RewardSignal
from clawloop.layers.harness import Insight
from enterprise.evolution.backends.skydiscover_cloud import (
    CloudAdaEvolve,
    RunStatus,
)

_PATCH_TARGET = (
    "enterprise.evolution.backends.skydiscover_adaevolve"
    ".SkyDiscoverAdaEvolve._get_run_discovery"
)


def _make_snapshot() -> HarnessSnapshot:
    return HarnessSnapshot(
        system_prompts={"default": "You are a helpful agent."},
        playbook_entries=[
            {"id": "e1", "content": "Be safe.", "tags": ["safety"], "helpful": 1, "harmful": 0},
        ],
        pareto_fronts={},
        playbook_generation=1,
        playbook_version=2,
    )


def _make_context() -> EvolverContext:
    return EvolverContext(reward_history=[0.5], iteration=1)


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


def _make_mock_run_discovery(tmp_path: Path, evolved_program: dict | None = None) -> MagicMock:
    """Create a mock run_discovery that returns after a short delay."""
    program = evolved_program or {
        "system_prompt": "You are a helpful agent.",
        "playbook": [
            {"content": "Be safe.", "tags": ["safety"], "helpful": 1, "harmful": 0},
        ],
    }
    evolved_path = tmp_path / "evolved.json"
    evolved_path.write_text(json.dumps(program))

    result = MagicMock()
    result.best_program = str(evolved_path)
    result.version = "0.1.0"
    result.tokens_used = 500

    def slow_discovery(**kwargs: Any) -> MagicMock:
        time.sleep(0.05)  # Simulate some work
        return result

    return MagicMock(side_effect=slow_discovery)


class TestCloudAdaEvolve:
    def test_evolve_returns_run_id_immediately(self, tmp_path: Path) -> None:
        """evolve() should return a run_id without blocking."""
        mock_run = _make_mock_run_discovery(tmp_path)

        adapter = MagicMock()
        adapter.run_episode.return_value = _make_episode()
        factory = MagicMock(return_value=MagicMock())

        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            start = time.time()
            result = cloud.evolve([_make_episode()], _make_snapshot(), _make_context())
            elapsed = time.time() - start

        assert result.run_id.startswith("sky-")
        assert elapsed < 0.03  # Should return near-instantly
        assert result.provenance.backend == "skydiscover_adaevolve_cloud"

        # Wait for background thread to finish
        time.sleep(0.2)

    def test_poll_status_lifecycle(self, tmp_path: Path) -> None:
        """Status should progress through running -> completed."""
        gate = threading.Event()

        def gated_discovery(**kwargs: Any) -> MagicMock:
            gate.wait(timeout=5)
            evolved_path = tmp_path / "evolved.json"
            evolved_path.write_text(json.dumps({
                "system_prompt": "You are a helpful agent.",
                "playbook": [{"content": "Be safe.", "tags": ["safety"], "helpful": 1, "harmful": 0}],
            }))
            result = MagicMock()
            result.best_program = str(evolved_path)
            result.version = "0.1.0"
            result.tokens_used = 100
            return result

        mock_run = MagicMock(side_effect=gated_discovery)

        adapter = MagicMock()
        adapter.run_episode.return_value = _make_episode()
        factory = MagicMock(return_value=MagicMock())

        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            result = cloud.evolve([_make_episode()], _make_snapshot(), _make_context())
            run_id = result.run_id

            # Should be running
            time.sleep(0.05)
            status = cloud.poll_status(run_id)
            assert status["status"] == "running"
            assert status["run_id"] == run_id

            # Release the gate
            gate.set()
            time.sleep(0.2)

            # Should be completed
            status = cloud.poll_status(run_id)
            assert status["status"] == "completed"
            assert status["elapsed_s"] > 0

    def test_get_result_after_completion(self, tmp_path: Path) -> None:
        """get_result() should return the EvolverResult after completion."""
        mock_run = _make_mock_run_discovery(tmp_path)

        adapter = MagicMock()
        adapter.run_episode.return_value = _make_episode()
        factory = MagicMock(return_value=MagicMock())

        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            result = cloud.evolve([_make_episode()], _make_snapshot(), _make_context())
            run_id = result.run_id

            # Not ready yet
            assert cloud.get_result(run_id) is None

            # Wait for completion
            time.sleep(0.3)

            final = cloud.get_result(run_id)
            assert final is not None
            assert isinstance(final, EvolverResult)

    def test_cancel_running_evolution(self, tmp_path: Path) -> None:
        """cancel() should stop a running evolution."""
        gate = threading.Event()

        def slow_discovery(**kwargs: Any) -> MagicMock:
            gate.wait(timeout=5)
            evolved_path = tmp_path / "evolved.json"
            evolved_path.write_text(json.dumps({
                "system_prompt": "test", "playbook": [],
            }))
            result = MagicMock()
            result.best_program = str(evolved_path)
            return result

        mock_run = MagicMock(side_effect=slow_discovery)

        adapter = MagicMock()
        adapter.run_episode.return_value = _make_episode()
        factory = MagicMock(return_value=MagicMock())

        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            result = cloud.evolve([_make_episode()], _make_snapshot(), _make_context())
            run_id = result.run_id

            time.sleep(0.05)
            assert cloud.cancel(run_id) is True

            status = cloud.poll_status(run_id)
            assert status["status"] == "cancelled"

            # Release gate to let thread exit
            gate.set()
            time.sleep(0.1)

    def test_cancel_unknown_run_id(self) -> None:
        adapter = MagicMock()
        factory = MagicMock()
        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )
        assert cloud.cancel("nonexistent") is False

    def test_poll_unknown_run_id(self) -> None:
        adapter = MagicMock()
        factory = MagicMock()
        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )
        status = cloud.poll_status("nonexistent")
        assert status["status"] == "failed"
        assert "Unknown" in status["error"]

    def test_concurrency_limit(self, tmp_path: Path) -> None:
        """Should reject new runs when at max_concurrent."""
        gate = threading.Event()

        def slow_discovery(**kwargs: Any) -> MagicMock:
            gate.wait(timeout=5)
            evolved_path = tmp_path / "evolved.json"
            evolved_path.write_text(json.dumps({
                "system_prompt": "test", "playbook": [],
            }))
            result = MagicMock()
            result.best_program = str(evolved_path)
            return result

        mock_run = MagicMock(side_effect=slow_discovery)

        adapter = MagicMock()
        adapter.run_episode.return_value = _make_episode()
        factory = MagicMock(return_value=MagicMock())

        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
            max_concurrent=1,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            # First run should succeed
            r1 = cloud.evolve([_make_episode()], _make_snapshot(), _make_context())
            assert r1.run_id.startswith("sky-")

            time.sleep(0.05)

            # Second run should be rejected (empty run_id)
            r2 = cloud.evolve([_make_episode()], _make_snapshot(), _make_context())
            assert r2.run_id == ""

            # Release and cleanup
            gate.set()
            time.sleep(0.2)

    def test_name(self) -> None:
        adapter = MagicMock()
        factory = MagicMock()
        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )
        assert cloud.name() == "skydiscover_adaevolve_cloud"

    def test_active_runs(self, tmp_path: Path) -> None:
        gate = threading.Event()

        def slow_discovery(**kwargs: Any) -> MagicMock:
            gate.wait(timeout=5)
            evolved_path = tmp_path / "evolved.json"
            evolved_path.write_text(json.dumps({
                "system_prompt": "test", "playbook": [],
            }))
            result = MagicMock()
            result.best_program = str(evolved_path)
            return result

        mock_run = MagicMock(side_effect=slow_discovery)

        adapter = MagicMock()
        adapter.run_episode.return_value = _make_episode()
        factory = MagicMock(return_value=MagicMock())

        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
            max_concurrent=3,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            cloud.evolve([_make_episode()], _make_snapshot(), _make_context())
            time.sleep(0.05)

            assert len(cloud.active_runs()) == 1

            gate.set()
            time.sleep(0.2)

            assert len(cloud.active_runs()) == 0

    def test_failed_evolution_captured(self, tmp_path: Path) -> None:
        """If run_discovery raises, the run should be marked failed."""
        def exploding_discovery(**kwargs: Any) -> None:
            raise RuntimeError("SkyDiscover crashed!")

        mock_run = MagicMock(side_effect=exploding_discovery)

        adapter = MagicMock()
        adapter.run_episode.return_value = _make_episode()
        factory = MagicMock(return_value=MagicMock())

        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            result = cloud.evolve([_make_episode()], _make_snapshot(), _make_context())
            run_id = result.run_id

            time.sleep(0.3)

            status = cloud.poll_status(run_id)
            assert status["status"] == "failed"
            assert "SkyDiscover crashed" in status["error"]
