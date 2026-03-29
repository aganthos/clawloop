"""Tests for CloudAdaEvolve async wrapper."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from clawloop.core.evolver import EvolverResult, Provenance
from enterprise.evolution.backends.skydiscover_cloud import (
    CloudAdaEvolve,
    RunStatus,
)
from enterprise.tests.conftest import make_context, make_episode, make_snapshot

_PATCH_TARGET = (
    "enterprise.evolution.backends.skydiscover_adaevolve"
    ".SkyDiscoverAdaEvolve._get_run_discovery"
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
    result.best_solution = json.dumps(program)
    result.best_program = MagicMock()
    result.best_program.solution = json.dumps(program)
    result.best_score = 0.8
    result.metrics = {"total_tokens": 500}
    result.output_dir = str(tmp_path)
    result.initial_score = 0.3

    def slow_discovery(**kwargs: Any) -> MagicMock:
        time.sleep(0.05)  # Simulate some work
        return result

    return MagicMock(side_effect=slow_discovery)


class TestCloudAdaEvolve:
    def test_evolve_returns_run_id_immediately(self, tmp_path: Path) -> None:
        """evolve() should return a run_id without blocking."""
        mock_run = _make_mock_run_discovery(tmp_path)

        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode()
        factory = MagicMock(return_value=MagicMock())

        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            start = time.time()
            result = cloud.evolve([make_episode()], make_snapshot(), make_context())
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
            program = {
                "system_prompt": "You are a helpful agent.",
                "playbook": [{"content": "Be safe.", "tags": ["safety"], "helpful": 1, "harmful": 0}],
            }
            result = MagicMock()
            result.best_solution = json.dumps(program)
            result.best_program = MagicMock()
            result.best_program.solution = json.dumps(program)
            result.best_score = 0.8
            result.metrics = {"total_tokens": 100}
            result.output_dir = str(tmp_path)
            return result

        mock_run = MagicMock(side_effect=gated_discovery)

        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode()
        factory = MagicMock(return_value=MagicMock())

        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            result = cloud.evolve([make_episode()], make_snapshot(), make_context())
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
        adapter.run_episode.return_value = make_episode()
        factory = MagicMock(return_value=MagicMock())

        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            result = cloud.evolve([make_episode()], make_snapshot(), make_context())
            run_id = result.run_id

            # Not ready yet
            assert cloud.get_result(run_id) is None

            # Wait for completion
            time.sleep(0.3)

            final = cloud.get_result(run_id)
            assert final is not None
            assert isinstance(final, EvolverResult)

    def test_cancel_running_evolution(self, tmp_path: Path) -> None:
        """cancel() requests cancellation; thread resolves as cancelled after unblocking."""
        gate = threading.Event()

        def slow_discovery(**kwargs: Any) -> MagicMock:
            gate.wait(timeout=5)
            program = {"system_prompt": "test", "playbook": []}
            result = MagicMock()
            result.best_solution = json.dumps(program)
            result.best_program = MagicMock()
            result.best_program.solution = json.dumps(program)
            result.best_score = 0.5
            result.metrics = {}
            result.output_dir = str(tmp_path)
            return result

        mock_run = MagicMock(side_effect=slow_discovery)

        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode()
        factory = MagicMock(return_value=MagicMock())

        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            result = cloud.evolve([make_episode()], make_snapshot(), make_context())
            run_id = result.run_id

            time.sleep(0.05)
            assert cloud.cancel(run_id) is True

            # While blocked on gate, thread is still running
            status = cloud.poll_status(run_id)
            assert status["status"] == "running"  # honest — still executing

            # Release gate — thread checks cancel event post-evolve
            gate.set()
            time.sleep(0.2)

            # Now it should be cancelled
            status = cloud.poll_status(run_id)
            assert status["status"] == "cancelled"

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
            program = {"system_prompt": "test", "playbook": []}
            result = MagicMock()
            result.best_solution = json.dumps(program)
            result.best_program = MagicMock()
            result.best_program.solution = json.dumps(program)
            result.best_score = 0.5
            result.metrics = {}
            result.output_dir = str(tmp_path)
            return result

        mock_run = MagicMock(side_effect=slow_discovery)

        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode()
        factory = MagicMock(return_value=MagicMock())

        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
            max_concurrent=1,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            # First run should succeed
            r1 = cloud.evolve([make_episode()], make_snapshot(), make_context())
            assert r1.run_id.startswith("sky-")

            time.sleep(0.05)

            # Second run should be rejected (empty run_id)
            r2 = cloud.evolve([make_episode()], make_snapshot(), make_context())
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
            program = {"system_prompt": "test", "playbook": []}
            result = MagicMock()
            result.best_solution = json.dumps(program)
            result.best_program = MagicMock()
            result.best_program.solution = json.dumps(program)
            result.best_score = 0.5
            result.metrics = {}
            result.output_dir = str(tmp_path)
            return result

        mock_run = MagicMock(side_effect=slow_discovery)

        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode()
        factory = MagicMock(return_value=MagicMock())

        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
            max_concurrent=3,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            cloud.evolve([make_episode()], make_snapshot(), make_context())
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
        adapter.run_episode.return_value = make_episode()
        factory = MagicMock(return_value=MagicMock())

        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            result = cloud.evolve([make_episode()], make_snapshot(), make_context())
            run_id = result.run_id

            time.sleep(0.3)

            status = cloud.poll_status(run_id)
            assert status["status"] == "failed"
            assert "SkyDiscover crashed" in status["error"]

    def test_cancel_then_release_resolves_cancelled(self, tmp_path: Path) -> None:
        """cancel() + release gate → thread finishes as cancelled, not completed."""
        gate = threading.Event()

        def blocked_discovery(**kwargs: Any) -> MagicMock:
            gate.wait(timeout=5)
            program = {"system_prompt": "test", "playbook": []}
            result = MagicMock()
            result.best_solution = json.dumps(program)
            result.best_program = MagicMock()
            result.best_program.solution = json.dumps(program)
            result.best_score = 0.5
            result.metrics = {}
            result.output_dir = str(tmp_path)
            return result

        mock_run = MagicMock(side_effect=blocked_discovery)

        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode()
        factory = MagicMock(return_value=MagicMock())

        cloud = CloudAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            result = cloud.evolve([make_episode()], make_snapshot(), make_context())
            run_id = result.run_id

            time.sleep(0.05)
            assert cloud.cancel(run_id) is True

            # Release gate so thread can finish and check cancel event
            gate.set()
            time.sleep(0.3)

            # Thread should resolve as cancelled, not completed
            status = cloud.poll_status(run_id)
            assert status["status"] == "cancelled"
