"""Integration tests — SkyDiscoverAdaEvolve plugged into a real Harness.

Validates the full pipeline:
  Harness(evolver=SkyDiscoverAdaEvolve(...))
      → forward_backward(episodes)
      → evolver.evolve() produces insights + candidates
      → optim_step() applies them to playbook + pareto fronts

Uses mocked SkyDiscover (no real evolution) but real Harness/Layer Protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from clawloop.core.types import Datum
from clawloop.layers.harness import Harness, PlaybookEntry
from enterprise.evolution.backends.skydiscover_adaevolve import SkyDiscoverAdaEvolve
from enterprise.evolution.backends.skydiscover_cloud import CloudAdaEvolve
from enterprise.tests.conftest import make_context, make_episode, make_factory

_PATCH_TARGET = (
    "enterprise.evolution.backends.skydiscover_adaevolve"
    ".SkyDiscoverAdaEvolve._get_run_discovery"
)


def _make_discovery_result_with_new_entry(
    tmp_path: Path,
    original_prompt: str = "Solve math problems.",
    evolved_prompt: str | None = None,
    new_entries: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a mock DiscoveryResult with evolved program content."""
    playbook = []
    if new_entries:
        playbook = new_entries

    program = {
        "system_prompt": evolved_prompt or original_prompt,
        "playbook": playbook,
        "model": "",
    }

    result = MagicMock()
    result.best_solution = json.dumps(program)
    result.best_program = MagicMock()
    result.best_program.solution = json.dumps(program)
    result.best_score = 0.8
    result.metrics = {"total_tokens": 500}
    result.output_dir = str(tmp_path)
    result.initial_score = 0.3
    return result


class TestHarnessWithSkyDiscover:
    """Integration: SkyDiscoverAdaEvolve as the Harness evolver."""

    def test_forward_backward_collects_insights(self, tmp_path: Path) -> None:
        """Harness.forward_backward → evolver.evolve → insights accumulated."""
        mock_result = _make_discovery_result_with_new_entry(
            tmp_path,
            new_entries=[
                {"content": "Always double-check arithmetic.", "tags": ["math"], "helpful": 0, "harmful": 0},
            ],
        )
        mock_run = MagicMock(return_value=mock_result)

        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode(0.7)

        evolver = SkyDiscoverAdaEvolve(
            adapter=adapter,
            tasks=["math-1"],
            agent_state_factory=make_factory(),
            iterations=1,
        )

        h = Harness(
            system_prompts={"math": "Solve math problems."},
            evolver=evolver,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            datum = Datum(episodes=[make_episode(-0.3)])
            fb = h.forward_backward(datum).result()

        assert fb.status == "ok"
        assert fb.metrics["insights_generated"] == 1
        assert fb.metrics.get("backend") == "skydiscover_adaevolve"

    def test_optim_step_applies_evolved_playbook(self, tmp_path: Path) -> None:
        """forward_backward → optim_step → new playbook entry appears."""
        mock_result = _make_discovery_result_with_new_entry(
            tmp_path,
            new_entries=[
                {"content": "Show work step by step.", "tags": ["reasoning"], "helpful": 0, "harmful": 0},
            ],
        )
        mock_run = MagicMock(return_value=mock_result)

        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode(0.7)

        evolver = SkyDiscoverAdaEvolve(
            adapter=adapter,
            tasks=["math-1"],
            agent_state_factory=make_factory(),
            iterations=1,
        )

        h = Harness(
            system_prompts={"math": "Solve math problems."},
            evolver=evolver,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            h.forward_backward(Datum(episodes=[make_episode(-0.5)]))
            result = h.optim_step().result()

        assert result.status == "ok"
        assert result.updates_applied >= 1

        # The evolved entry should be in the playbook
        contents = [e.content for e in h.playbook.entries]
        assert "Show work step by step." in contents

    def test_evolved_prompt_creates_candidate(self, tmp_path: Path) -> None:
        """Evolved system prompt surfaces as a PromptCandidate in pareto fronts."""
        mock_result = _make_discovery_result_with_new_entry(
            tmp_path,
            original_prompt="Solve math problems.",
            evolved_prompt="You are an expert math tutor. Show all work.",
        )
        mock_run = MagicMock(return_value=mock_result)

        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode(0.7)

        evolver = SkyDiscoverAdaEvolve(
            adapter=adapter,
            tasks=["math-1"],
            agent_state_factory=make_factory(),
            iterations=1,
        )

        h = Harness(
            system_prompts={"math": "Solve math problems."},
            evolver=evolver,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            h.forward_backward(Datum(episodes=[make_episode(-0.5)]))
            result = h.optim_step().result()

        assert result.status == "ok"
        # The evolved prompt should be in pareto fronts for "math"
        assert "math" in h.pareto_fronts
        candidates = h.pareto_fronts["math"].candidates
        assert any("expert math tutor" in c.text for c in candidates)

    def test_no_evolution_changes_is_noop(self, tmp_path: Path) -> None:
        """When evolution produces no diff, optim_step applies only reward signals."""
        mock_result = _make_discovery_result_with_new_entry(
            tmp_path,
            original_prompt="Solve math problems.",
            # Include the existing entry so the diff is empty
            new_entries=[
                {"content": "Check your work.", "tags": [], "helpful": 0, "harmful": 0},
            ],
        )
        mock_run = MagicMock(return_value=mock_result)

        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode(0.7)

        evolver = SkyDiscoverAdaEvolve(
            adapter=adapter,
            tasks=["math-1"],
            agent_state_factory=make_factory(),
            iterations=1,
        )

        h = Harness(
            system_prompts={"math": "Solve math problems."},
            evolver=evolver,
        )
        h.playbook.add(PlaybookEntry(id="p-1", content="Check your work."))

        with patch(_PATCH_TARGET, return_value=mock_run):
            h.forward_backward(Datum(episodes=[make_episode(0.8)]))
            result = h.optim_step().result()

        assert result.status == "ok"
        # Playbook unchanged (only signal update, no new entries)
        assert len(h.playbook.entries) == 1
        assert h.playbook.entries[0].content == "Check your work."

    def test_full_cycle_forward_backward_optim_twice(self, tmp_path: Path) -> None:
        """Two forward_backward + optim_step cycles — playbook grows."""
        # First evolution: add entry A
        result_1 = _make_discovery_result_with_new_entry(
            tmp_path,
            new_entries=[
                {"content": "Verify units in physics problems.", "tags": ["physics"], "helpful": 0, "harmful": 0},
            ],
        )
        # Second evolution: add entry B
        result_2 = _make_discovery_result_with_new_entry(
            tmp_path,
            new_entries=[
                {"content": "Verify units in physics problems.", "tags": ["physics"], "helpful": 0, "harmful": 0},
                {"content": "Draw diagrams for geometry.", "tags": ["geometry"], "helpful": 0, "harmful": 0},
            ],
        )
        mock_run = MagicMock(side_effect=[result_1, result_2])

        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode(0.6)

        evolver = SkyDiscoverAdaEvolve(
            adapter=adapter,
            tasks=["math-1"],
            agent_state_factory=make_factory(),
            iterations=1,
        )

        h = Harness(
            system_prompts={"math": "Solve math problems."},
            evolver=evolver,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            # Cycle 1
            h.forward_backward(Datum(episodes=[make_episode(-0.3)]))
            h.optim_step()

            assert len(h.playbook.entries) == 1

            # Cycle 2
            h.forward_backward(Datum(episodes=[make_episode(-0.2)]))
            h.optim_step()

            assert len(h.playbook.entries) == 2
            contents = {e.content for e in h.playbook.entries}
            assert "Verify units in physics problems." in contents
            assert "Draw diagrams for geometry." in contents


class TestHarnessWithCloudAdaEvolve:
    """Integration: CloudAdaEvolve returns run_id, Harness sees status=running."""

    def test_cloud_evolve_returns_run_id_in_metrics(self, tmp_path: Path) -> None:
        """CloudAdaEvolve.evolve returns run_id, Harness includes it in FBResult."""
        import time
        import threading

        gate = threading.Event()

        def gated_discovery(**kwargs: Any) -> MagicMock:
            gate.wait(timeout=5)
            result = _make_discovery_result_with_new_entry(tmp_path)
            return result

        mock_run = MagicMock(side_effect=gated_discovery)

        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode(0.7)

        evolver = CloudAdaEvolve(
            adapter=adapter,
            tasks=["math-1"],
            agent_state_factory=make_factory(),
            iterations=1,
        )

        h = Harness(
            system_prompts={"math": "Solve math problems."},
            evolver=evolver,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            fb = h.forward_backward(Datum(episodes=[make_episode(-0.3)])).result()

        assert fb.status == "ok"
        # Cloud evolver returns run_id — Harness should propagate it
        run_id = fb.metrics.get("run_id", "")
        assert run_id.startswith("sky-")
        assert fb.metrics.get("backend") == "skydiscover_adaevolve_cloud"

        # No insights yet (async — still running)
        assert fb.metrics.get("insights_generated", 0) == 0

        # Cleanup
        gate.set()
        evolver.cleanup(timeout=2)
