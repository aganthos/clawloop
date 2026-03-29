"""Tests for SkyDiscoverAdaEvolve backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from enterprise.tests.conftest import make_context, make_episode, make_snapshot
from enterprise.evolution.backends.skydiscover_adaevolve import (
    SkyDiscoverAdaEvolve,
    _build_config,
    _extract_best_program_path,
)

_PATCH_TARGET = (
    "enterprise.evolution.backends.skydiscover_adaevolve"
    ".SkyDiscoverAdaEvolve._get_run_discovery"
)


def _make_discovery_result(
    evolved_program_path: str,
    best_score: float = 0.8,
) -> MagicMock:
    """Create a mock DiscoveryResult matching the real API shape."""
    result = MagicMock()
    # Real DiscoveryResult has best_solution (str), best_program (Program obj),
    # best_score, metrics, output_dir, initial_score
    result.best_solution = Path(evolved_program_path).read_text()
    result.best_program = MagicMock()
    result.best_program.code = Path(evolved_program_path).read_text()
    result.best_score = best_score
    result.metrics = {"total_tokens": 1500, "iterations_completed": 5}
    result.output_dir = str(Path(evolved_program_path).parent)
    result.initial_score = 0.3
    return result


class TestBuildConfig:
    def test_returns_dict_without_skydiscover(self) -> None:
        """Without skydiscover installed, returns a dict fallback."""
        cfg = _build_config(num_islands=3, population_size=30)
        assert cfg["num_islands"] == 3
        assert cfg["population_size"] == 30
        assert cfg["search"] == "adaevolve"


class TestExtractBestProgramPath:
    def test_extracts_from_best_solution_json(self, tmp_path: Path) -> None:
        """best_solution that is valid JSON gets written to file."""
        program = {"system_prompt": "Evolved!", "playbook": [], "model": ""}
        result = MagicMock()
        result.best_solution = json.dumps(program)

        path = _extract_best_program_path(result, str(tmp_path))
        data = json.loads(Path(path).read_text())
        assert data["system_prompt"] == "Evolved!"

    def test_extracts_from_best_program_code(self, tmp_path: Path) -> None:
        """Falls back to best_program.code if best_solution is None."""
        program = {"system_prompt": "From code", "playbook": []}
        result = MagicMock()
        result.best_solution = None
        result.best_program.code = json.dumps(program)

        path = _extract_best_program_path(result, str(tmp_path))
        data = json.loads(Path(path).read_text())
        assert data["system_prompt"] == "From code"

    def test_wraps_plain_text_as_prompt(self, tmp_path: Path) -> None:
        """Non-JSON text gets wrapped as a system_prompt."""
        result = MagicMock()
        result.best_solution = "You are an expert agent."

        path = _extract_best_program_path(result, str(tmp_path))
        data = json.loads(Path(path).read_text())
        assert data["system_prompt"] == "You are an expert agent."
        assert data["playbook"] == []

    def test_returns_existing_file_path(self, tmp_path: Path) -> None:
        """If best_solution is already a file path, return it."""
        prog_file = tmp_path / "existing.json"
        prog_file.write_text(json.dumps({"system_prompt": "test"}))

        result = MagicMock()
        result.best_solution = str(prog_file)

        path = _extract_best_program_path(result, str(tmp_path / "out"))
        assert path == str(prog_file)

    def test_raises_on_no_solution(self, tmp_path: Path) -> None:
        result = MagicMock()
        result.best_solution = None
        result.best_program = None

        with pytest.raises(ValueError, match="no best_solution"):
            _extract_best_program_path(result, str(tmp_path))


class TestSkyDiscoverAdaEvolve:
    def test_evolve_returns_evolver_result(self, tmp_path: Path) -> None:
        """End-to-end: mock run_discovery, verify EvolverResult shape."""
        snap = make_snapshot()
        ctx = make_context()
        episodes = [make_episode()]

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
        adapter.run_episode.return_value = make_episode(0.8)
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

        # Verify run_discovery called with correct API params
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["search"] == "adaevolve"
        assert call_kwargs.kwargs["iterations"] == 5
        assert "config" in call_kwargs.kwargs  # Config object, not raw params
        assert "num_islands" not in call_kwargs.kwargs  # NOT direct kwarg
        assert "population_size" not in call_kwargs.kwargs  # NOT direct kwarg

        # Verify result shape
        assert result.provenance.backend == "skydiscover_adaevolve"
        assert result.provenance.tokens_used == 1500

        # Should have 1 add insight (new entry) + 1 prompt candidate
        add_insights = [i for i in result.insights if i.action == "add"]
        assert len(add_insights) == 1
        assert add_insights[0].content == "Think step by step."

        assert "default" in result.candidates
        assert result.candidates["default"][0].text == "You are an expert problem-solver."

    def test_evolve_no_changes(self, tmp_path: Path) -> None:
        """When evolution produces no changes, result should be empty."""
        snap = make_snapshot()

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
        adapter.run_episode.return_value = make_episode()
        factory = MagicMock(return_value=MagicMock())

        backend = SkyDiscoverAdaEvolve(
            adapter=adapter,
            tasks=["task-a"],
            agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            result = backend.evolve([make_episode()], snap, make_context())

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
        snap = make_snapshot()

        evolved_path = tmp_path / "evolved.json"
        evolved_path.write_text(json.dumps({
            "system_prompt": "You are a helpful agent.",
            "playbook": [
                {"content": "Always verify inputs.", "tags": ["safety"], "helpful": 5, "harmful": 1},
            ],
        }))

        mock_result = _make_discovery_result(str(evolved_path))
        seed_contents: list[dict[str, Any]] = []

        def capture_seed(**kwargs: Any) -> MagicMock:
            seed_data = json.loads(Path(kwargs["initial_program"]).read_text())
            seed_contents.append(seed_data)
            return mock_result

        mock_run = MagicMock(side_effect=capture_seed)

        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode()
        factory = MagicMock(return_value=MagicMock())

        backend = SkyDiscoverAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            backend.evolve([make_episode()], snap, make_context())

        assert len(seed_contents) == 1
        assert seed_contents[0]["system_prompt"] == "You are a helpful agent."

    def test_cleanup_removes_work_dir(self) -> None:
        """cleanup() should remove the temp directory."""
        adapter = MagicMock()
        factory = MagicMock()
        backend = SkyDiscoverAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )
        work_dir = Path(backend._work_dir)
        assert work_dir.exists()
        backend.cleanup()
        assert not work_dir.exists()
        backend.cleanup()  # idempotent

    def test_run_dir_cleaned_after_evolve(self, tmp_path: Path) -> None:
        """Per-run temp dir should be cleaned up after evolve completes."""
        snap = make_snapshot()

        evolved_path = tmp_path / "evolved.json"
        evolved_path.write_text(json.dumps({
            "system_prompt": "You are a helpful agent.",
            "playbook": [
                {"content": "Always verify inputs.", "tags": ["safety"], "helpful": 5, "harmful": 1},
            ],
        }))

        mock_result = _make_discovery_result(str(evolved_path))
        run_dirs: list[str] = []

        def capture_run_dir(**kwargs: Any) -> MagicMock:
            seed_path = Path(kwargs["initial_program"])
            run_dirs.append(str(seed_path.parent))
            assert seed_path.exists(), "Seed should exist during run_discovery"
            return mock_result

        mock_run = MagicMock(side_effect=capture_run_dir)

        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode()
        factory = MagicMock(return_value=MagicMock())

        backend = SkyDiscoverAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            backend.evolve([make_episode()], snap, make_context())

        assert len(run_dirs) == 1
        assert not Path(run_dirs[0]).exists(), "Run dir should be cleaned up"

    def test_output_dir_passed_to_run_discovery(self, tmp_path: Path) -> None:
        """run_discovery should receive output_dir for SkyDiscover to write to."""
        snap = make_snapshot()

        evolved_path = tmp_path / "evolved.json"
        evolved_path.write_text(json.dumps({
            "system_prompt": "You are a helpful agent.",
            "playbook": [
                {"content": "Always verify inputs.", "tags": ["safety"], "helpful": 1, "harmful": 0},
            ],
        }))

        mock_result = _make_discovery_result(str(evolved_path))
        mock_run = MagicMock(return_value=mock_result)

        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode()
        factory = MagicMock(return_value=MagicMock())

        backend = SkyDiscoverAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            backend.evolve([make_episode()], snap, make_context())

        call_kwargs = mock_run.call_args.kwargs
        assert "output_dir" in call_kwargs
        assert call_kwargs["output_dir"] is not None

    def test_metrics_extracted_from_result(self, tmp_path: Path) -> None:
        """Provenance should pull token count from result.metrics."""
        snap = make_snapshot()

        evolved_path = tmp_path / "evolved.json"
        evolved_path.write_text(json.dumps({
            "system_prompt": "You are a helpful agent.",
            "playbook": [
                {"content": "Always verify inputs.", "tags": ["safety"], "helpful": 1, "harmful": 0},
            ],
        }))

        mock_result = _make_discovery_result(str(evolved_path))
        mock_result.metrics = {"total_tokens": 42000}
        mock_run = MagicMock(return_value=mock_result)

        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode()
        factory = MagicMock(return_value=MagicMock())

        backend = SkyDiscoverAdaEvolve(
            adapter=adapter, tasks=["t"], agent_state_factory=factory,
        )

        with patch(_PATCH_TARGET, return_value=mock_run):
            result = backend.evolve([make_episode()], snap, make_context())

        assert result.provenance.tokens_used == 42000
